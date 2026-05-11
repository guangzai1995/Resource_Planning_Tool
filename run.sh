#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 资源规划工具 — 容器管理脚本（启动 / 停止 / 重启 / 日志 / 状态）
# 用法:
#   bash run.sh                    # 启动容器（使用默认镜像 rpt:latest）
#   bash run.sh -t myrepo/rpt:1.0.0   # 指定镜像启动
#   bash run.sh -p 9000            # 指定宿主机端口（默认 8000）
#   bash run.sh stop               # 停止并删除容器
#   bash run.sh restart            # 重启容器
#   bash run.sh logs               # 跟踪容器日志
#   bash run.sh status             # 查看容器状态
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 默认配置 ─────────────────────────────────────────────────
IMAGE_NAME="rpt"
IMAGE_TAG="latest"
CONTAINER_NAME="rpt"
HOST_PORT="8000"
CONTAINER_PORT="8000"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 子命令 & 参数解析 ─────────────────────────────────────────
SUBCMD="start"
[[ $# -gt 0 && "$1" =~ ^(start|stop|restart|logs|status)$ ]] && { SUBCMD="$1"; shift; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tag)
            FULL_TAG="$2"
            IMAGE_NAME="${FULL_TAG%%:*}"
            IMAGE_TAG="${FULL_TAG##*:}"
            [[ "$IMAGE_NAME" == "$IMAGE_TAG" ]] && IMAGE_TAG="latest"
            shift 2
            ;;
        -p|--port)
            HOST_PORT="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '3,11p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            error "未知参数: $1"
            ;;
    esac
done

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v docker &>/dev/null || error "未找到 docker 命令，请先安装 Docker。"

# ── 工具函数 ──────────────────────────────────────────────────

_is_running() {
    docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"
}

_exists() {
    docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"
}

_stop() {
    if _exists; then
        if _is_running; then
            info "停止容器: ${CONTAINER_NAME}"
            docker stop "${CONTAINER_NAME}" >/dev/null
        fi
        info "删除容器: ${CONTAINER_NAME}"
        docker rm "${CONTAINER_NAME}" >/dev/null
        info "完成。"
    else
        warn "容器 ${CONTAINER_NAME} 不存在。"
    fi
}

_wait_healthy() {
    local max_wait=60
    local interval=3
    local elapsed=0
    local url="http://localhost:${HOST_PORT}/healthz"

    info "等待服务就绪  ${url}"
    while (( elapsed < max_wait )); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            echo ""
            info "服务就绪 ✓"
            return 0
        fi
        printf "."
        sleep "${interval}"
        (( elapsed += interval ))
    done
    echo ""
    warn "服务在 ${max_wait}s 内未就绪，请查看日志："
    warn "  bash run.sh logs"
}

_print_status() {
    local state
    if _is_running; then
        state="${GREEN}运行中${NC}"
    elif _exists; then
        state="${YELLOW}已停止${NC}"
    else
        state="${RED}不存在${NC}"
    fi

    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  容器状态${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo -e "  状态      :  ${state}"
    echo    "  容器名称  :  ${CONTAINER_NAME}"
    echo    "  镜像      :  ${FULL_IMAGE}"
    echo    "  访问地址  :  http://localhost:${HOST_PORT}"
    echo    "  健康检查  :  http://localhost:${HOST_PORT}/healthz"
    echo    "  数据目录  :  ${SCRIPT_DIR}/backend/data  →  /data"
    echo ""
    echo    "  bash run.sh logs     # 查看日志"
    echo    "  bash run.sh stop     # 停止服务"
    echo    "  bash run.sh restart  # 重启服务"
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo ""
}

_start() {
    # 检查镜像是否存在
    docker image inspect "${FULL_IMAGE}" >/dev/null 2>&1 \
        || error "本地不存在镜像 ${FULL_IMAGE}，请先执行: bash build.sh -t ${FULL_IMAGE}"

    # 如已有同名容器先清理
    if _exists; then
        warn "发现已有容器 ${CONTAINER_NAME}，先停止并移除..."
        _stop
    fi

    # 准备宿主机目录
    local data_dir="${SCRIPT_DIR}/backend/data"
    local model_dir="${SCRIPT_DIR}/model"
    [[ -d "$data_dir" ]] || { info "创建数据目录: $data_dir"; mkdir -p "$data_dir"; }
    [[ -d "$model_dir" ]] || warn "模型目录不存在: $model_dir（分词器功能不可用）"

    step "启动容器: ${CONTAINER_NAME}  (${FULL_IMAGE}  →  0.0.0.0:${HOST_PORT})"

    local run_args=(
        -d
        --name "${CONTAINER_NAME}"
        -p "${HOST_PORT}:${CONTAINER_PORT}"
        -v "${data_dir}:/data"
        -e "SQLITE_PATH=/data/rpt.db"
        --restart unless-stopped
    )

    [[ -d "$model_dir" ]] && run_args+=(-v "${model_dir}:/model")

    # 自动探测 Excel 文件
    if compgen -G "${data_dir}/*.xlsx" >/dev/null 2>&1; then
        local xlsx_file
        xlsx_file="$(ls "${data_dir}"/*.xlsx | head -1 | xargs basename)"
        run_args+=(-e "EXCEL_DATA_PATH=/data/${xlsx_file}")
        info "检测到 Excel 数据文件: ${xlsx_file}"
    else
        warn "未在 ${data_dir} 找到 .xlsx 文件，启动后数据库将为空。"
        warn "请将数据文件放入该目录后重启容器，例如："
        warn "  cp 资源规划工具.xlsx ${data_dir}/"
        warn "  bash run.sh restart"
    fi

    run_args+=("${FULL_IMAGE}")
    docker run "${run_args[@]}"

    _wait_healthy
    _print_status
}

# ── 子命令分发 ────────────────────────────────────────────────
case "$SUBCMD" in
    start)   _start ;;
    stop)    _stop ;;
    restart) _stop; _start ;;
    logs)
        _is_running || error "容器 ${CONTAINER_NAME} 未在运行。"
        exec docker logs -f "${CONTAINER_NAME}"
        ;;
    status)  _print_status ;;
esac
