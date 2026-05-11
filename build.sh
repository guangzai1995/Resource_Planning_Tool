#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# 资源规划工具 — 一键构建 & 部署脚本
# 用法:
#   bash build.sh                          # 仅构建镜像 rpt:latest
#   bash build.sh --run                    # 构建并启动容器
#   bash build.sh --run-only               # 不构建，直接启动（使用已有镜像）
#   bash build.sh --stop                   # 停止并删除容器
#   bash build.sh --logs                   # 跟踪容器日志
#   bash build.sh -t myrepo/rpt:1.0.0 --run --push  # 构建+启动+推送
#   bash build.sh --no-cache --run         # 无缓存构建并启动
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 默认配置 ─────────────────────────────────────────────────
IMAGE_NAME="rpt"
IMAGE_TAG="latest"
CONTAINER_NAME="rpt"
HOST_PORT="8000"
CONTAINER_PORT="8000"
PUSH=false
RUN_AFTER_BUILD=false
RUN_ONLY=false
NO_CACHE=false
ACTION_STOP=false
ACTION_LOGS=false

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
step()    { echo -e "${CYAN}[STEP]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 参数解析 ──────────────────────────────────────────────────
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
        --push)
            PUSH=true
            shift
            ;;
        --run)
            RUN_AFTER_BUILD=true
            shift
            ;;
        --run-only)
            RUN_ONLY=true
            shift
            ;;
        --stop)
            ACTION_STOP=true
            shift
            ;;
        --logs)
            ACTION_LOGS=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
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

# ── 前置检查 ──────────────────────────────────────────────────
command -v docker &>/dev/null || error "未找到 docker 命令，请先安装 Docker。"
cd "$SCRIPT_DIR"

# ── 工具函数 ──────────────────────────────────────────────────

# 停止并删除容器
_stop_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            info "停止容器: ${CONTAINER_NAME}"
            docker stop "${CONTAINER_NAME}" >/dev/null
        fi
        info "删除容器: ${CONTAINER_NAME}"
        docker rm "${CONTAINER_NAME}" >/dev/null
    else
        warn "容器 ${CONTAINER_NAME} 不存在，跳过。"
    fi
}

# 确保宿主机数据目录存在
_prepare_volumes() {
    local data_dir="${SCRIPT_DIR}/backend/data"
    local model_dir="${SCRIPT_DIR}/model"
    [[ -d "$data_dir" ]] || { info "创建数据目录: $data_dir"; mkdir -p "$data_dir"; }
    [[ -d "$model_dir" ]] || { warn "模型目录不存在: $model_dir（分词器功能不可用）"; }
}

# 启动容器
_run_container() {
    _prepare_volumes

    local data_dir="${SCRIPT_DIR}/backend/data"
    local model_dir="${SCRIPT_DIR}/model"

    # 如已有同名容器先清理
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        warn "发现已有容器 ${CONTAINER_NAME}，将先停止并移除..."
        _stop_container
    fi

    step "启动容器: ${CONTAINER_NAME}  (${FULL_IMAGE}  →  0.0.0.0:${HOST_PORT})"

    local run_args=(
        -d
        --name "${CONTAINER_NAME}"
        -p "${HOST_PORT}:${CONTAINER_PORT}"
        -v "${data_dir}:/data"
        -e "SQLITE_PATH=/data/rpt.db"
        --restart unless-stopped
    )

    # 仅当模型目录存在时才挂载
    if [[ -d "$model_dir" ]]; then
        run_args+=(-v "${model_dir}:/model")
    fi

    # 若 Excel 文件存在于 data 目录则自动指定路径
    if compgen -G "${data_dir}/*.xlsx" >/dev/null 2>&1; then
        local xlsx_file
        xlsx_file="$(ls "${data_dir}"/*.xlsx | head -1 | xargs basename)"
        run_args+=(-e "EXCEL_DATA_PATH=/data/${xlsx_file}")
        info "检测到 Excel 数据文件: ${xlsx_file}"
    fi

    run_args+=("${FULL_IMAGE}")

    docker run "${run_args[@]}"
    info "容器已启动，等待服务就绪..."

    _wait_healthy
}

# 等待健康检查通过
_wait_healthy() {
    local max_wait=60
    local interval=3
    local elapsed=0
    local url="http://localhost:${HOST_PORT}/healthz"

    while (( elapsed < max_wait )); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            echo ""
            info "服务就绪 ✓  ${url}"
            _print_status
            return 0
        fi
        printf "."
        sleep "${interval}"
        (( elapsed += interval ))
    done

    echo ""
    warn "服务在 ${max_wait}s 内未就绪，请检查日志："
    warn "  docker logs ${CONTAINER_NAME}"
}

# 打印当前运行状态
_print_status() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  服务信息${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo "  容器名称  :  ${CONTAINER_NAME}"
    echo "  镜像      :  ${FULL_IMAGE}"
    echo "  访问地址  :  http://localhost:${HOST_PORT}"
    echo "  健康检查  :  http://localhost:${HOST_PORT}/healthz"
    echo "  数据目录  :  ${SCRIPT_DIR}/backend/data  →  /data"
    echo ""
    echo "  docker logs -f ${CONTAINER_NAME}       # 查看日志"
    echo "  docker exec -it ${CONTAINER_NAME} sh   # 进入容器"
    echo "  bash build.sh --stop                   # 停止服务"
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo ""
}

# ── 动作：--stop ──────────────────────────────────────────────
if [[ "$ACTION_STOP" == "true" ]]; then
    _stop_container
    info "完成。"
    exit 0
fi

# ── 动作：--logs ──────────────────────────────────────────────
if [[ "$ACTION_LOGS" == "true" ]]; then
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$" \
        || error "容器 ${CONTAINER_NAME} 未在运行。"
    exec docker logs -f "${CONTAINER_NAME}"
fi

# ── 动作：--run-only（不构建直接启动） ───────────────────────
if [[ "$RUN_ONLY" == "true" ]]; then
    docker image inspect "${FULL_IMAGE}" >/dev/null 2>&1 \
        || error "本地不存在镜像 ${FULL_IMAGE}，请先构建。"
    _run_container
    exit 0
fi

# ── 构建 ──────────────────────────────────────────────────────
step "开始构建镜像: ${FULL_IMAGE}"
echo ""

BUILD_START=$(date +%s)

BUILD_FLAGS=()
[[ "$NO_CACHE" == "true" ]] && BUILD_FLAGS+=(--no-cache)

docker build \
    --progress=plain \
    "${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"}" \
    -t "${FULL_IMAGE}" \
    -f Dockerfile \
    .

BUILD_END=$(date +%s)
ELAPSED=$(( BUILD_END - BUILD_START ))

echo ""
info "构建完成 (耗时 ${ELAPSED}s): ${FULL_IMAGE}"

# ── 可选：推送 ────────────────────────────────────────────────
if [[ "$PUSH" == "true" ]]; then
    step "推送镜像: ${FULL_IMAGE}"
    docker push "${FULL_IMAGE}"
    info "推送完成: ${FULL_IMAGE}"
fi

# ── 可选：启动服务 ────────────────────────────────────────────
if [[ "$RUN_AFTER_BUILD" == "true" ]]; then
    _run_container
else
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  下一步${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  # 启动服务（持久化数据）"
    echo "  bash build.sh --run-only"
    echo ""
    echo "  # 或直接运行 docker 命令"
    echo "  docker run -d \\"
    echo "    -p ${HOST_PORT}:${CONTAINER_PORT} \\"
    echo "    -v \$(pwd)/backend/data:/data \\"
    echo "    -v \$(pwd)/model:/model \\"
    echo "    --name ${CONTAINER_NAME} \\"
    echo "    --restart unless-stopped \\"
    echo "    ${FULL_IMAGE}"
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo ""
fi
