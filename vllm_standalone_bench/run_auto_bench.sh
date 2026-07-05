#!/usr/bin/env bash
# =============================================================================
# run_auto_bench.sh — 离线 vLLM / SGLang 自动化压测启动脚本
#
# 使用方法:
#   chmod +x run_auto_bench.sh
#   ./run_auto_bench.sh                         # 默认后台启动 smoke 配置
#   DETACH=false ./run_auto_bench.sh            # 前台运行，便于直接看日志
#   RUN_ID=qwen_smoke_001 ./run_auto_bench.sh   # 指定 run_id，方便后续查询
#   DRY_RUN=true ./run_auto_bench.sh            # 只打印 Docker 命令，不启动容器
#
# 后台任务控制:
#   RUN_ID=qwen_smoke_001 ./run_auto_bench.sh status
#   RUN_ID=qwen_smoke_001 ./run_auto_bench.sh logs
#   RUN_ID=qwen_smoke_001 FOLLOW=true ./run_auto_bench.sh logs
#   RUN_ID=qwen_smoke_001 ./run_auto_bench.sh stop
#   RUN_ID=qwen_smoke_001 ./run_auto_bench.sh cleanup
#   RUN_ID=qwen_smoke_001 ./run_auto_bench.sh resume
#   RUN_ID=qwen_smoke_001 DETACH=false ./run_auto_bench.sh resume
#   RUN_ID=qwen_smoke_001 ./run_auto_bench.sh postprocess
#
# Mooncake (sglang PD HiCache 的 L3 KV 存储后端，可选):
#   sglang PD + hicache(mooncake) 配置需要一个 mooncake_master(内嵌 metadata)
#   在控制机本地跑。run 子命令会按需自动起；也可手动控制:
#   ./run_auto_bench.sh mooncake start    # 本地起 mooncake_master
#   ./run_auto_bench.sh mooncake stop     # 停
#   ./run_auto_bench.sh mooncake status   # 看状态
#   ./run_auto_bench.sh mooncake restart
#
# 说明:
#   该脚本只负责拼装并调用 auto_bench.py。vLLM 镜像、bench-runner 镜像、
#   本地模型目录和 Docker network 策略均以 JSON 配置文件为准。
# =============================================================================

set -euo pipefail

# =============================================================================
# ▌ 一、默认参数
# =============================================================================

# 脚本所在目录（自动识别，支持从任意工作目录执行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 项目根目录；auto_bench 配置里的相对 results_dir 需要从这里解析
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Python 解释器；如主机 Python 命令不同，可通过 PYTHON_BIN 覆盖
PYTHON_BIN="${PYTHON_BIN:-/aipaas_test/miniconda3/bin/python3}"

# auto_bench.py 入口
AUTO_BENCH="${AUTO_BENCH:-${SCRIPT_DIR}/auto_bench.py}"

# 默认 smoke 配置；可通过 CONFIG=/path/to/config.json 覆盖
#CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/auto_bench.vllm_pd_nixl_remote_minimax.json}"
#CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/auto_bench.minimx_compare.json}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/auto_bench.sglang_pd_hicache_remote_minimax_nobase.json}"

# status/logs/stop 使用的结果目录；通常应与配置中的 run.results_dir 一致
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

# 可选：指定 run_id；为空时由 auto_bench.py 根据配置自动生成
RUN_ID="${RUN_ID:-}"

# 默认后台运行，避免 SSH 断开影响测试；需要前台时设为 false
DETACH="${DETACH:-true}"

# 只做 dry-run，不启动 Docker 容器
DRY_RUN="${DRY_RUN:-false}"

# logs 子命令是否持续跟随当前 case 的 bench.log；无当前 case 时回退 controller.log
FOLLOW="${FOLLOW:-false}"

# ── Mooncake (sglang PD HiCache L3 KV 后端) ──
# sglang PD + hicache(mooncake) 需要 mooncake_master(内嵌 metadata) 在控制机本地跑。
# worker 自带 store（设了 MOONCAKE_GLOBAL_SEGMENT_SIZE），所以只起 master 即可。
MOONCAKE_IMAGE="${MOONCAKE_IMAGE:-lmsysorg/sglang:latest-0705}"
MOONCAKE_MASTER_NAME="${MOONCAKE_MASTER_NAME:-mooncake-master}"
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-50051}"
MOONCAKE_META_PORT="${MOONCAKE_META_PORT:-8080}"

# =============================================================================
# ▌ 二、Mooncake 辅助函数
# =============================================================================

# 检查配置是否需要 mooncake（拓扑里有 MOONCAKE_ env 或 hicache.storage_backend=mooncake）
_config_needs_mooncake() {
    [[ -f "${CONFIG}" ]] || return 1
    "${PYTHON_BIN}" - "${CONFIG}" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
for p in d.get("topology_profiles", []) or []:
    env = p.get("env", {}) or {}
    if any(str(k).startswith("MOONCAKE_") for k in env):
        sys.exit(0)
    hc = p.get("sglang_hicache", {}) or {}
    if hc.get("storage_backend") == "mooncake":
        sys.exit(0)
sys.exit(1)
PY
}

_mooncake_running() {
    docker inspect -f '{{.State.Running}}' "${MOONCAKE_MASTER_NAME}" 2>/dev/null | grep -q true
}

# 等待本机端口监听（mooncake_master 用 --network host，端口在控制机本机）
_mooncake_wait_port() {
    local port="$1" tries=0
    while [[ ${tries} -lt 30 ]]; do
        if "${PYTHON_BIN}" -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1',${port}))==0 else 1)" 2>/dev/null; then
            return 0
        fi
        sleep 1
        tries=$((tries + 1))
    done
    return 1
}

_mooncake_start() {
    command -v docker >/dev/null 2>&1 || { echo "[mooncake] [ERROR] 找不到 docker" >&2; return 1; }
    if _mooncake_running; then
        echo "[mooncake] master 已在运行 (${MOONCAKE_MASTER_NAME})"
        return 0
    fi
    echo "[mooncake] 启动 master(内嵌 metadata) — 镜像 ${MOONCAKE_IMAGE}"
    docker run -d --name "${MOONCAKE_MASTER_NAME}" \
        --label vllm_auto_bench.managed=true \
        --network host \
        --entrypoint mooncake_master \
        "${MOONCAKE_IMAGE}" \
        --enable_http_metadata_server=true \
        --http_metadata_server_port="${MOONCAKE_META_PORT}" \
        --eviction_high_watermark_ratio=0.95 >/dev/null
    echo -n "[mooncake] 等待 master(${MOONCAKE_MASTER_PORT})/metadata(${MOONCAKE_META_PORT}) 就绪 ..."
    if _mooncake_wait_port "${MOONCAKE_MASTER_PORT}" && _mooncake_wait_port "${MOONCAKE_META_PORT}"; then
        echo " 就绪"
    else
        echo " 未就绪"
        echo "[mooncake] [WARN] 端口未起来，查看: docker logs ${MOONCAKE_MASTER_NAME}" >&2
        docker logs --tail 20 "${MOONCAKE_MASTER_NAME}" >&2 || true
        return 1
    fi
}

_mooncake_stop() {
    command -v docker >/dev/null 2>&1 || { echo "[mooncake] [ERROR] 找不到 docker" >&2; return 1; }
    if docker rm -f "${MOONCAKE_MASTER_NAME}" >/dev/null 2>&1; then
        echo "[mooncake] master 已停止 (${MOONCAKE_MASTER_NAME})"
    else
        echo "[mooncake] master 未在运行"
    fi
}

_mooncake_status() {
    command -v docker >/dev/null 2>&1 || { echo "[mooncake] [ERROR] 找不到 docker" >&2; return 1; }
    if _mooncake_running; then
        echo "[mooncake] master 运行中 (${MOONCAKE_MASTER_NAME})"
        echo "    master gRPC     : <本机IP>:${MOONCAKE_MASTER_PORT}"
        echo "    metadata server : http://<本机IP>:${MOONCAKE_META_PORT}/metadata"
        docker ps --filter "name=^${MOONCAKE_MASTER_NAME}\$" --format '    容器状态: {{.Status}}'
    else
        echo "[mooncake] master 未运行"
        return 1
    fi
}

# =============================================================================
# ▌ 三、解析子命令
# =============================================================================

COMMAND="run"
if [[ "$#" -gt 0 ]]; then
    case "$1" in
        run|status|logs|stop|cleanup|resume|postprocess)
            COMMAND="$1"
            shift
            ;;
        mooncake)
            COMMAND="mooncake"
            shift
            ;;
    esac
fi

# mooncake 子命令不走 auto_bench.py，直接处理本地 docker
if [[ "${COMMAND}" == "mooncake" ]]; then
    MC_ACTION="${1:-status}"
    case "${MC_ACTION}" in
        start)   _mooncake_start ;;
        stop)    _mooncake_stop ;;
        restart) _mooncake_stop; _mooncake_start ;;
        status)  _mooncake_status ;;
        *) echo "[ERROR] 用法: $0 mooncake {start|stop|restart|status}" >&2; exit 1 ;;
    esac
    exit 0
fi

# =============================================================================
# ▌ 四、前置检查
# =============================================================================

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] 找不到 Python 解释器: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${AUTO_BENCH}" ]]; then
    echo "[ERROR] 找不到 auto_bench.py: ${AUTO_BENCH}" >&2
    exit 1
fi

if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "[ERROR] 找不到项目根目录: ${PROJECT_ROOT}" >&2
    exit 1
fi

if [[ "${COMMAND}" == "run" || "${COMMAND}" == "postprocess" ]] && [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] 找不到自动压测配置: ${CONFIG}" >&2
    exit 1
fi

if [[ "${COMMAND}" != "run" && -z "${RUN_ID}" ]]; then
    echo "[ERROR] ${COMMAND} 需要指定 RUN_ID，例如: RUN_ID=qwen_smoke_001 $0 ${COMMAND}" >&2
    exit 1
fi

# =============================================================================
# ▌ 五、拼装 auto_bench.py 命令
# =============================================================================

case "${COMMAND}" in
    run)
        # sglang PD + mooncake 配置：按需自动起本地 mooncake_master（幂等；dry-run 跳过）
        if [[ "${DRY_RUN}" != "true" ]] && _config_needs_mooncake; then
            _mooncake_start || echo "[mooncake] [WARN] master 未就绪，sglang worker 可能起不来" >&2
            # controller 跑完(成功/中断/失败都走 finally)时自动清掉 mooncake_master，避免遗留
            export AUTO_BENCH_FINAL_LOCAL_CLEANUP="${MOONCAKE_MASTER_NAME}"
        fi

        CMD=("${PYTHON_BIN}" "${AUTO_BENCH}" run --config "${CONFIG}")

        if [[ -n "${RUN_ID}" ]]; then
            CMD+=(--run-id "${RUN_ID}")
        fi

        if [[ "${DETACH}" == "true" ]]; then
            CMD+=(--detach)
        fi

        if [[ "${DRY_RUN}" == "true" ]]; then
            CMD+=(--dry-run)
        fi
        ;;
    status|stop|cleanup)
        CMD=("${PYTHON_BIN}" "${AUTO_BENCH}" "${COMMAND}" --results-dir "${RESULTS_DIR}" --run-id "${RUN_ID}")
        ;;
    resume)
        CMD=("${PYTHON_BIN}" "${AUTO_BENCH}" resume --results-dir "${RESULTS_DIR}" --run-id "${RUN_ID}")
        if [[ "${DETACH}" == "true" ]]; then
            CMD+=(--detach)
        fi
        ;;
    postprocess)
        CMD=("${PYTHON_BIN}" "${AUTO_BENCH}" postprocess --config "${CONFIG}" --results-dir "${RESULTS_DIR}" --run-id "${RUN_ID}" --container)
        ;;
    logs)
        CMD=("${PYTHON_BIN}" "${AUTO_BENCH}" logs --results-dir "${RESULTS_DIR}" --run-id "${RUN_ID}")
        if [[ "${FOLLOW}" == "true" ]]; then
            CMD+=(--follow)
        fi
        ;;
    *)
        echo "[ERROR] 不支持的子命令: ${COMMAND}" >&2
        exit 1
        ;;
esac

# 额外参数原样透传给 auto_bench.py，便于临时追加 --dry-run 等选项
if [[ "$#" -gt 0 ]]; then
    CMD+=("$@")
fi

# =============================================================================
# ▌ 六、打印摘要并执行
# =============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                离线自动化压测控制器                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  子命令    : %-48s║\n" "${COMMAND}"
if [[ "${COMMAND}" == "run" || "${COMMAND}" == "postprocess" ]]; then
    printf "║  配置文件  : %-48s║\n" "${CONFIG}"
fi
if [[ "${COMMAND}" == "run" ]]; then
    printf "║  工作目录  : %-48s║\n" "${PROJECT_ROOT}"
    printf "║  后台运行  : %-48s║\n" "${DETACH}"
    printf "║  Dry-run   : %-48s║\n" "${DRY_RUN}"
else
    printf "║  结果目录  : %-48s║\n" "${RESULTS_DIR}"
    if [[ "${COMMAND}" == "resume" ]]; then
        printf "║  后台运行  : %-48s║\n" "${DETACH}"
    fi
fi
printf "║  Run ID    : %-48s║\n" "${RUN_ID:-自动生成}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "执行命令:"
echo "  ${CMD[*]}"
echo ""

cd "${PROJECT_ROOT}"
# stop/cleanup: 跑完顺带停 mooncake_master；其余子命令照常 exec
if [[ "${COMMAND}" == "stop" || "${COMMAND}" == "cleanup" ]]; then
    "${CMD[@]}" || true
    _mooncake_stop || true
else
    exec "${CMD[@]}"
fi
