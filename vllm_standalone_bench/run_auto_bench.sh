#!/usr/bin/env bash
# =============================================================================
# run_auto_bench.sh — 离线 vLLM 自动化压测启动脚本
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
#   RUN_ID=qwen_smoke_001 ./run_auto_bench.sh resume
#   RUN_ID=qwen_smoke_001 DETACH=false ./run_auto_bench.sh resume
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
PYTHON_BIN="${PYTHON_BIN:-python3}"

# auto_bench.py 入口
AUTO_BENCH="${AUTO_BENCH:-${SCRIPT_DIR}/auto_bench.py}"

# 默认 smoke 配置；可通过 CONFIG=/path/to/config.json 覆盖
CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/auto_bench.qwen2_5_1_5b.smoke.json}"

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

# =============================================================================
# ▌ 二、解析子命令
# =============================================================================

COMMAND="run"
if [[ "$#" -gt 0 ]]; then
    case "$1" in
        run|status|logs|stop|resume)
            COMMAND="$1"
            shift
            ;;
    esac
fi

# =============================================================================
# ▌ 三、前置检查
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

if [[ "${COMMAND}" == "run" && ! -f "${CONFIG}" ]]; then
    echo "[ERROR] 找不到自动压测配置: ${CONFIG}" >&2
    exit 1
fi

if [[ "${COMMAND}" != "run" && -z "${RUN_ID}" ]]; then
    echo "[ERROR] ${COMMAND} 需要指定 RUN_ID，例如: RUN_ID=qwen_smoke_001 $0 ${COMMAND}" >&2
    exit 1
fi

# =============================================================================
# ▌ 四、拼装 auto_bench.py 命令
# =============================================================================

case "${COMMAND}" in
    run)
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
    status|stop)
        CMD=("${PYTHON_BIN}" "${AUTO_BENCH}" "${COMMAND}" --results-dir "${RESULTS_DIR}" --run-id "${RUN_ID}")
        ;;
    resume)
        CMD=("${PYTHON_BIN}" "${AUTO_BENCH}" resume --results-dir "${RESULTS_DIR}" --run-id "${RUN_ID}")
        if [[ "${DETACH}" == "true" ]]; then
            CMD+=(--detach)
        fi
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
# ▌ 五、打印摘要并执行
# =============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                vLLM 离线自动化压测控制器                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  子命令    : %-48s║\n" "${COMMAND}"
if [[ "${COMMAND}" == "run" ]]; then
    printf "║  配置文件  : %-48s║\n" "${CONFIG}"
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
exec "${CMD[@]}"
