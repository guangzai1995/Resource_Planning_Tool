#!/usr/bin/env bash
# =============================================================================
# run_asr_bench.sh — 使用内置 LibriSpeech 数据集做 ASR 接口压测
#
# 使用方法:
#   ./run_asr_bench.sh
#   PARALLEL_NUMS="1 4 8 16" EPOCHS=8 ./run_asr_bench.sh
#
# 说明:
#   这是非自动化压测脚本：不启动 vLLM、不拉起 Docker、不管理服务生命周期。
#   它只假设 ASR 服务已可访问，然后调用 run_bench_multi.py 的 openai-audio 后端。
# =============================================================================

set -euo pipefail

# =============================================================================
# ▌ 一、服务连接配置
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BENCH_SCRIPT="${BENCH_SCRIPT:-${SCRIPT_DIR}/run_bench_multi.py}"

# 完整 ASR transcription 接口；默认来自 scripts/asr.sh
API_URL="${API_URL:-http://10.86.0.32:13001/v1/audio/transcriptions}"

# run_bench_multi.py 需要 /v1 级别 base-url，脚本从完整接口自动推导。
BASE_URL="${BASE_URL:-}"
if [[ -z "${BASE_URL}" ]]; then
    case "${API_URL}" in
        */audio/transcriptions)
            BASE_URL="${API_URL%/audio/transcriptions}"
            ;;
        *)
            echo "[ERROR] API_URL 必须以 /audio/transcriptions 结尾，或显式设置 BASE_URL" >&2
            exit 1
            ;;
    esac
fi

# 是否跳过 HTTPS 证书验证
INSECURE="${INSECURE:-false}"

# 模型字段需与服务端 /v1/audio/transcriptions 接口接受的 model 一致
MODEL="${MODEL:-Qwen3-ASR-1_7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

# 服务端开启鉴权时填写；未开启保持空
API_KEY="${API_KEY:-}"

# =============================================================================
# ▌ 二、内置 ASR 数据集
# =============================================================================

DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/assets/librispeech_test_clean_256/asr_smoke.jsonl}"
DATASET_NAME="${DATASET_NAME:-custom_audio}"
LANGUAGE="${LANGUAGE:-en}"

# 可选：运行时把内置短音频拼接成指定单条音频时长，单位秒。
# 不设置 AUDIO_DURATION_S 时使用数据集原始音频；设置后每个请求生成独立音频文件，
# 通过变换源片段顺序减少重复音频/前缀命中缓存对压测的影响。
AUDIO_DURATION_S="${AUDIO_DURATION_S:-}"
AUDIO_SILENCE_MS="${AUDIO_SILENCE_MS:-500}"
GENERATED_AUDIO_DIR="${GENERATED_AUDIO_DIR:-}"

# =============================================================================
# ▌ 三、压测维度
# =============================================================================

# ASR backend 会忽略 input-lens，保留 0 便于日志清晰。
INPUT_LENS="${INPUT_LENS:-0}"
OUTPUT_LENS="${OUTPUT_LENS:-128}"
PARALLEL_NUMS="${PARALLEL_NUMS:-1 4 8}"
EPOCHS="${EPOCHS:-4}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-2.0}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_OUTPUT_LEN="${WARMUP_OUTPUT_LEN:-32}"
MAX_TTFT_MS="${MAX_TTFT_MS:-10000}"

SEED="${SEED:-0}"
VARY_SEED_BY_CONFIG="${VARY_SEED_BY_CONFIG:-true}"

# =============================================================================
# ▌ 四、输出文件
# =============================================================================

RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_CSV="${OUTPUT_CSV:-${RESULT_DIR}/asr_bench_${TIMESTAMP}.csv}"
OUTPUT_XLSX="${OUTPUT_XLSX:-${RESULT_DIR}/asr_bench_${TIMESTAMP}.xlsx}"
if [[ -n "${AUDIO_DURATION_S}" && -z "${GENERATED_AUDIO_DIR}" ]]; then
    GENERATED_AUDIO_DIR="${RESULT_DIR}/asr_dynamic_audio_${TIMESTAMP}"
fi

# 设为 true 只打印命令，不实际发起压测
DRY_RUN="${DRY_RUN:-false}"

# =============================================================================
# ▌ 五、前置检查
# =============================================================================

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] 找不到 Python 解释器: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
    echo "[ERROR] 找不到压测脚本: ${BENCH_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "[ERROR] 找不到 ASR 数据集: ${DATASET_PATH}" >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}"

# =============================================================================
# ▌ 六、拼装并执行 run_bench_multi.py
# =============================================================================

CMD=(
    "${PYTHON_BIN}" "${BENCH_SCRIPT}"
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --backend "openai-audio"
    --dataset-name "${DATASET_NAME}"
    --dataset-path "${DATASET_PATH}"
    --language "${LANGUAGE}"
    --input-lens ${INPUT_LENS}
    --output-lens ${OUTPUT_LENS}
    --parallel-nums ${PARALLEL_NUMS}
    --epochs "${EPOCHS}"
    --sleep-between "${SLEEP_BETWEEN}"
    --warmup-requests "${WARMUP_REQUESTS}"
    --warmup-output-len "${WARMUP_OUTPUT_LEN}"
    --seed "${SEED}"
    --output-csv "${OUTPUT_CSV}"
    --output-xlsx "${OUTPUT_XLSX}"
)

if [[ -n "${SERVED_MODEL_NAME}" ]]; then
    CMD+=(--served-model-name "${SERVED_MODEL_NAME}")
fi

if [[ -n "${API_KEY}" ]]; then
    CMD+=(--api-key "${API_KEY}")
fi

if [[ "${INSECURE}" == "true" ]]; then
    CMD+=(--insecure)
fi

if [[ -n "${MAX_TTFT_MS}" ]]; then
    CMD+=(--max-ttft-ms "${MAX_TTFT_MS}")
fi

if [[ -n "${AUDIO_DURATION_S}" ]]; then
    CMD+=(--audio-duration-s "${AUDIO_DURATION_S}")
    CMD+=(--audio-silence-ms "${AUDIO_SILENCE_MS}")
    CMD+=(--generated-audio-dir "${GENERATED_AUDIO_DIR}")
fi

if [[ "${VARY_SEED_BY_CONFIG}" != "true" ]]; then
    CMD+=(--no-vary-seed-by-config)
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Qwen3-ASR 手动压测                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  接口      : %-48s║\n" "${API_URL}"
printf "║  Base URL  : %-48s║\n" "${BASE_URL}"
printf "║  模型      : %-48s║\n" "${MODEL}"
printf "║  数据集    : %-48s║\n" "${DATASET_PATH}"
if [[ -n "${AUDIO_DURATION_S}" ]]; then
    printf "║  单条时长  : %-48s║\n" "${AUDIO_DURATION_S}s"
    printf "║  拼接目录  : %-48s║\n" "${GENERATED_AUDIO_DIR}"
fi
printf "║  并发      : %-48s║\n" "${PARALLEL_NUMS}"
printf "║  轮数      : %-48s║\n" "${EPOCHS}"
printf "║  CSV       : %-48s║\n" "${OUTPUT_CSV}"
printf "║  XLSX      : %-48s║\n" "${OUTPUT_XLSX}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "执行命令:"
printf "  %q" "${CMD[@]}"
echo ""
echo ""

if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

exec "${CMD[@]}"
