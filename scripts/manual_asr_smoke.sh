#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://10.86.0.32:13001/v1/audio/transcriptions}"
MODEL_NAME="${MODEL_NAME:-Qwen3-ASR-1_7B}"
LANGUAGE="${LANGUAGE:-en}"
AUDIO_FILE="${AUDIO_FILE:-asr_en.wav}"
AUDIO_URL="${AUDIO_URL:-https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav}"
DOWNLOAD_MAX_TIME="${DOWNLOAD_MAX_TIME:-60}"
REQUEST_MAX_TIME="${REQUEST_MAX_TIME:-120}"
CURL_RETRY="${CURL_RETRY:-2}"
CURL_RETRY_DELAY="${CURL_RETRY_DELAY:-2}"

usage() {
    cat <<USAGE
Manual ASR smoke test.

Environment overrides:
  API_URL            ${API_URL}
  MODEL_NAME         ${MODEL_NAME}
  LANGUAGE           ${LANGUAGE}
  AUDIO_FILE         ${AUDIO_FILE}
  AUDIO_URL          ${AUDIO_URL}
  DOWNLOAD_MAX_TIME  ${DOWNLOAD_MAX_TIME}
  REQUEST_MAX_TIME   ${REQUEST_MAX_TIME}
  CURL_RETRY         ${CURL_RETRY}
  CURL_RETRY_DELAY   ${CURL_RETRY_DELAY}

Example:
  bash scripts/manual_asr_smoke.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

command -v curl >/dev/null 2>&1 || {
    echo "curl is required" >&2
    exit 127
}

command -v python3 >/dev/null 2>&1 || {
    echo "python3 is required" >&2
    exit 127
}

if [[ ! -f "${AUDIO_FILE}" ]]; then
    echo "Downloading test audio: ${AUDIO_URL}"
    curl -L --fail \
        --retry "${CURL_RETRY}" \
        --retry-delay "${CURL_RETRY_DELAY}" \
        --retry-connrefused \
        --max-time "${DOWNLOAD_MAX_TIME}" \
        -o "${AUDIO_FILE}" \
        "${AUDIO_URL}"
fi

response_file="$(mktemp -t asr-smoke-response.XXXXXX.json)"
trap 'rm -f "${response_file}"' EXIT

echo "Calling ASR endpoint"
echo "  url:      ${API_URL}"
echo "  model:    ${MODEL_NAME}"
echo "  language: ${LANGUAGE}"
echo "  audio:    ${AUDIO_FILE}"
echo

http_code="$(
    curl -sS --max-time "${REQUEST_MAX_TIME}" \
        --retry "${CURL_RETRY}" \
        --retry-delay "${CURL_RETRY_DELAY}" \
        --retry-connrefused \
        -o "${response_file}" \
        -w "%{http_code}" \
        -X POST \
        -F "model=${MODEL_NAME}" \
        -F "file=@${AUDIO_FILE}" \
        -F "language=${LANGUAGE}" \
        "${API_URL}"
)"

echo "HTTP ${http_code}"
python3 -m json.tool "${response_file}"

case "${http_code}" in
    2??)
        ;;
    *)
        echo "ASR request failed with HTTP ${http_code}" >&2
        exit 1
        ;;
esac
