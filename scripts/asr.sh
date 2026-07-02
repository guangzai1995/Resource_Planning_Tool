#!/bin/bash
# 测试 Qwen3-ASR-1_7B 语音转录接口

API_URL="http://10.86.0.32:13001/v1/audio/transcriptions"
MODEL_NAME="Qwen3-ASR-1_7B"
AUDIO_FILE="asr_en.wav"
AUDIO_URL="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"

# 如果音频文件不存在，则下载
if [ ! -f "${AUDIO_FILE}" ]; then
    echo "下载测试音频..."
    curl -L -o "${AUDIO_FILE}" "${AUDIO_URL}"
fi

echo "调用接口: ${API_URL}"
echo "模型: ${MODEL_NAME}"
echo "音频: ${AUDIO_FILE}"
echo "---"

# 调用转录接口
curl "${API_URL}" \
    -X POST \
    -H "Content-Type: multipart/form-data" \
    -F model="${MODEL_NAME}" \
    -F file="@${AUDIO_FILE}" \
    -F language="en" \
    -s | python3 -m json.tool

echo ""