#!/usr/bin/env bash
# =============================================================================
# check_sglang_serve.sh — 诊断 sglang serve 容器为什么起不来/没日志
#
# harness 抓到的 serve 日志是 0 字节, 所以这里用真实配置参数前台直接跑一次
# sglang launch_server, 抓全部 stdout/stderr, 看 sglang 到底是:
#   - 正常加载(只是慢, readiness 等不够) → 会看到 model loading 进度
#   - 参数报错(kv-cache-dtype / quantization / parser 等) → 看到 Error/Traceback
#   - 镜像/模块问题 → 看到 import error
#
# 用法(在 .14 或 .13 上跑, 需 docker + GPU):
#   bash check_sglang_serve.sh
#   TIMEOUT=300 bash check_sglang_serve.sh            # 等久一点(模型加载慢)
#   GPUS=4,5,6,7 bash check_sglang_serve.sh           # 换一组 GPU
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:latest-0705}"
GPUS="${GPUS:-0,1,2,3}"
MODEL="${MODEL:-/models/MiniMax-M2.7-NVFP4}"
MOUNT="${MOUNT:-/NVME1/models:/models:ro}"
PORT="${PORT:-4137}"
TIMEOUT="${TIMEOUT:-240}"   # 前台跑多久(秒), 够看到启动进度或报错
LOG="/tmp/sglang_diag.log"

echo "############ 1) 镜像 / 模型 / 模块预检 ############"
echo "--- 镜像在不在 ---"
docker images | grep -E "sglang.*0705|sglang:latest" || echo "  [警告] 没看到 $IMAGE"
echo "--- sglang 版本 + 能否 import ---"
docker run --rm --entrypoint python3 "$IMAGE" -c "import sglang; print('sglang', sglang.__version__)" 2>&1 | head -5
echo "--- 容器内模型目录存在? ---"
docker run --rm -v "$MOUNT" --entrypoint ls "$IMAGE" "$MODEL/config.json" 2>&1 | head -3

echo
echo "############ 2) 前台跑 sglang launch_server (真实参数, ${TIMEOUT}s 超时) ############"
echo "看下面输出:"
echo "  - 出现 'loading' / 'Capture' / 'server started' 等 = 正常, sglang 只是加载慢"
echo "  - 出现 Error / Traceback / invalid / unrecognized = 参数或环境报错(重点看)"
echo "  - 被超时杀(退出码124)且一直在 loading = sglang 没问题, 是 readiness 等不够"
echo
set +e
timeout "${TIMEOUT}" docker run --rm --network host --gpus "\"device=${GPUS}\"" \
  -v "$MOUNT" --entrypoint python3 "$IMAGE" -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.92 \
  --max-running-requests 128 \
  --context-length 172032 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax \
  2>&1 | tee "$LOG" | tail -100
rc=${PIPESTATUS[0]}
set -e

echo
echo "############ 3) 结果判定 ############"
echo "退出码: $rc  (124=超时杀=sglang 一直在加载[正常,只是慢]; 非0=报错)"
echo "完整日志: $LOG"
echo
echo "--- 错误关键词扫描 ---"
if grep -iqE "error|traceback|invalid|unrecognized|not support|valueerror|exception|failed|no such" "$LOG"; then
    echo ">>> 发现疑似报错, 重点行:"
    grep -inE "error|traceback|invalid|unrecognized|not support|valueerror|exception|failed|no such" "$LOG" | head -25
else
    echo "(无明显错误关键词 —— 若退出码124, 说明 sglang 在正常加载, 问题在 readiness 超时太短, 不是 sglang 崩)"
fi
echo
echo "--- 启动进度关键词(确认是否在加载) ---"
grep -iE "loading|initializing|capture|cuda|weight|kv cache|server start|application startup|avail|memory" "$LOG" | tail -15 || true
