#!/usr/bin/env bash
# =============================================================================
# check_ib_ucx.sh — 用 vllm 镜像拉临时容器，确认 PD 容器能否走 IB/RDMA。
#   A) /dev/infiniband/uverbs* 容器内可见？
#   B) UCX IB/rc 插件 (libuct_ib.so / rc_verbs / rc_mlx5)
#   C) UCX_LOG_LEVEL=info 用 nixl._api 建 agent，打印原始+过滤输出，看 UCX 加载的传输
# 用法: bash check_ib_ucx.sh [image]
# =============================================================================
set -euo pipefail

IMAGE="${1:-${IMAGE:-vllm/vllm-openai:latest-0616-msgpack}}"
echo "镜像: $IMAGE"
command -v docker >/dev/null 2>&1 || { echo "[ERROR] 缺少 docker"; exit 1; }

PROBE="$(mktemp)"
trap 'rm -f "$PROBE"' EXIT
cat > "$PROBE" <<'PROBE_EOF'
#!/usr/bin/env bash
echo "--- A) 容器内 /dev/infiniband 是否可见 ---"
if ls /dev/infiniband/uverbs* >/dev/null 2>&1; then
  echo "  可见 uverbs: $(ls /dev/infiniband/uverbs* 2>/dev/null | wc -l) 个"
else
  echo "  X 容器内看不到 /dev/infiniband（RDMA 设备没挂进来 → UCX 只能走 tcp）"
fi

echo; echo "--- B) UCX IB/rc 插件（接受新版合并的 libuct_ib.so）---"
ibplug=$(find / \( -name 'libuct_ib.so*' -o -name 'libuct_rc_verbs.so*' -o -name 'libuct_rc_mlx5.so*' -o -name 'libuct_rc.so*' \) 2>/dev/null)
if [ -n "$ibplug" ]; then echo "$ibplug" | sed 's/^/  /'; else echo "  X 没找到任何 IB/rc 插件"; fi

echo; echo "--- C) UCX_LOG_LEVEL=info 建 nixl agent（python3，30s 超时，先打印原始输出）---"
export UCX_LOG_LEVEL=info
cat > /tmp/nixl_probe.py <<'PY'
import os
os.environ.setdefault("UCX_LOG_LEVEL", "info")
try:
    from nixl._api import nixl_agent, nixl_agent_config
    cfg = nixl_agent_config(capture_telemetry=True)
    a = nixl_agent("probe", cfg)
    print("=== NIXL_AGENT_OK ===")
except Exception:
    import traceback; traceback.print_exc()
PY
PYBIN=$(command -v python3 || command -v python || true)
if [ -z "$PYBIN" ]; then echo "  X 容器内没有 python/python3"; ucx_log=""; else
  echo "  python: $PYBIN"
  echo "  --- 原始输出 (head 60) ---"
  ucx_log=$(timeout 30 "$PYBIN" /tmp/nixl_probe.py 2>&1) || ucx_log="${ucx_log}
=== TIMEOUT_OR_NONZERO (exit $?) ==="
  echo "$ucx_log" | head -60 | sed 's/^/    /'
  echo "  --- 传输相关行 ---"
  echo "$ucx_log" | grep -iE 'mlx5|rc_verbs|rc_mlx5|rdma|verbs|infiniband|transport|\btcp\b|cuda_copy|cuda_ipc|NIXL_AGENT_OK|error|exception|traceback|module|not found|timeout' | head -30 | sed 's/^/    /'
fi

echo; echo "=== 判定 ==="
has_ib_kw=$(echo "$ucx_log" | grep -ciE 'mlx5|rc_verbs|rc_mlx5|\brdma\b|verbs.*port|infiniband.*device')
if echo "$ucx_log" | grep -qi 'NIXL_AGENT_OK' && [ "$has_ib_kw" -gt 0 ]; then
  echo "RESULT: PASS — UCX 加载了 IB/rc(RDMA) 传输"
elif echo "$ucx_log" | grep -qi 'NIXL_AGENT_OK' && [ "$has_ib_kw" -eq 0 ]; then
  echo "RESULT: NO-RDMA-LOG — agent 建好但日志没出现 IB 传输（把 UCX_LOG_LEVEL 改 debug 重跑确认）"
elif echo "$ucx_log" | grep -qiE 'TIMEOUT_OR_NONZERO'; then
  echo "RESULT: TIMEOUT/HANG — nixl agent 创建超时或异常退出"
elif echo "$ucx_log" | grep -qiE 'error|exception|traceback|No module|not found'; then
  echo "RESULT: ERROR — 看 C 段原始输出"
elif ! ls /dev/infiniband/uverbs* >/dev/null 2>&1; then
  echo "RESULT: FAIL-DEVICE — 容器内无 /dev/infiniband（UCX 走不了 RDMA）"
else
  echo "RESULT: UNCERTAIN — 看 C 段原始输出人工判断"
fi
PROBE_EOF

echo
echo "########## 1) 宿主机 IB/RoCE 硬件 ##########"
echo "--- lspci Mellanox/IB ---"
lspci 2>/dev/null | grep -iE 'infiniband|mellanox|mlx' || echo "  (lspci 未发现 Mellanox/IB)"
echo "--- 宿主机 uverbs 设备数 ---"
ls /dev/infiniband/uverbs* 2>/dev/null | wc -l | xargs echo "  uverbs:"
echo "--- ibstat 摘要 ---"
ibstat 2>/dev/null | grep -iE '^CA |Rate:|State:|Link layer:' || echo "  (ibstat 不可用)"

run_probe() {  # $1 = 段标题, 其余 = docker 额外参数
  local label="$1"; shift
  echo
  echo "########## $label ##########"
  docker run --rm -i --network host --gpus all "$@" \
    -v "$PROBE:/probe.sh" --entrypoint bash "$IMAGE" /probe.sh
}

run_probe "2) 容器内（PD 等同访问：--network host --gpus all）"

if ls /dev/infiniband/* >/dev/null 2>&1; then
  run_probe "3) 对照：显式挂 /dev/infiniband + IPC_LOCK + memlock" \
    --device /dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1
else
  echo; echo "########## 3) 对照：宿主机无 /dev/infiniband，跳过 ##########"
fi

echo
echo "完成。看每段 RESULT。第 3 段 PASS = 挂 IB 后能走 RDMA → 改 harness；"
echo "NO-RDMA-LOG/UNCERTAIN = 把 UCX_LOG_LEVEL 改 debug 重跑；ERROR/TIMEOUT = 看原始输出。"
