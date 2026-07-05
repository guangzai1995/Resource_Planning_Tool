#!/usr/bin/env bash
# =============================================================================
# check_ib_ucx.sh — 在测试机上用 vllm 镜像拉起临时容器，确认 PD 容器能否走 IB/RDMA。
#
# 三路判定（不依赖 ucx_info，它常不在镜像里）：
#   A) /dev/infiniband/uverbs* 在容器内是否可见（RDMA 设备有没有挂进来）
#   B) UCX 是否带 IB/rc 插件 (libuct_ib.so / libuct_rc_verbs.so / libuct_rc_mlx5.so)
#   C) UCX_LOG_LEVEL=info 下用正确 API(nixl._api) 建 nixl agent，看 UCX 实际加载的传输
#
# 用法:
#   bash check_ib_ucx.sh                          # 默认镜像
#   bash check_ib_ucx.sh vllm/vllm-openai:0.23.0  # 指定镜像
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

echo; echo "--- C) UCX_LOG_LEVEL=info 建 nixl agent，看实际加载的传输 ---"
export UCX_LOG_LEVEL=info
ucx_log=$(python - <<'PY' 2>&1
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
)
echo "$ucx_log" | grep -iE 'mlx5|rc_verbs|rc_mlx5|rdma|verbs|infiniband|transport|\btcp\b|cuda_copy|cuda_ipc|NIXL_AGENT_OK|error|exception|traceback' | head -50

echo; echo "=== 判定 ==="
has_ib_kw=$(echo "$ucx_log" | grep -ciE 'mlx5|rc_verbs|rc_mlx5|\brdma\b|infiniband.*device|verbs.*port')
has_tcp=$(echo "$ucx_log" | grep -ciE 'transport.*tcp|\btcp\b')
if echo "$ucx_log" | grep -qi 'NIXL_AGENT_OK' && [ "$has_ib_kw" -gt 0 ]; then
  echo "RESULT: PASS — UCX 识别并加载了 IB/rc(RDMA) 传输"
elif echo "$ucx_log" | grep -qi 'NIXL_AGENT_OK' && [ "$has_ib_kw" -eq 0 ]; then
  echo "RESULT: TCP-ONLY — agent 建好但 UCX 只用 tcp/cuda，没启用 IB（镜像 UCX 没编/没识别 RDMA）"
elif echo "$ucx_log" | grep -qiE 'error|exception|traceback'; then
  echo "RESULT: ERROR — nixl agent 创建失败，看 C 段 traceback"
elif ! ls /dev/infiniband/uverbs* >/dev/null 2>&1; then
  echo "RESULT: FAIL-DEVICE — 容器内无 /dev/infiniband（UCX 走不了 RDMA）"
else
  echo "RESULT: UNCERTAIN — 看 C 段输出人工判断（可把 UCX_LOG_LEVEL 改 debug 重跑）"
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

run_probe() {  # $1 = 段落标题, 其余 = docker run 额外参数
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
echo "完成。看每段最后的 RESULT 行。"
echo "关键：第 2 段 FAIL-DEVICE = 现状(PD 没挂 IB，走 tcp)；"
echo "      第 3 段 PASS = 挂上 /dev/infiniband 后 UCX 能走 RDMA → 可改 harness 让 PD 走 IB。"
