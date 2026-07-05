#!/usr/bin/env bash
# =============================================================================
# check_ib_ucx.sh — 在测试机上用 vllm 镜像拉起临时容器，检查 PD 容器能否走 IB/RDMA。
#
# 不依赖 ucx_info（镜像里可能没装），改用三件事判定：
#   A) /dev/infiniband/uverbs* 在容器内是否可见（RDMA 设备有没有挂进来）
#   B) UCX 是否带 verbs/rc 插件 (libuct_rc_verbs.so / libuct_rc_mlx5.so)
#   C) UCX_LOG_LEVEL=info 下创建 nixl agent，看 UCX 实际选 rc 还是 tcp
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
  echo "  ❌ 容器内看不到 /dev/infiniband（RDMA 设备没挂进来 → UCX 只能走 tcp）"
fi

echo; echo "--- B) UCX 是否带 verbs/rc 插件（不依赖 ucx_info）---"
hits=$(find / \( -name 'libuct_rc_verbs.so*' -o -name 'libuct_rc_mlx5.so*' -o -name 'libuct_ib.so*' \) 2>/dev/null)
if [ -n "$hits" ]; then echo "$hits" | sed 's/^/  /'; else echo "  ❌ 没找到 rc_verbs/rc_mlx5/ib 插件（UCX 可能没编 RDMA 支持）"; fi

echo; echo "--- C) UCX_LOG_LEVEL=info 下建 nixl agent，看实际选的传输 ---"
UCX_LOG_LEVEL=info python - <<'PY' 2>&1 | grep -iE 'rc_verbs|rc_mlx5|\btcp\b|rdma|posix|cuda_copy|cuda_ipc|transport|md:|device|iface|agent|error|trace' | head -40
import os
try:
    from nixl import nixlAgent
except Exception:
    try:
        from nixl.nixl_agent import nixlAgent
    except Exception as e:
        print("NIXL_IMPORT_FAIL", repr(e)); raise SystemExit
try:
    a = nixlAgent("probe")
    print("NIXL_AGENT_CREATED")
except Exception as e:
    print("NIXL_AGENT_ERR", repr(e))
PY

echo; echo "=== 判定 ==="
log=$(UCX_LOG_LEVEL=info python - <<'PY' 2>&1
try:
    from nixl import nixlAgent
except Exception:
    from nixl.nixl_agent import nixlAgent
try:
    nixlAgent("p")
except Exception:
    pass
PY
)
if echo "$log" | grep -qiE 'rc_verbs|rc_mlx5'; then
  echo "RESULT: PASS — UCX 选中 rc(IB RDMA)，KV 迁移能走 IB"
elif ls /dev/infiniband/uverbs* >/dev/null 2>&1 && find / \( -name 'libuct_rc_verbs.so*' -o -name 'libuct_rc_mlx5.so*' \) 2>/dev/null | grep -q .; then
  echo "RESULT: LIKELY-PASS — IB 设备可见 + UCX 有 rc 插件（RDMA 可用；日志可能没打传输名）"
elif ! ls /dev/infiniband/uverbs* >/dev/null 2>&1; then
  echo "RESULT: FAIL-DEVICE — 容器内无 /dev/infiniband（UCX 走不了 RDMA，仅 tcp/共享内存）"
else
  echo "RESULT: FAIL-PLUGIN — IB 设备可见但 UCX 无 rc 插件（镜像 UCX 没编 verbs 支持）"
fi
PROBE_EOF

echo
echo "########## 1) 宿主机 IB/RoCE 硬件 ##########"
echo "--- lspci Mellanox/IB ---"
lspci 2>/dev/null | grep -iE 'infiniband|mellanox|mlx' || echo "  (lspci 未发现 Mellanox/IB)"
echo "--- 宿主机 uverbs 设备数 ---"
ls /dev/infiniband/uverbs* 2>/dev/null | wc -l | xargs echo "  uverbs:"
echo "--- ibstat 摘要（每个端口 Rate/State/LinkLayer）---"
ibstat 2>/dev/null | grep -iE '^CA |Rate:|State:|Link layer:' || echo "  (ibstat 不可用)"

run_probe() {  # $1 = 额外 docker 参数描述, 其余 = docker run 参数
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
echo "重点：第 2 段若 FAIL-DEVICE = PD 容器没挂 IB(现状，走tcp)；"
echo "      第 3 段若 PASS/LIKELY-PASS = 挂上 --device /dev/infiniband 后能走 IB，可改 harness。"
