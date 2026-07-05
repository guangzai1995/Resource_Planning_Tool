#!/usr/bin/env bash
# =============================================================================
# check_ib_ucx.sh — 在测试机上用 vllm 镜像拉起临时容器，检查 IB/RoCE 网卡
#                    以及镜像里的 UCX/NIXL 能否用 rc(RDMA) 传输。
#
# 判定 PD 的 KV 迁移到底走没走 IB：
#   - PASS  = UCX 支持 rc_verbs/rc_mlx5，KV 迁移走 IB/RoCE（快）
#   - FAIL  = UCX 只能用 tcp/共享内存，KV 迁移会慢（PD 性能差的嫌疑）
#
# 用法:
#   bash check_ib_ucx.sh                          # 用默认镜像
#   bash check_ib_ucx.sh vllm/vllm-openai:0.23.0  # 指定镜像
#   IMAGE=... bash check_ib_ucx.sh
# =============================================================================
set -euo pipefail

IMAGE="${1:-${IMAGE:-vllm/vllm-openai:latest-0616-msgpack}}"
echo "镜像: $IMAGE"
command -v docker >/dev/null 2>&1 || { echo "[ERROR] 缺少 docker"; exit 1; }

echo
echo "########## 1) 宿主机 IB/RoCE 硬件 ##########"
echo "--- /dev/infiniband ---"
ls -l /dev/infiniband* 2>/dev/null || echo "  (无 /dev/infiniband 设备)"
echo "--- lspci Mellanox / InfiniBand ---"
lspci 2>/dev/null | grep -iE 'infiniband|mellanox|mlx' || echo "  (lspci 未发现 Mellanox/IB)"
echo "--- ibstat ---"
ibstat 2>/dev/null || echo "  (ibstat 不可用)"
echo "--- ibv_devinfo ---"
ibv_devinfo 2>/dev/null | head -30 || echo "  (ibv_devinfo / libibverbs 不可用)"

echo
echo "########## 2) 容器内 UCX/NIXL（PD 等同访问：--network host --gpus all）##########"
docker run --rm --network host --gpus all --entrypoint bash "$IMAGE" -s <<'PROBE'
echo "--- ucx 版本 ---"
ucx_info -v 2>/dev/null | head -3 || echo "ucx_info 不在 PATH"
echo
echo "--- ucx_info -d（设备/传输；重点看 rc_verbs/rc_mlx5 + mlx5 设备）---"
ucx_info -d 2>/dev/null || echo "ucx_info 不可用"
echo
echo "--- 强制 UCX_TLS=rc 后能看到什么 ---"
UCX_TLS=rc ucx_info -d 2>/dev/null | grep -iE 'transport|device|md |rc' | head -25 || true
echo
echo "--- 容器内 /dev/infiniband 是否可见 ---"
ls -l /dev/infiniband* 2>/dev/null || echo "容器内看不到 /dev/infiniband（RDMA 设备没挂进来）"
echo
echo "--- NIXL 选传输日志（UCX_LOG_LEVEL=info）---"
UCX_LOG_LEVEL=info python -c "import nixl; print('nixl import OK')" 2>&1 \
  | grep -iE 'transport|rc_verbs|rc_mlx5|\btcp\b|rdma|posix|cuda|nixl import OK' | head -20 || true
echo
echo "=== 判定（PD 等同访问下）==="
if ucx_info -d 2>/dev/null | grep -qiE 'rc_verbs|rc_mlx5'; then
  echo "RESULT: PASS — UCX 支持 rc(RDMA) 传输，KV 迁移能走 IB/RoCE"
else
  echo "RESULT: FAIL — UCX 未检测到 rc(RDMA)，会回退 tcp/共享内存（PD 慢的嫌疑成立）"
fi
PROBE

echo
echo "########## 3) 对照：显式挂 /dev/infiniband + IPC_LOCK ##########"
if ls /dev/infiniband/* >/dev/null 2>&1; then
  docker run --rm --network host --gpus all \
    --device /dev/infiniband --cap-add=IPC_LOCK \
    --entrypoint bash "$IMAGE" -s <<'PROBE'
if ucx_info -d 2>/dev/null | grep -qiE 'rc_verbs|rc_mlx5'; then
  echo "对照 PASS: 加 --device /dev/infiniband 后 rc 可见"
  echo "  => 镜像/驱动没问题，PD 容器需要显式挂 IB 设备（--device /dev/infiniband --cap-add=IPC_LOCK）才能走 IB"
else
  echo "对照 FAIL: 即使挂了 /dev/infiniband 仍无 rc（可能缺 IB 驱动/verbs，或网卡不支持 RDMA）"
fi
PROBE
else
  echo "  宿主机无 /dev/infiniband，跳过（本机没有 IB/RoCE 卡，PD 无法走 IB）"
fi

echo
echo "完成。"
echo "解读：第 2 段 PASS = PD 已在用 IB（与之前 5.4GB/s 带宽一致）；"
echo "     第 2 段 FAIL 但第 3 段 PASS = 需要给 PD 容器挂 IB 设备；"
echo "     两段都 FAIL = 本机/镜像不支持 RDMA。"
