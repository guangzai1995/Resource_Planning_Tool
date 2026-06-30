#!/usr/bin/env bash
# =============================================================================
# build_bench_runner.sh — 构建离线 benchmark runner 镜像
#
# 使用方法:
#   chmod +x build_bench_runner.sh
#   ./build_bench_runner.sh
#
# 常用覆盖项:
#   IMAGE_NAME=vllm-bench-runner:offline ./build_bench_runner.sh
#   NO_CACHE=true ./build_bench_runner.sh
#   SAVE_TAR=/tmp/vllm-bench-runner.offline.tar ./build_bench_runner.sh
#
# 透传 docker build 参数:
#   ./build_bench_runner.sh --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#
# 说明:
#   这个镜像封装 run_bench_multi.py 运行依赖，适合在有网络环境构建，
#   然后用 docker save / docker load 搬运到无网络测试主机。
# =============================================================================

set -euo pipefail

# =============================================================================
# ▌ 一、构建参数
# =============================================================================

# 脚本所在目录（自动识别，支持从任意工作目录执行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# bench-runner 镜像名称，auto_bench 配置中的 run.bench_image 默认使用它
IMAGE_NAME="${IMAGE_NAME:-vllm-bench-runner:offline}"

# Dockerfile 路径，默认使用当前测试工程内的 Dockerfile.bench-runner
DOCKERFILE="${DOCKERFILE:-${SCRIPT_DIR}/Dockerfile.bench-runner}"

# Docker build 上下文目录，默认是 vllm_standalone_bench 目录
CONTEXT_DIR="${CONTEXT_DIR:-${SCRIPT_DIR}}"

# 可选：指定构建平台，例如 linux/amd64
PLATFORM="${PLATFORM:-}"

# 可选：是否禁用 Docker build cache
NO_CACHE="${NO_CACHE:-false}"

# 可选：是否在构建时拉取基础镜像最新版本
PULL_BASE="${PULL_BASE:-false}"

# 可选：构建完成后导出离线 tar 包；为空则不导出
SAVE_TAR="${SAVE_TAR:-}"

# =============================================================================
# ▌ 二、前置检查
# =============================================================================

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] 找不到 docker 命令，请先安装 Docker CLI。" >&2
    exit 1
fi

if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "[ERROR] 找不到 Dockerfile: ${DOCKERFILE}" >&2
    exit 1
fi

if [[ ! -d "${CONTEXT_DIR}" ]]; then
    echo "[ERROR] 找不到构建上下文目录: ${CONTEXT_DIR}" >&2
    exit 1
fi

# =============================================================================
# ▌ 三、拼装 docker build 命令
# =============================================================================

CMD=(docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}")

if [[ -n "${PLATFORM}" ]]; then
    CMD+=(--platform "${PLATFORM}")
fi

if [[ "${NO_CACHE}" == "true" ]]; then
    CMD+=(--no-cache)
fi

if [[ "${PULL_BASE}" == "true" ]]; then
    CMD+=(--pull)
fi

# 额外参数原样透传给 docker build，例如 --build-arg / --network / --progress
if [[ "$#" -gt 0 ]]; then
    CMD+=("$@")
fi

CMD+=("${CONTEXT_DIR}")

# =============================================================================
# ▌ 四、打印摘要并执行
# =============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              构建 vLLM bench-runner 镜像                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  镜像名称  : %-48s║\n" "${IMAGE_NAME}"
printf "║  Dockerfile: %-48s║\n" "${DOCKERFILE}"
printf "║  上下文    : %-48s║\n" "${CONTEXT_DIR}"
printf "║  平台      : %-48s║\n" "${PLATFORM:-默认}"
printf "║  禁用缓存  : %-48s║\n" "${NO_CACHE}"
printf "║  拉取基镜像: %-48s║\n" "${PULL_BASE}"
if [[ -n "${SAVE_TAR}" ]]; then
    printf "║  导出文件  : %-48s║\n" "${SAVE_TAR}"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "执行命令:"
echo "  ${CMD[*]}"
echo ""

"${CMD[@]}"

if [[ -n "${SAVE_TAR}" ]]; then
    mkdir -p "$(dirname "${SAVE_TAR}")"
    echo ""
    echo "导出离线镜像包:"
    echo "  docker save ${IMAGE_NAME} -o ${SAVE_TAR}"
    docker save "${IMAGE_NAME}" -o "${SAVE_TAR}"
fi
