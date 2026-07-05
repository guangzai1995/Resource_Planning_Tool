#!/usr/bin/env bash
# =============================================================================
# build_sgl_router.sh — 构建 SGLang PD 的 Python router 镜像 (sglang-router:offline)
#
# auto_bench harness 用 `python3 -m sglang_router.launch_router` 起 router,
# 所以 router 镜像需要 python3 + sglang_router 包。这个脚本用
# Dockerfile.sglang-router (python:3-slim + pip install sglang-router) 构建。
#
# sglang-router 在 PyPI 上是带预编译库的 wheel —— pip 装即可, 不需要 Rust/cargo/github。
# (注意: 不是 .magic/sglang-main/docker/sgl-router.Dockerfile 那个 Rust 二进制镜像,
#  那个没有 python3, harness 跑不起来。)
#
# 使用方法:
#   chmod +x build_sgl_router.sh
#   ./build_sgl_router.sh
#
# 常用覆盖项:
#   IMAGE_NAME=sglang-router:offline ./build_sgl_router.sh
#   NO_CACHE=true ./build_sgl_router.sh
#   SAVE_TAR=/tmp/sglang-router.offline.tar ./build_sgl_router.sh
#   MIRROR_PREFIX=xemegpzeib7tis.xuanyuan.run ./build_sgl_router.sh   # 内网镜像前缀拉 python 基础镜像
#
# PyPI 镜像源 (内网 pip):
#   ./build_sgl_router.sh --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#
# 说明:
#   构建出的 sglang-router:offline 对应 sglang PD 配置的 router_image。
#   router 跑在 controller 机(.14), build 后 docker save/load 搬过去即可。
# =============================================================================

set -euo pipefail

# =============================================================================
# ▌ 一、构建参数
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# router 镜像名称，sglang PD 配置的 router_image 默认用它
IMAGE_NAME="${IMAGE_NAME:-sglang-router:offline}"

# Dockerfile (本工程内的 Python 版 router Dockerfile)
DOCKERFILE="${DOCKERFILE:-${SCRIPT_DIR}/Dockerfile.sglang-router}"

# Docker build 上下文 = 脚本所在目录
CONTEXT_DIR="${CONTEXT_DIR:-${SCRIPT_DIR}}"

PLATFORM="${PLATFORM:-}"
NO_CACHE="${NO_CACHE:-false}"
PULL_BASE="${PULL_BASE:-false}"
SAVE_TAR="${SAVE_TAR:-}"

# 可选：内网镜像前缀。设了之后 build 前会用前缀拉基础镜像 python:3.12-slim 并 retag 成原名。
MIRROR_PREFIX="${MIRROR_PREFIX:-}"

# 基础镜像（python:3.12-slim），与 MIRROR_PREFIX 预拉保持一致
ROUTER_BASE_IMAGE="${ROUTER_BASE_IMAGE:-python:3.12-slim}"

# =============================================================================
# ▌ 二、前置检查
# =============================================================================

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] 找不到 docker 命令" >&2; exit 1
fi
if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "[ERROR] 找不到 Dockerfile: ${DOCKERFILE}" >&2; exit 1
fi
if [[ ! -d "${CONTEXT_DIR}" ]]; then
    echo "[ERROR] 找不到上下文目录: ${CONTEXT_DIR}" >&2; exit 1
fi

# =============================================================================
# ▌ 三、(可选) 内网镜像前缀预拉基础镜像并 retag
# =============================================================================
if [[ -n "${MIRROR_PREFIX}" ]]; then
    echo "通过镜像前缀预拉基础镜像并 retag: ${MIRROR_PREFIX}/"
    if docker pull "${MIRROR_PREFIX}/${ROUTER_BASE_IMAGE}"; then
        docker tag "${MIRROR_PREFIX}/${ROUTER_BASE_IMAGE}" "${ROUTER_BASE_IMAGE}"
        echo "  ✓ ${ROUTER_BASE_IMAGE}"
    else
        echo "  [WARN] 拉取 ${MIRROR_PREFIX}/${ROUTER_BASE_IMAGE} 失败" >&2
    fi
    echo ""
fi

# =============================================================================
# ▌ 四、拼装 docker build 命令
# =============================================================================

CMD=(docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}")
[[ -n "${PLATFORM}" ]] && CMD+=(--platform "${PLATFORM}")
[[ "${NO_CACHE}" == "true" ]] && CMD+=(--no-cache)
[[ "${PULL_BASE}" == "true" ]] && CMD+=(--pull)
# 透传额外参数，例如 --build-arg PIP_INDEX_URL=...
if [[ "$#" -gt 0 ]]; then
    CMD+=("$@")
fi
CMD+=("${CONTEXT_DIR}")

# =============================================================================
# ▌ 五、打印摘要并执行
# =============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          构建 SGLang PD Python router 镜像                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  镜像名称  : %-48s║\n" "${IMAGE_NAME}"
printf "║  Dockerfile: %-48s║\n" "${DOCKERFILE}"
printf "║  基础镜像  : %-48s║\n" "${ROUTER_BASE_IMAGE}"
printf "║  平台      : %-48s║\n" "${PLATFORM:-默认}"
printf "║  禁用缓存  : %-48s║\n" "${NO_CACHE}"
if [[ -n "${MIRROR_PREFIX}" ]]; then
    printf "║  镜像前缀  : %-48s║\n" "${MIRROR_PREFIX}"
fi
if [[ -n "${SAVE_TAR}" ]]; then
    printf "║  导出文件  : %-48s║\n" "${SAVE_TAR}"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "提示: pip install sglang-router (PyPI wheel, 含预编译库, 无需 Rust/github)。"
echo "      内网环境用 PyPI 镜像源: 加 --build-arg PIP_INDEX_URL=https://<pypi-mirror>/simple"
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
    echo ""
    echo "搬运到 router 所在主机(.14)并导入:"
    echo "  scp ${SAVE_TAR} 10.200.1.14:/tmp/"
    echo "  ssh 10.200.1.14 'docker load -i /tmp/$(basename "${SAVE_TAR}")'"
fi
