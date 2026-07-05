#!/usr/bin/env bash
# =============================================================================
# build_sgl_router.sh — 构建 SGLang PD router 镜像 (sglang-router:offline)
#
# sglang PD 分离部署需要一个 router (Rust 写的 sgl-router) 来在 prefill/decode
# 之间路由请求。这个脚本用 .magic/sglang-main/docker/sgl-router.Dockerfile 构建。
# 是 Rust 多阶段构建 (cargo-chef)，需要在有网络的环境构建（拉 crate），构建较慢。
#
# 使用方法:
#   chmod +x build_sgl_router.sh
#   ./build_sgl_router.sh
#
# 常用覆盖项:
#   IMAGE_NAME=sglang-router:offline ./build_sgl_router.sh
#   NO_CACHE=true ./build_sgl_router.sh                 # 禁用 docker 缓存
#   SAVE_TAR=/tmp/sglang-router.offline.tar ./build_sgl_router.sh   # 构建后导出离线包
#   SGLANG_ROOT=/path/to/sglang-main ./build_sgl_router.sh          # 指定 sglang 源码根
#
# 透传 docker build 参数 (例如固定 Rust/Debian 版本):
#   ./build_sgl_router.sh --build-arg RUST_VERSION=1.90 --build-arg DEBIAN_VERSION=bookworm
#
# 说明:
#   构建出的 sglang-router:offline 对应 sglang PD 配置里的 router_image。
#   router 跑在 controller 机上（sglang PD 配置 frontend.host 通常是 router=.14），
#   所以构建后用 docker save/load 搬到那台机即可。
# =============================================================================

set -euo pipefail

# =============================================================================
# ▌ 一、构建参数
# =============================================================================

# 脚本所在目录（vllm_standalone_bench），支持从任意工作目录执行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 项目根目录
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# sglang 源码根目录（Dockerfile 和构建上下文都在这下面）
SGLANG_ROOT="${SGLANG_ROOT:-${PROJECT_ROOT}/.magic/sglang-main}"

# router 镜像名称，sglang PD 配置中的 router_image 默认使用它
IMAGE_NAME="${IMAGE_NAME:-sglang-router:offline}"

# Dockerfile 路径
DOCKERFILE="${DOCKERFILE:-${SGLANG_ROOT}/docker/sgl-router.Dockerfile}"

# Docker build 上下文目录（sglang 仓库根，Dockerfile 里 COPY experimental/sgl-router/...）
CONTEXT_DIR="${CONTEXT_DIR:-${SGLANG_ROOT}}"

# 可选：指定构建平台，例如 linux/amd64
PLATFORM="${PLATFORM:-}"

# 可选：是否禁用 Docker build cache
NO_CACHE="${NO_CACHE:-false}"

# 可选：是否在构建时拉取基础镜像最新版本
PULL_BASE="${PULL_BASE:-false}"

# 可选：构建完成后导出离线 tar 包；为空则不导出
SAVE_TAR="${SAVE_TAR:-}"

# sgl-router 基础镜像版本（对应 Dockerfile 的 ARG RUST_VERSION / DEBIAN_VERSION）
RUST_VERSION="${RUST_VERSION:-1.90}"
DEBIAN_VERSION="${DEBIAN_VERSION:-bookworm}"

# 可选：内网镜像前缀。设了之后 build 前会用该前缀拉取基础镜像并 retag 成原名，
# 这样 docker build 不直连 Docker Hub / gcr.io。例如:
#   MIRROR_PREFIX=xemegpzeib7tis.xuanyuan.run ./build_sgl_router.sh
MIRROR_PREFIX="${MIRROR_PREFIX:-}"

# =============================================================================
# ▌ 二、前置检查
# =============================================================================

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] 找不到 docker 命令，请先安装 Docker CLI。" >&2
    exit 1
fi

if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "[ERROR] 找不到 sgl-router Dockerfile: ${DOCKERFILE}" >&2
    echo "       请确认 SGLANG_ROOT (.magic/sglang-main) 存在且包含 docker/sgl-router.Dockerfile" >&2
    echo "       可用 SGLANG_ROOT=/path/to/sglang-main 覆盖默认路径" >&2
    exit 1
fi

if [[ ! -d "${CONTEXT_DIR}" ]]; then
    echo "[ERROR] 找不到构建上下文目录: ${CONTEXT_DIR}" >&2
    exit 1
fi

if [[ ! -d "${SGLANG_ROOT}/experimental/sgl-router" ]]; then
    echo "[ERROR] ${SGLANG_ROOT}/experimental/sgl-router 不存在，sglang 源码不完整" >&2
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

# 固定基础镜像版本（与 MIRROR_PREFIX 预拉的版本一致）
CMD+=(--build-arg "RUST_VERSION=${RUST_VERSION}" --build-arg "DEBIAN_VERSION=${DEBIAN_VERSION}")

# 额外参数原样透传给 docker build，例如 --network / --progress
if [[ "$#" -gt 0 ]]; then
    CMD+=("$@")
fi

CMD+=("${CONTEXT_DIR}")

# =============================================================================
# ▌ 三点五、(可选) 通过内网镜像前缀预拉基础镜像并 retag 成原名
# =============================================================================
if [[ -n "${MIRROR_PREFIX}" ]]; then
    echo "通过镜像前缀预拉基础镜像并 retag 成原名: ${MIRROR_PREFIX}/"
    for img in "rust:${RUST_VERSION}-${DEBIAN_VERSION}" "gcr.io/distroless/cc-debian12:nonroot"; do
        if docker pull "${MIRROR_PREFIX}/${img}"; then
            docker tag "${MIRROR_PREFIX}/${img}" "${img}"
            echo "  ✓ ${img}"
        else
            echo "  [WARN] 拉取 ${MIRROR_PREFIX}/${img} 失败" >&2
            echo "         - 若该镜像本地已有，docker build 会直接复用，无碍；" >&2
            echo "         - 若前缀不支持 gcr.io，需手动: docker pull <别的方式> gcr.io/distroless/cc-debian12:nonroot" >&2
        fi
    done
    echo ""
fi

# =============================================================================
# ▌ 四、打印摘要并执行
# =============================================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              构建 SGLang PD router 镜像                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  镜像名称  : %-48s║\n" "${IMAGE_NAME}"
printf "║  Dockerfile: %-48s║\n" "${DOCKERFILE}"
printf "║  上下文    : %-48s║\n" "${CONTEXT_DIR}"
printf "║  Rust 版本 : %-48s║\n" "${RUST_VERSION}-${DEBIAN_VERSION}"
printf "║  平台      : %-48s║\n" "${PLATFORM:-默认}"
printf "║  禁用缓存  : %-48s║\n" "${NO_CACHE}"
printf "║  拉取基镜像: %-48s║\n" "${PULL_BASE}"
if [[ -n "${MIRROR_PREFIX}" ]]; then
    printf "║  镜像前缀  : %-48s║\n" "${MIRROR_PREFIX}"
fi
if [[ -n "${SAVE_TAR}" ]]; then
    printf "║  导出文件  : %-48s║\n" "${SAVE_TAR}"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "提示: 这是 Rust 多阶段构建 (cargo-chef)，需联网拉 crate，首次构建较慢。"
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
    echo "搬运到 router 所在主机（通常是 controller .14）并导入:"
    echo "  scp ${SAVE_TAR} <router-host>:/tmp/"
    echo "  ssh <router-host> 'docker load -i /tmp/$(basename "${SAVE_TAR}")'"
fi
