#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# 资源规划工具 — 一键开发启动脚本
# 用法: bash start.sh [--backend-only | --frontend-only | --test]
#       bash stop.sh  停止所有服务
# ──────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
ENV_FILE="$BACKEND_DIR/.env"
ENV_EXAMPLE="$BACKEND_DIR/.env.example"
LOG_DIR="$SCRIPT_DIR/logs/service"

# 国内镜像源
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
NPM_REGISTRY="https://registry.npmmirror.com"

# 依赖哈希印记文件（用于跳过未变更的安装）
PY_STAMP="$BACKEND_DIR/.venv/.deps_hash"
NODE_STAMP="$FRONTEND_DIR/node_modules/.deps_hash"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

MODE="${1:-}"

# ── 工具函数：计算文件 md5 ────────────────────────────────
_md5() { md5sum "$1" 2>/dev/null | awk '{print $1}' || md5 -q "$1" 2>/dev/null || echo ""; }

# ── 0. Check env file ─────────────────────────────────────
if [[ ! -f "$ENV_FILE" && -f "$ENV_EXAMPLE" ]]; then
    warn ".env not found — copying from .env.example"
    cp "$ENV_EXAMPLE" "$ENV_FILE"
fi

# ── 1. Python environment ─────────────────────────────────
setup_python() {
    info "Checking Python environment…"

    # 创建虚拟环境
    if [[ ! -d "$BACKEND_DIR/.venv" ]]; then
        info "Creating virtual environment at $BACKEND_DIR/.venv"
        python3 -m venv "$BACKEND_DIR/.venv"
    fi

    # shellcheck source=/dev/null
    source "$BACKEND_DIR/.venv/bin/activate"

    # 计算 pyproject.toml 哈希，与上次安装时的哈希对比
    local pyproject="$BACKEND_DIR/pyproject.toml"
    local current_hash
    current_hash="$(_md5 "$pyproject")"
    local saved_hash=""
    [[ -f "$PY_STAMP" ]] && saved_hash="$(cat "$PY_STAMP")"

    if [[ "$current_hash" == "$saved_hash" && -n "$current_hash" ]]; then
        info "Python deps up-to-date (hash unchanged), skipping install"
        return
    fi

    info "Installing Python dependencies (via 清华 PyPI 镜像)…"
    pip install --quiet --upgrade pip -i "$PIP_INDEX"

    # 优先用 pyproject.toml editable 安装；失败则手动列出
    if pip install --quiet -e "${BACKEND_DIR}[dev]" \
        -i "$PIP_INDEX" \
        --trusted-host pypi.tuna.tsinghua.edu.cn; then
        info "Installed via pyproject.toml [dev]"
    else
        warn "pyproject.toml editable install failed, falling back to explicit list"
        pip install --quiet \
            -i "$PIP_INDEX" --trusted-host pypi.tuna.tsinghua.edu.cn \
            "fastapi>=0.115" "uvicorn[standard]>=0.30" \
            sqlalchemy alembic "pydantic>=2.0" pydantic-settings \
            cachetools scipy numpy pandas openpyxl structlog \
            python-multipart aiofiles \
            pytest pytest-asyncio httpx pytest-cov
    fi

    # 保存哈希印记
    echo "$current_hash" > "$PY_STAMP"
    info "Python deps installed ✓"
}

# ── 2. Node environment ───────────────────────────────────
setup_node() {
    if [[ "$MODE" == "--backend-only" ]]; then return; fi
    info "Checking Node.js environment…"

    if ! command -v node &>/dev/null; then
        error "Node.js is not installed. Install from https://nodejs.org or via nvm."
    fi

    # 计算 package-lock.json 哈希，与上次安装时的哈希对比
    local lockfile="$FRONTEND_DIR/package-lock.json"
    local current_hash=""
    [[ -f "$lockfile" ]] && current_hash="$(_md5 "$lockfile")"
    local saved_hash=""
    [[ -f "$NODE_STAMP" ]] && saved_hash="$(cat "$NODE_STAMP")"

    if [[ -d "$FRONTEND_DIR/node_modules" && "$current_hash" == "$saved_hash" && -n "$current_hash" ]]; then
        info "Node deps up-to-date (hash unchanged), skipping install"
        return
    fi

    cd "$FRONTEND_DIR"
    if [[ ! -f "package-lock.json" ]]; then
        info "No package-lock.json found — running npm install to generate it…"
        npm install --registry "$NPM_REGISTRY"
    else
        info "Installing frontend dependencies (via npmmirror 镜像)…"
        npm ci --registry "$NPM_REGISTRY"
    fi

    # 重新计算（npm install 可能更新了 lockfile）
    [[ -f "package-lock.json" ]] && current_hash="$(_md5 "package-lock.json")"
    [[ -n "$current_hash" ]] && echo "$current_hash" > "$NODE_STAMP"
    info "Node deps installed ✓"
}

# ── 3. Run tests ──────────────────────────────────────────
run_tests() {
    info "Running pytest…"
    source "$BACKEND_DIR/.venv/bin/activate"
    cd "$SCRIPT_DIR"   # 从root运行保证路径一致
    PYTHONPATH="$BACKEND_DIR" python -m pytest "$BACKEND_DIR/tests/" -v --tb=short
    info "All tests passed ✓"
    exit 0
}

# ── 4. Start backend ──────────────────────────────────────
start_backend() {
    info "Starting FastAPI backend on http://localhost:8000 …"
    cd "$SCRIPT_DIR"          # 始终从项目根目录启动，确保相对路径正确解析
    source "$BACKEND_DIR/.venv/bin/activate"

    # Export env vars from .env
    if [[ -f "$ENV_FILE" ]]; then
        set -o allexport
        # shellcheck source=/dev/null
        source "$ENV_FILE"
        set +o allexport
    fi

    export PYTHONPATH="$BACKEND_DIR"
    mkdir -p "$LOG_DIR"

    if [[ "$MODE" == "--backend-only" ]]; then
        uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    else
        uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload \
            >> "$LOG_DIR/backend.log" 2>&1 &
        BACKEND_PID=$!
        echo $BACKEND_PID > /tmp/rpt_backend.pid
        info "Backend PID: $BACKEND_PID  日志: logs/service/backend.log"
    fi
}

# ── 5. Start frontend dev server ──────────────────────────
start_frontend() {
    if [[ "$MODE" == "--backend-only" ]]; then return; fi
    info "Starting Vite frontend dev server on http://localhost:5173 …"
    cd "$FRONTEND_DIR"
    mkdir -p "$LOG_DIR"

    if [[ "$MODE" == "--frontend-only" ]]; then
        npm run dev -- --host
    else
        npm run dev -- --host \
            >> "$LOG_DIR/frontend.log" 2>&1 &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > /tmp/rpt_frontend.pid
        info "Frontend PID: $FRONTEND_PID  日志: logs/service/frontend.log"
    fi
}

# ── 停止已有服务 ─────────────────────────────────────────
stop_existing() {
    local killed=0

    # 通过 PID 文件停止（同时杀掉整个进程组，覆盖 uvicorn reloader 的子进程）
    for pidfile in /tmp/rpt_backend.pid /tmp/rpt_frontend.pid; do
        if [[ -f "$pidfile" ]]; then
            local pid
            pid="$(cat "$pidfile")"
            if kill -0 "$pid" 2>/dev/null; then
                info "Stopping existing process (PID $pid)…"
                # 先杀进程组（负号表示 PGID），再杀单个 PID 兜底
                kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null
                killed=1
            fi
            rm -f "$pidfile"
        fi
    done

    # 通过端口兜底：杀掉占用 8000 / 5173 的所有进程
    for port in 8000 5173; do
        local pids
        pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
        if [[ -n "$pids" ]]; then
            info "Port $port in use — stopping PID(s): $pids"
            echo "$pids" | xargs kill 2>/dev/null || true
            killed=1
        fi
    done

    # 轮询等待端口真正释放（最多 5 秒），超时则强制 kill -9
    if [[ $killed -eq 1 ]]; then
        info "Waiting for ports to release…"
        local waited=0
        while [[ $waited -lt 10 ]]; do
            local busy=0
            for port in 8000 5173; do
                lsof -ti tcp:"$port" &>/dev/null && busy=1 && break
            done
            [[ $busy -eq 0 ]] && break
            sleep 0.5
            waited=$((waited + 1))
        done
        # 如果仍有进程占用，强制 SIGKILL
        for port in 8000 5173; do
            local pids
            pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
            if [[ -n "$pids" ]]; then
                warn "Port $port still busy — force killing PID(s): $pids"
                echo "$pids" | xargs kill -9 2>/dev/null || true
                sleep 0.5
            fi
        done
    fi
}

# ── Cleanup on exit ───────────────────────────────────────
# (仅供手动调用，不注册 trap，避免后台进程随脚本退出被杀掉)
cleanup() {
    [[ -f /tmp/rpt_backend.pid ]]  && kill "$(cat /tmp/rpt_backend.pid)"  2>/dev/null || true
    [[ -f /tmp/rpt_frontend.pid ]] && kill "$(cat /tmp/rpt_frontend.pid)" 2>/dev/null || true
    rm -f /tmp/rpt_backend.pid /tmp/rpt_frontend.pid
}

# ── Main ──────────────────────────────────────────────────
info "=== 资源规划工具 启动脚本 ==="

stop_existing
setup_python

case "$MODE" in
    --test)
        run_tests
        ;;
    --frontend-only)
        setup_node
        start_frontend
        ;;
    --backend-only)
        start_backend
        ;;
    *)
        setup_node
        start_backend
        start_frontend
        info "═══════════════════════════════════════════"
        info "  Backend:  http://localhost:8000"
        info "  Frontend: http://localhost:5173"
        info "  API Docs: http://localhost:8000/docs"
        info "  停止服务: bash stop.sh"
        info "═══════════════════════════════════════════"
        ;;
esac
