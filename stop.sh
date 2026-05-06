#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# 资源规划工具 — 停止服务脚本
# 用法: bash stop.sh
# ──────────────────────────────────────────────────────────
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

stopped=0

# ── 通过 PID 文件停止 ─────────────────────────────────────
for pidfile in /tmp/rpt_backend.pid /tmp/rpt_frontend.pid; do
    if [[ -f "$pidfile" ]]; then
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            info "Stopping process (PID $pid)…"
            kill "$pid" 2>/dev/null && stopped=1
        else
            warn "PID $pid not running (stale pidfile)"
        fi
        rm -f "$pidfile"
    fi
done

# ── 通过端口兜底 ──────────────────────────────────────────
for port in 8000 5173; do
    pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        info "Port $port still in use — stopping PID(s): $pids"
        echo "$pids" | xargs kill 2>/dev/null && stopped=1
    fi
done

if [[ $stopped -eq 1 ]]; then
    info "Waiting for ports to release…"
    local waited=0
    while [[ $waited -lt 10 ]]; do
        busy=0
        for port in 8000 5173; do
            lsof -ti tcp:"$port" &>/dev/null && busy=1 && break
        done
        [[ $busy -eq 0 ]] && break
        sleep 0.5
        waited=$((waited + 1))
    done
    # 超时仍占用则强制杀
    for port in 8000 5173; do
        pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
        if [[ -n "$pids" ]]; then
            warn "Port $port still busy — force killing PID(s): $pids"
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    done
    info "All services stopped ✓"
else
    info "No running services found."
fi
