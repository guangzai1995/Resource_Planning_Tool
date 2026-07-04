"""编排器统一日志门面：零第三方依赖的结构化彩色日志。

设计见 docs/superpowers/specs/2026-07-04-orchestrator-logging-design.md。
- FileHandler 始终写 controller.log（纯文本 + 完整时间戳，结构化全量）
- ConsoleHandler 仅 sys.stderr.isatty() 时挂（彩色）
- case_scope 注入 [case idx/total][phase] 或 [phase][label] 前缀（contextvars）

本模块已完整实现：颜色常量、`_CTX` ContextVar、File/Console Formatter、
`case_scope`、`setup_logging`、`get_logger`。
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import sys
from pathlib import Path


# case 上下文：case_scope 运行时 set 为 (idx, total, phase, label)。
# idx 为 None 时表示无 case 序号、仅 phase+label。默认 None 表示无上下文。
_CTX: contextvars.ContextVar[tuple | None] = contextvars.ContextVar(
    "bench_log_case_ctx", default=None,
)


class _Color:
    RESET = "\033[0m"
    GREY = "\033[90m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    # GREEN 预留给 task 7 status 展示上色，非日志级别使用（LEVEL_COLOR 不引用）。
    GREEN = "\033[32m"


LEVEL_COLOR = {
    "DEBUG": _Color.GREY,
    "INFO": _Color.GREY,
    "WARNING": _Color.YELLOW,
    "ERROR": _Color.RED,
    "CRITICAL": _Color.RED,
}


# File/Console Formatter 共用的行格式：时间 + 级别 + case 前缀 + 消息。
# case_prefix 非空时末尾自带空格（见 _current_prefix），故此处紧贴 %(message)s。
_LINE_FMT = "%(asctime)s %(levelname)-5s %(case_prefix)s%(message)s"


def _current_prefix() -> str:
    """读取当前 case 上下文前缀；无上下文时返回空串。

    非空前缀末尾带一个空格，便于与 message 分隔（对齐 spec 示例
    ``INFO [case 3/8][bench] bench input=...``）。
    """
    ctx = _CTX.get()
    if not ctx:
        return ""
    idx, total, phase, label = ctx
    if idx is not None:
        return f"[case {idx}/{total}][{phase}] "
    return f"[{phase}][{label}] "


@contextlib.contextmanager
def case_scope(*, total: int, phase: str, idx: int | None = None,
               label: str | None = None):
    """with 块内所有日志自动带前缀。

    idx 非 None（bench per-case 阶段）→ ``[case idx/total][phase]``；
    idx 为 None（serve per-group 阶段）→ ``[phase][label]``。
    调用方需保证 idx 与 label 至少一个非 None（idx 非 None 走 bench 前缀，否则走 serve 前缀）。
    基于 contextvars，同线程内跨函数自动传播；with 块退出后前缀复位。
    """
    token = _CTX.set((idx, total, phase, label))
    try:
        yield
    finally:
        _CTX.reset(token)


class _BaseFormatter(logging.Formatter):
    """把当前 case 前缀注入 record.case_prefix 后再按 fmt 渲染。"""

    def format(self, record: logging.LogRecord) -> str:
        record.case_prefix = _current_prefix()
        return super().format(record)


class FileFormatter(_BaseFormatter):
    """controller.log 用：纯文本、完整日期时间、无 ANSI 转义。"""

    def __init__(self) -> None:
        super().__init__(
            fmt=_LINE_FMT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class ConsoleFormatter(_BaseFormatter):
    """控制台用：仅时间、按级别上色（color=False 时退化为纯文本）。"""

    def __init__(self, color: bool = True) -> None:
        super().__init__(
            fmt=_LINE_FMT,
            datefmt="%H:%M:%S",
        )
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not self.color:
            return msg
        c = LEVEL_COLOR.get(record.levelname, "")
        return f"{c}{msg}{_Color.RESET}" if c else msg


def setup_logging(run_dir: Path, level: str = "INFO", *, color: bool | None = None,
                  log_file: Path | None = None) -> None:
    """初始化 root logger：FileHandler（始终）+ ConsoleHandler（仅 color=True）。

    color=None 时按 sys.stderr.isatty() 自动决定；isatty 检测失败回退 False。
    detach 模式 stderr 被重定向（非 tty）→ color=False → 不挂彩色 ConsoleHandler，
    避免与 FileHandler 双写 controller.log。幂等：先清 root 旧 handler 再挂。
    """
    logging.raiseExceptions = False
    if color is None:
        try:
            color = sys.stderr.isatty()
        except (OSError, ValueError):
            color = False
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    root.setLevel(level)
    target = Path(log_file) if log_file else (Path(run_dir) / "controller.log")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(target, mode="a", encoding="utf-8")
        fh.setFormatter(FileFormatter())
        fh.setLevel(level)
        root.addHandler(fh)
    except OSError:
        # FileHandler 创建失败（磁盘满/权限）→ 静默降级；color=True 时下面仍挂 ConsoleHandler
        pass
    if color:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(ConsoleFormatter(color=True))
        ch.setLevel(level)
        root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    """返回已配置好的 logger，调用方直接 .info/.warning/.error。"""
    return logging.getLogger(name)
