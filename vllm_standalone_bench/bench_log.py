"""编排器统一日志门面：零第三方依赖的结构化彩色日志。

设计见 docs/superpowers/specs/2026-07-04-orchestrator-logging-design.md。
- FileHandler 始终写 controller.log（纯文本 + 完整时间戳，结构化全量）
- ConsoleHandler 仅 sys.stderr.isatty() 时挂（彩色）
- case_scope 注入 [case idx/total][phase] 或 [phase][label] 前缀（contextvars）

本模块只实现颜色常量、_CTX ContextVar 占位、File/Console Formatter。
case_scope / setup_logging / get_logger 由后续任务追加。
"""
from __future__ import annotations

import contextvars
import logging


# case 上下文占位：case_scope（后续任务）会 set 为 (idx, total, phase, label)。
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
