"""bench_log 门面单测。"""
import logging
import re

import pytest

import bench_log


def _make_record(msg="hello", level=logging.INFO, name="auto_bench"):
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )


def test_file_formatter_has_timestamp_level_message():
    fmt = bench_log.FileFormatter()
    line = fmt.format(_make_record("starting serve"))
    assert "\033[" not in line
    assert "INFO" in line
    assert "starting serve" in line
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)


def test_console_formatter_color_when_enabled():
    fmt = bench_log.ConsoleFormatter(color=True)
    line = fmt.format(_make_record("boom", logging.ERROR))
    assert "\033[31m" in line
    assert "\033[0m" in line


def test_console_formatter_no_color_when_disabled():
    fmt = bench_log.ConsoleFormatter(color=False)
    line = fmt.format(_make_record("boom", logging.ERROR))
    assert "\033[" not in line


def test_case_scope_bench_prefix():
    rec = _make_record("bench input=4096")
    with bench_log.case_scope(total=8, phase="bench", idx=3):
        line = bench_log.FileFormatter().format(rec)
    assert "[case 3/8][bench]" in line


def test_case_scope_serve_prefix_no_idx():
    rec = _make_record("starting serve")
    with bench_log.case_scope(total=8, phase="serve", label="glm52_fp8"):
        line = bench_log.FileFormatter().format(rec)
    assert "[serve][glm52_fp8]" in line
    assert "[case" not in line


def test_case_scope_resets_after_block():
    rec = _make_record("outside")
    with bench_log.case_scope(total=8, phase="bench", idx=1):
        pass
    line = bench_log.FileFormatter().format(rec)
    assert "[case" not in line


def test_case_scope_propagates_across_function_call():
    """同线程内跨函数调用应继承当前前缀。"""
    captured = {}

    def inner():
        captured["line"] = bench_log.FileFormatter().format(_make_record("deep"))

    with bench_log.case_scope(total=4, phase="bench", idx=2):
        inner()
    assert "[case 2/4][bench]" in captured["line"]


def test_case_scope_resets_on_exception():
    rec = _make_record("after")
    with pytest.raises(RuntimeError):
        with bench_log.case_scope(total=8, phase="bench", idx=1):
            raise RuntimeError("boom")
    assert "[case" not in bench_log.FileFormatter().format(rec)
