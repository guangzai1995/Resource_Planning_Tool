"""bench_log 门面单测。"""
import logging
import re

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
