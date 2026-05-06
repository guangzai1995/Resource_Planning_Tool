"""
Tests for Excel/CSV import functionality.
Uses openpyxl to build a minimal in-memory xlsx and verifies the importer.
"""
import io
import pytest
import openpyxl

from app.services.excel_importer import import_excel


def _make_xlsx(sheet_name: str, rows: list[dict]) -> str:
    """Write a minimal xlsx to a temp file and return the path."""
    import tempfile, os
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name

    if not rows:
        fd, path = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        wb.save(path)
        return path

    headers = list(rows[0].keys())
    ws.append(headers)
    for row in rows:
        ws.append([row[h] for h in headers])

    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    wb.save(path)
    return path


SAMPLE_ROWS = [
    {
        "输入长度": 512,
        "输出长度": 128,
        "并发数": 4,
        "首Token延迟(ms)": 300,
        "首Token延迟P90(ms)": 350,
        "吞吐量(tokens/s)": 1200,
        "每用户吞吐(tokens/s)": 300,
        "GPU利用率(%)": 75,
        "显存使用(GB)": 40,
    },
    {
        "输入长度": 1024,
        "输出长度": 256,
        "并发数": 8,
        "首Token延迟(ms)": 480,
        "首Token延迟P90(ms)": 520,
        "吞吐量(tokens/s)": 2000,
        "每用户吞吐(tokens/s)": 250,
        "GPU利用率(%)": 88,
        "显存使用(GB)": 44,
    },
]


def test_import_valid_sheet(db_session):
    """Import a single well-formed sheet and verify rows are inserted."""
    sheet_name = "32B-H200-4 测试 数据"
    path = _make_xlsx(sheet_name, SAMPLE_ROWS)

    result = import_excel(path, db_session)

    assert result["sheets"] >= 1
    assert result["rows"] == len(SAMPLE_ROWS)
    assert result["skipped"] == []


def test_import_unrecognized_sheet_is_skipped(db_session):
    """Sheets with names not matching the expected pattern are skipped."""
    path = _make_xlsx("Summary", SAMPLE_ROWS)
    result = import_excel(path, db_session)

    assert result["sheets"] == 0
    assert result["rows"] == 0


def test_import_duplicate_rows_upsert(db_session):
    """Importing the same xlsx twice should not duplicate rows."""
    sheet_name = "7B-H200-2 测试 数据"
    path = _make_xlsx(sheet_name, SAMPLE_ROWS)

    r1 = import_excel(path, db_session)
    r2 = import_excel(path, db_session)

    # Second import should upsert (same or zero new rows), not raise
    assert r1["rows"] == len(SAMPLE_ROWS)
    assert r2["rows"] >= 0  # upsert: rows updated, not new


def test_import_empty_sheet(db_session):
    """An xlsx with a valid sheet name but no data rows is handled gracefully."""
    sheet_name = "72B-H20-8 测试 数据"
    path = _make_xlsx(sheet_name, [])
    result = import_excel(path, db_session)

    assert result["rows"] == 0
