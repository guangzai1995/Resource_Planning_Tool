import json
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_perf_auto_dry_run_writes_plan(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "scripts/perf_auto.py"),
            "--config",
            str(PACKAGE_ROOT / "configs/openai_chat.json"),
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--limit",
            "3",
        ],
        cwd=PACKAGE_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads((tmp_path / "dry_run_plan.json").read_text(encoding="utf-8"))
    assert plan["target_type"] == "openai-chat"
    assert plan["request_count"] == 3
    assert plan["requests"][0]["body"]["messages"][0]["content"]


def test_shell_wrapper_dry_run_writes_plan(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(PACKAGE_ROOT / "scripts/run_auto.sh"),
            "--config",
            str(PACKAGE_ROOT / "configs/generic_http.json"),
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--limit",
            "2",
        ],
        cwd=PACKAGE_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads((tmp_path / "dry_run_plan.json").read_text(encoding="utf-8"))
    assert plan["target_type"] == "generic-http"
    assert plan["request_count"] == 2


def test_perf_manual_dry_run_prints_prepared_request():
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "scripts/perf_manual.py"),
            "--url",
            "http://127.0.0.1:8000/echo",
            "--body",
            '{"prompt": "hello"}',
            "--header",
            "X-Test: yes",
            "--dry-run",
        ],
        cwd=PACKAGE_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["url"] == "http://127.0.0.1:8000/echo"
    assert payload["headers"]["X-Test"] == "yes"
    assert payload["body"]["prompt"] == "hello"
