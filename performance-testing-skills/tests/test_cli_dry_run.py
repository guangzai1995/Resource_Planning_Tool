import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_manual_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", "scripts/perf_manual.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def run_auto_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", "scripts/perf_auto.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_manual_cli_dry_run_outputs_request_and_curl():
    result = run_manual_cli("--dry-run", "--input", "hello dry run")

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "protocol: openai_chat" in result.stdout
    assert "method: POST" in result.stdout
    assert "url: http://127.0.0.1:8000/v1/chat/completions" in result.stdout
    assert "curl" in result.stdout
    assert "hello dry run" in result.stdout


def test_manual_cli_mode_curl_only_outputs_curl():
    result = run_manual_cli("--mode", "curl", "--input", "hello curl")

    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("curl ")
    assert "DRY RUN" not in result.stdout
    assert "hello curl" in result.stdout


@pytest.mark.parametrize("timeout_value", ["nan", "inf"])
def test_manual_cli_rejects_non_finite_timeout_values(timeout_value):
    result = run_manual_cli("--dry-run", "--timeout-seconds", timeout_value)

    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "argument --timeout-seconds" in result.stderr


def test_manual_cli_saves_failed_non_dry_run_response_for_missing_asr_audio(tmp_path):
    save_path = tmp_path / "responses.json"

    result = run_manual_cli(
        "--config",
        "configs/openai_asr.json",
        "--save-response",
        str(save_path),
        "--timeout-seconds",
        "1",
    )

    assert result.returncode == 1
    assert "success=False" in result.stdout
    assert "file_not_found" in result.stdout
    assert save_path.exists()
    payload = json.loads(save_path.read_text(encoding="utf-8"))
    assert isinstance(payload["requests"], list)
    assert payload["requests"][0]["success"] is False
    assert payload["requests"][0]["error_type"] == "file_not_found"


def test_manual_cli_dry_run_save_response_creates_parent_directories(tmp_path):
    save_path = tmp_path / "nested" / "out.json"

    result = run_manual_cli("--dry-run", "--save-response", str(save_path))

    assert result.returncode == 0, result.stderr
    assert save_path.parent.is_dir()
    assert json.loads(save_path.read_text(encoding="utf-8")) == {"requests": []}


def test_run_manual_shell_wrapper_forwards_to_manual_cli():
    result = subprocess.run(
        ["scripts/run_manual.sh", "--dry-run", "--input", "hello wrapper"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "curl" in result.stdout
    assert "hello wrapper" in result.stdout


def test_auto_cli_dry_run_outputs_plan_for_each_concurrency():
    result = run_auto_cli("--dry-run", "--concurrency", "1,4", "--epochs", "3")

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "config path:" in result.stdout
    assert "configs/openai_chat.json" in result.stdout
    assert "dataset path:" in result.stdout
    assert "output dir: reports/latest" in result.stdout
    assert "concurrency=1 request_count=3" in result.stdout
    assert "concurrency=4 request_count=12" in result.stdout


def test_run_auto_shell_wrapper_forwards_to_auto_cli():
    result = subprocess.run(
        ["scripts/run_auto.sh", "--dry-run", "--concurrency", "2", "--epochs", "2"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "concurrency=2 request_count=4" in result.stdout


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--concurrency", "0"),
        ("--concurrency", "1,0"),
        ("--concurrency", "1,nope"),
        ("--timeout-seconds", "0"),
        ("--timeout-seconds", "nan"),
        ("--timeout-seconds", "inf"),
        ("--max-error-rate", "-0.01"),
        ("--max-error-rate", "1.01"),
        ("--max-error-rate", "nan"),
        ("--max-p90-latency-ms", "0"),
        ("--max-p90-latency-ms", "inf"),
    ],
)
def test_auto_cli_rejects_invalid_numeric_arguments(option, value):
    result = run_auto_cli("--dry-run", option, value)

    assert result.returncode != 0
    assert "error:" in result.stderr
