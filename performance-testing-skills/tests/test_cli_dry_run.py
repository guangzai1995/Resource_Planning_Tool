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
