import json
from pathlib import Path
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from scripts import perf_auto, perf_manual


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLI_TIMEOUT_SECONDS = 15


class OpenAIChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        json.loads(body.decode("utf-8"))

        if self.path != "/v1/chat/completions":
            encoded = b"not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return

        response = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


def run_manual_cli(
    *args: str,
    timeout: float = DEFAULT_CLI_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", "scripts/perf_manual.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def run_auto_cli(
    *args: str,
    timeout: float = DEFAULT_CLI_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", "scripts/perf_auto.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
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
def test_manual_cli_rejects_non_finite_timeout_values(timeout_value, capsys):
    with pytest.raises(SystemExit) as excinfo:
        perf_manual.main(["--dry-run", "--timeout-seconds", timeout_value])

    captured = capsys.readouterr()
    assert excinfo.value.code != 0
    assert "error:" in captured.err
    assert "argument --timeout-seconds" in captured.err


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
        timeout=DEFAULT_CLI_TIMEOUT_SECONDS,
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
    assert f"output dir: {ROOT / 'reports/latest'}" in result.stdout
    assert "concurrency=1 request_count=3" in result.stdout
    assert "concurrency=4 request_count=12" in result.stdout


def test_auto_cli_duration_dry_run_marks_duration_mode():
    result = run_auto_cli(
        "--dry-run",
        "--duration-seconds",
        "30",
        "--concurrency",
        "2",
    )

    assert result.returncode == 0, result.stderr
    assert "duration_seconds=30" in result.stdout
    assert "concurrency=2" in result.stdout
    assert "mode=duration" in result.stdout
    assert "initial_request_count=2" in result.stdout


def test_auto_cli_dry_run_resolves_relative_output_dir_to_package_root():
    result = run_auto_cli("--dry-run", "--output-dir", "relative-reports")

    assert result.returncode == 0, result.stderr
    assert f"output dir: {ROOT / 'relative-reports'}" in result.stdout


def test_auto_cli_all_failed_non_dry_run_returns_nonzero_and_writes_reports(tmp_path):
    output_dir = tmp_path / "reports"

    result = run_auto_cli(
        "--config",
        "configs/openai_asr.json",
        "--concurrency",
        "1",
        "--epochs",
        "1",
        "--timeout-seconds",
        "1",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode != 0
    assert "success_rate=0.000" in result.stdout
    assert "threshold_violation concurrency=1" in result.stderr
    assert (output_dir / "summary.md").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics[0]["n_requests"] == 1
    assert metrics[0]["n_failed"] == 1
    request_lines = (output_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(request_lines) == 1
    assert json.loads(request_lines[0])["error_type"] == "file_not_found"
    error_lines = (output_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(error_lines) == 1
    assert json.loads(error_lines[0])["error_type"] == "file_not_found"


def test_perf_auto_writes_reports_with_local_server(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "hello local", "expected_output_len": 8}) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "reports"
    server = HTTPServer(("127.0.0.1", 0), OpenAIChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "protocol": "openai_chat",
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                    "model": "local-test-model",
                    "dataset": {
                        "type": "text_prompts",
                        "path": str(dataset_path),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        result = run_auto_cli(
            "--config",
            str(config_path),
            "--concurrency",
            "1,2",
            "--epochs",
            "2",
            "--timeout-seconds",
            "2",
            "--output-dir",
            str(output_dir),
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert result.returncode == 0, result.stderr
    assert "concurrency=1 request_count=2" in result.stdout
    assert "concurrency=2 request_count=4" in result.stdout

    for report_name in [
        "summary.md",
        "metrics.json",
        "metrics.csv",
        "requests.jsonl",
        "errors.jsonl",
    ]:
        assert (output_dir / report_name).exists()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics) == 2
    metrics_by_concurrency = {row["concurrency"]: row for row in metrics}
    assert set(metrics_by_concurrency) == {1, 2}
    assert metrics_by_concurrency[1]["n_requests"] == 2
    assert metrics_by_concurrency[2]["n_requests"] == 4
    for row in metrics:
        assert row["n_success"] == row["n_requests"]
        assert row["n_failed"] == 0
        assert row["success_rate"] == 1.0

    request_lines = (output_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(request_lines) == 6
    request_rows = [json.loads(line) for line in request_lines]
    assert all(row["status_code"] == 200 for row in request_rows)
    assert all(row["input_tokens"] == 3 for row in request_rows)
    assert all(row["output_tokens"] == 2 for row in request_rows)
    assert all(row["response_summary"] == "ok" for row in request_rows)
    assert (output_dir / "errors.jsonl").read_text(encoding="utf-8") == ""


def test_auto_cli_tiny_duration_submits_at_least_one_batch(tmp_path):
    output_dir = tmp_path / "duration-reports"

    result = run_auto_cli(
        "--config",
        "configs/openai_asr.json",
        "--duration-seconds",
        "1e-300",
        "--timeout-seconds",
        "1",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode != 0
    assert "request_count=1" in result.stdout
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics[0]["n_requests"] >= 1
    assert metrics[0]["n_failed"] >= 1


def test_run_auto_shell_wrapper_forwards_to_auto_cli():
    result = subprocess.run(
        ["scripts/run_auto.sh", "--dry-run", "--concurrency", "2", "--epochs", "2"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=DEFAULT_CLI_TIMEOUT_SECONDS,
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
def test_auto_cli_rejects_invalid_numeric_arguments(option, value, capsys):
    with pytest.raises(SystemExit) as excinfo:
        perf_auto.main(["--dry-run", option, value])

    captured = capsys.readouterr()
    assert excinfo.value.code != 0
    assert "error:" in captured.err
