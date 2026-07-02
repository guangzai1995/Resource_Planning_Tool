import os
import subprocess
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script_name", "expected_fragments"),
    [
        (
            "build_bench_runner.sh",
            [
                "Dockerfile.bench-runner",
                "vllm-bench-runner:offline",
                "docker build",
            ],
        ),
        (
            "run_auto_bench.sh",
            [
                "auto_bench.py",
                "auto_bench.qwen2_5_1_5b.smoke.json",
                "--detach",
                "resume",
            ],
        ),
    ],
)
def test_helper_shell_scripts_are_ready_to_use(script_name, expected_fragments):
    script = SCRIPTS_DIR / script_name

    assert script.exists()
    assert os.access(script, os.X_OK)
    subprocess.run(["bash", "-n", str(script)], check=True)

    text = script.read_text(encoding="utf-8")
    assert any(ord(char) > 127 for char in text)
    for fragment in expected_fragments:
        assert fragment in text


def test_run_auto_bench_uses_project_root_as_working_directory(tmp_path):
    capture = tmp_path / "capture.txt"
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"pwd > {capture}",
            f"printf '%s\\n' \"$@\" >> {capture}",
        ]),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": str(fake_python),
        "RUN_ID": "cwd_check",
        "DRY_RUN": "true",
    })

    subprocess.run(
        [str(SCRIPTS_DIR / "run_auto_bench.sh")],
        cwd=SCRIPTS_DIR,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    lines = capture.read_text(encoding="utf-8").splitlines()
    assert Path(lines[0]) == SCRIPTS_DIR.parent
    assert lines[1] == str(SCRIPTS_DIR / "auto_bench.py")
    assert lines[2] == "run"


def test_run_auto_bench_resume_forwards_detach(tmp_path):
    capture = tmp_path / "capture.txt"
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"printf '%s\\n' \"$@\" > {capture}",
        ]),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": str(fake_python),
        "RUN_ID": "resume_check",
        "DETACH": "true",
    })

    subprocess.run(
        [str(SCRIPTS_DIR / "run_auto_bench.sh"), "resume"],
        cwd=SCRIPTS_DIR,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    lines = capture.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [str(SCRIPTS_DIR / "auto_bench.py"), "resume"]
    assert "--results-dir" in lines
    assert "--run-id" in lines
    assert "resume_check" in lines
    assert "--detach" in lines
