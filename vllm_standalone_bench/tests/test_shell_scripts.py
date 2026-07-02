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
            ],
        ),
        (
            "run_asr_bench.sh",
            [
                "run_bench_multi.py",
                "http://10.86.0.32:13001/v1/audio/transcriptions",
                "assets/librispeech_test_clean_256/asr_smoke.jsonl",
                "--backend",
                "openai-audio",
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


def test_run_asr_bench_executes_builtin_audio_dataset_command(tmp_path):
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
        "AUDIO_DURATION_S": "60",
        "AUDIO_SILENCE_MS": "750",
        "DRY_RUN": "false",
        "PARALLEL_NUMS": "1",
        "EPOCHS": "1",
    })

    subprocess.run(
        [str(SCRIPTS_DIR / "run_asr_bench.sh")],
        cwd=tmp_path,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    args = capture.read_text(encoding="utf-8").splitlines()
    dataset_path = SCRIPTS_DIR / "assets" / "librispeech_test_clean_256" / "asr_smoke.jsonl"

    assert args[0] == str(SCRIPTS_DIR / "run_bench_multi.py")
    assert args[args.index("--base-url") + 1] == "http://10.86.0.32:13001/v1"
    assert args[args.index("--model") + 1] == "Qwen3-ASR-1_7B"
    assert args[args.index("--backend") + 1] == "openai-audio"
    assert args[args.index("--dataset-name") + 1] == "custom_audio"
    assert args[args.index("--dataset-path") + 1] == str(dataset_path)
    assert args[args.index("--audio-duration-s") + 1] == "60"
    assert args[args.index("--audio-silence-ms") + 1] == "750"
    generated_audio_dir = Path(args[args.index("--generated-audio-dir") + 1])
    assert generated_audio_dir.parent == SCRIPTS_DIR / "results"
    assert generated_audio_dir.name.startswith("asr_dynamic_audio_")
