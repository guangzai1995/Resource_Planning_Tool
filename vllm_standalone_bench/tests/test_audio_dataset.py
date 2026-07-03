import argparse
import json
import os
import sys
from pathlib import Path

import pytest

import run_bench_serve as rbs


def _write_tiny_audio_dataset(tmp_path: Path, *, count: int = 2) -> Path:
    """Create a minimal custom_audio dataset of short WAV files; return jsonl path."""
    import numpy as np
    import soundfile

    src = tmp_path / "src"
    src.mkdir()
    sample_rate = 16_000
    for idx in range(count):
        soundfile.write(
            src / f"s{idx}.wav",
            np.full(int(sample_rate * 0.25), 0.1, dtype=np.float32),
            sample_rate,
        )
    jsonl = tmp_path / "asr.jsonl"
    jsonl.write_text(
        "".join(
            json.dumps({"audio": f"src/s{i}.wav", "output_tokens": 8}) + "\n"
            for i in range(count)
        ),
        encoding="utf-8",
    )
    return jsonl


def test_custom_audio_dataset_reads_jsonl_and_repeats_to_num_prompts(tmp_path):
    audio = tmp_path / "audio" / "sample.flac"
    audio.parent.mkdir()
    audio.write_bytes(b"fake")
    jsonl = tmp_path / "asr_smoke.jsonl"
    jsonl.write_text(
        json.dumps({
            "prompt": "Transcribe the audio in English.",
            "audio": "audio/sample.flac",
            "output_tokens": 96,
            "reference": "REFERENCE TEXT",
        }) + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        dataset_name="custom_audio",
        dataset_path=str(jsonl),
        num_prompts=3,
        custom_output_len=None,
        random_output_len=128,
        seed=123,
    )

    requests = rbs.get_samples(args, tokenizer=None)

    assert len(requests) == 3
    assert requests[0].prompt == "Transcribe the audio in English."
    assert requests[0].prompt_len == 0
    assert requests[0].expected_output_len == 96
    assert requests[0].multi_modal_data == {"audio_path": str(audio)}
    assert [request.expected_output_len for request in requests] == [96, 96, 96]
    assert [request.multi_modal_data for request in requests] == [
        {"audio_path": str(audio)},
        {"audio_path": str(audio)},
        {"audio_path": str(audio)},
    ]
    assert requests[0].request_id.endswith("-0")
    assert requests[1].request_id.endswith("-1")


def test_custom_audio_dataset_custom_output_len_overrides_rows(tmp_path):
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"fake")
    jsonl = tmp_path / "asr.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "p", "audio": "sample.flac", "output_tokens": 32}) + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        dataset_name="custom_audio",
        dataset_path=str(jsonl),
        num_prompts=1,
        custom_output_len=144,
        random_output_len=128,
        seed=123,
    )

    requests = rbs.get_samples(args, tokenizer=None)

    assert requests[0].expected_output_len == 144


def test_custom_audio_dataset_generates_duration_target_audio(tmp_path):
    import numpy as np
    import soundfile

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    sample_rate = 16_000
    rows = []
    for idx, amplitude in enumerate((0.1, 0.2, 0.3, 0.4)):
        audio_path = src_dir / f"sample_{idx}.wav"
        data = np.full(int(sample_rate * 0.25), amplitude, dtype=np.float32)
        soundfile.write(audio_path, data, sample_rate, format="WAV")
        rows.append({
            "prompt": "Transcribe the audio in English.",
            "audio": str(audio_path.relative_to(tmp_path)),
            "output_tokens": 64,
            "reference": f"REFERENCE {idx}",
        })

    jsonl = tmp_path / "asr.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    args = argparse.Namespace(
        dataset_name="custom_audio",
        dataset_path=str(jsonl),
        num_prompts=3,
        custom_output_len=None,
        random_output_len=128,
        seed=123,
        audio_duration_s=0.8,
        audio_silence_ms=100,
        generated_audio_dir=str(generated_dir),
    )

    requests = rbs.get_samples(args, tokenizer=None)

    audio_paths = [
        Path(request.multi_modal_data["audio_path"])
        for request in requests
    ]
    assert len(audio_paths) == 3
    assert len(set(audio_paths)) == 3
    assert all(path.exists() for path in audio_paths)
    assert all(0.79 <= soundfile.info(path).duration <= 0.81 for path in audio_paths)
    assert [request.expected_output_len for request in requests] == [64, 64, 64]

    manifest = json.loads((generated_dir / "manifest.json").read_text(
        encoding="utf-8",
    ))
    assert manifest["target_duration_s"] == 0.8
    assert manifest["sample_count"] == 3
    assert manifest["silence_ms"] == 100

    generated_rows = [
        json.loads(line)
        for line in (generated_dir / "asr_dynamic.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
    ]
    assert len(generated_rows) == 3
    assert len({tuple(row["source_audio"][:1]) for row in generated_rows}) == 3


def test_custom_audio_dataset_malformed_json_reports_file_and_line(tmp_path):
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text('{"audio": "sample.flac"\n', encoding="utf-8")
    args = argparse.Namespace(
        dataset_name="custom_audio",
        dataset_path=str(jsonl),
        num_prompts=1,
        custom_output_len=None,
        random_output_len=128,
        seed=123,
    )

    with pytest.raises(ValueError) as excinfo:
        rbs.get_samples(args, tokenizer=None)

    assert f"{jsonl}:1 JSON 格式错误:" in str(excinfo.value)


def test_custom_audio_dataset_non_object_json_reports_file_and_line(tmp_path):
    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text(json.dumps(["sample.flac"]) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        dataset_name="custom_audio",
        dataset_path=str(jsonl),
        num_prompts=1,
        custom_output_len=None,
        random_output_len=128,
        seed=123,
    )

    with pytest.raises(ValueError) as excinfo:
        rbs.get_samples(args, tokenizer=None)

    assert str(excinfo.value) == f"{jsonl}:1 必须是 JSON 对象"


def test_dataset_path_help_mentions_custom_audio():
    parser = argparse.ArgumentParser()

    rbs.add_dataset_parser(parser)

    dataset_path_action = next(
        action
        for action in parser._actions
        if "--dataset-path" in action.option_strings
    )
    assert "custom_audio" in dataset_path_action.help


def test_dynamic_audio_auto_cleaned_when_no_generated_dir(tmp_path, monkeypatch):
    """No --generated-audio-dir => temp dir under TMPDIR, scheduled for cleanup."""
    # isolate TMPDIR so the auto-generated dir lands under tmp_path, not real /tmp
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    # drain any state left over from earlier tests in this same process
    rbs._cleanup_dynamic_audio_dirs()

    jsonl = _write_tiny_audio_dataset(tmp_path)
    args = argparse.Namespace(
        dataset_name="custom_audio",
        dataset_path=str(jsonl),
        num_prompts=2,
        custom_output_len=None,
        random_output_len=128,
        seed=123,
        audio_duration_s=0.5,
        audio_silence_ms=100,
        generated_audio_dir=None,  # <-- the leak path: falls back to /tmp default
    )

    requests = rbs.get_samples(args, tokenizer=None)
    audio_paths = [Path(r.multi_modal_data["audio_path"]) for r in requests]
    generated_dir = audio_paths[0].parent

    # landed under TMPDIR (not results/), files exist for the duration of the run
    assert tmp_path in generated_dir.parents
    assert all(path.exists() for path in audio_paths)
    # registered for auto-cleanup at process exit
    assert generated_dir in rbs._dynamic_audio_cleanup_dirs

    # simulate process-exit cleanup
    rbs._cleanup_dynamic_audio_dirs()

    assert not generated_dir.exists()
    assert rbs._dynamic_audio_cleanup_dirs == set()


def test_dynamic_audio_keeps_configured_generated_dir(tmp_path, monkeypatch):
    """Explicit --generated-audio-dir is retained (never auto-cleaned)."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    rbs._cleanup_dynamic_audio_dirs()

    jsonl = _write_tiny_audio_dataset(tmp_path)
    configured = tmp_path / "keep" / "generated"
    args = argparse.Namespace(
        dataset_name="custom_audio",
        dataset_path=str(jsonl),
        num_prompts=2,
        custom_output_len=None,
        random_output_len=128,
        seed=123,
        audio_duration_s=0.5,
        audio_silence_ms=100,
        generated_audio_dir=str(configured),
    )

    requests = rbs.get_samples(args, tokenizer=None)
    generated_dir = Path(requests[0].multi_modal_data["audio_path"]).parent

    assert generated_dir == configured
    assert generated_dir not in rbs._dynamic_audio_cleanup_dirs

    rbs._cleanup_dynamic_audio_dirs()  # must NOT touch the configured dir
    assert generated_dir.exists()


def test_dynamic_audio_auto_cleanup_runs_at_process_exit(tmp_path):
    """End-to-end: atexit must remove the auto-generated temp dir when the
    benchmark process exits (the exact scenario the user reported leaking)."""
    import subprocess
    import textwrap

    bench_dir = Path(__file__).resolve().parent.parent
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            """
            import argparse, json, os
            import numpy as np
            import soundfile
            import run_bench_serve as rbs

            work = os.environ["WORK_DIR"]
            src = os.path.join(work, "src")
            os.makedirs(src)
            sr = 16000
            for i in range(2):
                soundfile.write(os.path.join(src, f"s{i}.wav"),
                                np.full(int(sr * 0.25), 0.1, dtype=np.float32), sr)
            jsonl = os.path.join(work, "asr.jsonl")
            with open(jsonl, "w", encoding="utf-8") as f:
                for i in range(2):
                    f.write(json.dumps({"audio": f"src/s{i}.wav", "output_tokens": 8}) + "\\n")
            args = argparse.Namespace(
                dataset_name="custom_audio", dataset_path=jsonl, num_prompts=2,
                custom_output_len=None, random_output_len=128, seed=123,
                audio_duration_s=0.5, audio_silence_ms=100, generated_audio_dir=None)
            reqs = rbs.get_samples(args, tokenizer=None)
            print(os.path.dirname(reqs[0].multi_modal_data["audio_path"]))
            """
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(bench_dir)
    env["WORK_DIR"] = str(tmp_path)
    env["TMPDIR"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr

    generated_dir = Path(proc.stdout.strip().splitlines()[-1])
    # the path was produced during the run; after the process exited, atexit
    # must have removed it — otherwise it is exactly the reported leak.
    assert not generated_dir.exists(), (
        f"dynamic audio temp dir leaked after process exit: {generated_dir}"
    )
