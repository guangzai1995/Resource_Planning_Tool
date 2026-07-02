import argparse
import json
from pathlib import Path

import pytest

import run_bench_serve as rbs


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
