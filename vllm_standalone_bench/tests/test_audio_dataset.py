import argparse
import json

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
