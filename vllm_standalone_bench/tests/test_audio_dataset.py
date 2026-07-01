import argparse
import json

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
