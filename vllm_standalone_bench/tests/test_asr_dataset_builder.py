import json
from pathlib import Path

from vllm_standalone_bench.tools import build_librispeech_asr_smoke as builder


def _sample(idx: int, duration_s: float, size_bytes: int = 10) -> builder.LibriSpeechSample:
    return builder.LibriSpeechSample(
        speaker_id="1089",
        chapter_id="134686",
        utterance_id=f"1089-134686-{idx:04d}",
        audio_path=Path(f"/src/{idx}.flac"),
        text=f"sample transcript {idx}",
        duration_s=duration_s,
        size_bytes=size_bytes,
    )


def test_duration_bucket_uses_medium_long_xlong_ranges():
    assert builder.duration_bucket(4.99) is None
    assert builder.duration_bucket(5.0) == "medium"
    assert builder.duration_bucket(9.99) == "medium"
    assert builder.duration_bucket(10.0) == "long"
    assert builder.duration_bucket(19.99) == "long"
    assert builder.duration_bucket(20.0) == "xlong"
    assert builder.duration_bucket(30.0) == "xlong"
    assert builder.duration_bucket(30.01) is None


def test_duration_bucket_uses_configured_outer_duration_bounds(monkeypatch):
    monkeypatch.setattr(builder, "DEFAULT_MIN_DURATION_S", 6.0)
    monkeypatch.setattr(builder, "DEFAULT_MAX_DURATION_S", 25.0)

    assert builder.duration_bucket(5.99) is None
    assert builder.duration_bucket(6.0) == "medium"
    assert builder.duration_bucket(25.0) == "xlong"
    assert builder.duration_bucket(25.01) is None


def test_select_samples_is_seeded_and_balanced():
    samples = (
        [_sample(i, 5.5) for i in range(20)]
        + [_sample(100 + i, 12.0) for i in range(20)]
        + [_sample(200 + i, 24.0) for i in range(20)]
    )

    first = builder.select_samples(samples, target_count=20, seed=20260701)
    second = builder.select_samples(samples, target_count=20, seed=20260701)

    assert [s.utterance_id for s in first] == [s.utterance_id for s in second]
    counts = builder.bucket_counts(first)
    assert counts == {"medium": 7, "long": 9, "xlong": 4}
    assert all(5.0 <= s.duration_s <= 30.0 for s in first)


def test_select_samples_falls_back_when_a_bucket_is_short():
    samples = (
        [_sample(i, 5.5) for i in range(2)]
        + [_sample(100 + i, 12.0) for i in range(20)]
        + [_sample(200 + i, 24.0) for i in range(20)]
    )

    selected = builder.select_samples(samples, target_count=12, seed=20260701)

    assert len(selected) == 12
    assert builder.bucket_counts(selected)["medium"] == 2


def test_write_dataset_copies_audio_and_writes_relative_jsonl(tmp_path):
    source = tmp_path / "source.flac"
    source.write_bytes(b"fake-audio")
    sample = builder.LibriSpeechSample(
        speaker_id="1089",
        chapter_id="134686",
        utterance_id="1089-134686-0001",
        audio_path=source,
        text="A LONG ENOUGH TRANSCRIPT",
        duration_s=12.5,
        size_bytes=source.stat().st_size,
    )

    manifest = builder.write_dataset([sample], tmp_path / "out", seed=20260701)

    jsonl_path = tmp_path / "out" / "asr_smoke.jsonl"
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {
            "prompt": "Transcribe the audio in English.",
            "audio": "audio/1089-134686-0001.flac",
            "output_tokens": 128,
            "reference": "A LONG ENOUGH TRANSCRIPT",
        }
    ]
    assert (tmp_path / "out" / "audio" / "1089-134686-0001.flac").read_bytes() == (
        b"fake-audio"
    )
    assert manifest["sample_count"] == 1
    assert manifest["duration_buckets"] == {"medium": 0, "long": 1, "xlong": 0}
    assert manifest["seed"] == 20260701
    assert "OpenSLR" in (tmp_path / "out" / "ATTRIBUTION.md").read_text(
        encoding="utf-8"
    )
    assert "Creative Commons Attribution 4.0" in (
        tmp_path / "out" / "LICENSE.LibriSpeech.txt"
    ).read_text(encoding="utf-8")


def test_apply_size_budget_removes_longest_samples_first():
    samples = [
        _sample(1, 8.0, size_bytes=30),
        _sample(2, 25.0, size_bytes=40),
        _sample(3, 12.0, size_bytes=35),
    ]

    trimmed = builder.apply_size_budget(samples, max_bytes=65)

    assert [s.utterance_id for s in trimmed] == [
        "1089-134686-0001",
        "1089-134686-0003",
    ]
