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
