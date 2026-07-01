import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def _write_tar(tar_path: Path, member: tarfile.TarInfo, data: bytes = b"") -> None:
    if data:
        member.size = len(data)
        fileobj = io.BytesIO(data)
    else:
        fileobj = None
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.addfile(member, fileobj=fileobj)


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

    manifest = builder.write_dataset(
        [sample],
        tmp_path / "out",
        seed=20260701,
        requested_sample_count=1,
    )

    jsonl_path = tmp_path / "out" / "asr_smoke.jsonl"
    manifest_path = tmp_path / "out" / "manifest.json"
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    assert manifest["requested_sample_count"] == 1
    assert disk_manifest == manifest
    assert {
        key: disk_manifest[key]
        for key in (
            "sample_count",
            "duration_buckets",
            "seed",
            "requested_sample_count",
        )
    } == {
        "sample_count": 1,
        "duration_buckets": {"medium": 0, "long": 1, "xlong": 0},
        "seed": 20260701,
        "requested_sample_count": 1,
    }
    assert "OpenSLR" in (tmp_path / "out" / "ATTRIBUTION.md").read_text(
        encoding="utf-8"
    )
    assert "Creative Commons Attribution 4.0" in (
        tmp_path / "out" / "LICENSE.LibriSpeech.txt"
    ).read_text(encoding="utf-8")


def test_write_dataset_rejects_path_escape_utterance_id(tmp_path):
    source = tmp_path / "source.flac"
    source.write_bytes(b"fake-audio")
    sample = builder.LibriSpeechSample(
        speaker_id="1089",
        chapter_id="134686",
        utterance_id="../evil",
        audio_path=source,
        text="A LONG ENOUGH TRANSCRIPT",
        duration_s=12.5,
        size_bytes=source.stat().st_size,
    )

    with pytest.raises(ValueError, match="Invalid LibriSpeech utterance id"):
        builder.write_dataset([sample], tmp_path / "out", seed=20260701)

    assert not (tmp_path / "out" / "evil.flac").exists()


def test_write_dataset_rejects_unsupported_audio_suffix(tmp_path):
    source = tmp_path / "source.mp3"
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

    with pytest.raises(ValueError, match="Unsupported audio suffix"):
        builder.write_dataset([sample], tmp_path / "out", seed=20260701)


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


def test_build_dataset_passes_requested_sample_count_to_writer(monkeypatch, tmp_path):
    sample = _sample(1, 12.0)
    captured = {}

    def fake_write_dataset(samples, output_dir, **kwargs):
        captured["samples"] = samples
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return {"requested_sample_count": kwargs["requested_sample_count"]}

    monkeypatch.setattr(builder, "scan_librispeech", lambda _root: [sample])
    monkeypatch.setattr(builder, "select_samples", lambda samples, **_kwargs: samples)
    monkeypatch.setattr(builder, "apply_size_budget", lambda samples, **_kwargs: samples)
    monkeypatch.setattr(builder, "write_dataset", fake_write_dataset)

    args = builder.build_arg_parser().parse_args(
        [
            "--source-dir",
            str(tmp_path / "source"),
            "--output-dir",
            str(tmp_path / "out"),
            "--target-count",
            "10",
        ]
    )

    manifest = builder.build_dataset(args)

    assert manifest["requested_sample_count"] == 10
    assert captured["requested_sample_count"] == 10
    assert captured["samples"] == [sample]
    assert captured["output_dir"] == tmp_path / "out"


def test_scan_librispeech_reads_transcripts_and_audio_metadata(monkeypatch, tmp_path):
    root = tmp_path / "LibriSpeech" / "test-clean"
    transcript_dir = root / "1089" / "134686"
    transcript_dir.mkdir(parents=True)
    audio_path = transcript_dir / "1089-134686-0001.flac"
    audio_path.write_bytes(b"fake-audio")
    (transcript_dir / "1089-134686.trans.txt").write_text(
        "1089-134686-0001 A LONG ENOUGH TRANSCRIPT\n",
        encoding="utf-8",
    )

    seen_audio_paths = []
    fake_soundfile = SimpleNamespace(
        info=lambda path: seen_audio_paths.append(path)
        or SimpleNamespace(duration=12.5)
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    samples = builder.scan_librispeech(root)

    assert samples == [
        builder.LibriSpeechSample(
            speaker_id="1089",
            chapter_id="134686",
            utterance_id="1089-134686-0001",
            audio_path=audio_path,
            text="A LONG ENOUGH TRANSCRIPT",
            duration_s=12.5,
            size_bytes=audio_path.stat().st_size,
        )
    ]
    assert seen_audio_paths == [audio_path]


def test_requirements_declares_soundfile_as_runtime_dependency():
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements.txt"
    ).read_text(encoding="utf-8")
    runtime_requirements, _, _dev_requirements = requirements.partition("# 开发/测试")

    assert "soundfile>=0.12.1" in runtime_requirements


def test_extract_archive_rejects_path_traversal_member(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    work_dir = tmp_path / "work"
    outside = tmp_path / "evil.txt"
    _write_tar(archive, tarfile.TarInfo("../evil.txt"), b"evil")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        builder._extract_archive(archive, work_dir)

    assert not outside.exists()


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_extract_archive_rejects_links_and_special_members(tmp_path, member_type):
    archive = tmp_path / "bad.tar.gz"
    work_dir = tmp_path / "work"
    member = tarfile.TarInfo("LibriSpeech/test-clean/bad")
    member.type = member_type
    member.linkname = "../evil.txt"
    _write_tar(archive, member)

    with pytest.raises(ValueError, match="Unsafe archive member"):
        builder._extract_archive(archive, work_dir)


def test_scan_librispeech_rejects_invalid_transcript_id(tmp_path):
    transcript_dir = tmp_path / "1089" / "134686"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "1089-134686.trans.txt").write_text(
        "../evil A LONG ENOUGH TRANSCRIPT\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid LibriSpeech utterance id"):
        builder.scan_librispeech(tmp_path)
