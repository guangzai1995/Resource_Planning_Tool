#!/usr/bin/env python3
"""Build the built-in LibriSpeech ASR smoke dataset for vLLM audio benchmarks."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_SOURCE_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
DEFAULT_TARGET_COUNT = 256
DEFAULT_SEED = 20260701
DEFAULT_MIN_DURATION_S = 5.0
DEFAULT_MAX_DURATION_S = 30.0
DEFAULT_MAX_BYTES = 100 * 1024 * 1024
BUCKET_RATIOS = {"medium": 0.35, "long": 0.45, "xlong": 0.20}
PROMPT = "Transcribe the audio in English."
DEFAULT_OUTPUT_TOKENS = 128


@dataclass(frozen=True)
class LibriSpeechSample:
    speaker_id: str
    chapter_id: str
    utterance_id: str
    audio_path: Path
    text: str
    duration_s: float
    size_bytes: int


def duration_bucket(duration_s: float) -> str | None:
    if DEFAULT_MIN_DURATION_S <= duration_s < 10.0:
        return "medium"
    if 10.0 <= duration_s < 20.0:
        return "long"
    if 20.0 <= duration_s <= DEFAULT_MAX_DURATION_S:
        return "xlong"
    return None


def _target_counts(total: int) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in BUCKET_RATIOS.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def bucket_counts(samples: Sequence[LibriSpeechSample]) -> dict[str, int]:
    counts = {"medium": 0, "long": 0, "xlong": 0}
    for sample in samples:
        bucket = duration_bucket(sample.duration_s)
        if bucket:
            counts[bucket] += 1
    return counts


def select_samples(
    samples: Sequence[LibriSpeechSample],
    *,
    target_count: int = DEFAULT_TARGET_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[LibriSpeechSample]:
    rng = random.Random(seed)
    eligible: dict[str, list[LibriSpeechSample]] = {"medium": [], "long": [], "xlong": []}
    for sample in samples:
        bucket = duration_bucket(sample.duration_s)
        if bucket:
            eligible[bucket].append(sample)

    selected: list[LibriSpeechSample] = []
    leftovers: list[LibriSpeechSample] = []
    for bucket, target in _target_counts(target_count).items():
        pool = list(eligible[bucket])
        rng.shuffle(pool)
        selected.extend(pool[:target])
        leftovers.extend(pool[target:])

    if len(selected) < target_count:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_count - len(selected)])

    rng.shuffle(selected)
    return selected[:target_count]


def apply_size_budget(
    samples: Sequence[LibriSpeechSample],
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[LibriSpeechSample]:
    selected = list(samples)
    while sum(sample.size_bytes for sample in selected) > max_bytes and selected:
        longest = max(selected, key=lambda sample: (sample.duration_s, sample.size_bytes))
        selected.remove(longest)
    return selected


def _jsonable_manifest(
    samples: Sequence[LibriSpeechSample],
    *,
    seed: int,
    source_url: str,
    max_bytes: int,
) -> dict:
    total_bytes = sum(sample.size_bytes for sample in samples)
    total_duration = sum(sample.duration_s for sample in samples)
    return {
        "name": "librispeech_test_clean_256",
        "source_url": source_url,
        "license": "CC BY 4.0",
        "seed": seed,
        "requested_sample_count": DEFAULT_TARGET_COUNT,
        "sample_count": len(samples),
        "duration_buckets": bucket_counts(samples),
        "min_duration_s": min((sample.duration_s for sample in samples), default=0.0),
        "max_duration_s": max((sample.duration_s for sample in samples), default=0.0),
        "total_duration_s": round(total_duration, 3),
        "total_audio_bytes": total_bytes,
        "max_audio_bytes": max_bytes,
    }


def write_dataset(
    samples: Sequence[LibriSpeechSample],
    output_dir: Path,
    *,
    seed: int,
    source_url: str = DEFAULT_SOURCE_URL,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = _jsonable_manifest(
        samples,
        seed=seed,
        source_url=source_url,
        max_bytes=max_bytes,
    )
    jsonl_path = output_dir / "asr_smoke.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for sample in samples:
            dest_name = f"{sample.utterance_id}{sample.audio_path.suffix}"
            dest = audio_dir / dest_name
            shutil.copy2(sample.audio_path, dest)
            row = {
                "prompt": PROMPT,
                "audio": f"audio/{dest_name}",
                "output_tokens": DEFAULT_OUTPUT_TOKENS,
                "reference": sample.text,
            }
            jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ATTRIBUTION.md").write_text(
        "# LibriSpeech ASR Smoke Dataset\n\n"
        "This subset is derived from LibriSpeech test-clean, distributed by OpenSLR.\n"
        "Source: https://www.openslr.org/12\n"
        "License: Creative Commons Attribution 4.0 International.\n",
        encoding="utf-8",
    )
    (output_dir / "LICENSE.LibriSpeech.txt").write_text(
        "LibriSpeech is distributed under the Creative Commons Attribution 4.0 "
        "International license.\nSee https://creativecommons.org/licenses/by/4.0/\n",
        encoding="utf-8",
    )
    return manifest


def _extract_archive(source_archive: Path, work_dir: Path) -> Path:
    with tarfile.open(source_archive, "r:gz") as tar:
        tar.extractall(work_dir)
    return work_dir / "LibriSpeech" / "test-clean"


def _download_source(source_url: str, output_path: Path) -> Path:
    urllib.request.urlretrieve(source_url, output_path)
    return output_path


def scan_librispeech(root: Path) -> list[LibriSpeechSample]:
    import soundfile

    samples: list[LibriSpeechSample] = []
    for transcript_path in sorted(root.glob("*/*/*.trans.txt")):
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            utterance_id, text = line.split(" ", 1)
            speaker_id, chapter_id, _ = utterance_id.split("-", 2)
            audio_path = transcript_path.parent / f"{utterance_id}.flac"
            info = soundfile.info(audio_path)
            samples.append(
                LibriSpeechSample(
                    speaker_id=speaker_id,
                    chapter_id=chapter_id,
                    utterance_id=utterance_id,
                    audio_path=audio_path,
                    text=text,
                    duration_s=float(info.duration),
                    size_bytes=audio_path.stat().st_size,
                )
            )
    return samples


def build_dataset(args: argparse.Namespace) -> dict:
    with tempfile.TemporaryDirectory(prefix="librispeech-asr-") as tmp:
        tmp_path = Path(tmp)
        if args.source_dir:
            source_dir = Path(args.source_dir)
        else:
            archive = (
                Path(args.source_archive)
                if args.source_archive
                else tmp_path / "test-clean.tar.gz"
            )
            if not archive.exists():
                _download_source(args.source_url, archive)
            source_dir = _extract_archive(archive, tmp_path)

        all_samples = scan_librispeech(source_dir)
        selected = select_samples(
            all_samples,
            target_count=args.target_count,
            seed=args.seed,
        )
        selected = apply_size_budget(selected, max_bytes=args.max_bytes)
        return write_dataset(
            selected,
            Path(args.output_dir),
            seed=args.seed,
            source_url=args.source_url,
            max_bytes=args.max_bytes,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--source-archive", default=None)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = build_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
