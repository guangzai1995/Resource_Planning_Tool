#!/usr/bin/env python3
"""Build the built-in LibriSpeech ASR smoke dataset for vLLM audio benchmarks."""

from __future__ import annotations

import random
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
