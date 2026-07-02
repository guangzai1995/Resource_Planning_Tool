"""Dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.lib.config import resolve_package_path


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = resolve_package_path(path)
    rows: list[dict[str, Any]] = []

    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row must be an object: {dataset_path}:{line_number}"
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    return rows


def load_dataset(dataset_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    if dataset_config is None:
        return [{"prompt": "Hello", "expected_output_len": 32}]

    dataset_type = dataset_config.get("type")
    dataset_path = dataset_config.get("path")
    if not dataset_path:
        raise ValueError("Dataset config must include path")

    rows = _read_jsonl(dataset_path)

    if dataset_type == "text_prompts":
        for row in rows:
            if "prompt" not in row:
                raise ValueError("text_prompts row must include prompt")
            row["expected_output_len"] = int(row.get("expected_output_len", 128))
        return rows

    if dataset_type == "audio_manifest":
        for row in rows:
            if "audio" not in row:
                raise ValueError("audio_manifest row must include audio")
            if "duration_s" in row:
                row["duration_s"] = float(row["duration_s"])
        return rows

    raise ValueError(f"Unsupported dataset type: {dataset_type}")
