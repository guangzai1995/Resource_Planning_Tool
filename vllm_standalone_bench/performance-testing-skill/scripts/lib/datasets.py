import json
from itertools import cycle, islice
from pathlib import Path


class DatasetError(ValueError):
    """Raised when a dataset file cannot be read or has invalid rows."""


def load_dataset(path):
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise DatasetError(f"Dataset file does not exist: {dataset_path}")

    samples = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{dataset_path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{dataset_path}:{line_number} must be a JSON object")
            samples.append(row)

    if not samples:
        raise DatasetError(f"Dataset file is empty: {dataset_path}")
    return samples


def expand_samples(samples, count):
    if count <= 0:
        return []
    if not samples:
        raise DatasetError("Cannot expand an empty sample list")
    return list(islice(cycle(samples), count))
