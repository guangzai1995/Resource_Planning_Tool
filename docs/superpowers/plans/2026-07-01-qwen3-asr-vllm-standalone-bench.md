# Qwen3-ASR vLLM Standalone Bench 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 只让 `vllm_standalone_bench` 支持 Qwen3-ASR-1.7B 通过 vLLM OpenAI Audio transcription 接口做自动化基准测试，并内置一个较小但长度分布合理的 LibriSpeech 测试子集。

**架构：** 在现有 `run_bench_multi.py -> run_bench_serve.py shim -> vllm_bench/serve.py -> endpoint_request_func.py` 链路中新增 `openai-audio` 分支。ASR 使用 `custom_audio` JSONL 数据集，样本解析为 `SampleRequest.multi_modal_data={"audio_path": ...}`，请求函数负责 multipart 上传到 `/v1/audio/transcriptions`，自动化脚本在 `backend=openai-audio` 时默认使用镜像内置数据集。

**技术栈：** Python 3.11、pytest、aiohttp multipart、soundfile/libsndfile、Dockerfile、LibriSpeech test-clean、CSV/XLSX 汇总。

---

## 文件结构

- 创建：`vllm_standalone_bench/tools/build_librispeech_asr_smoke.py`
  - 负责从 LibriSpeech `test-clean` 目录或官方归档构建固定种子 ASR 子集。
  - 暴露可单测的采样、manifest 写入、路径解析函数。
- 创建：`vllm_standalone_bench/tools/__init__.py`
  - 让 `tools` 目录可被 pytest 直接 import。
- 创建：`vllm_standalone_bench/tests/test_asr_dataset_builder.py`
  - 覆盖时长分桶、固定种子采样、JSONL/manifest 输出、大小预算裁剪。
- 创建：`vllm_standalone_bench/tests/test_audio_dataset.py`
  - 覆盖 `run_bench_serve.py` shim 的 `custom_audio` 数据集解析。
- 创建：`vllm_standalone_bench/tests/test_openai_audio_request.py`
  - 覆盖 `async_request_openai_audio` 的 `audio_path`、语言参数、multipart 成功路径。
- 修改：`vllm_standalone_bench/run_bench_serve.py`
  - 新增 `custom_audio` 数据集支持，解析 JSONL，循环复用样本到 `num_prompts`。
- 修改：`vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py`
  - `openai-audio` 兼容 `multi_modal_content["audio_path"]`，保留已有 ndarray 音频路径。
  - 使用 `RequestFuncInput.language`，默认 `en`。
- 修改：`vllm_standalone_bench/run_bench_multi.py`
  - CLI 支持 `--backend openai-audio`、`--dataset-name`、`--dataset-path`、`--language`。
  - ASR 分支 endpoint 指向 `/audio/transcriptions`，dataset 指向 `custom_audio`。
  - 结果表新增 ASR 列，现有文本列保留。
- 修改：`vllm_standalone_bench/auto_bench.py`
  - 配置支持 ASR profile 字段，构建 bench runner 命令时自动带数据集参数。
  - `backend=openai-audio` 且未指定 `dataset_path` 时使用镜像内置数据集。
  - 外部 `/datasets/...` 路径需要 `mounts.datasets`。
- 修改：`vllm_standalone_bench/tests/test_integration.py`
  - 覆盖 `run_bench_multi` ASR 参数映射和 ASR CSV 列。
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`
  - 覆盖 ASR profile 解析、内置数据集默认值、外部数据集挂载、文本路径不变。
- 修改：`vllm_standalone_bench/tests/test_shell_scripts.py`
  - 保持脚本静态检查通过；如果新增 shell 命令参数说明触发检查，在这里补断言。
- 修改：`vllm_standalone_bench/requirements.txt`
  - 新增 `soundfile>=0.12.1`。
- 修改：`vllm_standalone_bench/Dockerfile.bench-runner`
  - 安装 `libsndfile1`，复制 `assets/` 和 `tools/`。
- 创建：`vllm_standalone_bench/assets/librispeech_test_clean_256/asr_smoke.jsonl`
  - 内置 ASR JSONL，最多 256 条；如果 100 MiB 预算触发，实际条数由 manifest 记录。
- 创建：`vllm_standalone_bench/assets/librispeech_test_clean_256/manifest.json`
  - 固定种子、条数、时长桶、总大小、源信息。
- 创建：`vllm_standalone_bench/assets/librispeech_test_clean_256/ATTRIBUTION.md`
  - LibriSpeech/OpenSLR 归属说明和 CC BY 4.0 说明。
- 创建：`vllm_standalone_bench/assets/librispeech_test_clean_256/LICENSE.LibriSpeech.txt`
  - 数据集许可文本。
- 创建：`vllm_standalone_bench/configs/auto_bench.qwen3_asr_1_7b.smoke.json`
  - Qwen3-ASR-1.7B 自动化示例配置。
- 修改：`vllm_standalone_bench/README.md`
  - 增加 Qwen3-ASR 使用示例、数据集说明、动态批处理说明。

---

## 实现任务

### 任务 1：LibriSpeech ASR 子集生成器的纯逻辑

**文件：**
- 创建：`vllm_standalone_bench/tools/__init__.py`
- 创建：`vllm_standalone_bench/tools/build_librispeech_asr_smoke.py`
- 创建：`vllm_standalone_bench/tests/test_asr_dataset_builder.py`

- [ ] **步骤 1：编写失败的采样测试**

在 `vllm_standalone_bench/tests/test_asr_dataset_builder.py` 写入：

```python
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_asr_dataset_builder.py -q`

预期：FAIL，报错包含 `ModuleNotFoundError` 或 `AttributeError: module ... has no attribute 'LibriSpeechSample'`。

- [ ] **步骤 3：实现采样数据结构和分桶函数**

创建 `vllm_standalone_bench/tools/__init__.py`：

```python
"""Utility scripts for vllm_standalone_bench."""
```

创建 `vllm_standalone_bench/tools/build_librispeech_asr_smoke.py`，先实现纯逻辑：

```python
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


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
    if 5.0 <= duration_s < 10.0:
        return "medium"
    if 10.0 <= duration_s < 20.0:
        return "long"
    if 20.0 <= duration_s <= 30.0:
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
```

- [ ] **步骤 4：运行采样测试验证通过**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_asr_dataset_builder.py -q`

预期：PASS，输出包含 `3 passed`。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/tools/__init__.py \
        vllm_standalone_bench/tools/build_librispeech_asr_smoke.py \
        vllm_standalone_bench/tests/test_asr_dataset_builder.py
git commit -m "test(bench): add asr dataset sampling logic"
```

### 任务 2：生成器写出 JSONL、manifest 和许可文件

**文件：**
- 修改：`vllm_standalone_bench/tools/build_librispeech_asr_smoke.py`
- 修改：`vllm_standalone_bench/tests/test_asr_dataset_builder.py`

- [ ] **步骤 1：编写失败的写出测试**

追加到 `vllm_standalone_bench/tests/test_asr_dataset_builder.py`：

```python
import json


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
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{
        "prompt": "Transcribe the audio in English.",
        "audio": "audio/1089-134686-0001.flac",
        "output_tokens": 128,
        "reference": "A LONG ENOUGH TRANSCRIPT",
    }]
    assert (tmp_path / "out" / "audio" / "1089-134686-0001.flac").read_bytes() == b"fake-audio"
    assert manifest["sample_count"] == 1
    assert manifest["duration_buckets"] == {"medium": 0, "long": 1, "xlong": 0}
    assert manifest["seed"] == 20260701
    assert "OpenSLR" in (tmp_path / "out" / "ATTRIBUTION.md").read_text(encoding="utf-8")
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_asr_dataset_builder.py -q`

预期：FAIL，报错包含 `AttributeError: module ... has no attribute 'write_dataset'`。

- [ ] **步骤 3：实现写出、大小预算和 CLI 入口**

在 `vllm_standalone_bench/tools/build_librispeech_asr_smoke.py` 追加：

```python
PROMPT = "Transcribe the audio in English."
DEFAULT_OUTPUT_TOKENS = 128


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
            archive = Path(args.source_archive) if args.source_archive else tmp_path / "test-clean.tar.gz"
            if not archive.exists():
                _download_source(args.source_url, archive)
            source_dir = _extract_archive(archive, tmp_path)

        all_samples = scan_librispeech(source_dir)
        selected = select_samples(all_samples, target_count=args.target_count, seed=args.seed)
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
```

- [ ] **步骤 4：运行生成器单元测试验证通过**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_asr_dataset_builder.py -q`

预期：PASS，输出包含 `5 passed`。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/tools/build_librispeech_asr_smoke.py \
        vllm_standalone_bench/tests/test_asr_dataset_builder.py
git commit -m "feat(bench): write built-in asr dataset assets"
```

### 任务 3：run_bench_serve shim 支持 custom_audio 数据集

**文件：**
- 创建：`vllm_standalone_bench/tests/test_audio_dataset.py`
- 修改：`vllm_standalone_bench/run_bench_serve.py`

- [ ] **步骤 1：编写失败的 custom_audio 测试**

创建 `vllm_standalone_bench/tests/test_audio_dataset.py`：

```python
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_audio_dataset.py -q`

预期：FAIL，报错包含 `dataset 'custom_audio' 不在 shim 支持范围`。

- [ ] **步骤 3：实现 JSONL 解析函数**

在 `vllm_standalone_bench/run_bench_serve.py` 中添加导入：

```python
from pathlib import Path
```

在 `_load_sharegpt_requests` 后添加：

```python
def _load_custom_audio_requests(args: argparse.Namespace,
                                tokenizer) -> list[SampleRequest]:
    if not args.dataset_path:
        raise ValueError("custom_audio 数据集需要指定 --dataset-path")

    dataset_path = Path(args.dataset_path)
    base_dir = dataset_path.parent
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "audio" not in row:
                raise ValueError(f"{dataset_path}:{line_no} 缺少 audio 字段")
            rows.append(row)

    if not rows:
        raise ValueError(f"custom_audio 数据集为空: {dataset_path}")

    out_len = (
        getattr(args, "custom_output_len", None)
        or getattr(args, "random_output_len", 128)
        or 128
    )
    requests: list[SampleRequest] = []
    for i in range(args.num_prompts):
        row = rows[i % len(rows)]
        audio_path = Path(row["audio"])
        if not audio_path.is_absolute():
            audio_path = base_dir / audio_path
        expected_output_len = int(
            getattr(args, "custom_output_len", None)
            or row.get("output_tokens")
            or out_len
        )
        requests.append(SampleRequest(
            prompt=row.get("prompt") or "Transcribe the audio in English.",
            prompt_len=int(row.get("prompt_len") or 0),
            expected_output_len=expected_output_len,
            multi_modal_data={"audio_path": str(audio_path)},
            request_id=f"bench-audio-{uuid.uuid4().hex[:8]}-{i}",
        ))
    return requests
```

更新 `get_samples`：

```python
    elif name == "custom_audio":
        return _load_custom_audio_requests(args, tokenizer)
```

更新 `add_dataset_parser` 的 help 文案：

```python
help='数据集类型（shim 支持: random / sharegpt / custom_audio；其他类型请安装完整 vllm 包）'
```

- [ ] **步骤 4：运行 custom_audio 测试验证通过**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_audio_dataset.py -q`

预期：PASS，输出包含 `2 passed`。

- [ ] **步骤 5：运行现有 shim 测试确认文本路径未变**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_run_bench_serve.py vllm_standalone_bench/tests/test_audio_dataset.py -q`

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add vllm_standalone_bench/run_bench_serve.py \
        vllm_standalone_bench/tests/test_audio_dataset.py
git commit -m "feat(bench): add custom audio dataset shim"
```

### 任务 4：OpenAI Audio 请求函数支持 audio_path 和 language

**文件：**
- 创建：`vllm_standalone_bench/tests/test_openai_audio_request.py`
- 修改：`vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py`
- 修改：`vllm_standalone_bench/requirements.txt`

- [ ] **步骤 1：编写失败的请求函数测试**

创建 `vllm_standalone_bench/tests/test_openai_audio_request.py`：

```python
import json

import pytest

from vllm_bench.lib.endpoint_request_func import (
    RequestFuncInput,
    async_request_openai_audio,
)


class _Content:
    async def iter_any(self):
        usage = {"usage": {"completion_tokens": 5}}
        data = {"choices": [{"delta": {"content": "hello"}}]}
        yield f"data: {json.dumps(data)}\n\n".encode()
        yield f"data: {json.dumps(usage)}\n\n".encode()
        yield b"data: [DONE]\n\n"


class _Response:
    status = 200
    reason = "OK"
    content = _Content()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, **kwargs):
        self.calls.append(kwargs)
        return _Response()


@pytest.mark.asyncio
async def test_openai_audio_reads_audio_path_and_sets_language(tmp_path):
    pytest.importorskip("soundfile")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not-a-real-wav")

    class _Info:
        duration = 7.25

    import soundfile

    original_info = soundfile.info
    soundfile.info = lambda _: _Info()
    try:
        session = _Session()
        output = await async_request_openai_audio(
            RequestFuncInput(
                prompt="Transcribe the audio in English.",
                api_url="http://server/v1/audio/transcriptions",
                prompt_len=0,
                output_len=128,
                model="qwen3-asr",
                model_name=None,
                multi_modal_content={"audio_path": str(audio)},
                language="zh",
            ),
            session=session,
        )
    finally:
        soundfile.info = original_info

    assert output.success is True
    assert output.generated_text == "hello"
    assert output.output_tokens == 5
    assert output.input_audio_duration == 7.25
    assert session.calls[0]["url"].endswith("/audio/transcriptions")
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_openai_audio_request.py -q`

预期：FAIL，报错包含 `multi_modal_content must be a dict containing 'audio'`。

- [ ] **步骤 3：实现 audio_path 分支和语言参数**

修改 `vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py` 的 `async_request_openai_audio`：

```python
from pathlib import Path
```

在 payload 中替换语言硬编码：

```python
        "language": request_func_input.language or "en",
```

将发送音频文件的准备逻辑替换为：

```python
    mm_audio = request_func_input.multi_modal_content
    if not isinstance(mm_audio, dict):
        raise TypeError("multi_modal_content must be a dict for openai-audio")

    opened_file = None
    if "audio_path" in mm_audio:
        audio_path = Path(str(mm_audio["audio_path"]))
        opened_file = audio_path.open("rb")
        audio_file = opened_file
        audio_content_type = "audio/wav" if audio_path.suffix.lower() == ".wav" else "application/octet-stream"
    elif "audio" in mm_audio:
        audio_file = to_bytes(*mm_audio["audio"])
        audio_content_type = "audio/wav"
    else:
        raise TypeError("multi_modal_content must contain 'audio_path' or 'audio'")

    try:
        form = aiohttp.FormData()
        form.add_field("file", audio_file, filename=getattr(audio_file, "name", "audio.wav"), content_type=audio_content_type)
        for key, value in payload.items():
            form.add_field(key, str(value))

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.input_audio_duration = soundfile.info(audio_file).duration
        audio_file.seek(0)
        ...
    finally:
        audio_file.close()
        if opened_file is not None:
            opened_file = None
```

保留现有 response 解析主体，只把它缩进到 `try` 中；`content = [{"type": "text", ...}]` 当前未使用，可以删除以减少无效代码。

修改 `vllm_standalone_bench/requirements.txt`：

```text
soundfile>=0.12.1
```

- [ ] **步骤 4：运行请求函数测试验证通过**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_openai_audio_request.py -q`

预期：PASS。

- [ ] **步骤 5：运行 endpoint 现有测试验证文本请求未受影响**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_endpoint_parse.py vllm_standalone_bench/tests/test_openai_audio_request.py -q`

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/lib/endpoint_request_func.py \
        vllm_standalone_bench/requirements.txt \
        vllm_standalone_bench/tests/test_openai_audio_request.py
git commit -m "feat(bench): support openai audio file requests"
```

### 任务 5：run_bench_multi 支持 openai-audio 批量入口和 ASR 结果列

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py`
- 修改：`vllm_standalone_bench/tests/test_integration.py`

- [ ] **步骤 1：编写失败的 ASR 参数映射测试**

追加到 `vllm_standalone_bench/tests/test_integration.py`：

```python
def test_run_all_maps_openai_audio_to_custom_audio_dataset(monkeypatch):
    seen_cfgs = []

    async def fake_main_async(cfg):
        seen_cfgs.append(cfg)
        return {
            **_fake_result(0, 128, completed=2),
            "rtfx": 3.5,
            "duration": 4.0,
        }

    monkeypatch.setattr(serve, "main_async", fake_main_async)
    args = argparse.Namespace(
        model="Qwen/Qwen3-ASR-1.7B",
        served_model_name="qwen3-asr",
        backend="openai-audio",
        base_url="http://x/v1",
        host="127.0.0.1",
        port=8000,
        insecure=False,
        api_key=None,
        tokenizer=None,
        dataset_name="custom_audio",
        dataset_path="/opt/vllm_standalone_bench/assets/librispeech_test_clean_256/asr_smoke.jsonl",
        language="en",
        input_lens=[0],
        output_lens=[128],
        cross_product=False,
        parallel_nums=[2],
        epochs=1,
        sleep_between=0,
        warmup_requests=0,
        warmup_concurrency=None,
        warmup_output_len=None,
        prefix_ratio=0.0,
        seed=0,
        no_vary_seed_by_config=False,
        output_csv=None,
        output_xlsx=None,
        result_dir=None,
        max_ttft_ms=None,
        min_throughput_tok_s=None,
        min_output_compliance=0.0,
    )

    rows = m._run_all(args)

    cfg = seen_cfgs[0]
    assert cfg.backend == "openai-audio"
    assert cfg.endpoint == "/audio/transcriptions"
    assert cfg.dataset_name == "custom_audio"
    assert cfg.dataset_path.endswith("asr_smoke.jsonl")
    assert cfg.custom_output_len == 128
    assert cfg.language == "en"
    assert cfg.skip_tokenizer_init is True
    assert rows[0]["dataset_name"] == "custom_audio"
    assert rows[0]["audio_duration_s_total"] == 14.0
    assert rows[0]["audio_duration_s_avg"] == 7.0
    assert rows[0]["rtfx"] == 3.5


def test_csv_headers_include_asr_columns_without_removing_existing_columns():
    assert m.CSV_HEADERS[:3] == ["model", "backend", "dataset_name"]
    assert "audio_duration_s_total" in m.CSV_HEADERS
    assert "rtfx" in m.CSV_HEADERS
    assert "输入长度(token)" in m.CSV_HEADERS_ZH
    assert len(m.CSV_HEADERS) == len(m.CSV_HEADERS_ZH)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_integration.py -q`

预期：FAIL，报错包含 `Namespace` 缺少 `dataset_name` 或 backend choice 不包含 `openai-audio`。

- [ ] **步骤 3：实现 ASR 参数映射**

在 `vllm_standalone_bench/run_bench_multi.py` 添加常量：

```python
ASR_BACKEND = "openai-audio"
DEFAULT_ASR_DATASET_NAME = "custom_audio"
```

修改 `_build_base_args` 的 endpoint 分支：

```python
        if our_args.backend == "openai-chat":
            base.endpoint = "/chat/completions"
        elif our_args.backend == ASR_BACKEND:
            base.endpoint = "/audio/transcriptions"
        else:
            base.endpoint = "/completions"
```

host:port 分支同样使用：

```python
        if our_args.backend == "openai-chat":
            base.endpoint = "/v1/chat/completions"
        elif our_args.backend == ASR_BACKEND:
            base.endpoint = "/v1/audio/transcriptions"
        else:
            base.endpoint = "/v1/completions"
```

替换固定 random 数据集：

```python
    if our_args.backend == ASR_BACKEND:
        base.dataset_name = our_args.dataset_name or DEFAULT_ASR_DATASET_NAME
        base.dataset_path = our_args.dataset_path
        base.language = our_args.language or "en"
        base.skip_tokenizer_init = True
    else:
        base.dataset_name = "random"
```

在主循环设置 cfg 时对 ASR 分支使用 `custom_output_len`：

```python
            if our_args.backend == ASR_BACKEND:
                cfg.input_len = 0
                cfg.output_len = out_len
                cfg.custom_output_len = out_len
                cfg.random_prefix_len = 0
            else:
                cfg.input_len = in_len
                cfg.output_len = out_len
                cfg.random_prefix_len = prefix_tokens
```

- [ ] **步骤 4：实现 ASR 结果列**

修改 `_extract_row` 签名：

```python
def _extract_row(
    result: dict,
    in_len: int,
    out_len: int,
    parallel_num: int,
    epochs: int,
    model: str,
    backend: str,
    *,
    dataset_name: str = "random",
    language: str = "",
    prefix_tokens: int = 0,
    prefix_ratio: float = 0.0,
    has_tokenizer: bool = False,
    seed: int = 0,
) -> dict:
```

在 return 前计算：

```python
    rtfx = _f("rtfx")
    audio_duration_s_total = round(rtfx * duration_s, 4) if rtfx > 0 else 0.0
    audio_duration_s_avg = (
        round(audio_duration_s_total / completed, 4) if completed > 0 else 0.0
    )
```

在返回字典的配置区加入：

```python
        "dataset_name": dataset_name,
        "language": language,
```

在其他指标区加入：

```python
        "audio_duration_s_total": audio_duration_s_total,
        "audio_duration_s_avg": audio_duration_s_avg,
        "rtfx": rtfx,
```

调用 `_extract_row` 时传入：

```python
                               dataset_name=getattr(cfg, "dataset_name", "random"),
                               language=getattr(cfg, "language", ""),
```

更新 `CSV_HEADERS` 开头：

```python
CSV_HEADERS = [
    "model", "backend", "dataset_name", "language",
    "input_len", "output_len", ...
```

在末尾 `duration_s` 前加入：

```python
    "audio_duration_s_total", "audio_duration_s_avg", "rtfx",
```

同步更新 `CSV_HEADERS_ZH`：

```python
    "模型", "接口类型", "数据集", "语言",
```

并在末尾加入：

```python
    "音频总时长(s)", "平均音频时长(s)", "RTFx",
```

- [ ] **步骤 5：更新 CLI 参数**

修改 `build_arg_parser()`：

```python
conn.add_argument("--backend", choices=["openai", "openai-chat", "openai-audio"], default="openai", ...)
```

在测试配置组加入：

```python
    bench.add_argument("--dataset-name", default=None,
                       help="ASR 使用的数据集类型；openai-audio 默认 custom_audio")
    bench.add_argument("--dataset-path", default=None,
                       help="ASR JSONL 数据集路径；custom_audio 需要")
    bench.add_argument("--language", default="en",
                       help="OpenAI Audio transcription language 参数，默认 en")
```

在 `_run_all` 起始处加入 ASR 必填校验：

```python
    if our_args.backend == ASR_BACKEND and not our_args.dataset_path:
        raise ValueError("--backend openai-audio requires --dataset-path")
```

- [ ] **步骤 6：运行 integration 测试验证通过**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_integration.py -q`

预期：PASS。

- [ ] **步骤 7：运行 run_bench_multi 相关测试验证通过**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_integration.py vllm_standalone_bench/tests/test_cli.py -q`

预期：PASS。

- [ ] **步骤 8：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py \
        vllm_standalone_bench/tests/test_integration.py
git commit -m "feat(bench): add openai audio batch benchmark mode"
```

### 任务 6：auto_bench 自动化脚本接入 ASR profile 和数据集挂载

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的自动化配置测试**

追加到 `vllm_standalone_bench/tests/test_auto_bench.py`：

```python
def asr_config(tmp_path):
    data = minimal_config(tmp_path)
    data["models"][0]["name"] = "qwen3_asr_1_7b"
    data["models"][0]["served_model_name"] = "qwen3-asr"
    data["bench_profiles"][0] = {
        "name": "asr_smoke",
        "backend": "openai-audio",
        "output_lens": [128],
        "parallel_nums": [1, 4],
        "epochs": 1,
        "warmup_requests": 0,
        "dataset_name": "custom_audio",
        "language": "en",
    }
    return data


def test_asr_profile_defaults_to_builtin_dataset_path(tmp_path):
    config = ab.load_config(write_config(tmp_path, asr_config(tmp_path)))

    bench = config.bench_profiles[0]

    assert bench.backend == "openai-audio"
    assert bench.input_lens == [0]
    assert bench.dataset_name == "custom_audio"
    assert bench.dataset_path == ab.BUILTIN_ASR_DATASET_PATH
    assert bench.language == "en"


def test_build_bench_command_passes_asr_dataset_args(tmp_path):
    config = ab.load_config(write_config(tmp_path, asr_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    bench_dir = tmp_path / "results" / "run123" / "qwen3_asr_1_7b" / "bf16_default" / "asr_smoke"

    cmd = ab.build_bench_run_command(config, case, bench_dir)

    assert value_after(cmd, "--backend") == "openai-audio"
    assert value_after(cmd, "--dataset-name") == "custom_audio"
    assert value_after(cmd, "--dataset-path") == ab.BUILTIN_ASR_DATASET_PATH
    assert value_after(cmd, "--language") == "en"
    assert "--input-lens" not in cmd
    assert value_after(cmd, "--output-lens") == "128"


def test_external_asr_dataset_requires_datasets_mount(tmp_path):
    data = asr_config(tmp_path)
    data["bench_profiles"][0]["dataset_path"] = "/datasets/asr/custom.jsonl"

    with pytest.raises(ab.ConfigError, match="mounts.datasets"):
        ab.load_config(write_config(tmp_path, data))


def test_external_asr_dataset_mount_is_added(tmp_path):
    data = asr_config(tmp_path)
    host_datasets = tmp_path / "datasets"
    host_datasets.mkdir()
    data["mounts"]["datasets"] = str(host_datasets)
    data["bench_profiles"][0]["dataset_path"] = "/datasets/asr/custom.jsonl"

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    cmd = ab.build_bench_run_command(config, case, tmp_path / "bench")

    mounts = [cmd[index + 1] for index, value in enumerate(cmd) if value == "-v"]
    assert f"{host_datasets.resolve()}:/datasets:ro" in mounts
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q`

预期：FAIL，报错包含 `backend` 或 `BUILTIN_ASR_DATASET_PATH` 不存在。

- [ ] **步骤 3：扩展配置数据结构**

在 `vllm_standalone_bench/auto_bench.py` 添加：

```python
DATASET_CONTAINER_ROOT = PurePosixPath("/datasets")
BUILTIN_ASR_DATASET_PATH = "/opt/vllm_standalone_bench/assets/librispeech_test_clean_256/asr_smoke.jsonl"
```

修改 `SUPPORTED_BACKENDS`：

```python
SUPPORTED_BACKENDS = frozenset({"openai", "openai-chat", "openai-audio"})
```

扩展 `MountConfig`：

```python
@dataclass(frozen=True)
class MountConfig:
    models: Path
    datasets: Path | None = None
```

扩展 `BenchProfile`：

```python
    dataset_name: str = "random"
    dataset_path: str | None = None
    language: str = "en"
```

- [ ] **步骤 4：解析 mounts.datasets 和 ASR profile 默认值**

修改 `_parse_mounts`：

```python
    datasets_value = raw.get("datasets")
    datasets = (
        _resolve_path(datasets_value, base_dir=base_dir)
        if datasets_value is not None
        else None
    )
    return MountConfig(models=models, datasets=datasets)
```

在 `_parse_bench_profiles` 中加入 ASR 分支：

```python
        is_asr = backend == "openai-audio"
        if is_asr:
            input_lens = [0]
            dataset_name = str(raw.get("dataset_name") or "custom_audio")
            dataset_path = str(raw.get("dataset_path") or BUILTIN_ASR_DATASET_PATH)
            language = str(raw.get("language") or "en")
            prefix_ratio = 0.0
        else:
            input_lens = _parse_int_list(raw, "input_lens", min_value=1)
            dataset_name = "random"
            dataset_path = None
            language = ""
```

实例化 `BenchProfile` 时传入：

```python
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            language=language,
```

在 `load_config` 完成 profiles 解析后校验外部挂载：

```python
    for bench in bench_profiles:
        if bench.backend == "openai-audio" and bench.dataset_path:
            dataset_path = PurePosixPath(bench.dataset_path)
            if str(dataset_path).startswith("/datasets/") and mounts.datasets is None:
                raise ConfigError("ASR dataset_path under /datasets requires mounts.datasets")
```

- [ ] **步骤 5：构建 bench runner 命令时传 ASR 参数和数据集挂载**

修改 `build_bench_run_command` 的 mount 段：

```python
    if config.mounts.datasets is not None:
        cmd += ["-v", f"{config.mounts.datasets.resolve()}:/datasets:ro"]
```

构建 `bench_args` 时按 backend 分支：

```python
    if bench.backend == "openai-audio":
        bench_args += [
            "--dataset-name", bench.dataset_name,
            "--dataset-path", bench.dataset_path or BUILTIN_ASR_DATASET_PATH,
            "--language", bench.language,
        ]
    else:
        bench_args += ["--input-lens", *map(str, bench.input_lens)]
        bench_args += ["--prefix-ratio", str(bench.prefix_ratio)]
```

保持 `--output-lens`、`--parallel-nums`、`--epochs`、warmup 和约束参数对所有 backend 生效。

- [ ] **步骤 6：运行 auto_bench 测试验证通过**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q`

预期：PASS。

- [ ] **步骤 7：额外验证旧文本命令不变**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_build_bench_command_targets_container_dns -q`

预期：PASS，断言旧 `openai-chat` 命令仍包含 `--input-lens` 和 `--prefix-ratio`。

- [ ] **步骤 8：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py \
        vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): wire asr profiles into auto bench"
```

### 任务 7：内置数据集资产、镜像和示例配置

**文件：**
- 创建：`vllm_standalone_bench/assets/librispeech_test_clean_256/asr_smoke.jsonl`
- 创建：`vllm_standalone_bench/assets/librispeech_test_clean_256/manifest.json`
- 创建：`vllm_standalone_bench/assets/librispeech_test_clean_256/ATTRIBUTION.md`
- 创建：`vllm_standalone_bench/assets/librispeech_test_clean_256/LICENSE.LibriSpeech.txt`
- 创建：`vllm_standalone_bench/assets/librispeech_test_clean_256/audio/*.flac`
- 创建：`vllm_standalone_bench/configs/auto_bench.qwen3_asr_1_7b.smoke.json`
- 修改：`vllm_standalone_bench/Dockerfile.bench-runner`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的示例配置测试**

追加到 `vllm_standalone_bench/tests/test_auto_bench.py`：

```python
def test_qwen3_asr_sample_config_loads():
    path = CONFIG_DIR / "auto_bench.qwen3_asr_1_7b.smoke.json"

    config = ab.load_config(path)

    assert config.bench_profiles[0].backend == "openai-audio"
    assert config.bench_profiles[0].dataset_path == ab.BUILTIN_ASR_DATASET_PATH
    assert config.bench_profiles[0].parallel_nums == [1, 4, 8]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_qwen3_asr_sample_config_loads -q`

预期：FAIL，报错包含配置文件不存在。

- [ ] **步骤 3：生成内置 LibriSpeech 子集**

运行：

```bash
python3 vllm_standalone_bench/tools/build_librispeech_asr_smoke.py \
  --source-url https://www.openslr.org/resources/12/test-clean.tar.gz \
  --output-dir vllm_standalone_bench/assets/librispeech_test_clean_256 \
  --target-count 256 \
  --seed 20260701 \
  --max-bytes 104857600
```

预期：命令输出 JSON，其中：

```json
{
  "name": "librispeech_test_clean_256",
  "source_url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
  "license": "CC BY 4.0",
  "seed": 20260701
}
```

然后运行：

```bash
python3 -m json.tool vllm_standalone_bench/assets/librispeech_test_clean_256/manifest.json
```

预期：JSON 合法；`sample_count` 大于等于 `192`，`min_duration_s` 大于等于 `5.0`，`max_duration_s` 小于等于 `30.0`，`total_audio_bytes` 小于等于 `104857600`。

- [ ] **步骤 4：创建 Qwen3-ASR 自动化示例配置**

创建 `vllm_standalone_bench/configs/auto_bench.qwen3_asr_1_7b.smoke.json`：

```json
{
  "run": {
    "name": "qwen3_asr_smoke",
    "results_dir": "results",
    "vllm_image": "vllm/vllm-openai:latest",
    "bench_image": "vllm-bench-runner:offline",
    "network": "vllm-bench-net",
    "create_network": true,
    "cleanup_network": true,
    "container_port": 8000,
    "publish_host_port": false,
    "api_key": "local-bench-key",
    "ready_timeout_sec": 600,
    "cooldown_sec": 5
  },
  "mounts": {
    "models": "/models"
  },
  "models": [
    {
      "name": "qwen3_asr_1_7b",
      "model_path": "/models/Qwen3-ASR-1.7B",
      "tokenizer_path": "/models/Qwen3-ASR-1.7B",
      "served_model_name": "qwen3-asr"
    }
  ],
  "serve_profiles": [
    {
      "name": "bf16_default",
      "gpus": "all",
      "args": [
        "--dtype", "bfloat16",
        "--trust-remote-code"
      ]
    }
  ],
  "bench_profiles": [
    {
      "name": "asr_builtin_128",
      "backend": "openai-audio",
      "dataset_name": "custom_audio",
      "output_lens": [128],
      "parallel_nums": [1, 4, 8],
      "epochs": 16,
      "language": "en",
      "warmup_requests": 1,
      "warmup_output_len": 32,
      "max_ttft_ms": 10000
    }
  ]
}
```

- [ ] **步骤 5：更新 bench runner 镜像**

修改 `vllm_standalone_bench/Dockerfile.bench-runner`：

```dockerfile
FROM python:3.11-slim

WORKDIR /opt/vllm_standalone_bench

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY run_bench_multi.py run_bench_serve.py ./
COPY vllm_bench ./vllm_bench
COPY tools ./tools
COPY assets ./assets

RUN pip install --no-cache-dir -r requirements.txt openpyxl modelscope

CMD ["python", "/opt/vllm_standalone_bench/run_bench_multi.py", "--help"]
```

- [ ] **步骤 6：运行配置和 Dockerfile 测试**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_qwen3_asr_sample_config_loads \
  vllm_standalone_bench/tests/test_shell_scripts.py -q
```

预期：PASS。

- [ ] **步骤 7：检查数据集体积和 JSONL 行数**

运行：

```bash
python3 -c "from pathlib import Path; root=Path('vllm_standalone_bench/assets/librispeech_test_clean_256'); print(sum(p.stat().st_size for p in root.rglob('*') if p.is_file())); print(sum(1 for _ in (root/'asr_smoke.jsonl').open(encoding='utf-8')))"
```

预期：第一行小于等于 `104857600`，第二行大于等于 `192` 且小于等于 `256`。

- [ ] **步骤 8：Commit**

```bash
git add vllm_standalone_bench/assets/librispeech_test_clean_256 \
        vllm_standalone_bench/configs/auto_bench.qwen3_asr_1_7b.smoke.json \
        vllm_standalone_bench/Dockerfile.bench-runner \
        vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): embed librispeech asr benchmark dataset"
```

### 任务 8：文档、完整验证和收尾

**文件：**
- 修改：`vllm_standalone_bench/README.md`

- [ ] **步骤 1：更新 README 的 Qwen3-ASR 使用说明**

在 `vllm_standalone_bench/README.md` 增加：

```markdown
## Qwen3-ASR-1.7B benchmark

`vllm_standalone_bench` supports Qwen3-ASR through the OpenAI Audio transcription
endpoint by selecting `backend: "openai-audio"` in an auto bench profile.

When `backend` is `openai-audio` and `dataset_path` is not set, the bench runner
uses the built-in dataset:

```text
/opt/vllm_standalone_bench/assets/librispeech_test_clean_256/asr_smoke.jsonl
```

The built-in dataset is a deterministic LibriSpeech `test-clean` subset sampled
with seed `20260701`, filtered to 5-30 second clips, and balanced across 5-10s,
10-20s, and 20-30s duration buckets. The asset directory contains
`manifest.json`, `ATTRIBUTION.md`, and `LICENSE.LibriSpeech.txt`.

Example:

```bash
python auto_bench.py --config configs/auto_bench.qwen3_asr_1_7b.smoke.json
```

For external ASR data, mount the host directory as `/datasets`:

```json
{
  "mounts": {
    "models": "/models",
    "datasets": "/data/asr-bench"
  },
  "bench_profiles": [
    {
      "name": "asr_external",
      "backend": "openai-audio",
      "dataset_name": "custom_audio",
      "dataset_path": "/datasets/custom/asr.jsonl",
      "output_lens": [128],
      "parallel_nums": [1, 4, 8],
      "epochs": 16,
      "language": "en"
    }
  ]
}
```

For ASR, `parallel_nums` controls the maximum in-flight request concurrency.
The benchmark client sends up to that many concurrent HTTP transcription
requests, and the vLLM server performs its own continuous batching internally.
Selecting 128 samples means the client will submit 128 total requests for that
configuration when `parallel_num * epochs == 128`; it is not a single static
batch of 128 audio clips.
```

- [ ] **步骤 2：运行文档和空白检查**

运行：`git diff --check`

预期：无输出，退出码 0。

- [ ] **步骤 3：运行完整单元测试**

运行：`python3 -m pytest vllm_standalone_bench/tests -q`

预期：PASS，全部测试通过。

- [ ] **步骤 4：运行 ASR CLI smoke 解析**

运行：

```bash
python3 vllm_standalone_bench/run_bench_multi.py \
  --model qwen3-asr \
  --served-model-name qwen3-asr \
  --backend openai-audio \
  --base-url http://127.0.0.1:8000/v1 \
  --dataset-name custom_audio \
  --dataset-path vllm_standalone_bench/assets/librispeech_test_clean_256/asr_smoke.jsonl \
  --output-lens 128 \
  --parallel-nums 1 \
  --epochs 1 \
  --language en \
  --help
```

预期：输出 help 文本且退出码 0；命令只验证参数解析，不连接服务。

- [ ] **步骤 5：Commit 文档**

```bash
git add vllm_standalone_bench/README.md
git commit -m "docs(bench): document qwen3 asr benchmark flow"
```

- [ ] **步骤 6：最终状态检查**

运行：

```bash
git status --short --branch
git log --oneline --decorate -8
```

预期：工作树干净；最近提交包含本计划的 8 个实现提交。

---

## 自检记录

- 规格覆盖：
  - Qwen3-ASR 只接入 `vllm_standalone_bench`：任务 3-8。
  - 不影响现有文本 benchmark：任务 3、4、5、6 都包含现有测试回归。
  - 自动化脚本兼容：任务 6 和任务 7。
  - 内置较小、随机、不同长度、不太短的数据集：任务 1、2、7。
  - 镜像内置资产和依赖：任务 7。
  - 动态批处理说明：任务 8。
- 类型一致性：
  - JSONL 字段统一为 `prompt`、`audio`、`output_tokens`、`reference`。
  - shim 统一传递 `multi_modal_data={"audio_path": str(path)}`。
  - endpoint 统一读取 `RequestFuncInput.multi_modal_content["audio_path"]`。
  - 自动化配置统一使用 `dataset_name`、`dataset_path`、`language`。
