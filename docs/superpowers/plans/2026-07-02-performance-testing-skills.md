# 可迁移性能测试 Skills 包实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建一个可复制到任意项目运行的 `performance-testing-skills` 包，包含自动化性能测试 skill、非自动化接口测试 skill、Python 脚本、Shell 包装器、示例配置、示例数据集和本地验证测试。

**架构：** 包根目录独立于本仓库，所有脚本按包根解析相对路径。两个 skill 只负责指导 Codex/Claude 何时和如何调用脚本，脚本共享 `scripts/lib/` 中的配置、数据集、客户端、指标和报告模块。第一版使用 Python 标准库实现 HTTP、并发、CSV/JSON/Markdown 输出，避免强制第三方依赖。

**技术栈：** Python 3 标准库、Bash、JSON/JSONL、unittest/pytest 兼容测试、HTTP multipart/form-data、CSV/Markdown 报告。

---

## 文件结构

创建以下目录和文件：

- `performance-testing-skills/README.md`：包使用说明、复制方式、Codex/Claude 调用方式。
- `performance-testing-skills/skills/automated-performance-testing/SKILL.md`：自动化性能测试 skill。
- `performance-testing-skills/skills/manual-interface-performance-testing/SKILL.md`：非自动化接口测试 skill。
- `performance-testing-skills/scripts/perf_auto.py`：自动化压测 CLI。
- `performance-testing-skills/scripts/perf_manual.py`：非自动化接口测试 CLI。
- `performance-testing-skills/scripts/run_auto.sh`：自动化压测 shell 包装器。
- `performance-testing-skills/scripts/run_manual.sh`：非自动化测试 shell 包装器。
- `performance-testing-skills/scripts/lib/__init__.py`：共享库 package 标记。
- `performance-testing-skills/scripts/lib/config.py`：配置读取、环境变量展开、包相对路径解析。
- `performance-testing-skills/scripts/lib/datasets.py`：文本、音频 manifest 数据集读取。
- `performance-testing-skills/scripts/lib/clients.py`：请求构造、curl 生成、HTTP 发送、错误分类。
- `performance-testing-skills/scripts/lib/metrics.py`：请求级结果、聚合指标、瓶颈判断。
- `performance-testing-skills/scripts/lib/reporters.py`：JSON、CSV、Markdown 报告输出。
- `performance-testing-skills/configs/openai_chat.json`：OpenAI Chat 示例配置。
- `performance-testing-skills/configs/openai_completion.json`：OpenAI Completion 示例配置。
- `performance-testing-skills/configs/openai_asr.json`：OpenAI ASR 示例配置。
- `performance-testing-skills/configs/generic_http.json`：Generic HTTP 示例配置。
- `performance-testing-skills/datasets/text_prompts.example.jsonl`：文本示例数据集。
- `performance-testing-skills/datasets/audio_manifest.example.jsonl`：音频 manifest 示例数据集。
- `performance-testing-skills/reports/.gitkeep`：报告目录占位。
- `performance-testing-skills/tests/test_package_structure.py`：结构和 skill frontmatter 测试。
- `performance-testing-skills/tests/test_config_and_datasets.py`：配置和数据集读取测试。
- `performance-testing-skills/tests/test_clients.py`：请求构造、curl、错误分类、multipart 测试。
- `performance-testing-skills/tests/test_metrics_and_reporters.py`：指标、瓶颈判断、报告输出测试。
- `performance-testing-skills/tests/test_cli_dry_run.py`：两个 CLI 和两个 shell 包装器 dry-run 测试。
- `performance-testing-skills/tests/test_local_http_e2e.py`：标准库本地 HTTP 服务端端到端测试。

不修改本仓库已有压测代码。新包只作为可迁移工具包加入仓库。

## 任务 1：创建包结构和结构测试

**文件：**
- 创建：`performance-testing-skills/tests/test_package_structure.py`
- 创建：`performance-testing-skills/README.md`
- 创建：`performance-testing-skills/skills/automated-performance-testing/SKILL.md`
- 创建：`performance-testing-skills/skills/manual-interface-performance-testing/SKILL.md`
- 创建：`performance-testing-skills/reports/.gitkeep`

- [ ] **步骤 1：编写失败的结构测试**

创建 `performance-testing-skills/tests/test_package_structure.py`：

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_package_files_exist():
    expected = [
        "README.md",
        "skills/automated-performance-testing/SKILL.md",
        "skills/manual-interface-performance-testing/SKILL.md",
        "scripts/perf_auto.py",
        "scripts/perf_manual.py",
        "scripts/run_auto.sh",
        "scripts/run_manual.sh",
        "scripts/lib/__init__.py",
        "scripts/lib/config.py",
        "scripts/lib/datasets.py",
        "scripts/lib/clients.py",
        "scripts/lib/metrics.py",
        "scripts/lib/reporters.py",
        "configs/openai_chat.json",
        "configs/openai_completion.json",
        "configs/openai_asr.json",
        "configs/generic_http.json",
        "datasets/text_prompts.example.jsonl",
        "datasets/audio_manifest.example.jsonl",
        "reports/.gitkeep",
    ]
    missing = [path for path in expected if not (ROOT / path).exists()]
    assert missing == []


def test_skill_frontmatter_is_valid():
    skills = [
        ROOT / "skills/automated-performance-testing/SKILL.md",
        ROOT / "skills/manual-interface-performance-testing/SKILL.md",
    ]
    for path in skills:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "\nname: " in text
        assert "\ndescription: " in text
        assert "\n---\n" in text[4:]
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
PYTHONPATH=. pytest -q performance-testing-skills/tests/test_package_structure.py
```

预期：FAIL，缺少 `README.md`、两个 `SKILL.md` 和脚本/配置文件。

- [ ] **步骤 3：创建最小包结构和两个 skill 文档**

创建目录：

```bash
mkdir -p performance-testing-skills/skills/automated-performance-testing
mkdir -p performance-testing-skills/skills/manual-interface-performance-testing
mkdir -p performance-testing-skills/scripts/lib
mkdir -p performance-testing-skills/configs
mkdir -p performance-testing-skills/datasets
mkdir -p performance-testing-skills/reports
mkdir -p performance-testing-skills/tests
```

`performance-testing-skills/skills/automated-performance-testing/SKILL.md` 必须包含：

```markdown
---
name: automated-performance-testing
description: Use when running automated performance benchmarks for text, ASR, or generic HTTP model APIs, including concurrency sweeps, throughput analysis, latency analysis, and bottleneck detection.
---

# Automated Performance Testing

Use this skill when the user asks for load testing, performance benchmarking, concurrency sweeps, throughput, latency, bottleneck analysis, or regression performance testing.

## Required Workflow

1. Identify the protocol: `openai_chat`, `openai_completion`, `openai_asr`, or `generic_http`.
2. Confirm the benchmark scale: concurrency list plus either epochs or duration.
3. Confirm the dataset: text JSONL, audio manifest JSONL, or a user-provided path.
4. Run a smoke request before the benchmark.
5. Prefer the package scripts over ad-hoc request loops:

```bash
python3 scripts/perf_auto.py --config configs/openai_chat.json --dry-run
```

6. After execution, analyze success rate, throughput, latency percentiles, error types, stable concurrency, and bottleneck range.

## Safety Rules

- Do not treat 100% request failure as a performance bottleneck.
- If authentication, URL, or request-shape errors appear, switch to the manual interface testing skill first.
- Do not assume the target service lifecycle is managed by this package.
```

`performance-testing-skills/skills/manual-interface-performance-testing/SKILL.md` 必须包含：

```markdown
---
name: manual-interface-performance-testing
description: Use when manually validating text, ASR, or generic HTTP model API requests, including smoke tests, small batches, curl generation, and request-shape debugging.
---

# Manual Interface Performance Testing

Use this skill when the user asks to try an endpoint, send one or a few requests, generate curl, validate ASR upload format, or debug request errors before a benchmark.

## Required Workflow

1. Identify the protocol and endpoint.
2. Generate or inspect the request shape.
3. Run a smoke request or print an equivalent curl command.
4. Report status code, latency, response summary, and error diagnosis.
5. Use small batches only for stability checks, not full bottleneck conclusions.

## Preferred Commands

```bash
python3 scripts/perf_manual.py --config configs/openai_chat.json --mode smoke --dry-run
python3 scripts/perf_manual.py --config configs/openai_asr.json --audio-file sample.wav --print-curl
```
```

`performance-testing-skills/README.md` 必须说明：

```markdown
# Performance Testing Skills

Portable skills and scripts for automated and manual performance testing of model APIs.

## Quick Start

```bash
python3 scripts/perf_manual.py --config configs/openai_chat.json --dry-run
python3 scripts/perf_auto.py --config configs/openai_chat.json --dry-run
```

## Supported Protocols

- openai_chat
- openai_completion
- openai_asr
- generic_http
```

创建空文件：

```bash
touch performance-testing-skills/reports/.gitkeep
touch performance-testing-skills/scripts/lib/__init__.py
```

- [ ] **步骤 4：运行结构测试验证剩余失败**

运行：

```bash
PYTHONPATH=. pytest -q performance-testing-skills/tests/test_package_structure.py
```

预期：FAIL，只剩脚本、lib、配置和数据集文件缺失。

- [ ] **步骤 5：创建脚本和配置占位文件让结构测试通过**

创建最小文件：

```bash
touch performance-testing-skills/scripts/perf_auto.py
touch performance-testing-skills/scripts/perf_manual.py
touch performance-testing-skills/scripts/run_auto.sh
touch performance-testing-skills/scripts/run_manual.sh
touch performance-testing-skills/scripts/lib/config.py
touch performance-testing-skills/scripts/lib/datasets.py
touch performance-testing-skills/scripts/lib/clients.py
touch performance-testing-skills/scripts/lib/metrics.py
touch performance-testing-skills/scripts/lib/reporters.py
touch performance-testing-skills/configs/openai_chat.json
touch performance-testing-skills/configs/openai_completion.json
touch performance-testing-skills/configs/openai_asr.json
touch performance-testing-skills/configs/generic_http.json
touch performance-testing-skills/datasets/text_prompts.example.jsonl
touch performance-testing-skills/datasets/audio_manifest.example.jsonl
```

设置 shell 脚本可执行：

```bash
chmod +x performance-testing-skills/scripts/run_auto.sh
chmod +x performance-testing-skills/scripts/run_manual.sh
```

- [ ] **步骤 6：运行结构测试验证通过**

运行：

```bash
PYTHONPATH=. pytest -q performance-testing-skills/tests/test_package_structure.py
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add performance-testing-skills
git commit -m "feat: scaffold portable performance testing skills"
```

## 任务 2：实现配置读取和数据集读取

**文件：**
- 创建：`performance-testing-skills/tests/test_config_and_datasets.py`
- 修改：`performance-testing-skills/scripts/lib/config.py`
- 修改：`performance-testing-skills/scripts/lib/datasets.py`
- 修改：`performance-testing-skills/configs/openai_chat.json`
- 修改：`performance-testing-skills/configs/openai_completion.json`
- 修改：`performance-testing-skills/configs/openai_asr.json`
- 修改：`performance-testing-skills/configs/generic_http.json`
- 修改：`performance-testing-skills/datasets/text_prompts.example.jsonl`
- 修改：`performance-testing-skills/datasets/audio_manifest.example.jsonl`

- [ ] **步骤 1：编写配置和数据集失败测试**

创建 `performance-testing-skills/tests/test_config_and_datasets.py`：

```python
import json
import os
from pathlib import Path

from scripts.lib.config import load_config, package_root, resolve_package_path
from scripts.lib.datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_package_root_points_to_portable_package():
    assert package_root().name == "performance-testing-skills"


def test_load_config_expands_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_TEST_TOKEN", "secret-token")
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({
            "protocol": "openai_chat",
            "base_url": "http://127.0.0.1:8000/v1",
            "headers": {"Authorization": "Bearer ${PERF_TEST_TOKEN}"},
        }),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config["headers"]["Authorization"] == "Bearer secret-token"


def test_resolve_package_path_keeps_absolute_paths(tmp_path):
    assert resolve_package_path(str(tmp_path)) == tmp_path


def test_resolve_package_path_resolves_relative_to_package_root():
    resolved = resolve_package_path("datasets/text_prompts.example.jsonl")
    assert resolved == ROOT / "datasets/text_prompts.example.jsonl"


def test_load_text_prompts_dataset():
    rows = load_dataset({
        "type": "text_prompts",
        "path": "datasets/text_prompts.example.jsonl",
    })

    assert rows[0]["prompt"]
    assert isinstance(rows[0]["expected_output_len"], int)


def test_load_audio_manifest_dataset():
    rows = load_dataset({
        "type": "audio_manifest",
        "path": "datasets/audio_manifest.example.jsonl",
    })

    assert rows[0]["audio"].endswith(".wav")
    assert isinstance(rows[0]["duration_s"], float)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_config_and_datasets.py
```

预期：FAIL，`load_config`、`package_root`、`load_dataset` 未定义。

- [ ] **步骤 3：实现配置读取**

在 `performance-testing-skills/scripts/lib/config.py` 中写入：

```python
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = package_root() / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    data = expand_env(data)
    data["_config_path"] = str(config_path)
    return data


def resolve_package_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return package_root() / resolved
```

- [ ] **步骤 4：实现数据集读取**

在 `performance-testing-skills/scripts/lib/datasets.py` 中写入：

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import resolve_package_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON error: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def load_dataset(dataset_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not dataset_config:
        return [{"prompt": "Hello", "expected_output_len": 32}]
    dataset_type = dataset_config.get("type")
    dataset_path = dataset_config.get("path")
    if not dataset_path:
        raise ValueError("dataset.path is required")
    path = resolve_package_path(str(dataset_path))
    rows = _read_jsonl(path)
    if dataset_type == "text_prompts":
        for row in rows:
            if "prompt" not in row:
                raise ValueError(f"text prompt row missing prompt: {path}")
            row["expected_output_len"] = int(row.get("expected_output_len") or 128)
        return rows
    if dataset_type == "audio_manifest":
        for row in rows:
            if "audio" not in row:
                raise ValueError(f"audio manifest row missing audio: {path}")
            if "duration_s" in row:
                row["duration_s"] = float(row["duration_s"])
        return rows
    raise ValueError(f"unsupported dataset type: {dataset_type}")
```

- [ ] **步骤 5：填写示例配置和示例数据集**

`performance-testing-skills/datasets/text_prompts.example.jsonl`：

```jsonl
{"prompt": "Write a short introduction to API benchmarking.", "expected_output_len": 128, "metadata": {"case": "intro"}}
{"prompt": "Summarize why latency percentiles matter.", "expected_output_len": 96, "metadata": {"case": "latency"}}
```

`performance-testing-skills/datasets/audio_manifest.example.jsonl`：

```jsonl
{"audio": "datasets/audio/sample.wav", "prompt": "Transcribe the audio.", "reference": "example transcript", "duration_s": 12.3}
```

`performance-testing-skills/configs/openai_chat.json`：

```json
{
  "name": "openai_chat_example",
  "protocol": "openai_chat",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "example-chat-model",
  "headers": {
    "Authorization": "Bearer ${API_KEY}"
  },
  "request": {
    "max_tokens": 128,
    "temperature": 0
  },
  "dataset": {
    "type": "text_prompts",
    "path": "datasets/text_prompts.example.jsonl"
  },
  "bench": {
    "concurrency": [1, 4, 8],
    "epochs": 3,
    "timeout_seconds": 60
  },
  "report": {
    "output_dir": "reports",
    "formats": ["json", "csv", "md"]
  }
}
```

`openai_completion.json` must use `"protocol": "openai_completion"` and the same text dataset. `openai_asr.json` must use `"protocol": "openai_asr"` and audio manifest dataset. `generic_http.json` must use `"protocol": "generic_http"` with `method`, `url`, `headers`, and `body_template`.

- [ ] **步骤 6：运行测试验证通过**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_config_and_datasets.py
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add performance-testing-skills
git commit -m "feat: add portable benchmark config and datasets"
```

## 任务 3：实现请求构造、curl 生成和错误分类

**文件：**
- 创建：`performance-testing-skills/tests/test_clients.py`
- 修改：`performance-testing-skills/scripts/lib/clients.py`

- [ ] **步骤 1：编写客户端失败测试**

创建 `performance-testing-skills/tests/test_clients.py`：

```python
from scripts.lib.clients import (
    build_request,
    classify_error,
    make_curl,
)


def test_build_openai_chat_request():
    request = build_request(
        {"protocol": "openai_chat", "base_url": "http://host/v1", "model": "m", "request": {"max_tokens": 8}},
        {"prompt": "hello"},
    )

    assert request["method"] == "POST"
    assert request["url"] == "http://host/v1/chat/completions"
    assert request["json"]["model"] == "m"
    assert request["json"]["messages"][0]["content"] == "hello"


def test_build_openai_completion_request():
    request = build_request(
        {"protocol": "openai_completion", "base_url": "http://host/v1", "model": "m", "request": {"max_tokens": 8}},
        {"prompt": "hello"},
    )

    assert request["url"] == "http://host/v1/completions"
    assert request["json"]["prompt"] == "hello"


def test_build_openai_asr_request():
    request = build_request(
        {"protocol": "openai_asr", "base_url": "http://host/v1", "model": "asr", "request": {"language": "en"}},
        {"audio": "sample.wav", "prompt": "transcribe"},
    )

    assert request["url"] == "http://host/v1/audio/transcriptions"
    assert request["multipart"]["model"] == "asr"
    assert request["multipart"]["language"] == "en"
    assert request["multipart_file"] == "sample.wav"


def test_build_generic_http_request_replaces_template_values():
    request = build_request(
        {
            "protocol": "generic_http",
            "method": "POST",
            "url": "http://host/infer",
            "body_template": {"input": "${prompt}", "max_tokens": "${max_tokens}"},
        },
        {"prompt": "hello", "expected_output_len": 9},
    )

    assert request["json"] == {"input": "hello", "max_tokens": 9}


def test_make_curl_contains_method_url_and_json():
    curl = make_curl({
        "method": "POST",
        "url": "http://host/v1/chat/completions",
        "headers": {"Content-Type": "application/json"},
        "json": {"model": "m"},
    })

    assert "curl" in curl
    assert "-X POST" in curl
    assert "http://host/v1/chat/completions" in curl
    assert "'Content-Type: application/json'" in curl


def test_classify_error_maps_common_status_codes():
    assert classify_error(status_code=401, exception=None) == "auth_error"
    assert classify_error(status_code=404, exception=None) == "not_found"
    assert classify_error(status_code=422, exception=None) == "bad_request"
    assert classify_error(status_code=500, exception=None) == "http_5xx"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_clients.py
```

预期：FAIL，`build_request`、`make_curl`、`classify_error` 未定义。

- [ ] **步骤 3：实现请求构造和 curl 生成**

在 `performance-testing-skills/scripts/lib/clients.py` 中实现：

```python
from __future__ import annotations

import json
import shlex
from typing import Any


def _base_url(config: dict[str, Any]) -> str:
    return str(config.get("base_url", "")).rstrip("/")


def _headers(config: dict[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in (config.get("headers") or {}).items() if str(v)}


def _replace_template(value: Any, sample: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if value == "${prompt}":
            return sample.get("prompt", "")
        if value == "${max_tokens}":
            return int(sample.get("expected_output_len") or 128)
        result = value
        for key, sample_value in sample.items():
            result = result.replace("${" + key + "}", str(sample_value))
        return result
    if isinstance(value, list):
        return [_replace_template(item, sample) for item in value]
    if isinstance(value, dict):
        return {key: _replace_template(item, sample) for key, item in value.items()}
    return value


def build_request(config: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    protocol = config.get("protocol")
    headers = _headers(config)
    request_options = dict(config.get("request") or {})
    model = config.get("model")
    if protocol == "openai_chat":
        body = {"model": model, "messages": [{"role": "user", "content": sample.get("prompt", "")}]}
        body.update(request_options)
        return {"method": "POST", "url": f"{_base_url(config)}/chat/completions", "headers": headers, "json": body}
    if protocol == "openai_completion":
        body = {"model": model, "prompt": sample.get("prompt", "")}
        body.update(request_options)
        return {"method": "POST", "url": f"{_base_url(config)}/completions", "headers": headers, "json": body}
    if protocol == "openai_asr":
        multipart = {"model": model}
        multipart.update(request_options)
        if sample.get("prompt"):
            multipart["prompt"] = sample["prompt"]
        return {
            "method": "POST",
            "url": f"{_base_url(config)}/audio/transcriptions",
            "headers": headers,
            "multipart": multipart,
            "multipart_file": sample["audio"],
        }
    if protocol == "generic_http":
        body_template = config.get("body_template") or {}
        return {
            "method": str(config.get("method", "POST")).upper(),
            "url": str(config["url"]),
            "headers": headers,
            "json": _replace_template(body_template, sample),
        }
    raise ValueError(f"unsupported protocol: {protocol}")


def make_curl(request: dict[str, Any]) -> str:
    parts = ["curl", "-sS", "-X", request["method"]]
    for key, value in (request.get("headers") or {}).items():
        parts.extend(["-H", f"{key}: {value}"])
    if "json" in request:
        parts.extend(["-H", "Content-Type: application/json"])
        parts.extend(["--data", json.dumps(request["json"], ensure_ascii=False)])
    if "multipart" in request:
        for key, value in request["multipart"].items():
            parts.extend(["-F", f"{key}={value}"])
        parts.extend(["-F", f"file=@{request['multipart_file']}"])
    parts.append(request["url"])
    return " ".join(shlex.quote(str(part)) for part in parts)


def classify_error(status_code: int | None, exception: BaseException | None) -> str:
    if status_code in (401, 403):
        return "auth_error"
    if status_code == 404:
        return "not_found"
    if status_code in (400, 422):
        return "bad_request"
    if status_code is not None and status_code >= 500:
        return "http_5xx"
    if exception is not None:
        name = exception.__class__.__name__.lower()
        message = str(exception).lower()
        if "timeout" in name or "timed out" in message:
            return "timeout"
        if "broken pipe" in message:
            return "broken_pipe"
        if "disconnect" in message or "connection reset" in message:
            return "server_disconnected"
        return "unknown_error"
    return "unknown_error"
```

- [ ] **步骤 4：运行客户端测试验证通过**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_clients.py
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add performance-testing-skills/scripts/lib/clients.py performance-testing-skills/tests/test_clients.py
git commit -m "feat: build portable benchmark requests"
```

## 任务 4：实现指标聚合、瓶颈判断和报告输出

**文件：**
- 创建：`performance-testing-skills/tests/test_metrics_and_reporters.py`
- 修改：`performance-testing-skills/scripts/lib/metrics.py`
- 修改：`performance-testing-skills/scripts/lib/reporters.py`

- [ ] **步骤 1：编写指标和报告失败测试**

创建 `performance-testing-skills/tests/test_metrics_and_reporters.py`：

```python
import csv
import json

from scripts.lib.metrics import aggregate_results, analyze_bottleneck
from scripts.lib.reporters import write_reports


def test_aggregate_results_computes_latency_and_success_rate():
    rows = [
        {"success": True, "latency_ms": 100.0, "output_tokens": 10, "audio_duration_s": 30.0},
        {"success": True, "latency_ms": 200.0, "output_tokens": 20, "audio_duration_s": 30.0},
        {"success": False, "latency_ms": 300.0, "output_tokens": 0, "audio_duration_s": 30.0},
    ]

    metrics = aggregate_results(rows, concurrency=4, duration_s=10.0)

    assert metrics["concurrency"] == 4
    assert metrics["n_success"] == 2
    assert metrics["n_failed"] == 1
    assert metrics["success_rate"] == 2 / 3
    assert metrics["latency_p50_ms"] == 200.0
    assert metrics["request_throughput_req_s"] == 0.2
    assert metrics["audio_rtfx"] == 9.0


def test_analyze_bottleneck_finds_stable_and_overloaded_concurrency():
    metrics = [
        {"concurrency": 16, "success_rate": 1.0, "request_throughput_req_s": 0.50, "latency_p90_ms": 30000.0},
        {"concurrency": 32, "success_rate": 1.0, "request_throughput_req_s": 0.61, "latency_p90_ms": 58000.0},
        {"concurrency": 64, "success_rate": 0.93, "request_throughput_req_s": 0.65, "latency_p90_ms": 120000.0},
    ]

    analysis = analyze_bottleneck(metrics)

    assert analysis["stable_concurrency"] == 16
    assert analysis["peak_throughput_concurrency"] == 64
    assert analysis["overload_starts_at"] == 32


def test_write_reports_creates_summary_json_and_csv(tmp_path):
    metrics = [{"concurrency": 1, "n_success": 3, "n_failed": 0, "success_rate": 1.0, "request_throughput_req_s": 0.1}]
    requests = [{"request_id": "req-1", "success": True, "latency_ms": 100.0}]
    errors = []

    write_reports(tmp_path, metrics, requests, errors, {"stable_concurrency": 1})

    assert (tmp_path / "summary.md").exists()
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))[0]["concurrency"] == 1
    with (tmp_path / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["concurrency"] == "1"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_metrics_and_reporters.py
```

预期：FAIL，指标和报告函数未定义。

- [ ] **步骤 3：实现指标计算**

在 `performance-testing-skills/scripts/lib/metrics.py` 中实现 `aggregate_results`、`percentile` 和 `analyze_bottleneck`。核心语义：

```python
def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = int(round((len(values) - 1) * pct / 100.0))
    return values[index]
```

`aggregate_results` 必须：

- 只用成功请求计算 `request_throughput_req_s` 的分子。
- 使用所有请求计算成功率和失败率。
- 使用成功请求的 latency 计算 p50/p90/p99。
- 当存在 `audio_duration_s` 时计算 `audio_rtfx`。

`analyze_bottleneck` 必须：

- `stable_concurrency`：最后一个 `success_rate == 1.0` 且未出现延迟拐点的并发。
- `peak_throughput_concurrency`：`request_throughput_req_s` 最大的并发。
- `overload_starts_at`：第一个出现失败、吞吐提升低于 30% 且延迟 P90 增长超过 50%、或吞吐下降的并发。

- [ ] **步骤 4：实现报告输出**

在 `performance-testing-skills/scripts/lib/reporters.py` 中实现：

- `write_reports(output_dir, metrics, requests, errors, analysis)`
- `metrics.json`
- `metrics.csv`
- `requests.jsonl`
- `errors.jsonl`
- `summary.md`

`summary.md` 必须包含：

```markdown
## Conclusion

- Stable concurrency:
- Peak throughput concurrency:
- Overload starts at:

## Metrics
```

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_metrics_and_reporters.py
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add performance-testing-skills/scripts/lib/metrics.py performance-testing-skills/scripts/lib/reporters.py performance-testing-skills/tests/test_metrics_and_reporters.py
git commit -m "feat: report benchmark metrics and bottlenecks"
```

## 任务 5：实现 HTTP 发送和本地服务端端到端测试

**文件：**
- 创建：`performance-testing-skills/tests/test_local_http_e2e.py`
- 修改：`performance-testing-skills/scripts/lib/clients.py`

- [ ] **步骤 1：编写本地 HTTP 端到端失败测试**

创建 `performance-testing-skills/tests/test_local_http_e2e.py`，使用标准库 `http.server` 启动本地服务，验证 `send_request` 能 POST JSON 并解析响应：

```python
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from scripts.lib.clients import build_request, send_request


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        response = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "echo": payload,
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


def test_send_request_posts_json_to_local_server():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = {"protocol": "openai_chat", "base_url": f"http://127.0.0.1:{server.server_port}/v1", "model": "m"}
        request = build_request(config, {"prompt": "hello"})
        result = send_request("req-1", request, timeout_seconds=5)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert result["success"] is True
    assert result["status_code"] == 200
    assert result["output_tokens"] == 2
    assert result["latency_ms"] >= 0
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_local_http_e2e.py
```

预期：FAIL，`send_request` 未定义。

- [ ] **步骤 3：实现标准库 HTTP 发送**

在 `clients.py` 中实现 `send_request`：

- JSON 请求使用 `urllib.request.Request`。
- multipart 请求手工编码 `multipart/form-data`。
- 记录 `started = time.perf_counter()` 和 `latency_ms`。
- 成功响应解析 JSON；非 JSON 保存 `response_text` 摘要。
- 从 `usage.prompt_tokens` 和 `usage.completion_tokens` 提取 token。
- 异常时设置 `success=False` 和 `error_type=classify_error(...)`。

返回结构必须包含：

```python
{
    "request_id": request_id,
    "success": True,
    "status_code": 200,
    "latency_ms": 123.4,
    "input_tokens": 0,
    "output_tokens": 0,
    "response_summary": "...",
}
```

- [ ] **步骤 4：运行端到端测试验证通过**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_local_http_e2e.py
```

预期：PASS。

- [ ] **步骤 5：运行客户端全量测试验证通过**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_clients.py tests/test_local_http_e2e.py
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add performance-testing-skills/scripts/lib/clients.py performance-testing-skills/tests/test_local_http_e2e.py
git commit -m "feat: send portable benchmark HTTP requests"
```

## 任务 6：实现非自动化 CLI 和 Shell 包装器

**文件：**
- 创建：`performance-testing-skills/tests/test_cli_dry_run.py`
- 修改：`performance-testing-skills/scripts/perf_manual.py`
- 修改：`performance-testing-skills/scripts/run_manual.sh`

- [ ] **步骤 1：编写 manual CLI dry-run 失败测试**

在 `performance-testing-skills/tests/test_cli_dry_run.py` 中写入 manual 相关测试：

```python
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args):
    return subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def test_perf_manual_dry_run_outputs_request_plan():
    result = run_cmd([
        sys.executable,
        "scripts/perf_manual.py",
        "--config",
        "configs/openai_chat.json",
        "--dry-run",
        "--input",
        "hello",
    ])

    assert "DRY RUN" in result.stdout
    assert "openai_chat" in result.stdout
    assert "/chat/completions" in result.stdout


def test_perf_manual_curl_mode_outputs_curl():
    result = run_cmd([
        sys.executable,
        "scripts/perf_manual.py",
        "--config",
        "configs/openai_chat.json",
        "--mode",
        "curl",
        "--input",
        "hello",
    ])

    assert "curl" in result.stdout
    assert "/chat/completions" in result.stdout


def test_run_manual_shell_wrapper_dry_run():
    result = run_cmd([
        "bash",
        "scripts/run_manual.sh",
        "--config",
        "configs/openai_chat.json",
        "--dry-run",
    ])

    assert "DRY RUN" in result.stdout
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_cli_dry_run.py::test_perf_manual_dry_run_outputs_request_plan tests/test_cli_dry_run.py::test_perf_manual_curl_mode_outputs_curl tests/test_cli_dry_run.py::test_run_manual_shell_wrapper_dry_run
```

预期：FAIL，CLI 未实现。

- [ ] **步骤 3：实现 `perf_manual.py`**

`perf_manual.py` 必须：

- 使用 `argparse` 支持 `--config`、`--mode`、`--request-count`、`--input`、`--audio-file`、`--print-curl`、`--save-response`、`--dry-run`。
- 通过 `scripts.lib.config.load_config` 读取配置。
- 用命令行输入覆盖数据集首条记录。
- 用 `build_request` 生成请求。
- dry-run 输出 `DRY RUN`、protocol、method、url、curl。
- `mode=curl` 只输出 curl。
- 非 dry-run 调用 `send_request`。

脚本开头需要保证从包根运行和直接按路径运行都能导入：

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **步骤 4：实现 `run_manual.sh`**

`run_manual.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/perf_manual.py" "$@"
```

- [ ] **步骤 5：运行 manual CLI 测试验证通过**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_cli_dry_run.py::test_perf_manual_dry_run_outputs_request_plan tests/test_cli_dry_run.py::test_perf_manual_curl_mode_outputs_curl tests/test_cli_dry_run.py::test_run_manual_shell_wrapper_dry_run
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add performance-testing-skills/scripts/perf_manual.py performance-testing-skills/scripts/run_manual.sh performance-testing-skills/tests/test_cli_dry_run.py
git commit -m "feat: add manual interface testing CLI"
```

## 任务 7：实现自动化 CLI、并发执行和 dry-run

**文件：**
- 修改：`performance-testing-skills/tests/test_cli_dry_run.py`
- 修改：`performance-testing-skills/scripts/perf_auto.py`
- 修改：`performance-testing-skills/scripts/run_auto.sh`

- [ ] **步骤 1：补充 auto CLI 失败测试**

向 `test_cli_dry_run.py` 追加：

```python
def test_perf_auto_dry_run_outputs_concurrency_plan():
    result = run_cmd([
        sys.executable,
        "scripts/perf_auto.py",
        "--config",
        "configs/openai_chat.json",
        "--concurrency",
        "1,4",
        "--epochs",
        "2",
        "--dry-run",
    ])

    assert "DRY RUN" in result.stdout
    assert "concurrency=1 request_count=2" in result.stdout
    assert "concurrency=4 request_count=8" in result.stdout


def test_run_auto_shell_wrapper_dry_run():
    result = run_cmd([
        "bash",
        "scripts/run_auto.sh",
        "--config",
        "configs/openai_chat.json",
        "--concurrency",
        "1",
        "--epochs",
        "1",
        "--dry-run",
    ])

    assert "DRY RUN" in result.stdout
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_cli_dry_run.py::test_perf_auto_dry_run_outputs_concurrency_plan tests/test_cli_dry_run.py::test_run_auto_shell_wrapper_dry_run
```

预期：FAIL，auto CLI 未实现。

- [ ] **步骤 3：实现 `perf_auto.py` dry-run 和实际执行框架**

`perf_auto.py` 必须：

- 支持 `--config`、`--dataset`、`--concurrency`、`--epochs`、`--duration-seconds`、`--output-dir`、`--fail-fast`、`--max-error-rate`、`--max-p90-latency-ms`、`--dry-run`。
- 读取配置和数据集。
- 并发列表优先级：CLI `--concurrency` > 配置 `bench.concurrency` > `[1]`。
- epochs 优先级：CLI `--epochs` > 配置 `bench.epochs` > `1`。
- dry-run 打印每档 `concurrency=<n> request_count=<n*epochs>`。
- 非 dry-run 先发 1 条 smoke request。
- 每档使用 `concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)` 发请求。
- 每档调用 `aggregate_results`。
- 最后调用 `write_reports`。

第一版 `duration_seconds` 可以实现为时间循环：在时间未到前不断提交新请求，保持最多 `concurrency` 个 future 在途。

- [ ] **步骤 4：实现 `run_auto.sh`**

`run_auto.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/perf_auto.py" "$@"
```

- [ ] **步骤 5：运行 auto CLI dry-run 测试验证通过**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_cli_dry_run.py
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add performance-testing-skills/scripts/perf_auto.py performance-testing-skills/scripts/run_auto.sh performance-testing-skills/tests/test_cli_dry_run.py
git commit -m "feat: add automated performance testing CLI"
```

## 任务 8：实现 CLI 端到端报告验证

**文件：**
- 修改：`performance-testing-skills/tests/test_cli_dry_run.py`
- 修改：`performance-testing-skills/scripts/perf_auto.py`
- 修改：`performance-testing-skills/scripts/perf_manual.py`

- [ ] **步骤 1：编写实际报告输出测试**

向 `test_cli_dry_run.py` 追加一个使用本地 HTTP 服务的 auto CLI 报告测试。复用 `test_local_http_e2e.py` 的 server 模式，或在该测试文件中复制一个简短 handler：

```python
def test_perf_auto_writes_reports_with_local_server(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "protocol": "openai_chat",
        "base_url": f"http://127.0.0.1:{server_port}/v1",
        "model": "m",
        "dataset": {"type": "text_prompts", "path": str(dataset_path)},
        "bench": {"concurrency": [1], "epochs": 2, "timeout_seconds": 5},
        "report": {"output_dir": str(tmp_path / "reports"), "formats": ["json", "csv", "md"]},
    }), encoding="utf-8")

    result = run_cmd([sys.executable, "scripts/perf_auto.py", "--config", str(config)])

    assert "summary.md" in result.stdout
    assert (tmp_path / "reports" / "summary.md").exists()
    assert (tmp_path / "reports" / "metrics.csv").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()
```

Use the concrete local server setup from `test_local_http_e2e.py`, not an external endpoint.

- [ ] **步骤 2：运行测试验证失败或不完整**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_cli_dry_run.py::test_perf_auto_writes_reports_with_local_server
```

预期：FAIL，直到 `perf_auto.py` 能完整写报告并在 stdout 输出报告路径。

- [ ] **步骤 3：补齐 auto CLI 报告输出路径和错误文件**

确保 `perf_auto.py`：

- 将报告目录解析为配置中的 `report.output_dir` 或 CLI `--output-dir`。
- 输出 `summary.md`、`metrics.csv`、`metrics.json`、`requests.jsonl`、`errors.jsonl`。
- 终端打印报告路径。
- 如果 smoke request 失败，写一条 errors.jsonl 并以非零 exit code 退出。

- [ ] **步骤 4：运行端到端报告测试通过**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_cli_dry_run.py::test_perf_auto_writes_reports_with_local_server
```

预期：PASS。

- [ ] **步骤 5：运行所有包测试通过**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests
```

预期：全部 PASS。

- [ ] **步骤 6：Commit**

```bash
git add performance-testing-skills
git commit -m "feat: write portable benchmark reports"
```

## 任务 9：补全 README、skill 使用细节和可迁移验证

**文件：**
- 修改：`performance-testing-skills/README.md`
- 修改：`performance-testing-skills/skills/automated-performance-testing/SKILL.md`
- 修改：`performance-testing-skills/skills/manual-interface-performance-testing/SKILL.md`
- 修改：`performance-testing-skills/tests/test_package_structure.py`

- [ ] **步骤 1：补充 README 内容检查测试**

向 `test_package_structure.py` 追加：

```python
def test_readme_documents_copy_and_commands():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Copy" in text or "复制" in text
    assert "perf_auto.py" in text
    assert "perf_manual.py" in text
    assert "Codex" in text
    assert "Claude" in text


def test_shell_wrappers_are_executable():
    assert (ROOT / "scripts/run_auto.sh").stat().st_mode & 0o111
    assert (ROOT / "scripts/run_manual.sh").stat().st_mode & 0o111
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_package_structure.py
```

预期：FAIL，如果 README 未包含复制、Codex/Claude 或 CLI 说明。

- [ ] **步骤 3：补全 README**

README 必须包含：

- 包定位。
- 复制方式：

  ```bash
  cp -R performance-testing-skills /path/to/target-project/
  ```

- Codex/Claude 使用方式：让助手读取对应 `skills/*/SKILL.md`，再调用 `scripts/run_auto.sh` 或 `scripts/run_manual.sh`。
- 示例命令：

  ```bash
  python3 scripts/perf_manual.py --config configs/openai_chat.json --dry-run
  python3 scripts/perf_auto.py --config configs/openai_chat.json --concurrency 1,4 --epochs 2 --dry-run
  ```

- OpenAI Chat、Completion、ASR、Generic HTTP 配置说明。
- 报告输出说明。
- 依赖说明：Python 3 标准库优先，无强制第三方依赖。

- [ ] **步骤 4：补全两个 SKILL.md 的执行细节**

`automated-performance-testing/SKILL.md` 增加：

- 参数确认清单。
- smoke request 失败时切到 manual skill。
- 报告解读规则。
- 失败率和延迟拐点判断。

`manual-interface-performance-testing/SKILL.md` 增加：

- curl 模式说明。
- ASR 文件检查说明。
- HTTP 状态码诊断表。
- 小批量验证和完整压测的边界。

- [ ] **步骤 5：运行结构和 CLI 测试通过**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests/test_package_structure.py tests/test_cli_dry_run.py
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add performance-testing-skills
git commit -m "docs: document portable performance testing skills"
```

## 任务 10：最终验证、跨目录复制验证和合并准备

**文件：**
- 不新增功能文件。
- 可能修改：`performance-testing-skills/README.md` 或测试文件中的路径 bug。

- [ ] **步骤 1：运行包内完整测试**

运行：

```bash
cd performance-testing-skills
PYTHONPATH=. pytest -q tests
```

预期：全部 PASS。

- [ ] **步骤 2：运行仓库级相关测试**

从仓库根运行：

```bash
PYTHONPATH=. pytest -q performance-testing-skills/tests
```

预期：全部 PASS。

- [ ] **步骤 3：执行可迁移复制验证**

运行：

```bash
tmpdir="$(mktemp -d)"
cp -R performance-testing-skills "$tmpdir/"
cd "$tmpdir/performance-testing-skills"
python3 scripts/perf_manual.py --config configs/openai_chat.json --dry-run
python3 scripts/perf_auto.py --config configs/openai_chat.json --concurrency 1 --epochs 1 --dry-run
bash scripts/run_manual.sh --config configs/openai_chat.json --dry-run
bash scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1 --epochs 1 --dry-run
```

预期：四条命令 exit 0，并输出 `DRY RUN`。

- [ ] **步骤 4：运行空白检查**

运行：

```bash
git diff --check
```

预期：无输出，exit 0。

- [ ] **步骤 5：检查没有真实密钥和内网地址**

运行：

```bash
python3 - <<'PY'
from pathlib import Path

bad_fragments = [
    "Bearer real-secret",
    "Resource" + "_Planning_Tool",
    "/work" + "/development-code",
    "10" + ".",
]
matches = []
for path in Path("performance-testing-skills").rglob("*"):
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for fragment in bad_fragments:
            if fragment in text:
                matches.append(f"{path}: contains {fragment!r}")
if matches:
    print("\n".join(matches))
    raise SystemExit(1)
PY
```

预期：无输出。`${API_KEY}` 不算真实密钥。

- [ ] **步骤 6：提交最终修正**

如果步骤 1 到 5 暴露问题，修复后运行：

```bash
git add performance-testing-skills
git commit -m "test: verify portable performance testing skills"
```

如果没有任何修正，不创建空提交。

- [ ] **步骤 7：最终状态检查**

运行：

```bash
git status --short --branch
git log --oneline --max-count=10
```

预期：工作树干净，分支包含本计划中的功能提交。

## 自检

规格覆盖情况：

- 可迁移包目录：任务 1、任务 10。
- 两个 skill：任务 1、任务 9。
- 必要脚本：任务 6、任务 7。
- OpenAI Chat/Completion/ASR 和 Generic HTTP：任务 2、任务 3。
- 文本和语音数据集：任务 2。
- 自动化并发阶梯和报告：任务 4、任务 7、任务 8。
- 非自动化 smoke、curl、小批量：任务 6。
- 错误分类：任务 3、任务 5、任务 8。
- 可复制到任意目录运行：任务 10。

占位符扫描要求：执行者使用 writing-plans 技能文件中列出的禁止占位词扫描本计划，预期没有命中。如果有命中，把该位置改成明确步骤、命令或代码片段。

本计划已覆盖规格中的包结构、两个 skill、自动化脚本、非自动化脚本、配置、数据集、报告、错误分类和可迁移验证。
