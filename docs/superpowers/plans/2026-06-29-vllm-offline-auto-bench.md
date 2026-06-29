# vLLM Offline Auto Bench 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 `vllm_standalone_bench` 增加离线双镜像自动化压测编排能力，支持多模型、多 vLLM 启动参数、多 benchmark 参数、后台运行、模型准备、结果归档和 Docker 网络清理。

**架构：** 新增 `auto_bench.py` 作为只依赖 Python 标准库和 Docker CLI 的主机编排器，按 JSON 配置展开测试矩阵并调用 vLLM 容器和 bench-runner 容器。新增 bench-runner Dockerfile 和示例配置，复用现有 `run_bench_multi.py` 作为实际压测客户端。实现以可测试的小函数和 `DockerRunner` 抽象隔离 Docker 命令，单元测试使用 fake runner 覆盖命令构造、状态文件、失败恢复和后台控制。

**技术栈：** Python 3 标准库、pytest、Docker CLI、JSON、现有 `vllm_standalone_bench/run_bench_multi.py`、vLLM OpenAI 镜像、ModelScope。

---

## 执行前准备

实现应在隔离 worktree 中进行，完成后合并回 `main` 再做最终验证。

```bash
git worktree add .worktrees/vllm-offline-auto-bench -b feat/vllm-offline-auto-bench
cd .worktrees/vllm-offline-auto-bench
python -m pytest vllm_standalone_bench/tests -q
```

如果 `.worktrees` 未被 git ignore，先在主仓库把 `.worktrees/` 加入 `.gitignore` 并单独提交，再创建 worktree。若基线测试失败，记录失败输出并先判断是否为已有问题。

## 文件结构

- 创建：`vllm_standalone_bench/auto_bench.py`
  - 主机编排器。负责配置解析、校验、Docker 命令构造、前台运行、后台运行、状态查询、日志查看、停止、模型准备、结果归档和清理。
- 创建：`vllm_standalone_bench/configs/auto_bench.example.json`
  - 完整矩阵示例，覆盖多模型、多 serve profile、多 bench profile。
- 创建：`vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json`
  - 当前主机真实 smoke 配置，使用 vLLM 镜像 `009e4cb46541` 和 `Qwen/Qwen2.5-1.5B-Instruct` 本地目录。
- 创建：`vllm_standalone_bench/Dockerfile.bench-runner`
  - 联网环境构建的 bench-runner 镜像，预装 `openpyxl`、`modelscope` 和现有 benchmark 依赖。
- 创建：`vllm_standalone_bench/tests/test_auto_bench.py`
  - `auto_bench.py` 的单元测试和 fake Docker runner 测试。
- 修改：`vllm_standalone_bench/README.md`
  - 增加离线双镜像自动化压测、后台运行、模型准备、真实 smoke 验证说明。

## 任务 1：配置数据结构、JSON 解析和基础校验

**文件：**
- 创建：`vllm_standalone_bench/auto_bench.py`
- 创建：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的配置解析测试**

在 `vllm_standalone_bench/tests/test_auto_bench.py` 写入：

```python
import json

import pytest

import auto_bench as ab


def write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def minimal_config(tmp_path):
    model_root = tmp_path / "model"
    model_dir = model_root / "Qwen2.5-1.5B-Instruct"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return {
        "run": {
            "name": "smoke",
            "results_dir": str(tmp_path / "results"),
            "vllm_image": "009e4cb46541",
            "bench_image": "vllm-bench-runner:offline",
            "network": "vllm-bench-net",
            "create_network": True,
            "cleanup_network": True,
            "container_port": 8000,
            "publish_host_port": False,
            "api_key": "local-bench-key",
            "ready_timeout_sec": 30,
            "cooldown_sec": 0
        },
        "mounts": {"models": str(model_root)},
        "models": [{
            "name": "qwen2_5_1_5b",
            "model_path": "/models/Qwen2.5-1.5B-Instruct",
            "tokenizer_path": "/models/Qwen2.5-1.5B-Instruct",
            "served_model_name": "qwen2_5_1_5b"
        }],
        "serve_profiles": [{
            "name": "bf16_default",
            "gpus": "all",
            "args": ["--dtype", "bfloat16"]
        }],
        "bench_profiles": [{
            "name": "smoke",
            "backend": "openai-chat",
            "input_lens": [64],
            "output_lens": [32],
            "parallel_nums": [1],
            "epochs": 1,
            "prefix_ratio": 0.0,
            "warmup_requests": 0
        }]
    }


def test_load_config_applies_defaults_and_expands_cases(tmp_path):
    path = write_config(tmp_path, minimal_config(tmp_path))

    config = ab.load_config(path)
    cases = ab.expand_cases(config)

    assert config.run.name == "smoke"
    assert config.run.container_port == 8000
    assert config.run.publish_host_port is False
    assert config.models[0].host_model_path == tmp_path / "model" / "Qwen2.5-1.5B-Instruct"
    assert len(cases) == 1
    assert cases[0].api_model_name == "qwen2_5_1_5b"
    assert cases[0].container_name.startswith("bench-vllm-qwen2_5_1_5b-bf16_default-")


def test_invalid_name_is_rejected(tmp_path):
    data = minimal_config(tmp_path)
    data["models"][0]["name"] = "bad/name"

    with pytest.raises(ab.ConfigError, match="safe filename"):
        ab.load_config(write_config(tmp_path, data))


def test_output_lens_must_broadcast_or_match_input_lens(tmp_path):
    data = minimal_config(tmp_path)
    data["bench_profiles"][0]["input_lens"] = [64, 128, 256]
    data["bench_profiles"][0]["output_lens"] = [32, 64]

    with pytest.raises(ab.ConfigError, match="output_lens"):
        ab.load_config(write_config(tmp_path, data))
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q
```

预期：失败，`ModuleNotFoundError: No module named 'auto_bench'`。

- [ ] **步骤 3：实现配置数据结构和校验**

创建 `vllm_standalone_bench/auto_bench.py`，包含以下实现骨架和实际逻辑：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MODEL_CONTAINER_ROOT = Path("/models")


class ConfigError(ValueError):
    """Raised when the auto bench configuration is invalid."""


@dataclass(frozen=True)
class RunConfig:
    name: str
    results_dir: Path
    vllm_image: str
    bench_image: str
    network: str = "vllm-bench-net"
    create_network: bool = True
    cleanup_network: bool = True
    container_port: int = 8000
    publish_host_port: bool = False
    host_port: int | None = None
    api_key: str | None = None
    ready_timeout_sec: int = 1800
    cooldown_sec: float = 20.0


@dataclass(frozen=True)
class MountConfig:
    models: Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_path: str
    tokenizer_path: str | None
    served_model_name: str | None
    host_model_path: Path
    host_tokenizer_path: Path | None


@dataclass(frozen=True)
class ServeProfile:
    name: str
    gpus: str = "all"
    args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BenchProfile:
    name: str
    backend: str = "openai-chat"
    input_lens: tuple[int, ...] = (512,)
    output_lens: tuple[int, ...] = (128,)
    parallel_nums: tuple[int, ...] = (1, 4, 8)
    epochs: int = 3
    prefix_ratio: float = 0.0
    warmup_requests: int = 1
    cross_product: bool = False
    max_ttft_ms: float | None = None
    min_throughput_tok_s: float | None = None
    min_output_compliance: float | None = None


@dataclass(frozen=True)
class AutoBenchConfig:
    run: RunConfig
    mounts: MountConfig
    models: tuple[ModelConfig, ...]
    serve_profiles: tuple[ServeProfile, ...]
    bench_profiles: tuple[BenchProfile, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    model: ModelConfig
    serve_profile: ServeProfile
    bench_profile: BenchProfile
    run_id: str
    container_name: str
    api_model_name: str


def _require_mapping(data: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{field_name} must be an object")
    return data


def _safe_name(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not SAFE_NAME_RE.match(value):
        raise ConfigError(f"{field_name} must be a safe filename: {value!r}")
    return value


def _positive_int_list(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field_name} must be a non-empty list")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or item <= 0:
            raise ConfigError(f"{field_name} must contain positive integers")
        result.append(item)
    return tuple(result)


def _container_path_to_host(path_value: str, model_root: Path) -> Path:
    container_path = Path(path_value)
    if not container_path.is_absolute():
        raise ConfigError(f"model path must be absolute inside the container: {path_value}")
    try:
        relative = container_path.relative_to(MODEL_CONTAINER_ROOT)
    except ValueError as exc:
        raise ConfigError(f"model path must be under /models: {path_value}") from exc
    return model_root / relative


def _parse_run(data: dict[str, Any]) -> RunConfig:
    run = _require_mapping(data.get("run"), "run")
    return RunConfig(
        name=_safe_name(run["name"], "run.name"),
        results_dir=Path(run.get("results_dir", "vllm_standalone_bench/results")),
        vllm_image=str(run["vllm_image"]),
        bench_image=str(run["bench_image"]),
        network=str(run.get("network", "vllm-bench-net")),
        create_network=bool(run.get("create_network", True)),
        cleanup_network=bool(run.get("cleanup_network", True)),
        container_port=int(run.get("container_port", 8000)),
        publish_host_port=bool(run.get("publish_host_port", False)),
        host_port=run.get("host_port"),
        api_key=run.get("api_key"),
        ready_timeout_sec=int(run.get("ready_timeout_sec", 1800)),
        cooldown_sec=float(run.get("cooldown_sec", 20.0)),
    )


def _parse_mounts(data: dict[str, Any]) -> MountConfig:
    mounts = _require_mapping(data.get("mounts"), "mounts")
    return MountConfig(models=Path(str(mounts["models"])))


def _parse_models(data: dict[str, Any], mounts: MountConfig) -> tuple[ModelConfig, ...]:
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError("models must be a non-empty list")
    parsed: list[ModelConfig] = []
    for item in raw_models:
        model = _require_mapping(item, "models[]")
        name = _safe_name(str(model["name"]), "model.name")
        model_path = str(model["model_path"])
        tokenizer_path = model.get("tokenizer_path")
        parsed.append(ModelConfig(
            name=name,
            model_path=model_path,
            tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
            served_model_name=model.get("served_model_name"),
            host_model_path=_container_path_to_host(model_path, mounts.models),
            host_tokenizer_path=(
                _container_path_to_host(str(tokenizer_path), mounts.models)
                if tokenizer_path else None
            ),
        ))
    return tuple(parsed)


def _parse_serve_profiles(data: dict[str, Any]) -> tuple[ServeProfile, ...]:
    raw_profiles = data.get("serve_profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ConfigError("serve_profiles must be a non-empty list")
    parsed: list[ServeProfile] = []
    for item in raw_profiles:
        profile = _require_mapping(item, "serve_profiles[]")
        args = profile.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ConfigError("serve_profile.args must be a string array")
        parsed.append(ServeProfile(
            name=_safe_name(str(profile["name"]), "serve_profile.name"),
            gpus=str(profile.get("gpus", "all")),
            args=tuple(args),
        ))
    return tuple(parsed)


def _parse_bench_profiles(data: dict[str, Any]) -> tuple[BenchProfile, ...]:
    raw_profiles = data.get("bench_profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ConfigError("bench_profiles must be a non-empty list")
    parsed: list[BenchProfile] = []
    for item in raw_profiles:
        profile = _require_mapping(item, "bench_profiles[]")
        input_lens = _positive_int_list(profile.get("input_lens", [512]), "input_lens")
        output_lens = _positive_int_list(profile.get("output_lens", [128]), "output_lens")
        cross_product = bool(profile.get("cross_product", False))
        if not cross_product and len(output_lens) not in (1, len(input_lens)):
            raise ConfigError("output_lens length must be 1 or match input_lens unless cross_product=true")
        parsed.append(BenchProfile(
            name=_safe_name(str(profile["name"]), "bench_profile.name"),
            backend=str(profile.get("backend", "openai-chat")),
            input_lens=input_lens,
            output_lens=output_lens,
            parallel_nums=_positive_int_list(profile.get("parallel_nums", [1, 4, 8]), "parallel_nums"),
            epochs=int(profile.get("epochs", 3)),
            prefix_ratio=float(profile.get("prefix_ratio", 0.0)),
            warmup_requests=int(profile.get("warmup_requests", 1)),
            cross_product=cross_product,
            max_ttft_ms=profile.get("max_ttft_ms"),
            min_throughput_tok_s=profile.get("min_throughput_tok_s"),
            min_output_compliance=profile.get("min_output_compliance"),
        ))
    return tuple(parsed)


def load_config(path: str | Path) -> AutoBenchConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = _require_mapping(json.load(handle), "config")
    run = _parse_run(raw)
    mounts = _parse_mounts(raw)
    models = _parse_models(raw, mounts)
    serve_profiles = _parse_serve_profiles(raw)
    bench_profiles = _parse_bench_profiles(raw)
    return AutoBenchConfig(run, mounts, models, serve_profiles, bench_profiles)


def make_run_id(run_name: str, now: float | None = None) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now or time.time()))
    return f"{run_name}_{stamp}"


def make_container_name(model: ModelConfig, serve_profile: ServeProfile, run_id: str) -> str:
    return f"bench-vllm-{model.name}-{serve_profile.name}-{run_id}"


def expand_cases(config: AutoBenchConfig, run_id: str | None = None) -> tuple[BenchmarkCase, ...]:
    resolved_run_id = run_id or make_run_id(config.run.name)
    cases: list[BenchmarkCase] = []
    for model in config.models:
        for serve_profile in config.serve_profiles:
            for bench_profile in config.bench_profiles:
                api_model_name = model.served_model_name or model.name
                cases.append(BenchmarkCase(
                    model=model,
                    serve_profile=serve_profile,
                    bench_profile=bench_profile,
                    run_id=resolved_run_id,
                    container_name=make_container_name(model, serve_profile, resolved_run_id),
                    api_model_name=api_model_name,
                ))
    return tuple(cases)
```

- [ ] **步骤 4：运行配置解析测试验证通过**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_load_config_applies_defaults_and_expands_cases vllm_standalone_bench/tests/test_auto_bench.py::test_invalid_name_is_rejected vllm_standalone_bench/tests/test_auto_bench.py::test_output_lens_must_broadcast_or_match_input_lens -q
```

预期：3 个测试通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git -c user.name=Codex -c user.email=codex@example.local commit -m "feat(bench): 添加自动压测配置解析"
```

## 任务 2：Docker 命令构造、网络所有权和预检

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的 Docker 命令测试**

追加测试：

```python
def test_build_vllm_command_uses_bridge_network_without_host_port(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    run_dir = tmp_path / "results" / "run123"

    cmd = ab.build_vllm_run_command(config, case, run_dir)

    assert "--network" in cmd
    assert "vllm-bench-net" in cmd
    assert "--network=host" not in cmd
    assert "-p" not in cmd
    assert "vllm serve" not in " ".join(cmd)
    assert cmd[-10:] == [
        "vllm", "serve", "/models/Qwen2.5-1.5B-Instruct",
        "--served-model-name", "qwen2_5_1_5b",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", "local-bench-key",
        "--dtype", "bfloat16",
    ][-10:]


def test_build_bench_command_targets_container_dns(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    bench_dir = tmp_path / "results" / "run123" / "qwen2_5_1_5b" / "bf16_default" / "smoke"

    cmd = ab.build_bench_run_command(config, case, bench_dir)

    assert "--network" in cmd
    assert "vllm-bench-net" in cmd
    assert "--base-url" in cmd
    assert f"http://{case.container_name}:8000/v1" in cmd
    assert "--model" in cmd
    assert "qwen2_5_1_5b" in cmd
    assert "--output-csv" in cmd
    assert "/results/result.csv" in cmd


def test_network_cleanup_only_removes_owned_empty_network():
    assert ab.should_cleanup_network(owned=True, cleanup_enabled=True, connected_containers=[]) is True
    assert ab.should_cleanup_network(owned=False, cleanup_enabled=True, connected_containers=[]) is False
    assert ab.should_cleanup_network(owned=True, cleanup_enabled=False, connected_containers=[]) is False
    assert ab.should_cleanup_network(owned=True, cleanup_enabled=True, connected_containers=["external"]) is False
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_build_vllm_command_uses_bridge_network_without_host_port vllm_standalone_bench/tests/test_auto_bench.py::test_build_bench_command_targets_container_dns vllm_standalone_bench/tests/test_auto_bench.py::test_network_cleanup_only_removes_owned_empty_network -q
```

预期：失败，命令构造函数未定义。

- [ ] **步骤 3：实现 Docker 命令构造和预检接口**

在 `auto_bench.py` 中追加：

```python
@dataclass
class Completed:
    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class DockerRunner:
    def run(self, args: list[str], *, check: bool = False,
            capture: bool = True, text: bool = True,
            stdout: Any = None, stderr: Any = None) -> Completed:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=capture if stdout is None and stderr is None else False,
            text=text,
            stdout=stdout,
            stderr=stderr,
        )
        result = Completed(args=args, returncode=completed.returncode,
                           stdout=completed.stdout or "", stderr=completed.stderr or "")
        if check and result.returncode != 0:
            raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}\n{result.stderr}")
        return result


def build_vllm_run_command(config: AutoBenchConfig, case: BenchmarkCase, run_dir: Path) -> list[str]:
    cmd = [
        "docker", "run", "-d",
        "--name", case.container_name,
        "--gpus", case.serve_profile.gpus,
        "--network", config.run.network,
        "-v", f"{config.mounts.models}:/models:ro",
    ]
    if config.run.publish_host_port:
        if config.run.host_port is None:
            raise ConfigError("host_port is required when publish_host_port=true")
        cmd.extend(["-p", f"127.0.0.1:{config.run.host_port}:{config.run.container_port}"])
    cmd.extend([
        config.run.vllm_image,
        "vllm", "serve", case.model.model_path,
        "--served-model-name", case.api_model_name,
        "--host", "0.0.0.0",
        "--port", str(config.run.container_port),
    ])
    if config.run.api_key:
        cmd.extend(["--api-key", config.run.api_key])
    cmd.extend(case.serve_profile.args)
    return cmd


def _append_many(cmd: list[str], flag: str, values: tuple[int, ...]) -> None:
    cmd.append(flag)
    cmd.extend(str(value) for value in values)


def build_bench_run_command(config: AutoBenchConfig, case: BenchmarkCase, bench_dir: Path) -> list[str]:
    bench = case.bench_profile
    cmd = [
        "docker", "run", "--rm",
        "--network", config.run.network,
        "-v", f"{config.mounts.models}:/models:ro",
        "-v", f"{bench_dir}:/results",
        config.run.bench_image,
        "python", "/opt/vllm_standalone_bench/run_bench_multi.py",
        "--base-url", f"http://{case.container_name}:{config.run.container_port}/v1",
        "--model", case.api_model_name,
        "--served-model-name", case.api_model_name,
        "--backend", bench.backend,
        "--epochs", str(bench.epochs),
        "--warmup-requests", str(bench.warmup_requests),
        "--output-csv", "/results/result.csv",
        "--output-xlsx", "/results/result.xlsx",
    ]
    if config.run.api_key:
        cmd.extend(["--api-key", config.run.api_key])
    if case.model.tokenizer_path:
        cmd.extend(["--tokenizer", case.model.tokenizer_path])
    _append_many(cmd, "--input-lens", bench.input_lens)
    _append_many(cmd, "--output-lens", bench.output_lens)
    _append_many(cmd, "--parallel-nums", bench.parallel_nums)
    if bench.cross_product:
        cmd.append("--cross-product")
    if bench.prefix_ratio:
        cmd.extend(["--prefix-ratio", str(bench.prefix_ratio)])
    if bench.max_ttft_ms is not None:
        cmd.extend(["--max-ttft-ms", str(bench.max_ttft_ms)])
    if bench.min_throughput_tok_s is not None:
        cmd.extend(["--min-throughput-tok-s", str(bench.min_throughput_tok_s)])
    if bench.min_output_compliance is not None:
        cmd.extend(["--min-output-compliance", str(bench.min_output_compliance)])
    return cmd


def should_cleanup_network(*, owned: bool, cleanup_enabled: bool,
                           connected_containers: list[str]) -> bool:
    return owned and cleanup_enabled and not connected_containers


def validate_local_paths(config: AutoBenchConfig) -> None:
    if not config.mounts.models.is_dir():
        raise ConfigError(f"model root does not exist: {config.mounts.models}")
    for model in config.models:
        if not model.host_model_path.is_dir():
            raise ConfigError(f"model path does not exist: {model.host_model_path}")
        if model.host_tokenizer_path is not None and not model.host_tokenizer_path.is_dir():
            raise ConfigError(f"tokenizer path does not exist: {model.host_tokenizer_path}")
```

- [ ] **步骤 4：运行 Docker 命令测试验证通过**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_build_vllm_command_uses_bridge_network_without_host_port vllm_standalone_bench/tests/test_auto_bench.py::test_build_bench_command_targets_container_dns vllm_standalone_bench/tests/test_auto_bench.py::test_network_cleanup_only_removes_owned_empty_network -q
```

预期：3 个测试通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git -c user.name=Codex -c user.email=codex@example.local commit -m "feat(bench): 构造自动压测 Docker 命令"
```

## 任务 3：结果目录、状态文件和 manifest

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的状态文件测试**

追加测试：

```python
def test_case_paths_and_state_files_are_written(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)

    assert layout.run_dir == tmp_path / "results" / "run123"
    assert layout.serve_dir == layout.run_dir / "qwen2_5_1_5b" / "bf16_default"
    assert layout.bench_dir == layout.serve_dir / "smoke"

    ab.write_state(layout.run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": {"model": "qwen2_5_1_5b", "serve_profile": "bf16_default", "bench_profile": "smoke"},
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1}
    })
    state = json.loads((layout.run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "running"


def test_manifest_records_relative_artifact_paths(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    manifest = ab.Manifest(run_id="run123", total=1)

    manifest.record(case, layout, "passed")
    ab.write_manifest(layout.run_dir, manifest)

    data = json.loads((layout.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["run_id"] == "run123"
    assert data["status"] == "completed"
    assert data["cases"][0]["csv"] == "qwen2_5_1_5b/bf16_default/smoke/result.csv"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_case_paths_and_state_files_are_written vllm_standalone_bench/tests/test_auto_bench.py::test_manifest_records_relative_artifact_paths -q
```

预期：失败，`build_layout` 或 `Manifest` 未定义。

- [ ] **步骤 3：实现状态和 manifest 辅助函数**

追加：

```python
@dataclass(frozen=True)
class CaseLayout:
    run_dir: Path
    serve_dir: Path
    bench_dir: Path


def build_layout(config: AutoBenchConfig, run_id: str, case: BenchmarkCase) -> CaseLayout:
    run_dir = config.run.results_dir / run_id
    serve_dir = run_dir / case.model.name / case.serve_profile.name
    bench_dir = serve_dir / case.bench_profile.name
    return CaseLayout(run_dir=run_dir, serve_dir=serve_dir, bench_dir=bench_dir)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def write_state(run_dir: Path, state: dict[str, Any]) -> None:
    write_json_atomic(run_dir / "state.json", state)


@dataclass
class Manifest:
    run_id: str
    total: int
    cases: list[dict[str, Any]] = field(default_factory=list)

    def record(self, case: BenchmarkCase, layout: CaseLayout, status: str,
               error: str | None = None) -> None:
        row: dict[str, Any] = {
            "model": case.model.name,
            "serve_profile": case.serve_profile.name,
            "bench_profile": case.bench_profile.name,
            "status": status,
            "csv": str((layout.bench_dir / "result.csv").relative_to(layout.run_dir)),
            "xlsx": str((layout.bench_dir / "result.xlsx").relative_to(layout.run_dir)),
        }
        if error:
            row["error"] = error
        self.cases.append(row)

    def status(self) -> str:
        statuses = {case["status"] for case in self.cases}
        if len(self.cases) < self.total:
            return "running"
        if statuses == {"passed"}:
            return "completed"
        if "interrupted" in statuses:
            return "interrupted"
        if "failed" in statuses or "skipped" in statuses:
            return "completed_with_failures"
        return "completed"

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "status": self.status(), "cases": self.cases}


def write_manifest(run_dir: Path, manifest: Manifest) -> None:
    write_json_atomic(run_dir / "manifest.json", manifest.to_dict())
```

- [ ] **步骤 4：运行状态文件测试验证通过**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_case_paths_and_state_files_are_written vllm_standalone_bench/tests/test_auto_bench.py::test_manifest_records_relative_artifact_paths -q
```

预期：2 个测试通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git -c user.name=Codex -c user.email=codex@example.local commit -m "feat(bench): 写入自动压测状态和 manifest"
```

## 任务 4：前台编排流程、ready 检查、失败恢复和网络清理

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写 fake runner 编排测试**

追加：

```python
class FakeRunner:
    def __init__(self, failures=None):
        self.commands = []
        self.failures = failures or {}

    def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
        self.commands.append(list(args))
        key = " ".join(args[:3])
        if key in self.failures:
            return ab.Completed(list(args), self.failures[key], "", "forced failure")
        if args[:3] == ["docker", "network", "inspect"]:
            return ab.Completed(list(args), 1, "", "not found")
        if args[:3] == ["docker", "network", "create"]:
            return ab.Completed(list(args), 0, "network-id\n", "")
        if args[:3] == ["docker", "run", "-d"]:
            return ab.Completed(list(args), 0, "container-id\n", "")
        if args[:3] == ["docker", "logs", "--timestamps"]:
            return ab.Completed(list(args), 0, "vllm log\n", "")
        if args[:3] == ["docker", "inspect", "--format"]:
            return ab.Completed(list(args), 0, "[]\n", "")
        if args[:3] == ["docker", "network", "rm"]:
            return ab.Completed(list(args), 0, "", "")
        if args[:2] == ["docker", "stop"]:
            return ab.Completed(list(args), 0, "", "")
        if args[:3] == ["docker", "inspect", "--type=image"]:
            return ab.Completed(list(args), 0, "image\n", "")
        return ab.Completed(list(args), 0, "ok\n", "")


def test_controller_runs_case_and_cleans_owned_network(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    assert result == 0
    assert any("docker network create vllm-bench-net" in cmd for cmd in joined)
    assert any("docker run -d" in cmd for cmd in joined)
    assert any("run_bench_multi.py" in cmd for cmd in joined)
    assert any("docker stop bench-vllm-qwen2_5_1_5b-bf16_default-run123" in cmd for cmd in joined)
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_controller_skips_bench_when_vllm_not_ready(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: False)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 1
    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cases"][0]["status"] == "skipped"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_controller_runs_case_and_cleans_owned_network vllm_standalone_bench/tests/test_auto_bench.py::test_controller_skips_bench_when_vllm_not_ready -q
```

预期：失败，`run_controller` 未定义。

- [ ] **步骤 3：实现前台编排和 cleanup**

追加：

```python
class FakeRunnerProtocol:
    def run(self, args: list[str], *, check: bool = False,
            capture: bool = True, text: bool = True,
            stdout: Any = None, stderr: Any = None) -> Completed:
        raise NotImplementedError


def docker_network_exists(runner: DockerRunner | FakeRunnerProtocol, network: str) -> bool:
    result = runner.run(["docker", "network", "inspect", network], check=False)
    return result.returncode == 0


def ensure_network(config: AutoBenchConfig, runner: DockerRunner | FakeRunnerProtocol,
                   dry_run: bool) -> bool:
    if docker_network_exists(runner, config.run.network):
        return False
    if not config.run.create_network:
        raise RuntimeError(f"Docker network does not exist: {config.run.network}")
    if dry_run:
        print_cmd(["docker", "network", "create", config.run.network])
        return True
    runner.run(["docker", "network", "create", config.run.network], check=True)
    return True


def connected_network_containers(runner: DockerRunner | FakeRunnerProtocol,
                                 network: str) -> list[str]:
    result = runner.run([
        "docker", "inspect", "--format",
        "{{json .Containers}}", network,
    ], check=False)
    if result.returncode != 0 or result.stdout.strip() in ("", "null", "{}"):
        return []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ["unknown"]
    return list(data.keys())


def cleanup_network(config: AutoBenchConfig, runner: DockerRunner | FakeRunnerProtocol,
                    owned: bool, dry_run: bool) -> None:
    connected = connected_network_containers(runner, config.run.network)
    if should_cleanup_network(
        owned=owned,
        cleanup_enabled=config.run.cleanup_network,
        connected_containers=connected,
    ):
        cmd = ["docker", "network", "rm", config.run.network]
        if dry_run:
            print_cmd(cmd)
        else:
            runner.run(cmd, check=False)


def print_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))


def wait_for_ready(base_url: str, api_key: str | None, timeout_sec: int) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_sec
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    while time.time() < deadline:
        request = urllib.request.Request(f"{base_url}/models", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    return False


def save_vllm_artifacts(config: AutoBenchConfig,
                        runner: DockerRunner | FakeRunnerProtocol,
                        case: BenchmarkCase, layout: CaseLayout) -> None:
    layout.serve_dir.mkdir(parents=True, exist_ok=True)
    logs = runner.run(["docker", "logs", "--timestamps", case.container_name], check=False)
    (layout.serve_dir / "vllm.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
    inspect = runner.run(["docker", "inspect", case.container_name], check=False)
    (layout.serve_dir / "docker.inspect.json").write_text(inspect.stdout, encoding="utf-8")
    (layout.serve_dir / "serve_command.txt").write_text(
        " ".join(build_vllm_run_command(config, case, layout.run_dir)),
        encoding="utf-8",
    )
```

实现 `run_controller`：

```python
def run_controller(config: AutoBenchConfig, run_id: str,
                   runner: DockerRunner | FakeRunnerProtocol | None = None,
                   dry_run: bool = False) -> int:
    active_runner = runner or DockerRunner()
    cases = expand_cases(config, run_id=run_id)
    run_dir = config.run.results_dir / run_id
    manifest = Manifest(run_id=run_id, total=len(cases))
    write_json_atomic(run_dir / "config.resolved.json", config_to_dict(config))
    network_owned = False
    exit_code = 0
    try:
        network_owned = ensure_network(config, active_runner, dry_run)
        grouped: dict[tuple[str, str], list[BenchmarkCase]] = {}
        for case in cases:
            grouped.setdefault((case.model.name, case.serve_profile.name), []).append(case)
        completed = 0
        for group_cases in grouped.values():
            serve_case = group_cases[0]
            serve_layout = build_layout(config, run_id, serve_case)
            vllm_cmd = build_vllm_run_command(config, serve_case, serve_layout.run_dir)
            if dry_run:
                print_cmd(vllm_cmd)
                ready = True
            else:
                active_runner.run(["docker", "rm", "-f", serve_case.container_name], check=False)
                active_runner.run(vllm_cmd, check=True)
                ready = wait_for_ready(
                    f"http://{serve_case.container_name}:{config.run.container_port}/v1",
                    config.run.api_key,
                    config.run.ready_timeout_sec,
                )
            if not ready:
                exit_code = 1
                for case in group_cases:
                    layout = build_layout(config, run_id, case)
                    manifest.record(case, layout, "skipped", error="vLLM ready check timed out")
                    completed += 1
                write_manifest(run_dir, manifest)
                continue
            for case in group_cases:
                layout = build_layout(config, run_id, case)
                layout.bench_dir.mkdir(parents=True, exist_ok=True)
                write_state(run_dir, current_state(run_id, cases, completed, case, "running"))
                bench_cmd = build_bench_run_command(config, case, layout.bench_dir)
                if dry_run:
                    print_cmd(bench_cmd)
                    status = "passed"
                    error = None
                else:
                    with (layout.bench_dir / "bench.log").open("w", encoding="utf-8") as log:
                        result = active_runner.run(bench_cmd, check=False, capture=False, stdout=log, stderr=log)
                    status = "passed" if result.returncode == 0 else "failed"
                    error = None if result.returncode == 0 else f"benchmark exited {result.returncode}"
                if status != "passed":
                    exit_code = 1
                manifest.record(case, layout, status, error=error)
                write_json_atomic(layout.bench_dir / "status.json", {"status": status, "error": error})
                completed += 1
                write_manifest(run_dir, manifest)
            if not dry_run:
                save_vllm_artifacts(config, active_runner, serve_case, serve_layout)
                active_runner.run(["docker", "stop", serve_case.container_name], check=False)
                if config.run.cooldown_sec > 0:
                    time.sleep(config.run.cooldown_sec)
        write_state(run_dir, finished_state(run_id, manifest))
        return exit_code
    finally:
        cleanup_network(config, active_runner, network_owned, dry_run)
```

同时实现 `config_to_dict`、`current_state`、`finished_state`，字段与规格一致。

- [ ] **步骤 4：修复类型引用并运行编排测试**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_controller_runs_case_and_cleans_owned_network vllm_standalone_bench/tests/test_auto_bench.py::test_controller_skips_bench_when_vllm_not_ready -q
```

预期：2 个测试通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git -c user.name=Codex -c user.email=codex@example.local commit -m "feat(bench): 编排 vLLM 容器压测流程"
```

## 任务 5：CLI、dry-run、后台运行、status、logs、stop

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的 CLI 辅助测试**

追加：

```python
def test_detach_command_reinvokes_child_with_run_id(tmp_path):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    run_id = "smoke_20260629_120000"
    cmd = ab.build_detach_command(config_path, run_id)

    assert cmd[:2] == [sys.executable, str(Path(ab.__file__).resolve())]
    assert cmd[2:] == [
        "run", "--config", str(config_path),
        "--run-id", run_id,
        "--child"
    ]


def test_status_reads_state_file(tmp_path, capsys):
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": {"model": "m", "serve_profile": "s", "bench_profile": "b"},
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1}
    })

    exit_code = ab.print_status(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "running" in captured.out
    assert "m/s/b" in captured.out
```

在测试文件顶部补充：

```python
import sys
from pathlib import Path
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_detach_command_reinvokes_child_with_run_id vllm_standalone_bench/tests/test_auto_bench.py::test_status_reads_state_file -q
```

预期：失败，CLI 辅助函数未定义。

- [ ] **步骤 3：实现 CLI 和后台辅助函数**

追加：

```python
def build_detach_command(config_path: Path, run_id: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--config", str(config_path),
        "--run-id", run_id,
        "--child",
    ]


def start_detached(config_path: Path, config: AutoBenchConfig, run_id: str) -> int:
    run_dir = config.run.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_state(run_dir, {
        "run_id": run_id,
        "status": "starting",
        "current": None,
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 0,
                   "total": len(expand_cases(config, run_id=run_id))}
    })
    log_file = (run_dir / "controller.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        build_detach_command(config_path, run_id),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (run_dir / "controller.pid").write_text(str(process.pid), encoding="utf-8")
    print(f"run_id={run_id}")
    print(f"log={run_dir / 'controller.log'}")
    return 0


def print_status(run_dir: Path) -> int:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        print(f"state file not found: {state_path}", file=sys.stderr)
        return 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    current = state.get("current")
    current_text = "-"
    if current:
        current_text = f"{current['model']}/{current['serve_profile']}/{current['bench_profile']}"
    counts = state.get("counts", {})
    print(f"run_id: {state.get('run_id')}")
    print(f"status: {state.get('status')}")
    print(f"current: {current_text}")
    print(f"counts: passed={counts.get('passed', 0)} failed={counts.get('failed', 0)} skipped={counts.get('skipped', 0)} total={counts.get('total', 0)}")
    return 0


def follow_file(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if line:
                print(line, end="")
            else:
                time.sleep(1)


def stop_run(run_dir: Path) -> int:
    pid_path = run_dir / "controller.pid"
    if not pid_path.exists():
        print(f"pid file not found: {pid_path}", file=sys.stderr)
        return 1
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    os.kill(pid, signal.SIGTERM)
    print(f"sent SIGTERM to {pid}")
    return 0
```

实现 `main`：

```python
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline vLLM auto benchmark orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--run-id")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--detach", action="store_true")
    run_p.add_argument("--child", action="store_true")

    status_p = sub.add_parser("status")
    status_p.add_argument("--results-dir", default="vllm_standalone_bench/results")
    status_p.add_argument("--run-id", required=True)

    logs_p = sub.add_parser("logs")
    logs_p.add_argument("--results-dir", default="vllm_standalone_bench/results")
    logs_p.add_argument("--run-id", required=True)
    logs_p.add_argument("--follow", action="store_true")

    stop_p = sub.add_parser("stop")
    stop_p.add_argument("--results-dir", default="vllm_standalone_bench/results")
    stop_p.add_argument("--run-id", required=True)

    prepare_p = sub.add_parser("prepare-model")
    prepare_p.add_argument("--modelscope-id", required=True)
    prepare_p.add_argument("--target", required=True)
    prepare_p.add_argument("--bench-image", required=True)
    prepare_p.add_argument("--force", action="store_true")
    return parser.parse_args(argv)
```

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "run":
        config_path = Path(args.config)
        config = load_config(config_path)
        run_id = args.run_id or make_run_id(config.run.name)
        if args.detach and not args.child:
            return start_detached(config_path, config, run_id)
        return run_controller(config, run_id=run_id, dry_run=args.dry_run)
    if args.command == "status":
        return print_status(Path(args.results_dir) / args.run_id)
    if args.command == "logs":
        log_path = Path(args.results_dir) / args.run_id / "controller.log"
        if args.follow:
            return follow_file(log_path)
        print(log_path.read_text(encoding="utf-8"), end="")
        return 0
    if args.command == "stop":
        return stop_run(Path(args.results_dir) / args.run_id)
    if args.command == "prepare-model":
        return prepare_model(
            modelscope_id=args.modelscope_id,
            target=Path(args.target),
            bench_image=args.bench_image,
            force=args.force,
            runner=DockerRunner(),
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 4：运行 CLI 测试和 dry-run 冒烟**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_detach_command_reinvokes_child_with_run_id vllm_standalone_bench/tests/test_auto_bench.py::test_status_reads_state_file -q
python vllm_standalone_bench/auto_bench.py --help
```

预期：2 个测试通过，`--help` 输出包含 `run`、`status`、`logs`、`stop`、`prepare-model`。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git -c user.name=Codex -c user.email=codex@example.local commit -m "feat(bench): 添加自动压测 CLI 和后台运行"
```

## 任务 6：ModelScope 模型准备命令和权重完整性校验

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的模型校验和准备测试**

追加：

```python
def test_model_dir_requires_complete_safetensors(tmp_path):
    target = tmp_path / "Qwen2.5-1.5B-Instruct"
    target.mkdir()
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "tokenizer.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors.parts").mkdir()

    with pytest.raises(ab.ConfigError, match="safetensors"):
        ab.validate_prepared_model_dir(target)


def test_prepare_model_uses_bench_image_and_temp_dir(tmp_path):
    target = tmp_path / "Qwen2.5-1.5B-Instruct"
    tmp_download = tmp_path / "Qwen2.5-1.5B-Instruct.download-tmp"
    tmp_download.mkdir()
    (tmp_download / "config.json").write_text("{}", encoding="utf-8")
    (tmp_download / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_download / "model.safetensors").write_text("weights", encoding="utf-8")
    runner = FakeRunner()

    exit_code = ab.prepare_model(
        modelscope_id="Qwen/Qwen2.5-1.5B-Instruct",
        target=target,
        bench_image="vllm-bench-runner:offline",
        force=False,
        runner=runner,
    )

    assert exit_code == 0
    assert target.exists()
    joined = " ".join(" ".join(cmd) for cmd in runner.commands)
    assert "vllm-bench-runner:offline" in joined
    assert "Qwen/Qwen2.5-1.5B-Instruct" in joined
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_model_dir_requires_complete_safetensors vllm_standalone_bench/tests/test_auto_bench.py::test_prepare_model_uses_bench_image_and_temp_dir -q
```

预期：失败，模型准备函数未定义。

- [ ] **步骤 3：实现模型准备**

追加：

```python
def validate_prepared_model_dir(target: Path) -> None:
    required = ["config.json"]
    for filename in required:
        if not (target / filename).is_file():
            raise ConfigError(f"model directory missing {filename}: {target}")
    if not ((target / "tokenizer.json").is_file() or (target / "tokenizer_config.json").is_file()):
        raise ConfigError(f"model directory missing tokenizer files: {target}")
    safetensors = [path for path in target.glob("*.safetensors") if path.is_file()]
    if not safetensors:
        raise ConfigError(f"model directory missing complete safetensors weights: {target}")


def prepare_model(modelscope_id: str, target: Path, bench_image: str,
                  force: bool, runner: DockerRunner | FakeRunnerProtocol) -> int:
    parent = target.parent
    tmp_target = parent / f"{target.name}.download-tmp"
    if target.exists() and not force:
        validate_prepared_model_dir(target)
        print(f"model already prepared: {target}")
        return 0
    if target.exists() and force:
        backup = parent / f"{target.name}.backup-{time.strftime('%Y%m%d_%H%M%S')}"
        target.rename(backup)
    tmp_target.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{parent}:/model-target",
        bench_image,
        "python", "-c",
        (
            "import sys; "
            "from modelscope.hub.snapshot_download import snapshot_download; "
            "snapshot_download(sys.argv[1], local_dir=sys.argv[2])"
        ),
        modelscope_id,
        f"/model-target/{tmp_target.name}",
    ]
    result = runner.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"model download failed: {result.stderr}")
    validate_prepared_model_dir(tmp_target)
    if target.exists():
        raise ConfigError(f"target exists after download: {target}")
    tmp_target.rename(target)
    print(f"model prepared: {target}")
    return 0
```

- [ ] **步骤 4：运行模型准备测试验证通过**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_model_dir_requires_complete_safetensors vllm_standalone_bench/tests/test_auto_bench.py::test_prepare_model_uses_bench_image_and_temp_dir -q
```

预期：2 个测试通过。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git -c user.name=Codex -c user.email=codex@example.local commit -m "feat(bench): 添加 ModelScope 模型准备命令"
```

## 任务 7：bench-runner Dockerfile 和示例配置

**文件：**
- 创建：`vllm_standalone_bench/Dockerfile.bench-runner`
- 创建：`vllm_standalone_bench/configs/auto_bench.example.json`
- 创建：`vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的静态配置测试**

追加：

```python
def test_example_configs_are_parseable():
    root = Path(__file__).resolve().parents[1]
    for filename in [
        "auto_bench.example.json",
        "auto_bench.qwen2_5_1_5b.smoke.json",
    ]:
        path = root / "configs" / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run"]["network"] == "vllm-bench-net"
        assert data["run"]["publish_host_port"] is False
        assert data["models"]
        assert data["serve_profiles"]
        assert data["bench_profiles"]


def test_bench_runner_dockerfile_contains_offline_dependencies():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile.bench-runner").read_text(encoding="utf-8")
    assert "openpyxl" in dockerfile
    assert "modelscope" in dockerfile
    assert "run_bench_multi.py" in dockerfile
    assert "vllm_bench" in dockerfile
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_example_configs_are_parseable vllm_standalone_bench/tests/test_auto_bench.py::test_bench_runner_dockerfile_contains_offline_dependencies -q
```

预期：失败，配置文件和 Dockerfile 不存在。

- [ ] **步骤 3：创建 Dockerfile**

写入 `vllm_standalone_bench/Dockerfile.bench-runner`：

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/vllm_standalone_bench

COPY requirements.txt /opt/vllm_standalone_bench/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt openpyxl modelscope

COPY run_bench_multi.py /opt/vllm_standalone_bench/run_bench_multi.py
COPY run_bench_serve.py /opt/vllm_standalone_bench/run_bench_serve.py
COPY vllm_bench /opt/vllm_standalone_bench/vllm_bench

CMD ["python", "/opt/vllm_standalone_bench/run_bench_multi.py", "--help"]
```

- [ ] **步骤 4：创建示例配置**

写入 `vllm_standalone_bench/configs/auto_bench.example.json`：

```json
{
  "run": {
    "name": "offline_full_bench",
    "results_dir": "vllm_standalone_bench/results",
    "vllm_image": "009e4cb46541",
    "bench_image": "vllm-bench-runner:offline",
    "network": "vllm-bench-net",
    "create_network": true,
    "cleanup_network": true,
    "container_port": 8000,
    "publish_host_port": false,
    "host_port": 18000,
    "api_key": "local-bench-key",
    "ready_timeout_sec": 1800,
    "cooldown_sec": 20
  },
  "mounts": {
    "models": "/Resource_Planning_Tool/model"
  },
  "models": [
    {
      "name": "qwen2_5_1_5b",
      "model_path": "/models/Qwen2.5-1.5B-Instruct",
      "tokenizer_path": "/models/Qwen2.5-1.5B-Instruct",
      "served_model_name": "qwen2_5_1_5b"
    }
  ],
  "serve_profiles": [
    {
      "name": "bf16_default",
      "gpus": "all",
      "args": ["--dtype", "bfloat16", "--gpu-memory-utilization", "0.90"]
    },
    {
      "name": "bf16_prefix_on",
      "gpus": "all",
      "args": ["--dtype", "bfloat16", "--enable-prefix-caching", "--gpu-memory-utilization", "0.90"]
    }
  ],
  "bench_profiles": [
    {
      "name": "latency_matrix",
      "backend": "openai-chat",
      "input_lens": [128, 512, 1024],
      "output_lens": [1024],
      "parallel_nums": [1, 4, 8],
      "epochs": 3,
      "prefix_ratio": 0.8,
      "warmup_requests": 1,
      "max_ttft_ms": 15000,
      "min_throughput_tok_s": 5
    }
  ]
}
```

写入 `vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json`：

```json
{
  "run": {
    "name": "qwen2_5_1_5b_smoke",
    "results_dir": "vllm_standalone_bench/results",
    "vllm_image": "009e4cb46541",
    "bench_image": "vllm-bench-runner:offline",
    "network": "vllm-bench-net",
    "create_network": true,
    "cleanup_network": true,
    "container_port": 8000,
    "publish_host_port": false,
    "host_port": 18000,
    "api_key": "local-bench-key",
    "ready_timeout_sec": 1800,
    "cooldown_sec": 5
  },
  "mounts": {
    "models": "/Resource_Planning_Tool/model"
  },
  "models": [
    {
      "name": "qwen2_5_1_5b",
      "model_path": "/models/Qwen2.5-1.5B-Instruct",
      "tokenizer_path": "/models/Qwen2.5-1.5B-Instruct",
      "served_model_name": "qwen2_5_1_5b"
    }
  ],
  "serve_profiles": [
    {
      "name": "bf16_default",
      "gpus": "all",
      "args": ["--dtype", "bfloat16", "--gpu-memory-utilization", "0.90"]
    }
  ],
  "bench_profiles": [
    {
      "name": "smoke",
      "backend": "openai-chat",
      "input_lens": [64],
      "output_lens": [32],
      "parallel_nums": [1],
      "epochs": 1,
      "prefix_ratio": 0.0,
      "warmup_requests": 0
    }
  ]
}
```

- [ ] **步骤 5：运行静态配置测试验证通过**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_example_configs_are_parseable vllm_standalone_bench/tests/test_auto_bench.py::test_bench_runner_dockerfile_contains_offline_dependencies -q
```

预期：2 个测试通过。

- [ ] **步骤 6：Commit**

```bash
git add vllm_standalone_bench/Dockerfile.bench-runner vllm_standalone_bench/configs/auto_bench.example.json vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json vllm_standalone_bench/tests/test_auto_bench.py
git -c user.name=Codex -c user.email=codex@example.local commit -m "feat(bench): 添加离线压测镜像和配置"
```

## 任务 8：README 文档和 dry-run 验证

**文件：**
- 修改：`vllm_standalone_bench/README.md`

- [ ] **步骤 1：添加 README 章节**

在 `vllm_standalone_bench/README.md` 的“安装”章节后追加：

```markdown
## 离线双镜像自动化压测

自动化入口为 `auto_bench.py`。主机只需要 Docker CLI、GPU runtime 和 Python 3；benchmark 依赖预装在 `vllm-bench-runner:offline` 镜像里。

### 联网环境构建 bench-runner

```bash
docker build \
  -f vllm_standalone_bench/Dockerfile.bench-runner \
  -t vllm-bench-runner:offline \
  vllm_standalone_bench
docker save vllm-bench-runner:offline -o vllm-bench-runner-offline.tar
```

### 离线环境导入镜像

```bash
docker load -i vllm-bench-runner-offline.tar
docker load -i vllm-offline-image.tar
```

### 前台运行

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.example.json
```

### 后台运行

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.example.json \
  --detach
```

```bash
python3 vllm_standalone_bench/auto_bench.py status --run-id <run_id>
python3 vllm_standalone_bench/auto_bench.py logs --run-id <run_id> --follow
python3 vllm_standalone_bench/auto_bench.py stop --run-id <run_id>
```

默认使用 Docker bridge network，不使用 `--network host`，也不暴露主机端口。运行结束后，脚本会清理本次自动创建的 Docker network。

### ModelScope 准备 smoke 模型

```bash
python3 vllm_standalone_bench/auto_bench.py prepare-model \
  --modelscope-id Qwen/Qwen2.5-1.5B-Instruct \
  --target /Resource_Planning_Tool/model/Qwen2.5-1.5B-Instruct \
  --bench-image vllm-bench-runner:offline
```

### 当前主机 smoke 验证

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json
```
```

- [ ] **步骤 2：运行文档相关 dry-run**

运行：

```bash
python vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json \
  --dry-run
```

预期：输出 `docker network create`、`docker run -d`、bench-runner `docker run --rm`，不启动容器。

- [ ] **步骤 3：Commit**

```bash
git add vllm_standalone_bench/README.md
git -c user.name=Codex -c user.email=codex@example.local commit -m "docs(bench): 说明离线自动压测流程"
```

## 任务 9：全量测试、合并回 main 和真实验证

**文件：**
- 验证：`vllm_standalone_bench/tests`
- 验证：`vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json`

- [ ] **步骤 1：运行全量 Python 测试**

运行：

```bash
python -m pytest vllm_standalone_bench/tests -q
```

预期：全部测试通过。

- [ ] **步骤 2：检查 feature worktree diff**

运行：

```bash
git status --short
git log --oneline main..HEAD
```

预期：工作区没有未提交的实现文件；输出包含任务提交。

- [ ] **步骤 3：合并回 main**

在主仓库执行：

```bash
cd /Resource_Planning_Tool
git merge --ff-only feat/vllm-offline-auto-bench
```

预期：快进合并成功。

- [ ] **步骤 4：在 main 运行单元测试**

运行：

```bash
cd /Resource_Planning_Tool
python -m pytest vllm_standalone_bench/tests -q
```

预期：全部测试通过。

- [ ] **步骤 5：在 main 构建 bench-runner 镜像**

运行：

```bash
docker build \
  -f vllm_standalone_bench/Dockerfile.bench-runner \
  -t vllm-bench-runner:offline \
  vllm_standalone_bench
```

预期：镜像构建成功，日志包含 `Successfully tagged vllm-bench-runner:offline` 或 BuildKit 的成功输出。

- [ ] **步骤 6：确认 vLLM 镜像和模型完整性**

运行：

```bash
docker image inspect 009e4cb46541 --format '{{.Id}} {{.RepoTags}}'
python - <<'PY'
from pathlib import Path
model = Path("/Resource_Planning_Tool/model/Qwen2.5-1.5B-Instruct")
print("model_dir", model)
print("safetensors", [p.name for p in model.glob("*.safetensors")])
PY
```

预期：Docker image inspect 成功；模型目录输出至少一个完整 `*.safetensors` 文件。

- [ ] **步骤 7：如模型不完整，在联网环境准备模型**

运行：

```bash
python3 vllm_standalone_bench/auto_bench.py prepare-model \
  --modelscope-id Qwen/Qwen2.5-1.5B-Instruct \
  --target /Resource_Planning_Tool/model/Qwen2.5-1.5B-Instruct \
  --bench-image vllm-bench-runner:offline \
  --force
```

预期：命令成功，目标目录包含 `config.json`、tokenizer 文件和至少一个完整 `*.safetensors` 文件。

- [ ] **步骤 8：运行真实 smoke**

运行：

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json
```

预期：命令返回 0；对应 `vllm_standalone_bench/results/<run_id>/qwen2_5_1_5b/bf16_default/smoke/` 下生成 `result.csv`、`result.xlsx`、`bench.log`、`status.json`；serve profile 目录生成 `vllm.log`、`docker.inspect.json`、`serve_command.txt`；run 目录生成 `manifest.json` 和 `state.json`。

- [ ] **步骤 9：验证容器和网络清理**

运行：

```bash
docker ps --filter 'name=bench-vllm-' --format '{{.Names}}'
docker network inspect vllm-bench-net --format '{{json .Containers}}'
```

预期：没有当前 run 的 `bench-vllm-qwen2_5_1_5b-bf16_default-<run_id>` 容器仍在运行；如果 `vllm-bench-net` 是本次创建的网络，它已被删除，`docker network inspect` 返回非 0；如果运行前网络已存在，它未被删除。

- [ ] **步骤 10：最终提交状态检查**

运行：

```bash
git status --short
git log --oneline -5
```

预期：只有用户已有的未跟踪模型或结果文件可能存在；实现文件均已提交并合并到 `main`。
