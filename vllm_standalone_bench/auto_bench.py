#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MODEL_CONTAINER_ROOT = PurePosixPath("/models")
SUPPORTED_BACKENDS = frozenset({"openai", "openai-chat"})
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
NETWORK_MANAGED_LABEL = "vllm_auto_bench.managed"
NETWORK_RUN_ID_LABEL = "vllm_auto_bench.run_id"
CONTAINER_RUN_DIR_LABEL = "vllm_auto_bench.run_dir"
CONTAINER_MODEL_LABEL = "vllm_auto_bench.model"
CONTAINER_SERVE_PROFILE_LABEL = "vllm_auto_bench.serve_profile"
CONTAINER_BENCH_PROFILE_LABEL = "vllm_auto_bench.bench_profile"


class ConfigError(ValueError):
    """Raised when the auto bench configuration is invalid."""


class StopRequested(Exception):
    """Raised when the detached controller is asked to stop gracefully."""


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


@dataclass(frozen=True)
class CaseLayout:
    run_dir: Path
    serve_dir: Path
    bench_dir: Path


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
        result = Completed(
            args=args,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"command failed ({result.returncode}): {' '.join(args)}\n{result.stderr}"
            )
        return result


def _require_mapping(data: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{field_name} must be an object")
    return data


def _required(data: dict[str, Any], key: str, field_name: str) -> Any:
    if key not in data:
        raise ConfigError(f"{field_name} is required")
    return data[key]


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name)


def _safe_name(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not SAFE_NAME_RE.fullmatch(value)
        or value in {".", ".."}
    ):
        raise ConfigError(f"{field_name} must be a safe filename: {value!r}")
    return value


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a boolean")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ConfigError(f"{field_name} must be a non-negative integer")
    return value


def _finite_float(value: Any, field_name: str) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field_name} must be a finite number")
    return result


def _non_negative_float(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result < 0:
        raise ConfigError(f"{field_name} must be a non-negative number")
    return result


def _optional_non_negative_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _non_negative_float(value, field_name)


def _ratio(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if not 0 <= result <= 1:
        raise ConfigError(f"{field_name} must be a ratio between 0 and 1")
    return result


def _optional_ratio(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _ratio(value, field_name)


def _backend(value: Any, field_name: str) -> str:
    backend = _string(value, field_name)
    if backend not in SUPPORTED_BACKENDS:
        raise ConfigError(
            f"{field_name} must be one of: {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )
    return backend


def _network_name(value: Any, field_name: str) -> str:
    network = _string(value, field_name)
    if network in {"host", "none"}:
        raise ConfigError(f"{field_name} must be a Docker bridge network, not {network!r}")
    return network


def _positive_int_list(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field_name} must be a non-empty list")
    result: list[int] = []
    for item in value:
        if type(item) is not int or item <= 0:
            raise ConfigError(f"{field_name} must contain positive integers")
        result.append(item)
    return tuple(result)


def _container_path_to_host(path_value: Any, model_root: Path, field_name: str) -> Path:
    path_text = _string(path_value, field_name)
    container_path = PurePosixPath(path_text)
    if not container_path.is_absolute():
        raise ConfigError(f"{field_name} must be absolute inside the container: {path_text}")
    if ".." in container_path.parts:
        raise ConfigError(f"{field_name} must stay under /models: {path_text}")
    try:
        relative = container_path.relative_to(MODEL_CONTAINER_ROOT)
    except ValueError as exc:
        raise ConfigError(f"{field_name} must be under /models: {path_text}") from exc
    return model_root / relative


def _parse_run(data: dict[str, Any]) -> RunConfig:
    run = _require_mapping(data.get("run"), "run")
    host_port = run.get("host_port")
    return RunConfig(
        name=_safe_name(_required(run, "name", "run.name"), "run.name"),
        results_dir=Path(_string(run.get("results_dir", "vllm_standalone_bench/results"),
                                 "run.results_dir")),
        vllm_image=_string(_required(run, "vllm_image", "run.vllm_image"), "run.vllm_image"),
        bench_image=_string(_required(run, "bench_image", "run.bench_image"), "run.bench_image"),
        network=_network_name(run.get("network", "vllm-bench-net"), "run.network"),
        create_network=_bool(run.get("create_network", True), "run.create_network"),
        cleanup_network=_bool(run.get("cleanup_network", True), "run.cleanup_network"),
        container_port=_positive_int(run.get("container_port", 8000), "run.container_port"),
        publish_host_port=_bool(run.get("publish_host_port", False), "run.publish_host_port"),
        host_port=(
            _positive_int(host_port, "run.host_port")
            if host_port is not None else None
        ),
        api_key=_optional_string(run.get("api_key"), "run.api_key"),
        ready_timeout_sec=_positive_int(run.get("ready_timeout_sec", 1800),
                                        "run.ready_timeout_sec"),
        cooldown_sec=_non_negative_float(run.get("cooldown_sec", 20.0), "run.cooldown_sec"),
    )


def _parse_mounts(data: dict[str, Any], config_dir: Path) -> MountConfig:
    mounts = _require_mapping(data.get("mounts"), "mounts")
    models = Path(_string(_required(mounts, "models", "mounts.models"), "mounts.models"))
    if not models.is_absolute():
        models = config_dir / models
    return MountConfig(
        models=models.resolve()
    )


def _parse_models(data: dict[str, Any], mounts: MountConfig) -> tuple[ModelConfig, ...]:
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError("models must be a non-empty list")

    parsed: list[ModelConfig] = []
    for item in raw_models:
        model = _require_mapping(item, "models[]")
        name = _safe_name(_required(model, "name", "model.name"), "model.name")
        model_path = _string(_required(model, "model_path", "model.model_path"),
                             "model.model_path")
        tokenizer_path = _optional_string(model.get("tokenizer_path"), "model.tokenizer_path")
        served_model_name = _optional_string(model.get("served_model_name"),
                                             "model.served_model_name")
        parsed.append(ModelConfig(
            name=name,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            served_model_name=served_model_name,
            host_model_path=_container_path_to_host(model_path, mounts.models,
                                                    "model.model_path"),
            host_tokenizer_path=(
                _container_path_to_host(tokenizer_path, mounts.models,
                                        "model.tokenizer_path")
                if tokenizer_path is not None else None
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
            name=_safe_name(_required(profile, "name", "serve_profile.name"),
                            "serve_profile.name"),
            gpus=_string(profile.get("gpus", "all"), "serve_profile.gpus"),
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
        parallel_nums = _positive_int_list(profile.get("parallel_nums", [1, 4, 8]),
                                           "parallel_nums")
        cross_product = _bool(profile.get("cross_product", False), "cross_product")
        if not cross_product and len(output_lens) not in (1, len(input_lens)):
            raise ConfigError(
                "output_lens length must be 1 or match input_lens unless cross_product=true"
            )
        parsed.append(BenchProfile(
            name=_safe_name(_required(profile, "name", "bench_profile.name"),
                            "bench_profile.name"),
            backend=_backend(profile.get("backend", "openai-chat"), "bench_profile.backend"),
            input_lens=input_lens,
            output_lens=output_lens,
            parallel_nums=parallel_nums,
            epochs=_positive_int(profile.get("epochs", 3), "bench_profile.epochs"),
            prefix_ratio=_ratio(profile.get("prefix_ratio", 0.0),
                                "bench_profile.prefix_ratio"),
            warmup_requests=_non_negative_int(profile.get("warmup_requests", 1),
                                              "bench_profile.warmup_requests"),
            cross_product=cross_product,
            max_ttft_ms=_optional_non_negative_float(profile.get("max_ttft_ms"),
                                                     "bench_profile.max_ttft_ms"),
            min_throughput_tok_s=_optional_non_negative_float(
                profile.get("min_throughput_tok_s"),
                "bench_profile.min_throughput_tok_s",
            ),
            min_output_compliance=_optional_ratio(
                profile.get("min_output_compliance"),
                "bench_profile.min_output_compliance",
            ),
        ))
    return tuple(parsed)


def load_config(path: str | Path) -> AutoBenchConfig:
    config_path = Path(path).resolve()
    try:
        with config_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON config: {exc}") from exc

    config_data = _require_mapping(raw, "config")
    run = _parse_run(config_data)
    mounts = _parse_mounts(config_data, config_path.parent)
    models = _parse_models(config_data, mounts)
    serve_profiles = _parse_serve_profiles(config_data)
    bench_profiles = _parse_bench_profiles(config_data)
    return AutoBenchConfig(run, mounts, models, serve_profiles, bench_profiles)


def make_run_id(run_name: str, now: float | None = None) -> str:
    safe_run_name = _safe_name(run_name, "run.name")
    resolved_now = time.time() if now is None else now
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(resolved_now))
    return f"{safe_run_name}_{stamp}_{secrets.token_hex(3)}"


def make_container_name(model: ModelConfig, serve_profile: ServeProfile, run_id: str) -> str:
    safe_run_id = _safe_name(run_id, "run_id")
    return f"bench-vllm-{model.name}-{serve_profile.name}-{safe_run_id}"


def make_bench_container_name(case: BenchmarkCase) -> str:
    return (
        f"bench-runner-{case.model.name}-{case.serve_profile.name}-"
        f"{case.bench_profile.name}-{_safe_name(case.run_id, 'run_id')}"
    )


def expand_cases(config: AutoBenchConfig, run_id: str | None = None) -> tuple[BenchmarkCase, ...]:
    resolved_run_id = run_id or make_run_id(config.run.name)
    _safe_name(resolved_run_id, "run_id")

    cases: list[BenchmarkCase] = []
    for model in config.models:
        for serve_profile in config.serve_profiles:
            for bench_profile in config.bench_profiles:
                cases.append(BenchmarkCase(
                    model=model,
                    serve_profile=serve_profile,
                    bench_profile=bench_profile,
                    run_id=resolved_run_id,
                    container_name=make_container_name(model, serve_profile, resolved_run_id),
                    api_model_name=model.served_model_name or model.name,
                ))
    return tuple(cases)


def build_vllm_run_command(config: AutoBenchConfig, case: BenchmarkCase,
                           run_dir: Path) -> list[str]:
    resolved_run_dir = Path(run_dir).resolve()
    cmd = [
        "docker", "run", "-d",
        "--name", case.container_name,
        "--label", f"{NETWORK_MANAGED_LABEL}=true",
        "--label", f"{NETWORK_RUN_ID_LABEL}={case.run_id}",
        "--label", f"{CONTAINER_RUN_DIR_LABEL}={resolved_run_dir}",
        "--label", f"{CONTAINER_MODEL_LABEL}={case.model.name}",
        "--label", f"{CONTAINER_SERVE_PROFILE_LABEL}={case.serve_profile.name}",
        "--gpus", case.serve_profile.gpus,
        "--network", config.run.network,
        "-v", f"{config.mounts.models}:/models:ro",
        "--entrypoint", "vllm",
    ]
    if config.run.publish_host_port:
        if config.run.host_port is None:
            raise ConfigError("host_port is required when publish_host_port=true")
        cmd.extend([
            "-p",
            f"127.0.0.1:{config.run.host_port}:{config.run.container_port}",
        ])
    cmd.extend([
        config.run.vllm_image,
        "serve", case.model.model_path,
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


def build_bench_run_command(config: AutoBenchConfig, case: BenchmarkCase,
                            bench_dir: Path) -> list[str]:
    bench = case.bench_profile
    resolved_bench_dir = Path(bench_dir).resolve()
    resolved_run_dir = resolved_bench_dir.parents[2]
    cmd = [
        "docker", "run", "--rm",
        "--name", make_bench_container_name(case),
        "--label", f"{NETWORK_MANAGED_LABEL}=true",
        "--label", f"{NETWORK_RUN_ID_LABEL}={case.run_id}",
        "--label", f"{CONTAINER_RUN_DIR_LABEL}={resolved_run_dir}",
        "--label", f"{CONTAINER_MODEL_LABEL}={case.model.name}",
        "--label", f"{CONTAINER_SERVE_PROFILE_LABEL}={case.serve_profile.name}",
        "--label", f"{CONTAINER_BENCH_PROFILE_LABEL}={case.bench_profile.name}",
        "--network", config.run.network,
        "-v", f"{config.mounts.models}:/models:ro",
        "-v", f"{resolved_bench_dir}:/results",
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
        if model.host_tokenizer_path is not None and not model.host_tokenizer_path.exists():
            raise ConfigError(f"tokenizer path does not exist: {model.host_tokenizer_path}")


def build_layout(config: AutoBenchConfig, run_id: str, case: BenchmarkCase) -> CaseLayout:
    run_dir = config.run.results_dir / run_id
    serve_dir = run_dir / case.model.name / case.serve_profile.name
    bench_dir = serve_dir / case.bench_profile.name
    return CaseLayout(run_dir=run_dir, serve_dir=serve_dir, bench_dir=bench_dir)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def write_state(run_dir: Path, state: dict[str, Any]) -> None:
    write_json_atomic(run_dir / "state.json", state)


@dataclass
class Manifest:
    run_id: str
    total: int
    cases: list[dict[str, Any]] = field(default_factory=list)
    terminal_status: str | None = None

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
        if self.terminal_status:
            return self.terminal_status
        statuses = {case["status"] for case in self.cases}
        if "interrupted" in statuses:
            return "interrupted"
        if len(self.cases) < self.total:
            return "running"
        if statuses == {"passed"}:
            return "completed"
        if "failed" in statuses or "skipped" in statuses:
            return "completed_with_failures"
        return "completed"

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "status": self.status(), "cases": self.cases}


def write_manifest(run_dir: Path, manifest: Manifest) -> None:
    write_json_atomic(run_dir / "manifest.json", manifest.to_dict())


class RunnerProtocol(Protocol):
    def run(self, args: list[str], *, check: bool = False,
            capture: bool = True, text: bool = True,
            stdout: Any = None, stderr: Any = None) -> Completed:
        ...


FakeRunnerProtocol = RunnerProtocol
Runner = DockerRunner | RunnerProtocol


def _check_result(result: Completed) -> Completed:
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(result.args)}\n{result.stderr}"
        )
    return result


def docker_network_exists(runner: Runner, network: str) -> bool:
    result = runner.run(["docker", "network", "inspect", network], check=False)
    return result.returncode == 0


def docker_network_driver(runner: Runner, network: str) -> str | None:
    result = runner.run([
        "docker", "network", "inspect", "--format", "{{.Driver}}", network,
    ], check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def validate_bridge_network_driver(runner: Runner, network: str) -> None:
    driver = docker_network_driver(runner, network)
    if driver != "bridge":
        raise RuntimeError(f"Docker network must use bridge driver: {network} ({driver or 'unknown'})")


def build_network_create_command(config: AutoBenchConfig, run_id: str) -> list[str]:
    return [
        "docker", "network", "create",
        "--label", f"{NETWORK_MANAGED_LABEL}=true",
        "--label", f"{NETWORK_RUN_ID_LABEL}={run_id}",
        config.run.network,
    ]


def ensure_network(config: AutoBenchConfig, runner: Runner,
                   dry_run: bool, run_id: str | None = None) -> bool:
    if dry_run:
        if config.run.create_network:
            print_cmd(build_network_create_command(config, run_id or config.run.name))
            return True
        return False
    if docker_network_exists(runner, config.run.network):
        validate_bridge_network_driver(runner, config.run.network)
        return False
    if not config.run.create_network:
        raise RuntimeError(f"Docker network does not exist: {config.run.network}")
    cmd = build_network_create_command(config, run_id or config.run.name)
    if dry_run:
        print_cmd(cmd)
    else:
        _check_result(runner.run(cmd, check=False))
    return True


def connected_network_containers(runner: Runner, network: str) -> list[str]:
    result = runner.run([
        "docker", "inspect", "--format",
        "{{json .Containers}}", network,
    ], check=False)
    payload = result.stdout.strip()
    if result.returncode != 0 or payload in ("", "null", "{}"):
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ["unknown"]
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        return [str(key) for key in data.keys()]
    return ["unknown"]


def network_has_run_labels(runner: Runner, network: str, run_id: str) -> bool:
    result = runner.run([
        "docker", "network", "inspect", "--format",
        "{{json .Labels}}", network,
    ], check=False)
    payload = result.stdout.strip()
    if result.returncode != 0 or payload in ("", "null"):
        return False
    try:
        labels = json.loads(payload)
    except json.JSONDecodeError:
        return False
    if not isinstance(labels, dict):
        return False
    return (
        labels.get(NETWORK_MANAGED_LABEL) == "true"
        and labels.get(NETWORK_RUN_ID_LABEL) == run_id
    )


def inspect_container_labels(runner: Runner, container_name: str) -> dict[str, str] | None:
    result = runner.run([
        "docker", "inspect", "--format",
        "{{json .Config.Labels}}", container_name,
    ], check=False)
    if result.returncode != 0:
        return None
    try:
        labels = json.loads(result.stdout.strip() or "null")
    except json.JSONDecodeError:
        return {}
    if not isinstance(labels, dict):
        return {}
    return {str(key): str(value) for key, value in labels.items()}


def vllm_container_labels_match(labels: dict[str, str],
                                case: BenchmarkCase,
                                run_dir: Path) -> bool:
    return (
        labels.get(NETWORK_MANAGED_LABEL) == "true"
        and labels.get(NETWORK_RUN_ID_LABEL) == case.run_id
        and labels.get(CONTAINER_RUN_DIR_LABEL) == str(Path(run_dir).resolve())
    )


def bench_container_labels_match(labels: dict[str, str],
                                 case: BenchmarkCase,
                                 run_dir: Path) -> bool:
    return (
        vllm_container_labels_match(labels, case, run_dir)
        and labels.get(CONTAINER_MODEL_LABEL) == case.model.name
        and labels.get(CONTAINER_SERVE_PROFILE_LABEL) == case.serve_profile.name
        and labels.get(CONTAINER_BENCH_PROFILE_LABEL) == case.bench_profile.name
    )


def remove_existing_vllm_container_if_owned(runner: Runner, case: BenchmarkCase,
                                            run_dir: Path) -> None:
    labels = inspect_container_labels(runner, case.container_name)
    if labels is None:
        return
    if not vllm_container_labels_match(labels, case, run_dir):
        raise RuntimeError(
            f"vLLM container exists but is not owned by this run: {case.container_name}"
        )
    runner.run(["docker", "rm", "-f", case.container_name], check=False)


def stop_and_remove_vllm_container_if_owned(runner: Runner, case: BenchmarkCase,
                                           run_dir: Path, dry_run: bool) -> None:
    if dry_run:
        stop_and_remove_container(runner, case.container_name, dry_run=True)
        return
    labels = inspect_container_labels(runner, case.container_name)
    if labels is None:
        return
    if not vllm_container_labels_match(labels, case, run_dir):
        return
    stop_and_remove_container(runner, case.container_name, dry_run=False)


def remove_existing_bench_container_if_owned(runner: Runner, case: BenchmarkCase,
                                             run_dir: Path) -> bool:
    container_name = make_bench_container_name(case)
    labels = inspect_container_labels(runner, container_name)
    if labels is None:
        return True
    if not bench_container_labels_match(labels, case, run_dir):
        return False
    stop_and_remove_container(runner, container_name, dry_run=False)
    return True


def cleanup_bench_container_if_owned(runner: Runner, case: BenchmarkCase,
                                     run_dir: Path) -> None:
    container_name = make_bench_container_name(case)
    labels = inspect_container_labels(runner, container_name)
    if labels is None:
        return
    if not bench_container_labels_match(labels, case, run_dir):
        return
    stop_and_remove_container(runner, container_name, dry_run=False)


def cleanup_network(config: AutoBenchConfig, runner: Runner,
                    owned: bool, dry_run: bool,
                    run_id: str | None = None) -> bool:
    if not owned or not config.run.cleanup_network:
        return False
    stop_requested = False
    if not dry_run:
        if run_id is None:
            print("warning: network cleanup skipped because run_id is unknown", file=sys.stderr)
            return False
        try:
            if not network_has_run_labels(runner, config.run.network, run_id):
                print(
                    f"warning: network cleanup skipped because labels do not match: {config.run.network}",
                    file=sys.stderr,
                )
                return False
        except StopRequested:
            return True
        except Exception as exc:
            print(
                f"warning: network cleanup skipped after label inspect failed: {exc}",
                file=sys.stderr,
            )
            return False
    try:
        connected = [] if dry_run else connected_network_containers(runner, config.run.network)
    except StopRequested:
        return True
    except Exception as exc:
        print(f"warning: network cleanup skipped after inspect failed: {exc}", file=sys.stderr)
        return False
    if connected:
        print(
            f"warning: network cleanup skipped because containers are still connected: {connected}",
            file=sys.stderr,
        )
        return False
    if should_cleanup_network(
        owned=owned,
        cleanup_enabled=config.run.cleanup_network,
        connected_containers=connected,
    ):
        cmd = ["docker", "network", "rm", config.run.network]
        if dry_run:
            print_cmd(cmd)
        else:
            try:
                result = runner.run(cmd, check=False)
                if result.returncode != 0:
                    print(
                        f"warning: network cleanup failed ({result.returncode}): {result.stderr}",
                        file=sys.stderr,
                    )
            except StopRequested:
                stop_requested = True
            except Exception as exc:
                print(f"warning: network cleanup failed: {exc}", file=sys.stderr)
    return stop_requested


def print_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))


def install_signal_handlers() -> None:
    def request_stop(signum: int, frame: Any) -> None:
        _ = frame
        raise StopRequested(f"received signal {signum}")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def wait_for_ready(base_url: str, api_key: str | None, timeout_sec: int) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_sec
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    while time.time() < deadline:
        request = urllib.request.Request(f"{base_url}/models", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return True
        except (OSError, TimeoutError, urllib.error.URLError):
            time.sleep(2)
    return False


READY_PROBE_SCRIPT = """import sys
import time
import urllib.error
import urllib.request

url = sys.argv[1]
api_key = sys.argv[2]
timeout = float(sys.argv[3])
deadline = time.time() + timeout
headers = {}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
while time.time() < deadline:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status == 200:
                sys.exit(0)
    except (OSError, TimeoutError, urllib.error.URLError):
        time.sleep(2)
sys.exit(1)
"""


def make_ready_probe_container_name(case: BenchmarkCase) -> str:
    return f"bench-ready-{case.model.name}-{case.serve_profile.name}-{case.run_id}"


def build_ready_probe_run_command(config: AutoBenchConfig, case: BenchmarkCase,
                                  run_dir: Path) -> list[str]:
    resolved_run_dir = Path(run_dir).resolve()
    url = f"http://{case.container_name}:{config.run.container_port}/v1/models"
    return [
        "docker", "run", "--rm",
        "--name", make_ready_probe_container_name(case),
        "--label", f"{NETWORK_MANAGED_LABEL}=true",
        "--label", f"{NETWORK_RUN_ID_LABEL}={case.run_id}",
        "--label", f"{CONTAINER_RUN_DIR_LABEL}={resolved_run_dir}",
        "--network", config.run.network,
        config.run.bench_image,
        "python", "-c", READY_PROBE_SCRIPT,
        url,
        config.run.api_key or "",
        str(config.run.ready_timeout_sec),
    ]


def cleanup_ready_probe_container_if_owned(runner: Runner, case: BenchmarkCase,
                                           run_dir: Path) -> None:
    probe_name = make_ready_probe_container_name(case)
    labels = inspect_container_labels(runner, probe_name)
    if labels is None:
        return
    if not vllm_container_labels_match(labels, case, run_dir):
        return
    stop_and_remove_container(runner, probe_name, dry_run=False)


def remove_existing_ready_probe_container_if_owned(runner: Runner, case: BenchmarkCase,
                                                   run_dir: Path) -> bool:
    probe_name = make_ready_probe_container_name(case)
    labels = inspect_container_labels(runner, probe_name)
    if labels is None:
        return True
    if not vllm_container_labels_match(labels, case, run_dir):
        return False
    stop_and_remove_container(runner, probe_name, dry_run=False)
    return True


def wait_for_container_ready(config: AutoBenchConfig, case: BenchmarkCase,
                             runner: Runner) -> bool:
    run_dir = build_layout(config, case.run_id, case).run_dir
    interrupted: BaseException | None = None
    try:
        if not remove_existing_ready_probe_container_if_owned(runner, case, run_dir):
            return False
        result = runner.run(
            build_ready_probe_run_command(config, case, run_dir),
            check=False,
        )
        return result.returncode == 0
    except (StopRequested, KeyboardInterrupt) as exc:
        interrupted = exc
        raise
    finally:
        try:
            cleanup_ready_probe_container_if_owned(runner, case, run_dir)
        except (StopRequested, KeyboardInterrupt):
            if interrupted is None:
                raise
        except Exception:
            pass


def save_vllm_artifacts(config: AutoBenchConfig, runner: Runner,
                        case: BenchmarkCase, layout: CaseLayout) -> None:
    layout.serve_dir.mkdir(parents=True, exist_ok=True)
    logs = runner.run(["docker", "logs", "--timestamps", case.container_name], check=False)
    (layout.serve_dir / "vllm.log").write_text(
        logs.stdout + logs.stderr,
        encoding="utf-8",
    )
    inspect = runner.run(["docker", "inspect", case.container_name], check=False)
    (layout.serve_dir / "docker.inspect.json").write_text(inspect.stdout, encoding="utf-8")
    (layout.serve_dir / "serve_command.txt").write_text(
        " ".join(build_vllm_run_command(config, case, layout.run_dir)),
        encoding="utf-8",
    )


def save_vllm_artifacts_best_effort(config: AutoBenchConfig, runner: Runner,
                                    case: BenchmarkCase, layout: CaseLayout) -> None:
    try:
        save_vllm_artifacts(config, runner, case, layout)
    except StopRequested:
        raise
    except Exception as exc:
        try:
            layout.serve_dir.mkdir(parents=True, exist_ok=True)
            (layout.serve_dir / "artifact.warning.txt").write_text(
                f"failed to save vLLM artifacts: {exc}",
                encoding="utf-8",
            )
        except StopRequested:
            raise
        except Exception:
            pass


def stop_and_remove_container(runner: Runner, container_name: str, dry_run: bool) -> None:
    commands = [
        ["docker", "stop", container_name],
        ["docker", "rm", "-f", container_name],
    ]
    stop_requested: StopRequested | None = None
    for command in commands:
        try:
            if dry_run:
                print_cmd(command)
            else:
                runner.run(command, check=False)
        except StopRequested as exc:
            stop_requested = exc
        except Exception:
            pass
    if stop_requested is not None:
        raise stop_requested


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


RUN_LOCK_FILE = ".run.lock"
TERMINAL_RUN_STATUSES = frozenset({
    "completed",
    "completed_with_failures",
    "failed",
    "interrupted",
})


@dataclass(frozen=True)
class RunLock:
    run_dir: Path
    token: str


def run_lock_path(run_dir: Path) -> Path:
    return run_dir / RUN_LOCK_FILE


def _run_lock_payload(token: str, pid: int | None = None) -> dict[str, Any]:
    return {
        "pid": os.getpid() if pid is None else pid,
        "token": token,
        "created_at": time.time(),
    }


def _read_run_lock(run_dir: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(run_lock_path(run_dir).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_run_state(run_dir: Path) -> dict[str, Any] | None:
    try:
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def run_lock_token_matches(run_dir: Path, token: str | None) -> bool:
    if token is None:
        return False
    payload = _read_run_lock(run_dir)
    return payload is not None and payload.get("token") == token


def acquire_run_lock(run_dir: Path, token: str | None = None) -> RunLock:
    run_dir.mkdir(parents=True, exist_ok=True)
    if token is not None:
        if not run_lock_token_matches(run_dir, token):
            raise RuntimeError(f"run already active: {run_dir}")
        write_json_atomic(run_lock_path(run_dir), _run_lock_payload(token))
        return RunLock(run_dir=run_dir, token=token)

    new_token = secrets.token_hex(16)
    lock_path = run_lock_path(run_dir)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"run already active: {run_dir}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(_run_lock_payload(new_token), handle)
    return RunLock(run_dir=run_dir, token=new_token)


def update_run_lock_owner(lock: RunLock, pid: int) -> None:
    if run_lock_token_matches(lock.run_dir, lock.token):
        write_json_atomic(run_lock_path(lock.run_dir), _run_lock_payload(lock.token, pid=pid))


def release_run_lock(lock: RunLock) -> None:
    lock_path = run_lock_path(lock.run_dir)
    if not run_lock_token_matches(lock.run_dir, lock.token):
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def release_run_lock_for_token(run_dir: Path | None, token: str | None) -> None:
    if run_dir is None or token is None:
        return
    release_run_lock(RunLock(run_dir=run_dir, token=token))


def cleanup_stale_terminal_run_lock(run_dir: Path, lock_token: str | None = None) -> bool:
    if run_lock_token_matches(run_dir, lock_token):
        return False
    state = _read_run_state(run_dir)
    if state is None or state.get("status") not in TERMINAL_RUN_STATUSES:
        return False
    payload = _read_run_lock(run_dir)
    if payload is None:
        return False
    pid = payload.get("pid")
    if type(pid) is not int or pid <= 1:
        return False
    if is_process_running(pid):
        return False
    try:
        run_lock_path(run_dir).unlink()
    except OSError:
        return False
    print(f"cleaned stale run lock: {run_dir}", file=sys.stderr)
    return True


def active_state_blocks_start(run_dir: Path, *, allow_pid: int | None = None,
                              lock_token: str | None = None) -> bool:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(state, dict) or state.get("status") not in ("starting", "running"):
        return False
    if run_lock_token_matches(run_dir, lock_token):
        return False
    pid_path = run_dir / "controller.pid"
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeDecodeError, ValueError):
        print(
            f"run is already active but controller pid is unavailable: {run_dir}",
            file=sys.stderr,
        )
        return True
    if allow_pid is not None and pid == allow_pid:
        return False
    if pid <= 1:
        print(f"run is already active with invalid controller pid: {run_dir}", file=sys.stderr)
        return True
    if is_process_running(pid):
        print(f"run is already active: {run_dir} (pid {pid})", file=sys.stderr)
        return True
    return False


def active_run_pid(run_dir: Path) -> int | None:
    state_path = run_dir / "state.json"
    pid_path = run_dir / "controller.pid"
    if not state_path.exists() or not pid_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(state, dict):
        return None
    if state.get("status") not in ("starting", "running"):
        return None
    if pid <= 1 or not is_process_running(pid):
        return None
    return pid


def reject_active_run(run_dir: Path, *, allow_pid: int | None = None,
                      lock_token: str | None = None) -> bool:
    if active_state_blocks_start(run_dir, allow_pid=allow_pid, lock_token=lock_token):
        return True
    if run_lock_path(run_dir).exists() and not run_lock_token_matches(run_dir, lock_token):
        if cleanup_stale_terminal_run_lock(run_dir, lock_token=lock_token):
            return False
        print(f"run is already active: {run_dir}", file=sys.stderr)
        return True
    pid = active_run_pid(run_dir)
    if pid is None or pid == allow_pid:
        return False
    print(f"run is already active: {run_dir} (pid {pid})", file=sys.stderr)
    return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _jsonable(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    return value


def config_to_dict(config: AutoBenchConfig) -> dict[str, Any]:
    return _jsonable(config)


def _case_ref(case: BenchmarkCase) -> dict[str, str]:
    return {
        "model": case.model.name,
        "serve_profile": case.serve_profile.name,
        "bench_profile": case.bench_profile.name,
    }


def _case_status_counts(rows: list[dict[str, Any]], total: int) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "running": 0, "total": total}
    for row in rows:
        status = row.get("status")
        if status in ("passed", "failed", "skipped"):
            counts[status] += 1
    return counts


def current_state(run_id: str, cases: tuple[BenchmarkCase, ...], completed: int,
                  case: BenchmarkCase, status: str,
                  manifest: Manifest | None = None) -> dict[str, Any]:
    counts = (
        _case_status_counts(manifest.cases, len(cases))
        if manifest is not None else
        {"passed": 0, "failed": 0, "skipped": 0, "running": 0, "total": len(cases)}
    )
    counts["running"] = 1 if status == "running" else 0
    counts["completed"] = (
        counts["passed"] + counts["failed"] + counts["skipped"]
        if manifest is not None else
        completed
    )
    return {
        "run_id": run_id,
        "status": status,
        "current": _case_ref(case),
        "counts": counts,
    }


def finished_state(run_id: str, manifest: Manifest) -> dict[str, Any]:
    counts = _case_status_counts(manifest.cases, manifest.total)
    if manifest.status() == "running":
        counts["running"] = max(manifest.total - len(manifest.cases), 0)
    return {
        "run_id": run_id,
        "status": manifest.status(),
        "current": None,
        "counts": counts,
    }


def write_terminal_state(run_dir: Path, run_id: str, manifest: Manifest,
                         status: str, error: str | None = None) -> None:
    manifest.terminal_status = status
    write_manifest(run_dir, manifest)
    state = finished_state(run_id, manifest)
    if error:
        state["error"] = error
    write_state(run_dir, state)


def _case_key(case: BenchmarkCase) -> tuple[str, str, str]:
    return (case.model.name, case.serve_profile.name, case.bench_profile.name)


def _manifest_case_keys(manifest: Manifest) -> set[tuple[str, str, str]]:
    return {
        (str(row["model"]), str(row["serve_profile"]), str(row["bench_profile"]))
        for row in manifest.cases
    }


def _record_group_status(manifest: Manifest, run_dir: Path, run_id: str,
                         group_cases: list[BenchmarkCase], config: AutoBenchConfig,
                         status: str, error: str) -> int:
    recorded = _manifest_case_keys(manifest)
    added = 0
    for case in group_cases:
        if _case_key(case) in recorded:
            continue
        layout = build_layout(config, run_id, case)
        layout.bench_dir.mkdir(parents=True, exist_ok=True)
        manifest.record(case, layout, status, error=error)
        recorded.add(_case_key(case))
        added += 1
        write_json_atomic(layout.bench_dir / "status.json", {
            "status": status,
            "error": error,
        })
    write_manifest(run_dir, manifest)
    return added


def _record_skipped_group(manifest: Manifest, run_dir: Path, run_id: str,
                          group_cases: list[BenchmarkCase], config: AutoBenchConfig,
                          error: str) -> int:
    return _record_group_status(
        manifest,
        run_dir,
        run_id,
        group_cases,
        config,
        "skipped",
        error,
    )


def _record_interrupted_group(manifest: Manifest, run_dir: Path, run_id: str,
                              group_cases: list[BenchmarkCase],
                              config: AutoBenchConfig, error: str) -> int:
    return _record_group_status(
        manifest,
        run_dir,
        run_id,
        group_cases,
        config,
        "interrupted",
        error,
    )


def _group_cases_by_serve(cases: tuple[BenchmarkCase, ...]) -> dict[tuple[str, str], list[BenchmarkCase]]:
    grouped: dict[tuple[str, str], list[BenchmarkCase]] = {}
    for case in cases:
        grouped.setdefault((case.model.name, case.serve_profile.name), []).append(case)
    return grouped


def _run_controller_dry_run(config: AutoBenchConfig, run_id: str) -> int:
    cases = expand_cases(config, run_id=run_id)
    run_dir = config.run.results_dir / run_id
    network_owned = config.run.create_network
    write_json_atomic(run_dir / "config.resolved.json", config_to_dict(config))
    try:
        if config.run.create_network:
            print_cmd(build_network_create_command(config, run_id))
        for group_cases in _group_cases_by_serve(cases).values():
            serve_case = group_cases[0]
            serve_layout = build_layout(config, run_id, serve_case)
            print_cmd(build_vllm_run_command(config, serve_case, serve_layout.run_dir))
            for case in group_cases:
                layout = build_layout(config, run_id, case)
                print_cmd(build_bench_run_command(config, case, layout.bench_dir))
        return 0
    finally:
        if should_cleanup_network(
            owned=network_owned,
            cleanup_enabled=config.run.cleanup_network,
            connected_containers=[],
        ):
            print_cmd(["docker", "network", "rm", config.run.network])


def run_controller(config: AutoBenchConfig, run_id: str,
                   runner: Runner | None = None,
                   dry_run: bool = False,
                   lock_token: str | None = None) -> int:
    active_runner: Runner = runner or DockerRunner()
    if dry_run:
        return _run_controller_dry_run(config, run_id)

    cases = expand_cases(config, run_id=run_id)
    run_dir = config.run.results_dir / run_id
    if reject_active_run(run_dir, allow_pid=os.getpid(), lock_token=lock_token):
        return 1
    run_lock: RunLock | None = None
    try:
        run_lock = acquire_run_lock(run_dir, token=lock_token)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if reject_active_run(run_dir, allow_pid=os.getpid(), lock_token=run_lock.token):
        release_run_lock(run_lock)
        return 1
    manifest = Manifest(run_id=run_id, total=len(cases))

    network_owned = False
    exit_code = 0
    completed = 0
    interrupted = False
    try:
        write_json_atomic(run_dir / "config.resolved.json", config_to_dict(config))
        validate_local_paths(config)

        if docker_network_exists(active_runner, config.run.network):
            validate_bridge_network_driver(active_runner, config.run.network)
            network_owned = False
        else:
            if not config.run.create_network:
                raise RuntimeError(f"Docker network does not exist: {config.run.network}")
            _check_result(active_runner.run(
                build_network_create_command(config, run_id),
                check=False,
            ))
            network_owned = True
        grouped = _group_cases_by_serve(cases)

        for group_cases in grouped.values():
            serve_case = group_cases[0]
            serve_layout = build_layout(config, run_id, serve_case)
            vllm_cmd = build_vllm_run_command(config, serve_case, serve_layout.run_dir)
            started = False
            cleanup_container = False
            try:
                if dry_run:
                    print_cmd(vllm_cmd)
                    ready = True
                else:
                    cleanup_container = True
                    remove_existing_vllm_container_if_owned(
                        active_runner,
                        serve_case,
                        serve_layout.run_dir,
                    )
                    start_result = active_runner.run(vllm_cmd, check=False)
                    started = start_result.returncode == 0
                    if not started:
                        ready = False
                    elif config.run.publish_host_port:
                        ready = wait_for_ready(
                            f"http://127.0.0.1:{config.run.host_port}/v1",
                            config.run.api_key,
                            config.run.ready_timeout_sec,
                        )
                    else:
                        ready = wait_for_container_ready(config, serve_case, active_runner)

                if not ready:
                    exit_code = 1
                    error = (
                        "vLLM container failed to start"
                        if not started and not dry_run else
                        "vLLM ready check timed out"
                    )
                    completed += _record_skipped_group(
                        manifest,
                        run_dir,
                        run_id,
                        group_cases,
                        config,
                        error,
                    )
                    continue

                for case in group_cases:
                    layout = build_layout(config, run_id, case)
                    layout.bench_dir.mkdir(parents=True, exist_ok=True)
                    write_state(
                        run_dir,
                        current_state(
                            run_id,
                            cases,
                            completed,
                            case,
                            "running",
                            manifest=manifest,
                        ),
                    )
                    bench_cmd = build_bench_run_command(config, case, layout.bench_dir)
                    if dry_run:
                        print_cmd(bench_cmd)
                        status = "passed"
                        error = None
                    else:
                        if not remove_existing_bench_container_if_owned(
                            active_runner,
                            case,
                            layout.run_dir,
                        ):
                            raise RuntimeError(
                                "benchmark container exists but is not owned by this run: "
                                f"{make_bench_container_name(case)}"
                            )
                        bench_interrupted: BaseException | None = None
                        try:
                            with (layout.bench_dir / "bench.log").open(
                                "w",
                                encoding="utf-8",
                            ) as log:
                                result = active_runner.run(
                                    bench_cmd,
                                    check=False,
                                    capture=False,
                                    stdout=log,
                                    stderr=log,
                                )
                        except (StopRequested, KeyboardInterrupt) as exc:
                            bench_interrupted = exc
                            raise
                        finally:
                            try:
                                cleanup_bench_container_if_owned(
                                    active_runner,
                                    case,
                                    layout.run_dir,
                                )
                            except (StopRequested, KeyboardInterrupt):
                                if bench_interrupted is None:
                                    raise
                            except Exception:
                                pass
                        status = "passed" if result.returncode == 0 else "failed"
                        error = None if result.returncode == 0 else (
                            f"benchmark exited {result.returncode}"
                        )
                    if status != "passed":
                        exit_code = 1
                    manifest.record(case, layout, status, error=error)
                    write_json_atomic(layout.bench_dir / "status.json", {
                        "status": status,
                        "error": error,
                    })
                    completed += 1
                    write_manifest(run_dir, manifest)
            except StopRequested as exc:
                exit_code = 130
                interrupted = True
                manifest.terminal_status = "interrupted"
                completed += _record_interrupted_group(
                    manifest,
                    run_dir,
                    run_id,
                    group_cases,
                    config,
                    str(exc) or "stop requested",
                )
            except Exception as exc:
                exit_code = 1
                completed += _record_skipped_group(
                    manifest,
                    run_dir,
                    run_id,
                    group_cases,
                    config,
                    str(exc),
                )
            finally:
                if not dry_run and cleanup_container:
                    stop_requested: StopRequested | None = None
                    if started:
                        try:
                            save_vllm_artifacts_best_effort(
                                config,
                                active_runner,
                                serve_case,
                                serve_layout,
                            )
                        except StopRequested as exc:
                            stop_requested = exc
                    try:
                        stop_and_remove_vllm_container_if_owned(
                            active_runner,
                            serve_case,
                            serve_layout.run_dir,
                            dry_run,
                        )
                    except StopRequested as exc:
                        if stop_requested is None:
                            stop_requested = exc
                    if config.run.cooldown_sec > 0:
                        time.sleep(config.run.cooldown_sec)
                    if stop_requested is not None:
                        raise stop_requested
            if interrupted:
                break

        write_state(run_dir, finished_state(run_id, manifest))
    except StopRequested as exc:
        manifest.terminal_status = "interrupted"
        _record_interrupted_group(
            manifest,
            run_dir,
            run_id,
            list(cases),
            config,
            str(exc) or "stop requested",
        )
        write_terminal_state(
            run_dir,
            run_id,
            manifest,
            "interrupted",
            error=str(exc) or "stop requested",
        )
        exit_code = 130
    except Exception as exc:
        write_terminal_state(run_dir, run_id, manifest, "failed", error=str(exc))
        exit_code = 1
    finally:
        try:
            cleanup_interrupted = cleanup_network(
                config,
                active_runner,
                network_owned,
                False,
                run_id=run_id,
            )
            if cleanup_interrupted:
                write_terminal_state(
                    run_dir,
                    run_id,
                    manifest,
                    "interrupted",
                    error="stop requested during network cleanup",
                )
                exit_code = 130
        finally:
            if run_lock is not None:
                release_run_lock(run_lock)
    return exit_code


def build_detach_command(config_path: Path, run_id: str,
                         results_dir: Path,
                         lock_token: str | None = None) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--child",
        "--results-dir",
        str(results_dir),
    ]
    if lock_token is not None:
        cmd.extend(["--lock-token", lock_token])
    return cmd


def _detached_state(run_id: str, status: str, total: int,
                    error: str | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "current": None,
        "counts": {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "running": 0,
            "total": total,
        },
    }
    if error:
        state["error"] = error
    return state


def start_detached(config_path: Path, config: AutoBenchConfig, run_id: str) -> int:
    _safe_name(run_id, "run_id")
    cases = expand_cases(config, run_id=run_id)
    run_dir = config.run.results_dir / run_id
    if reject_active_run(run_dir):
        return 1
    run_lock: RunLock | None = None
    try:
        run_lock = acquire_run_lock(run_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if reject_active_run(run_dir, lock_token=run_lock.token):
        release_run_lock(run_lock)
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    total = len(cases)

    def fail_start(error: str) -> int:
        try:
            write_state(run_dir, _detached_state(run_id, "failed", total, error=error))
            print(f"failed to start detached controller: {error}", file=sys.stderr)
            return 1
        finally:
            if run_lock is not None:
                release_run_lock(run_lock)

    try:
        validate_local_paths(config)
    except (ConfigError, OSError) as exc:
        return fail_start(str(exc))

    try:
        write_state(run_dir, _detached_state(run_id, "starting", total))
    except OSError as exc:
        return fail_start(str(exc))

    log_path = run_dir / "controller.log"
    command = build_detach_command(
        config_path,
        run_id,
        config.run.results_dir,
        lock_token=run_lock.token,
    )
    process: subprocess.Popen[Any] | None = None
    try:
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        (run_dir / "controller.pid").write_text(f"{process.pid}\n", encoding="utf-8")
        write_json_atomic(run_dir / "controller.json", {
            "pid": process.pid,
            "run_id": run_id,
            "command": command,
            "config_path": str(config_path),
            "started_at": time.time(),
        })
        update_run_lock_owner(run_lock, process.pid)
    except OSError as exc:
        if process is not None:
            try:
                process.terminate()
            except Exception:
                pass
        return fail_start(str(exc))
    print(f"run_id: {run_id}")
    print(f"log: {log_path}")
    return 0


def _format_current(current: Any) -> str:
    if not isinstance(current, dict):
        return "-"
    return "/".join([
        str(current.get("model", "-")),
        str(current.get("serve_profile", "-")),
        str(current.get("bench_profile", "-")),
    ])


def _format_counts(counts: Any) -> str:
    if not isinstance(counts, dict):
        return "-"
    keys = ["passed", "failed", "skipped", "running", "completed", "total"]
    return " ".join(
        f"{key}={counts[key]}"
        for key in keys
        if key in counts
    )


def print_status(run_dir: Path) -> int:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        print(f"state file not found: {state_path}", file=sys.stderr)
        return 1
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"state file invalid: {exc}", file=sys.stderr)
        return 1
    if not isinstance(state, dict):
        print(f"state file invalid: expected object in {state_path}", file=sys.stderr)
        return 1
    print(f"run_id: {state.get('run_id', run_dir.name)}")
    print(f"status: {state.get('status', 'unknown')}")
    print(f"current: {_format_current(state.get('current'))}")
    print(f"counts: {_format_counts(state.get('counts'))}")
    exit_code = 0
    pid_path = run_dir / "controller.pid"
    if pid_path.exists():
        try:
            pid = pid_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"warning: pid file invalid: {exc}", file=sys.stderr)
            pid = "-"
            exit_code = 1
        if not pid:
            pid = "-"
    else:
        pid = "-"
    print(f"pid: {pid}")

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print("manifest: -")
        return exit_code
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"warning: manifest file invalid: {exc}", file=sys.stderr)
        return 1
    if not isinstance(manifest, dict):
        print(f"warning: manifest file invalid: expected object in {manifest_path}", file=sys.stderr)
        return 1
    cases = manifest.get("cases")
    case_count = len(cases) if isinstance(cases, list) else 0
    print(f"manifest: {manifest.get('status', 'unknown')} cases={case_count}")
    return exit_code


def follow_file(path: Path) -> int:
    if not path.exists():
        print(f"log file not found: {path}", file=sys.stderr)
        return 1
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            while True:
                line = handle.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(1)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"failed to read log file: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


def print_log(path: Path) -> int:
    if not path.exists():
        print(f"log file not found: {path}", file=sys.stderr)
        return 1
    try:
        sys.stdout.write(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeDecodeError) as exc:
        print(f"failed to read log file: {exc}", file=sys.stderr)
        return 1
    return 0


def read_process_cmdline(pid: int) -> list[str]:
    data = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [
        part.decode("utf-8", "replace")
        for part in data.split(b"\0")
        if part
    ]


def _is_current_script_arg(arg: str) -> bool:
    try:
        return Path(arg).resolve() == Path(__file__).resolve()
    except (OSError, RuntimeError):
        return arg == str(Path(__file__).resolve())


def _flag_indexes(command: list[str], flag: str) -> list[int]:
    return [index for index, value in enumerate(command) if value == flag]


def _single_flag_value(command: list[str], flag: str) -> str | None:
    indexes = _flag_indexes(command, flag)
    if len(indexes) != 1:
        return None
    index = indexes[0]
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def controller_command_matches(command: list[str], run_id: str,
                               results_dir: Path | None = None) -> bool:
    if len(command) < 3:
        return False
    if not _is_current_script_arg(command[1]):
        return False
    if command[2] != "run":
        return False
    if "--child" not in command:
        return False
    if _single_flag_value(command, "--run-id") != run_id:
        return False
    if results_dir is not None:
        results_value = _single_flag_value(command, "--results-dir")
        if results_value is None:
            return False
        try:
            if Path(results_value).resolve() != Path(results_dir).resolve():
                return False
        except (OSError, RuntimeError):
            return False
    return True


def _cmdline_matches_controller(cmdline: list[str], run_id: str) -> bool:
    return controller_command_matches(cmdline, run_id)


def _controller_metadata_command(run_dir: Path, pid: int,
                                 run_id: str) -> list[str]:
    metadata_path = run_dir / "controller.json"
    if not metadata_path.exists():
        raise ValueError(f"controller metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"controller metadata invalid: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("controller metadata invalid: expected object")
    command = metadata.get("command")
    if metadata.get("pid") != pid:
        raise ValueError("controller metadata pid mismatch")
    if metadata.get("run_id") != run_id:
        raise ValueError("controller metadata run_id mismatch")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("controller metadata command invalid")
    if not controller_command_matches(command, run_id, run_dir.parent):
        raise ValueError("controller metadata command is not an auto_bench child")
    return command


def is_controller_process(pid: int, run_id: str,
                          expected_command: list[str] | None = None,
                          results_dir: Path | None = None) -> bool:
    try:
        cmdline = read_process_cmdline(pid)
    except OSError:
        return False
    if expected_command is not None:
        return (
            cmdline == expected_command
            and controller_command_matches(cmdline, run_id, results_dir)
        )
    return controller_command_matches(cmdline, run_id, results_dir)


def _run_state_is_active(run_dir: Path) -> bool:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        print(f"state invalid: {state_path} not found", file=sys.stderr)
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"state invalid: {exc}", file=sys.stderr)
        return False
    if not isinstance(state, dict):
        print(f"state invalid: expected object in {state_path}", file=sys.stderr)
        return False
    status = state.get("status")
    if status not in ("starting", "running"):
        print(f"run is not active: {status}", file=sys.stderr)
        return False
    return True


def stop_run(run_dir: Path) -> int:
    pid_path = run_dir / "controller.pid"
    if not pid_path.exists():
        print(f"pid file not found: {pid_path}", file=sys.stderr)
        return 1
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"invalid pid file: {pid_path}: {exc}", file=sys.stderr)
        return 1
    if pid <= 1:
        print(f"unsafe pid in {pid_path}: {pid}", file=sys.stderr)
        return 1
    if not _run_state_is_active(run_dir):
        return 1
    try:
        expected_command = _controller_metadata_command(run_dir, pid, run_dir.name)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not is_controller_process(
        pid,
        run_dir.name,
        expected_command=expected_command,
        results_dir=run_dir.parent,
    ):
        print(f"process does not match controller or stale pid: {pid}", file=sys.stderr)
        return 1
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"process not found or not running: {pid}", file=sys.stderr)
        return 1
    except PermissionError as exc:
        print(f"failed to stop {pid}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"failed to stop {pid}: {exc}", file=sys.stderr)
        return 1
    print(f"sent SIGTERM to {pid}")
    return 0


def validate_prepared_model_dir(target: Path) -> None:
    model_dir = Path(target)
    if not model_dir.is_dir():
        raise ConfigError(f"model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise ConfigError(f"model directory missing config.json: {model_dir}")
    if not (
        (model_dir / "tokenizer.json").is_file()
        or (model_dir / "tokenizer_config.json").is_file()
    ):
        raise ConfigError(
            f"model directory missing tokenizer.json or tokenizer_config.json: {model_dir}"
        )
    index_paths = sorted(model_dir.glob("*.safetensors.index.json"))
    if index_paths:
        shard_names: set[str] = set()
        for index_path in index_paths:
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ConfigError(f"safetensors index invalid: {index_path}: {exc}") from exc
            if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
                raise ConfigError(f"safetensors index weight_map invalid: {index_path}")
            for shard_name in index["weight_map"].values():
                if not isinstance(shard_name, str):
                    raise ConfigError(f"safetensors index shard invalid: {index_path}")
                shard_path = Path(shard_name)
                if shard_path.is_absolute() or ".." in shard_path.parts:
                    raise ConfigError(f"safetensors shard path invalid: {shard_name}")
                if shard_path.suffix != ".safetensors":
                    raise ConfigError(f"safetensors shard filename invalid: {shard_name}")
                shard_names.add(shard_name)
        if not shard_names:
            raise ConfigError(f"model directory missing complete safetensors weights: {model_dir}")
        for shard_name in sorted(shard_names):
            shard_path = model_dir / shard_name
            if not shard_path.is_file() or shard_path.stat().st_size <= 0:
                raise ConfigError(f"safetensors shard missing or empty: {shard_name}")
        return
    safetensors = [
        path for path in model_dir.glob("*.safetensors")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not safetensors:
        raise ConfigError(f"model directory missing complete safetensors weights: {model_dir}")


PREPARE_MODEL_SCRIPT = """import sys
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(sys.argv[1], local_dir=sys.argv[2])
"""


def _download_tmp_for_target(target: Path) -> Path:
    return target.with_name(f"{target.name}.download-tmp")


def _backup_path_for_target(target: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    candidate = target.with_name(f"{target.name}.backup-{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = target.with_name(f"{target.name}.backup-{stamp}-{suffix}")
        suffix += 1
    return candidate


def _remove_download_tmp(tmp_dir: Path) -> None:
    if not (tmp_dir.exists() or tmp_dir.is_symlink()):
        return
    if tmp_dir.is_dir() and not tmp_dir.is_symlink():
        shutil.rmtree(tmp_dir)
    else:
        tmp_dir.unlink()


def build_prepare_model_command(modelscope_id: str, target: Path,
                                bench_image: str) -> list[str]:
    resolved_target = Path(target).resolve()
    tmp_dir = _download_tmp_for_target(resolved_target)
    return [
        "docker", "run", "--rm",
        "-v", f"{resolved_target.parent}:/model-target",
        bench_image,
        "python", "-c", PREPARE_MODEL_SCRIPT,
        modelscope_id,
        f"/model-target/{tmp_dir.name}",
    ]


def prepare_model(*, modelscope_id: str, target: Path,
                  bench_image: str, force: bool = False,
                  runner: Runner | None = None) -> int:
    resolved_target = Path(target).resolve()
    active_runner: Runner = runner or DockerRunner()
    tmp_dir = _download_tmp_for_target(resolved_target)

    if resolved_target.exists() and not force:
        validate_prepared_model_dir(resolved_target)
        print(f"model already exists: {resolved_target}")
        return 0

    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    _remove_download_tmp(tmp_dir)
    result = active_runner.run(
        build_prepare_model_command(modelscope_id, resolved_target, bench_image),
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr or result.stdout or "model download failed"
        raise RuntimeError(message)

    validate_prepared_model_dir(tmp_dir)

    if resolved_target.exists():
        if not force:
            raise ConfigError(f"target already exists: {resolved_target}")
        shutil.move(str(resolved_target), str(_backup_path_for_target(resolved_target)))
    tmp_dir.rename(resolved_target)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline vLLM auto benchmark controller",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run benchmark cases")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--detach", action="store_true")
    run_parser.add_argument("--child", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--results-dir", type=Path, help=argparse.SUPPRESS)
    run_parser.add_argument("--lock-token", help=argparse.SUPPRESS)

    status_parser = subparsers.add_parser("status", help="show run status")
    status_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    status_parser.add_argument("--run-id", required=True)

    logs_parser = subparsers.add_parser("logs", help="show controller log")
    logs_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    logs_parser.add_argument("--run-id", required=True)
    logs_parser.add_argument("-f", "--follow", action="store_true")

    stop_parser = subparsers.add_parser("stop", help="stop detached controller")
    stop_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    stop_parser.add_argument("--run-id", required=True)

    prepare_parser = subparsers.add_parser("prepare-model", help="prepare model assets")
    prepare_parser.add_argument("--modelscope-id", required=True)
    prepare_parser.add_argument("--target", required=True)
    prepare_parser.add_argument("--bench-image", required=True)
    prepare_parser.add_argument("--force", action="store_true")

    return parser.parse_args(argv)


def write_child_startup_state(results_dir: Path | None, run_id: str | None,
                              status: str, error: str) -> None:
    if results_dir is None or run_id is None:
        return
    state = _detached_state(run_id, status, 0, error=error)
    write_state(results_dir / run_id, state)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        if args.child:
            try:
                install_signal_handlers()
                config = load_config(args.config)
                run_id = args.run_id or make_run_id(config.run.name)
            except StopRequested as exc:
                error = str(exc) or "stop requested"
                write_child_startup_state(args.results_dir, args.run_id, "interrupted", error)
                if args.results_dir is not None and args.run_id is not None:
                    release_run_lock_for_token(args.results_dir / args.run_id, args.lock_token)
                print(error, file=sys.stderr)
                return 130
            except Exception as exc:
                error = str(exc)
                write_child_startup_state(args.results_dir, args.run_id, "failed", error)
                if args.results_dir is not None and args.run_id is not None:
                    release_run_lock_for_token(args.results_dir / args.run_id, args.lock_token)
                print(error, file=sys.stderr)
                return 1
            controller_kwargs: dict[str, Any] = {"dry_run": args.dry_run}
            if args.lock_token is not None:
                controller_kwargs["lock_token"] = args.lock_token
            return run_controller(config, run_id=run_id, **controller_kwargs)

        config = load_config(args.config)
        run_id = args.run_id or make_run_id(config.run.name)
        if args.dry_run:
            return run_controller(config, run_id=run_id, dry_run=True)
        if args.detach and not args.child:
            return start_detached(args.config, config, run_id)
        install_signal_handlers()
        return run_controller(config, run_id=run_id, dry_run=args.dry_run)
    if args.command == "status":
        return print_status(args.results_dir / args.run_id)
    if args.command == "logs":
        log_path = args.results_dir / args.run_id / "controller.log"
        return follow_file(log_path) if args.follow else print_log(log_path)
    if args.command == "stop":
        return stop_run(args.results_dir / args.run_id)
    if args.command == "prepare-model":
        return prepare_model(
            modelscope_id=args.modelscope_id,
            target=Path(args.target),
            bench_image=args.bench_image,
            force=args.force,
            runner=DockerRunner(),
        )
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
