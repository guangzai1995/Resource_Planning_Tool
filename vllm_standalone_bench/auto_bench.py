#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MODEL_CONTAINER_ROOT = PurePosixPath("/models")
SUPPORTED_BACKENDS = frozenset({"openai", "openai-chat"})


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
        network=_string(run.get("network", "vllm-bench-net"), "run.network"),
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
    return f"{safe_run_name}_{stamp}"


def make_container_name(model: ModelConfig, serve_profile: ServeProfile, run_id: str) -> str:
    safe_run_id = _safe_name(run_id, "run_id")
    return f"bench-vllm-{model.name}-{serve_profile.name}-{safe_run_id}"


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
