#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from bench_compare import aggregate_compare
from remote_topology import RemoteAuth, TopologyProfile, parse_topology_profiles
from resource_monitor import ResourceMonitor, append_summary_to_result_files

logger = logging.getLogger("auto_bench")


SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MODEL_CONTAINER_ROOT = PurePosixPath("/models")
DATASET_CONTAINER_ROOT = PurePosixPath("/datasets")
BUILTIN_ASR_DATASET_PATH = (
    "/opt/vllm_standalone_bench/assets/librispeech_test_clean_256/asr_smoke.jsonl"
)
SUPPORTED_BACKENDS = frozenset({"openai", "openai-chat", "openai-audio"})
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
NETWORK_MANAGED_LABEL = "vllm_auto_bench.managed"
NETWORK_RUN_ID_LABEL = "vllm_auto_bench.run_id"
CONTAINER_RUN_DIR_LABEL = "vllm_auto_bench.run_dir"
CONTAINER_MODEL_LABEL = "vllm_auto_bench.model"
CONTAINER_SERVE_PROFILE_LABEL = "vllm_auto_bench.serve_profile"
CONTAINER_BENCH_PROFILE_LABEL = "vllm_auto_bench.bench_profile"


class ConfigError(ValueError):
    """Raised when the auto bench configuration is invalid."""


SUPPORTED_ENGINES = ("vllm", "sglang")


class StopRequested(Exception):
    """Raised when the detached controller is asked to stop gracefully."""


@dataclass(frozen=True)
class VllmCacheConfig:
    enabled: bool = False
    root: Path | None = None
    container_path: str = "/vllm-cache"
    set_default_env: bool = True


@dataclass(frozen=True)
class ResourceMonitorRunConfig:
    enabled: bool = True
    backend: str = "nvidia-smi"
    interval_sec: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    name: str
    results_dir: Path
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
    vllm_image: str | None = None
    images: Mapping[str, str] = field(
        default_factory=lambda: types.MappingProxyType({})
    )
    vllm_cache: VllmCacheConfig = field(default_factory=VllmCacheConfig)
    resource_monitor: ResourceMonitorRunConfig = field(
        default_factory=ResourceMonitorRunConfig
    )


@dataclass(frozen=True)
class MountConfig:
    models: Path
    datasets: Path | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_path: str
    tokenizer_path: str | None
    served_model_name: str | None
    host_model_path: Path
    host_tokenizer_path: Path | None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    length_policy: str = "exact"
    input_len_tolerance: float = 0.2
    on_bucket_shortage: str = "error"
    sampling: str = "shuffle"


@dataclass(frozen=True)
class ServeProfile:
    name: str
    engine: str = "vllm"
    gpus: str = "all"
    args: tuple[str, ...] = field(default_factory=tuple)
    cache_key: str | None = None


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
    warmup_concurrency: int | None = None
    warmup_output_len: int | None = None
    cross_product: bool = False
    max_ttft_ms: float | None = None
    min_throughput_tok_s: float | None = None
    min_output_compliance: float | None = None
    dataset: DatasetConfig | None = None
    dataset_name: str = "random"
    dataset_path: str | None = None
    language: str = "en"


@dataclass(frozen=True)
class AutoBenchConfig:
    run: RunConfig
    mounts: MountConfig
    models: tuple[ModelConfig, ...]
    serve_profiles: tuple[ServeProfile, ...]
    bench_profiles: tuple[BenchProfile, ...]
    topology_profiles: tuple[TopologyProfile, ...] = ()


@dataclass(frozen=True)
class BenchmarkCase:
    model: ModelConfig
    bench_profile: BenchProfile
    run_id: str
    api_model_name: str
    serve_profile: ServeProfile | None = None
    topology_profile: TopologyProfile | None = None
    container_name: str | None = None

    def __post_init__(self) -> None:
        if (self.serve_profile is None) == (self.topology_profile is None):
            raise ConfigError(
                "BenchmarkCase requires exactly one of serve_profile or topology_profile"
            )

    @property
    def serving_name(self) -> str:
        if self.serve_profile is not None:
            return self.serve_profile.name
        if self.topology_profile is not None:
            return self.topology_profile.name
        raise ConfigError("BenchmarkCase is missing serving profile information")


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


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{field_name} must be a positive integer or None, got {value!r}")
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


def _container_abs_path(value: Any, field_name: str) -> str:
    path_text = _string(value, field_name)
    container_path = PurePosixPath(path_text)
    if not container_path.is_absolute():
        raise ConfigError(f"{field_name} must be absolute inside the container: {path_text}")
    if ".." in container_path.parts:
        raise ConfigError(f"{field_name} must not contain '..': {path_text}")
    if container_path == PurePosixPath("/"):
        raise ConfigError(f"{field_name} must not be the container root: {path_text}")
    try:
        container_path.relative_to(MODEL_CONTAINER_ROOT)
    except ValueError:
        pass
    else:
        raise ConfigError(f"{field_name} must not overlap the /models mount: {path_text}")
    return str(container_path)


def _dataset_container_path(value: Any, field_name: str) -> str:
    path_text = _string(value, field_name)
    if path_text.startswith("//"):
        raise ConfigError(f"{field_name} must not start with '//': {path_text}")
    container_path = PurePosixPath(path_text)
    if not container_path.is_absolute():
        raise ConfigError(f"{field_name} must be absolute inside the container: {path_text}")
    if ".." in container_path.parts:
        raise ConfigError(f"{field_name} must not contain '..': {path_text}")
    return str(container_path)


def _parse_vllm_cache(run: dict[str, Any], config_dir: Path) -> VllmCacheConfig:
    if "vllm_cache" not in run:
        return VllmCacheConfig()
    raw = run["vllm_cache"]
    cache = _require_mapping(raw, "run.vllm_cache")
    enabled = _bool(cache.get("enabled", False), "run.vllm_cache.enabled")
    container_path = _container_abs_path(
        cache.get("container_path", "/vllm-cache"),
        "run.vllm_cache.container_path",
    )
    set_default_env = _bool(
        cache.get("set_default_env", True),
        "run.vllm_cache.set_default_env",
    )
    if not enabled:
        return VllmCacheConfig(
            enabled=False,
            root=None,
            container_path=container_path,
            set_default_env=set_default_env,
        )
    if "root" in cache:
        root = Path(_string(cache["root"], "run.vllm_cache.root"))
    else:
        root = config_dir / ".cache" / "vllm_auto_bench"
    if not root.is_absolute():
        root = config_dir / root
    return VllmCacheConfig(
        enabled=True,
        root=root.resolve(),
        container_path=container_path,
        set_default_env=set_default_env,
    )


def _parse_resource_monitor(run: dict[str, Any]) -> ResourceMonitorRunConfig:
    raw = run.get("resource_monitor")
    if raw is None:
        return ResourceMonitorRunConfig()
    data = _require_mapping(raw, "run.resource_monitor")
    enabled = _bool(data.get("enabled", True), "run.resource_monitor.enabled")
    backend = _string(data.get("backend", "nvidia-smi"), "run.resource_monitor.backend")
    if backend != "nvidia-smi":
        raise ConfigError("run.resource_monitor.backend only supports nvidia-smi")
    interval_sec = _finite_float(
        data.get("interval_sec", 1.0),
        "run.resource_monitor.interval_sec",
    )
    if interval_sec <= 0:
        raise ConfigError("run.resource_monitor.interval_sec must be > 0")
    return ResourceMonitorRunConfig(
        enabled=enabled,
        backend=backend,
        interval_sec=interval_sec,
    )


def _parse_run(data: dict[str, Any], config_dir: Path) -> RunConfig:
    run = _require_mapping(data.get("run"), "run")
    host_port = run.get("host_port")
    vllm_image = _optional_string(run.get("vllm_image"), "run.vllm_image")
    images = _parse_engine_images(run, vllm_image)
    return RunConfig(
        name=_safe_name(_required(run, "name", "run.name"), "run.name"),
        results_dir=Path(_string(run.get("results_dir", "vllm_standalone_bench/results"),
                                 "run.results_dir")),
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
        vllm_image=vllm_image,
        images=images,
        vllm_cache=_parse_vllm_cache(run, config_dir),
        resource_monitor=_parse_resource_monitor(run),
    )


def _parse_engine_images(
    run: dict[str, Any], vllm_image: str | None
) -> Mapping[str, str]:
    raw = run.get("images")
    images: dict[str, str] = {}
    if raw is not None:
        if not isinstance(raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
        ):
            raise ConfigError("run.images must be an object mapping engine -> image string")
        images = dict(raw)
    if vllm_image is not None:
        images.setdefault("vllm", vllm_image)
    return types.MappingProxyType(images)


def _parse_mounts(data: dict[str, Any], config_dir: Path) -> MountConfig:
    mounts = _require_mapping(data.get("mounts"), "mounts")
    models = Path(_string(_required(mounts, "models", "mounts.models"), "mounts.models"))
    if not models.is_absolute():
        models = config_dir / models
    raw_datasets = mounts.get("datasets")
    datasets = None
    if raw_datasets is not None:
        datasets = Path(_string(raw_datasets, "mounts.datasets"))
        if not datasets.is_absolute():
            datasets = config_dir / datasets
    return MountConfig(
        models=models.resolve(),
        datasets=datasets.resolve() if datasets is not None else None,
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
    if raw_profiles is None:
        return ()
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ConfigError("serve_profiles must be a non-empty list")

    parsed: list[ServeProfile] = []
    for item in raw_profiles:
        profile = _require_mapping(item, "serve_profiles[]")
        args = profile.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ConfigError("serve_profile.args must be a string array")
        _validate_serve_args(args, "serve_profile")
        engine = _string(profile.get("engine", "vllm"), "serve_profile.engine")
        if engine not in SUPPORTED_ENGINES:
            raise ConfigError(
                f"serve_profile.engine must be one of {SUPPORTED_ENGINES}, got {engine!r}"
            )
        cache_key = profile.get("cache_key")
        parsed.append(ServeProfile(
            name=_safe_name(_required(profile, "name", "serve_profile.name"),
                            "serve_profile.name"),
            engine=engine,
            gpus=_string(profile.get("gpus", "all"), "serve_profile.gpus"),
            args=tuple(args),
            cache_key=(
                _safe_name(cache_key, "serve_profile.cache_key")
                if cache_key is not None else None
            ),
        ))
    return tuple(parsed)


def _parse_topology_profiles(data: dict[str, Any]) -> tuple[TopologyProfile, ...]:
    return parse_topology_profiles(
        data,
        error=ConfigError,
        safe_name=_safe_name,
        supported_engines=SUPPORTED_ENGINES,
    )


def _validate_serving_profile_names_unique(
    serve_profiles: tuple[ServeProfile, ...],
    topology_profiles: tuple[TopologyProfile, ...],
) -> None:
    all_names = [profile.name for profile in serve_profiles]
    all_names.extend(profile.name for profile in topology_profiles)
    duplicates = sorted({
        name for name in all_names if all_names.count(name) > 1
    })
    if duplicates:
        raise ConfigError(
            "duplicate serving profile name: " + ", ".join(duplicates)
        )


def _validate_serve_args(args: Sequence[str], path: str) -> None:
    for value in args:
        if value.startswith("speculative-config."):
            raise ConfigError(
                f"{path}.args contains {value!r}; use '--{value}' for vLLM dotted flags"
            )


def _parse_dataset_config(raw: object, path: str) -> DatasetConfig | None:
    if raw is None:
        return None
    dataset = _require_mapping(raw, path)
    name = _string(_required(dataset, "name", f"{path}.name"), f"{path}.name")
    if name not in {"random", "builtin_mtp_chat"}:
        raise ConfigError(f"{path}.name unsupported dataset: {name}")

    length_policy = _string(dataset.get("length_policy", "exact"),
                            f"{path}.length_policy")
    if length_policy not in {"exact", "bucket"}:
        raise ConfigError(f"{path}.length_policy must be exact or bucket")

    tolerance = _finite_float(dataset.get("input_len_tolerance", 0.2),
                              f"{path}.input_len_tolerance")
    if tolerance < 0 or tolerance >= 1:
        raise ConfigError(f"{path}.input_len_tolerance must be >= 0 and < 1")

    shortage = _string(dataset.get("on_bucket_shortage", "error"),
                       f"{path}.on_bucket_shortage")
    if shortage != "error":
        raise ConfigError(f"{path}.on_bucket_shortage only supports error")

    sampling = _string(dataset.get("sampling", "shuffle"), f"{path}.sampling")
    if sampling not in {"shuffle", "round_robin"}:
        raise ConfigError(f"{path}.sampling must be shuffle or round_robin")

    return DatasetConfig(
        name=name,
        length_policy=length_policy,
        input_len_tolerance=tolerance,
        on_bucket_shortage=shortage,
        sampling=sampling,
    )


def _parse_bench_profiles(data: dict[str, Any]) -> tuple[BenchProfile, ...]:
    raw_profiles = data.get("bench_profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ConfigError("bench_profiles must be a non-empty list")

    parsed: list[BenchProfile] = []
    for item in raw_profiles:
        profile = _require_mapping(item, "bench_profiles[]")
        backend = _backend(profile.get("backend", "openai-chat"), "bench_profile.backend")
        output_lens = _positive_int_list(profile.get("output_lens", [128]), "output_lens")
        parallel_nums = _positive_int_list(profile.get("parallel_nums", [1, 4, 8]),
                                           "parallel_nums")
        cross_product = _bool(profile.get("cross_product", False), "cross_product")
        if backend == "openai-audio":
            input_lens = (0,)
            prefix_ratio = 0.0
            dataset = None
            dataset_name = (
                _string(profile["dataset_name"], "bench_profile.dataset_name")
                if "dataset_name" in profile else "custom_audio"
            )
            if "dataset_path" in profile:
                dataset_path = _dataset_container_path(
                    profile["dataset_path"],
                    "bench_profile.dataset_path",
                )
                if not _is_under_container_path(dataset_path, DATASET_CONTAINER_ROOT):
                    raise ConfigError(
                        "bench_profile.dataset_path for openai-audio must be under "
                        "/datasets so mounts.datasets can expose it to the bench "
                        "container"
                    )
            else:
                dataset_path = BUILTIN_ASR_DATASET_PATH
            language = (
                _string(profile["language"], "bench_profile.language")
                if "language" in profile else "en"
            )
        else:
            input_lens = _positive_int_list(profile.get("input_lens", [512]), "input_lens")
            prefix_ratio = _ratio(profile.get("prefix_ratio", 0.0),
                                  "bench_profile.prefix_ratio")
            dataset_name = "random"
            dataset_path = None
            language = ""
            dataset = _parse_dataset_config(
                profile.get("dataset"),
                "bench_profile.dataset",
            )
        if (
            backend != "openai-audio"
            and not cross_product
            and len(output_lens) not in (1, len(input_lens))
        ):
            raise ConfigError(
                "output_lens length must be 1 or match input_lens unless cross_product=true"
            )
        parsed.append(BenchProfile(
            name=_safe_name(_required(profile, "name", "bench_profile.name"),
                            "bench_profile.name"),
            backend=backend,
            input_lens=input_lens,
            output_lens=output_lens,
            parallel_nums=parallel_nums,
            epochs=_positive_int(profile.get("epochs", 3), "bench_profile.epochs"),
            prefix_ratio=prefix_ratio,
            warmup_requests=_non_negative_int(profile.get("warmup_requests", 1),
                                              "bench_profile.warmup_requests"),
            warmup_concurrency=_optional_positive_int(
                profile.get("warmup_concurrency"), "bench_profile.warmup_concurrency"),
            warmup_output_len=_optional_positive_int(
                profile.get("warmup_output_len"), "bench_profile.warmup_output_len"),
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
            dataset=dataset,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            language=language,
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
    run = _parse_run(config_data, config_path.parent)
    mounts = _parse_mounts(config_data, config_path.parent)
    models = _parse_models(config_data, mounts)
    serve_profiles = _parse_serve_profiles(config_data)
    topology_profiles = _parse_topology_profiles(config_data)
    if not serve_profiles and not topology_profiles:
        raise ConfigError("serve_profiles or topology_profiles must be configured")
    _validate_serving_profile_names_unique(serve_profiles, topology_profiles)
    bench_profiles = _parse_bench_profiles(config_data)
    config = AutoBenchConfig(
        run,
        mounts,
        models,
        serve_profiles,
        bench_profiles,
        topology_profiles,
    )
    _validate_asr_dataset_mounts(config)
    _validate_images_cover_engines(config)
    return config


def _is_under_container_path(path: str, root: PurePosixPath) -> bool:
    try:
        PurePosixPath(path).relative_to(root)
    except ValueError:
        return False
    return True


def _validate_asr_dataset_mounts(config: AutoBenchConfig) -> None:
    for bench in config.bench_profiles:
        if (
            bench.backend == "openai-audio"
            and bench.dataset_path is not None
            and _is_under_container_path(bench.dataset_path, DATASET_CONTAINER_ROOT)
            and config.mounts.datasets is None
        ):
            raise ConfigError(
                "bench_profile.dataset_path under /datasets requires mounts.datasets"
            )


def _validate_images_cover_engines(config: AutoBenchConfig) -> None:
    engines = {profile.engine for profile in config.serve_profiles}
    missing = sorted(engine for engine in engines if engine not in config.run.images)
    if missing:
        raise ConfigError(
            f"run.images missing image for engine(s): {', '.join(missing)}"
        )


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
        f"bench-runner-{case.model.name}-{case.serving_name}-"
        f"{case.bench_profile.name}-{_safe_name(case.run_id, 'run_id')}"
    )


def require_legacy_case(case: BenchmarkCase) -> ServeProfile:
    if case.serve_profile is None or case.container_name is None:
        raise ConfigError("legacy serve profile case required")
    return case.serve_profile


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def vllm_cache_key_inputs(config: AutoBenchConfig,
                          case: BenchmarkCase) -> dict[str, Any]:
    serve_profile = require_legacy_case(case)
    return {
        "vllm_image_ref": config.run.images["vllm"],
        "model": {
            "name": case.model.name,
            "model_path": case.model.model_path,
            "tokenizer_path": case.model.tokenizer_path,
            "served_model_name": case.model.served_model_name,
        },
        "serve_profile": {
            "name": serve_profile.name,
            "gpus": serve_profile.gpus,
            "args": list(serve_profile.args),
        },
    }


def _short_json_fingerprint(data: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _short_hash(canonical)


def default_vllm_cache_key(config: AutoBenchConfig, case: BenchmarkCase) -> str:
    serve_profile = require_legacy_case(case)
    fingerprint = _short_json_fingerprint(vllm_cache_key_inputs(config, case))
    return f"{case.model.name}__{serve_profile.name}__{fingerprint}"


def vllm_cache_key(config: AutoBenchConfig, case: BenchmarkCase) -> str | None:
    if not config.run.vllm_cache.enabled:
        return None
    serve_profile = require_legacy_case(case)
    if serve_profile.engine != "vllm":
        return None
    return serve_profile.cache_key or default_vllm_cache_key(config, case)


def vllm_cache_key_source(config: AutoBenchConfig, case: BenchmarkCase) -> str | None:
    if not config.run.vllm_cache.enabled:
        return None
    serve_profile = require_legacy_case(case)
    if serve_profile.engine != "vllm":
        return None
    return "explicit" if serve_profile.cache_key else "default"


def resolve_vllm_cache_dir(config: AutoBenchConfig, case: BenchmarkCase) -> Path | None:
    cache_key = vllm_cache_key(config, case)
    if cache_key is None:
        return None
    if config.run.vllm_cache.root is None:
        raise ConfigError("run.vllm_cache.root is required when enabled=true")
    return config.run.vllm_cache.root / cache_key


def build_vllm_cache_env(config: AutoBenchConfig) -> dict[str, str]:
    cache = config.run.vllm_cache
    if not cache.enabled or not cache.set_default_env:
        return {}
    root = cache.container_path
    root_path = PurePosixPath(root)
    return {
        "VLLM_CACHE_ROOT": root,
        "DG_JIT_CACHE_DIR": str(root_path / "deep_gemm"),
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(root_path / "flashinfer_autotune"),
    }


def vllm_cache_metadata(config: AutoBenchConfig,
                        case: BenchmarkCase) -> dict[str, Any] | None:
    cache_dir = resolve_vllm_cache_dir(config, case)
    key = vllm_cache_key(config, case)
    if cache_dir is None or key is None:
        return None
    return {
        "enabled": True,
        "cache_key": key,
        "cache_key_source": vllm_cache_key_source(config, case),
        "cache_key_inputs": vllm_cache_key_inputs(config, case),
        "host_dir": str(cache_dir),
        "container_path": config.run.vllm_cache.container_path,
        "env": build_vllm_cache_env(config),
    }


def ensure_vllm_cache_dirs(config: AutoBenchConfig) -> None:
    if not config.run.vllm_cache.enabled:
        return
    seen: set[Path] = set()
    for case in expand_cases(config, run_id="cache-validation"):
        cache_dir = resolve_vllm_cache_dir(config, case)
        if cache_dir is None or cache_dir in seen:
            continue
        seen.add(cache_dir)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"cannot create vllm cache dir: {cache_dir}") from exc


def expand_cases(config: AutoBenchConfig, run_id: str | None = None) -> tuple[BenchmarkCase, ...]:
    resolved_run_id = run_id or make_run_id(config.run.name)
    _safe_name(resolved_run_id, "run_id")

    cases: list[BenchmarkCase] = []
    for model in config.models:
        for serve_profile in config.serve_profiles:
            for bench_profile in config.bench_profiles:
                cases.append(BenchmarkCase(
                    model=model,
                    bench_profile=bench_profile,
                    run_id=resolved_run_id,
                    api_model_name=model.served_model_name or model.name,
                    serve_profile=serve_profile,
                    container_name=make_container_name(model, serve_profile, resolved_run_id),
                ))
        for topology_profile in config.topology_profiles:
            for bench_profile in config.bench_profiles:
                cases.append(BenchmarkCase(
                    model=model,
                    bench_profile=bench_profile,
                    run_id=resolved_run_id,
                    api_model_name=model.served_model_name or model.name,
                    topology_profile=topology_profile,
                    container_name=None,
                ))
    return tuple(cases)


def build_vllm_run_command(config: AutoBenchConfig, case: BenchmarkCase,
                           run_dir: Path) -> list[str]:
    serve_profile = require_legacy_case(case)
    resolved_run_dir = Path(run_dir).resolve()
    cmd = [
        "docker", "run", "-d",
        "--name", case.container_name,
        "--label", f"{NETWORK_MANAGED_LABEL}=true",
        "--label", f"{NETWORK_RUN_ID_LABEL}={case.run_id}",
        "--label", f"{CONTAINER_RUN_DIR_LABEL}={resolved_run_dir}",
        "--label", f"{CONTAINER_MODEL_LABEL}={case.model.name}",
        "--label", f"{CONTAINER_SERVE_PROFILE_LABEL}={serve_profile.name}",
        "--gpus", serve_profile.gpus,
        "--network", config.run.network,
        "-v", f"{config.mounts.models}:/models:ro",
    ]
    cache_dir = resolve_vllm_cache_dir(config, case)
    if cache_dir is not None:
        cmd.extend(["-v", f"{cache_dir}:{config.run.vllm_cache.container_path}:rw"])
        for name, value in build_vllm_cache_env(config).items():
            cmd.extend(["-e", f"{name}={value}"])
    cmd.extend([
        "--entrypoint", "vllm",
    ])
    if config.run.publish_host_port:
        if config.run.host_port is None:
            raise ConfigError("host_port is required when publish_host_port=true")
        cmd.extend([
            "-p",
            f"127.0.0.1:{config.run.host_port}:{config.run.container_port}",
        ])
    cmd.extend([
        config.run.images["vllm"],
        "serve", case.model.model_path,
        "--served-model-name", case.api_model_name,
        "--host", "0.0.0.0",
        "--port", str(config.run.container_port),
    ])
    if config.run.api_key:
        cmd.extend(["--api-key", config.run.api_key])
    cmd.extend(serve_profile.args)
    return cmd


def build_serve_run_command(config: AutoBenchConfig, case: BenchmarkCase,
                            run_dir: Path) -> list[str]:
    """按 serve_profile.engine 分派服务启动命令。args 原样透传，不做参数翻译。"""
    serve_profile = require_legacy_case(case)
    if serve_profile.engine == "sglang":
        return _build_sglang_run_command(config, case, run_dir)
    return build_vllm_run_command(config, case, run_dir)


def _build_sglang_run_command(config: AutoBenchConfig, case: BenchmarkCase,
                              run_dir: Path) -> list[str]:
    serve_profile = require_legacy_case(case)
    resolved_run_dir = Path(run_dir).resolve()
    cmd = [
        "docker", "run", "-d",
        "--name", case.container_name,
        "--label", f"{NETWORK_MANAGED_LABEL}=true",
        "--label", f"{NETWORK_RUN_ID_LABEL}={case.run_id}",
        "--label", f"{CONTAINER_RUN_DIR_LABEL}={resolved_run_dir}",
        "--label", f"{CONTAINER_MODEL_LABEL}={case.model.name}",
        "--label", f"{CONTAINER_SERVE_PROFILE_LABEL}={serve_profile.name}",
        "--gpus", serve_profile.gpus,
        "--network", config.run.network,
        "-v", f"{config.mounts.models}:/models:ro",
        "--entrypoint", "python3",
    ]
    if config.run.publish_host_port:
        if config.run.host_port is None:
            raise ConfigError("host_port is required when publish_host_port=true")
        cmd.extend(["-p", f"127.0.0.1:{config.run.host_port}:{config.run.container_port}"])
    cmd.extend([
        config.run.images["sglang"],
        "-m", "sglang.launch_server",
        "--model-path", case.model.model_path,
        "--host", "0.0.0.0",
        "--port", str(config.run.container_port),
        "--served-model-name", case.api_model_name,
    ])
    if config.run.api_key:
        cmd.extend(["--api-key", config.run.api_key])
    cmd.extend(serve_profile.args)
    return cmd


def _append_many(cmd: list[str], flag: str, values: tuple[int, ...]) -> None:
    cmd.append(flag)
    cmd.extend(str(value) for value in values)


def build_bench_run_command(config: AutoBenchConfig, case: BenchmarkCase,
                            bench_dir: Path) -> list[str]:
    serve_profile = require_legacy_case(case)
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
        "--label", f"{CONTAINER_SERVE_PROFILE_LABEL}={serve_profile.name}",
        "--label", f"{CONTAINER_BENCH_PROFILE_LABEL}={case.bench_profile.name}",
        "--network", config.run.network,
        "-v", f"{config.mounts.models}:/models:ro",
    ]
    if config.mounts.datasets is not None:
        cmd.extend(["-v", f"{config.mounts.datasets}:/datasets:ro"])
    cmd.extend([
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
    ])
    if bench.warmup_concurrency is not None:
        cmd.extend(["--warmup-concurrency", str(bench.warmup_concurrency)])
    if bench.warmup_output_len is not None:
        cmd.extend(["--warmup-output-len", str(bench.warmup_output_len)])
    if config.run.api_key:
        cmd.extend(["--api-key", config.run.api_key])
    if bench.backend == "openai-audio":
        cmd.extend([
            "--dataset-name", bench.dataset_name,
            "--dataset-path", bench.dataset_path or BUILTIN_ASR_DATASET_PATH,
            "--language", bench.language,
        ])
    else:
        if case.model.tokenizer_path:
            cmd.extend(["--tokenizer", case.model.tokenizer_path])
        if bench.dataset is not None:
            cmd.extend(["--dataset", bench.dataset.name])
            cmd.extend(["--dataset-length-policy", bench.dataset.length_policy])
            cmd.extend(["--dataset-input-len-tolerance",
                        str(bench.dataset.input_len_tolerance)])
            cmd.extend(["--dataset-on-bucket-shortage",
                        bench.dataset.on_bucket_shortage])
            cmd.extend(["--dataset-sampling", bench.dataset.sampling])
        _append_many(cmd, "--input-lens", bench.input_lens)
    _append_many(cmd, "--output-lens", bench.output_lens)
    _append_many(cmd, "--parallel-nums", bench.parallel_nums)
    if bench.cross_product and bench.backend != "openai-audio":
        cmd.append("--cross-product")
    if bench.backend != "openai-audio" and bench.prefix_ratio:
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
    if config.mounts.datasets is not None and not config.mounts.datasets.is_dir():
        raise ConfigError(f"datasets root does not exist: {config.mounts.datasets}")
    for bench in config.bench_profiles:
        if (
            bench.backend == "openai-audio"
            and bench.dataset_path is not None
            and _is_under_container_path(bench.dataset_path, DATASET_CONTAINER_ROOT)
        ):
            if config.mounts.datasets is None:
                raise ConfigError(
                    "bench_profile.dataset_path under /datasets requires mounts.datasets"
                )
            relative_dataset = PurePosixPath(bench.dataset_path).relative_to(
                DATASET_CONTAINER_ROOT
            )
            host_dataset = config.mounts.datasets.joinpath(*relative_dataset.parts)
            if not host_dataset.is_file():
                raise ConfigError(f"dataset file does not exist: {host_dataset}")
    for model in config.models:
        if not model.host_model_path.is_dir():
            raise ConfigError(f"model path does not exist: {model.host_model_path}")
        if model.host_tokenizer_path is not None and not model.host_tokenizer_path.exists():
            raise ConfigError(f"tokenizer path does not exist: {model.host_tokenizer_path}")
    ensure_vllm_cache_dirs(config)


def build_layout(config: AutoBenchConfig, run_id: str, case: BenchmarkCase) -> CaseLayout:
    run_dir = config.run.results_dir / run_id
    serve_dir = run_dir / case.model.name / case.serving_name
    bench_dir = serve_dir / case.bench_profile.name
    return CaseLayout(run_dir=run_dir, serve_dir=serve_dir, bench_dir=bench_dir)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        raise


def write_vllm_cache_metadata(config: AutoBenchConfig, case: BenchmarkCase,
                              layout: CaseLayout) -> None:
    payload = vllm_cache_metadata(config, case)
    if payload is None:
        return
    write_json_atomic(layout.serve_dir / "vllm_cache.json", payload)


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
            "serve_profile": case.serving_name,
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
    serve_profile = require_legacy_case(case)
    return (
        vllm_container_labels_match(labels, case, run_dir)
        and labels.get(CONTAINER_MODEL_LABEL) == case.model.name
        and labels.get(CONTAINER_SERVE_PROFILE_LABEL) == serve_profile.name
        and labels.get(CONTAINER_BENCH_PROFILE_LABEL) == case.bench_profile.name
    )


def remove_existing_vllm_container_if_owned(runner: Runner, case: BenchmarkCase,
                                            run_dir: Path) -> None:
    require_legacy_case(case)
    container_name = case.container_name
    labels = inspect_container_labels(runner, container_name)
    if labels is None:
        return
    if not vllm_container_labels_match(labels, case, run_dir):
        raise RuntimeError(
            f"vLLM container exists but is not owned by this run: {container_name}"
        )
    runner.run(["docker", "rm", "-f", container_name], check=False)


def stop_and_remove_vllm_container_if_owned(runner: Runner, case: BenchmarkCase,
                                           run_dir: Path, dry_run: bool) -> None:
    require_legacy_case(case)
    container_name = case.container_name
    if dry_run:
        stop_and_remove_container(runner, container_name, dry_run=True)
        return
    labels = inspect_container_labels(runner, container_name)
    if labels is None:
        return
    if not vllm_container_labels_match(labels, case, run_dir):
        return
    stop_and_remove_container(runner, container_name, dry_run=False)


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
    serve_profile = require_legacy_case(case)
    return f"bench-ready-{case.model.name}-{serve_profile.name}-{case.run_id}"


def build_ready_probe_run_command(config: AutoBenchConfig, case: BenchmarkCase,
                                  run_dir: Path) -> list[str]:
    require_legacy_case(case)
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
    require_legacy_case(case)
    container_name = case.container_name
    layout.serve_dir.mkdir(parents=True, exist_ok=True)
    logs = runner.run(["docker", "logs", "--timestamps", container_name], check=False)
    (layout.serve_dir / "vllm.log").write_text(
        logs.stdout + logs.stderr,
        encoding="utf-8",
    )
    inspect = runner.run(["docker", "inspect", container_name], check=False)
    (layout.serve_dir / "docker.inspect.json").write_text(inspect.stdout, encoding="utf-8")
    (layout.serve_dir / "serve_command.txt").write_text(
        " ".join(build_serve_run_command(config, case, layout.run_dir)),
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
RESUME_STARTUP_STATE_FILE = ".resume-startup-state.json"
TERMINAL_RUN_STATUSES = frozenset({
    "completed",
    "completed_with_failures",
    "failed",
    "interrupted",
})
RESUMABLE_RUN_STATUSES = frozenset({
    "interrupted",
    "failed",
    "completed_with_failures",
})


@dataclass(frozen=True)
class RunLock:
    run_dir: Path
    token: str


@dataclass(frozen=True)
class ResumeContext:
    config: AutoBenchConfig
    run_id: str
    run_dir: Path
    initial_manifest: Manifest
    pending_cases: tuple[BenchmarkCase, ...]
    unknown_manifest_cases: tuple[tuple[str, str, str], ...]


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


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{label} invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{label} must be a JSON object: {path}")
    return payload


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


def _remove_run_lock_matching_identity(run_dir: Path, identity: tuple[int, str, Any],
                                       purpose: str) -> bool:
    lock_path = run_lock_path(run_dir)
    marker_path = run_dir / f"{RUN_LOCK_FILE}.{purpose}.{os.getpid()}.{secrets.token_hex(8)}"
    try:
        lock_path.replace(marker_path)
    except OSError:
        return False
    marker_payload = _read_lock_marker_payload(marker_path)
    if _run_lock_identity(marker_payload) != identity:
        _restore_reclaimed_lock(marker_path, lock_path)
        return False
    try:
        marker_path.unlink()
    except OSError:
        _restore_reclaimed_lock(marker_path, lock_path)
        return False
    return True


def release_run_lock(lock: RunLock) -> None:
    payload = _read_run_lock(lock.run_dir)
    identity = _run_lock_identity(payload)
    if identity is None or identity[1] != lock.token:
        return
    if _run_lock_identity(_read_run_lock(lock.run_dir)) != identity:
        return
    _remove_run_lock_matching_identity(lock.run_dir, identity, "release")


def release_run_lock_for_token(run_dir: Path | None, token: str | None) -> None:
    if run_dir is None or token is None:
        return
    release_run_lock(RunLock(run_dir=run_dir, token=token))


def _run_lock_identity(payload: dict[str, Any] | None) -> tuple[int, str, Any] | None:
    if payload is None:
        return None
    pid = payload.get("pid")
    token = payload.get("token")
    created_at = payload.get("created_at")
    if type(pid) is not int or pid <= 1:
        return None
    if not isinstance(token, str) or not token:
        return None
    if type(created_at) not in (int, float) or not math.isfinite(float(created_at)):
        return None
    return (pid, token, float(created_at))


def _restore_reclaimed_lock(reclaim_path: Path, lock_path: Path) -> None:
    try:
        os.link(reclaim_path, lock_path)
    except FileExistsError:
        pass
    except OSError:
        pass
    try:
        reclaim_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _read_lock_marker_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def cleanup_stale_terminal_run_lock(run_dir: Path, lock_token: str | None = None) -> bool:
    if run_lock_token_matches(run_dir, lock_token):
        return False
    state = _read_run_state(run_dir)
    if state is None or state.get("status") not in TERMINAL_RUN_STATUSES:
        return False
    payload = _read_run_lock(run_dir)
    identity = _run_lock_identity(payload)
    if identity is None:
        return False
    pid = identity[0]
    if is_process_running(pid):
        return False
    if _run_lock_identity(_read_run_lock(run_dir)) != identity:
        return False
    lock_path = run_lock_path(run_dir)
    reclaim_path = run_dir / f"{RUN_LOCK_FILE}.reclaim.{os.getpid()}.{secrets.token_hex(8)}"
    try:
        lock_path.replace(reclaim_path)
    except OSError:
        return False
    claimed_payload = _read_lock_marker_payload(reclaim_path)
    if _run_lock_identity(claimed_payload) != identity:
        _restore_reclaimed_lock(reclaim_path, lock_path)
        return False
    state = _read_run_state(run_dir)
    if state is None or state.get("status") not in TERMINAL_RUN_STATUSES:
        _restore_reclaimed_lock(reclaim_path, lock_path)
        return False
    if is_process_running(pid):
        _restore_reclaimed_lock(reclaim_path, lock_path)
        return False
    try:
        reclaim_path.unlink()
    except OSError:
        _restore_reclaimed_lock(reclaim_path, lock_path)
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


def _config_with_results_dir(config: AutoBenchConfig, results_dir: Path) -> AutoBenchConfig:
    return replace(config, run=replace(config.run, results_dir=results_dir))


def _resume_context_from_state(results_dir: Path, run_id: str,
                               state: dict[str, Any]) -> ResumeContext:
    resolved_results_dir = Path(results_dir)
    run_dir = resolved_results_dir / run_id
    status = state.get("status")
    if state.get("run_id") not in (None, run_id):
        raise ConfigError(f"state run_id mismatch: expected {run_id}, got {state.get('run_id')}")
    if status in ("starting", "running"):
        raise ConfigError(f"run is active and cannot be resumed: {status}")

    config = load_config(run_dir / "config.resolved.json")
    config = _config_with_results_dir(config, resolved_results_dir)
    cases = expand_cases(config, run_id=run_id)
    manifest_data = _read_json_object(run_dir / "manifest.json", "manifest")
    initial_manifest, pending, unknown = plan_resume_cases(
        run_id=run_id,
        cases=cases,
        manifest_data=manifest_data,
    )
    if status == "completed" and not pending:
        pass
    elif status in RESUMABLE_RUN_STATUSES:
        pass
    else:
        raise ConfigError(f"run status cannot be resumed: {status}")
    return ResumeContext(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        initial_manifest=initial_manifest,
        pending_cases=pending,
        unknown_manifest_cases=tuple(unknown),
    )


def load_resume_context(results_dir: Path, run_id: str) -> ResumeContext:
    _safe_name(run_id, "run_id")
    run_dir = Path(results_dir) / run_id
    state = _read_json_object(run_dir / "state.json", "state")
    return _resume_context_from_state(results_dir, run_id, state)


def load_resume_child_startup_context(results_dir: Path, run_id: str,
                                      lock_token: str | None) -> ResumeContext:
    _safe_name(run_id, "run_id")
    run_dir = Path(results_dir) / run_id
    try:
        state = _read_json_object(run_dir / "state.json", "state")
    except ConfigError:
        return load_resume_context(results_dir, run_id)
    if state.get("status") == "starting" and run_lock_token_matches(run_dir, lock_token):
        startup_state = _read_json_object(
            run_dir / RESUME_STARTUP_STATE_FILE,
            "resume startup state",
        )
        return _resume_context_from_state(results_dir, run_id, startup_state)
    return load_resume_context(results_dir, run_id)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, RemoteAuth):
        payload = {
            field_name: _jsonable(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
        if value.password is not None:
            payload["password"] = "***"
        return payload
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _jsonable(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    return value


def config_to_dict(config: AutoBenchConfig) -> dict[str, Any]:
    payload = _jsonable(config)
    if not config.serve_profiles:
        payload.pop("serve_profiles", None)
    if not config.topology_profiles:
        payload.pop("topology_profiles", None)
    return payload


def _case_ref(case: BenchmarkCase) -> dict[str, str]:
    return {
        "model": case.model.name,
        "serve_profile": case.serving_name,
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
    return (case.model.name, case.serving_name, case.bench_profile.name)


def _manifest_case_keys(manifest: Manifest) -> set[tuple[str, str, str]]:
    return {
        (str(row["model"]), str(row["serve_profile"]), str(row["bench_profile"]))
        for row in manifest.cases
    }


def _manifest_row_key(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    model = row.get("model")
    serve_profile = row.get("serve_profile")
    bench_profile = row.get("bench_profile")
    if not all(isinstance(value, str) for value in (model, serve_profile, bench_profile)):
        return None
    return (model, serve_profile, bench_profile)


def _copy_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in row.items():
        if not isinstance(key, str):
            raise ConfigError("manifest case row keys must be strings")
        copied[key] = value
    return copied


def plan_resume_cases(
    *,
    run_id: str,
    cases: tuple[BenchmarkCase, ...],
    manifest_data: Mapping[str, Any],
) -> tuple[Manifest, tuple[BenchmarkCase, ...], list[tuple[str, str, str]]]:
    rows = manifest_data.get("cases")
    if not isinstance(rows, list):
        raise ConfigError("manifest cases must be a list")
    if manifest_data.get("run_id") not in (None, run_id):
        raise ConfigError(
            f"manifest run_id mismatch: expected {run_id}, got {manifest_data.get('run_id')}"
        )

    full_keys = {_case_key(case) for case in cases}
    passed_rows: list[dict[str, Any]] = []
    passed_keys: set[tuple[str, str, str]] = set()
    unknown_keys: list[tuple[str, str, str]] = []

    for row in rows:
        if not isinstance(row, dict):
            raise ConfigError("manifest cases must contain objects")
        key = _manifest_row_key(row)
        if key is None:
            raise ConfigError("manifest case row is missing model/serve_profile/bench_profile")
        if key not in full_keys:
            unknown_keys.append(key)
            continue
        if row.get("status") == "passed":
            passed_keys.add(key)
            passed_rows.append(_copy_manifest_row(row))

    initial_manifest = Manifest(run_id=run_id, total=len(cases), cases=passed_rows)
    pending = tuple(case for case in cases if _case_key(case) not in passed_keys)
    return initial_manifest, pending, unknown_keys


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
        grouped.setdefault((case.model.name, case.serving_name), []).append(case)
    return grouped


def _reject_topology_profiles_until_runner_supported(config: AutoBenchConfig) -> None:
    if config.topology_profiles:
        raise ConfigError("topology_profiles are parsed but not runnable yet")


def _run_controller_dry_run(config: AutoBenchConfig, run_id: str) -> int:
    _reject_topology_profiles_until_runner_supported(config)
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
            print_cmd(build_serve_run_command(config, serve_case, serve_layout.run_dir))
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
                   lock_token: str | None = None,
                   initial_manifest: Manifest | None = None,
                   cases_to_run: tuple[BenchmarkCase, ...] | None = None) -> int:
    active_runner: Runner = runner or DockerRunner()
    _reject_topology_profiles_until_runner_supported(config)
    if dry_run:
        return _run_controller_dry_run(config, run_id)

    all_cases = expand_cases(config, run_id=run_id)
    cases = all_cases if cases_to_run is None else cases_to_run
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
    manifest = initial_manifest or Manifest(run_id=run_id, total=len(all_cases))

    network_owned = False
    exit_code = 0
    completed = len(manifest.cases)
    interrupted = False
    try:
        write_json_atomic(run_dir / "config.resolved.json", config_to_dict(config))
        validate_local_paths(config)

        if not cases:
            write_manifest(run_dir, manifest)
            write_state(run_dir, finished_state(run_id, manifest))
            return 0

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
            if not dry_run:
                write_vllm_cache_metadata(config, serve_case, serve_layout)
            serve_cmd = build_serve_run_command(config, serve_case, serve_layout.run_dir)
            started = False
            cleanup_container = False
            try:
                if dry_run:
                    print_cmd(serve_cmd)
                    ready = True
                else:
                    cleanup_container = True
                    remove_existing_vllm_container_if_owned(
                        active_runner,
                        serve_case,
                        serve_layout.run_dir,
                    )
                    start_result = active_runner.run(serve_cmd, check=False)
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
                            all_cases,
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
                        monitor = (
                            ResourceMonitor(
                                output_dir=layout.bench_dir,
                                interval_sec=config.run.resource_monitor.interval_sec,
                                enabled=True,
                                backend=config.run.resource_monitor.backend,
                            )
                            if config.run.resource_monitor.enabled
                            else None
                        )
                        try:
                            try:
                                if monitor is not None:
                                    try:
                                        monitor.start()
                                    except (StopRequested, KeyboardInterrupt):
                                        raise
                                    except Exception as exc:
                                        logger.warning(
                                            "resource monitor start failed: %s",
                                            exc,
                                        )
                                        monitor = None
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
                                if monitor is not None:
                                    resource_summary = None
                                    try:
                                        resource_summary = monitor.stop()
                                    except (StopRequested, KeyboardInterrupt):
                                        if bench_interrupted is None:
                                            raise
                                    except Exception as exc:
                                        logger.warning(
                                            "resource monitor stop failed: %s",
                                            exc,
                                        )
                                    if resource_summary is not None:
                                        try:
                                            append_summary_to_result_files(
                                                layout.bench_dir,
                                                resource_summary,
                                            )
                                        except (StopRequested, KeyboardInterrupt):
                                            if bench_interrupted is None:
                                                raise
                                        except Exception as exc:
                                            logger.warning(
                                                "resource monitor result merge failed: %s",
                                                exc,
                                            )
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

        if not dry_run and not interrupted:
            try:
                aggregate_compare(config, run_dir)
            except Exception as exc:
                logger.warning("结果对比聚合失败：%s", exc)

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


def build_detach_command(config_path: Path | None, run_id: str,
                         results_dir: Path,
                         lock_token: str | None = None,
                         command_name: str = "run") -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), command_name]
    if command_name == "run":
        if config_path is None:
            raise ValueError("run detached command requires config_path")
        cmd.extend(["--config", str(config_path)])
    elif command_name != "resume":
        raise ValueError(f"unsupported detached command: {command_name}")
    cmd.extend([
        "--run-id",
        run_id,
        "--child",
        "--results-dir",
        str(results_dir),
    ])
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


def _wait_for_process_exit(process: Any, timeout_sec: float) -> bool:
    try:
        process.wait(timeout=timeout_sec)
        return True
    except StopRequested:
        raise
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def terminate_detached_child(process: Any, timeout_sec: float = 5.0) -> bool:
    try:
        process.terminate()
    except StopRequested:
        raise
    except ProcessLookupError:
        return True
    except Exception:
        pass

    if _wait_for_process_exit(process, timeout_sec):
        return True

    try:
        process.kill()
    except StopRequested:
        raise
    except ProcessLookupError:
        return True
    except Exception:
        return False

    return _wait_for_process_exit(process, timeout_sec)


def start_detached(config_path: Path | None, config: AutoBenchConfig, run_id: str,
                   command_name: str = "run") -> int:
    _safe_name(run_id, "run_id")
    _reject_topology_profiles_until_runner_supported(config)
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

    def fail_start(error: str, *, release_lock: bool = True) -> int:
        try:
            write_state(run_dir, _detached_state(run_id, "failed", total, error=error))
            print(f"failed to start detached controller: {error}", file=sys.stderr)
            return 1
        finally:
            if release_lock and run_lock is not None:
                release_run_lock(run_lock)

    try:
        validate_local_paths(config)
    except (ConfigError, OSError) as exc:
        return fail_start(str(exc))

    if command_name == "resume":
        try:
            startup_state = _read_json_object(run_dir / "state.json", "state")
            write_json_atomic(run_dir / RESUME_STARTUP_STATE_FILE, startup_state)
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
        command_name=command_name,
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
            "config_path": str(config_path) if config_path is not None else None,
            "started_at": time.time(),
        })
        update_run_lock_owner(run_lock, process.pid)
    except OSError as exc:
        release_lock = True
        if process is not None:
            release_lock = terminate_detached_child(process)
        return fail_start(str(exc), release_lock=release_lock)
    print(f"run_id: {run_id}")
    print(f"controller_log: {log_path.resolve()}")
    print(
        "logs: "
        f"python3 {Path(__file__).resolve()} logs "
        f"--results-dir {config.run.results_dir.resolve()} "
        f"--run-id {run_id} --follow"
    )
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


def _safe_log_component(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if not SAFE_NAME_RE.fullmatch(value):
        return None
    if value in (".", ".."):
        return None
    return value


def current_bench_log_path(run_dir: Path) -> Path | None:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    current = state.get("current")
    if not isinstance(current, dict):
        return None
    model = _safe_log_component(current.get("model"))
    serve_profile = _safe_log_component(current.get("serve_profile"))
    bench_profile = _safe_log_component(current.get("bench_profile"))
    if not all([model, serve_profile, bench_profile]):
        return None
    log_path = run_dir / model / serve_profile / bench_profile / "bench.log"
    return log_path if log_path.exists() else None


def select_log_path(run_dir: Path, *, controller: bool = False) -> Path:
    if controller:
        return run_dir / "controller.log"
    return current_bench_log_path(run_dir) or (run_dir / "controller.log")


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
    if command[2] not in {"run", "resume"}:
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


def resume_run(results_dir: Path, run_id: str, *,
               runner: Runner | None = None,
               lock_token: str | None = None) -> int:
    context = load_resume_child_startup_context(results_dir, run_id, lock_token)
    if context.unknown_manifest_cases:
        print(
            f"warning: ignoring manifest cases not in resolved config: {context.unknown_manifest_cases}",
            file=sys.stderr,
        )
    if not context.pending_cases:
        print(f"nothing to resume: {run_id}")
        write_manifest(context.run_dir, context.initial_manifest)
        write_state(context.run_dir, finished_state(run_id, context.initial_manifest))
        return 0
    return run_controller(
        context.config,
        run_id=run_id,
        runner=runner,
        lock_token=lock_token,
        initial_manifest=context.initial_manifest,
        cases_to_run=context.pending_cases,
    )


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

    logs_parser = subparsers.add_parser("logs", help="show run log")
    logs_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    logs_parser.add_argument("--run-id", required=True)
    logs_parser.add_argument("-f", "--follow", action="store_true")
    logs_parser.add_argument("--controller", action="store_true")

    stop_parser = subparsers.add_parser("stop", help="stop detached controller")
    stop_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    stop_parser.add_argument("--run-id", required=True)

    resume_parser = subparsers.add_parser("resume", help="resume interrupted benchmark cases")
    resume_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--detach", action="store_true")
    resume_parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    resume_parser.add_argument("--lock-token", help=argparse.SUPPRESS)

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


def release_child_startup_lock(results_dir: Path | None, run_id: str | None,
                               lock_token: str | None) -> None:
    if results_dir is None or run_id is None:
        return
    release_run_lock_for_token(results_dir / run_id, lock_token)


def write_child_startup_state_best_effort(results_dir: Path | None, run_id: str | None,
                                          status: str, error: str) -> None:
    try:
        write_child_startup_state(results_dir, run_id, status, error)
    except OSError as exc:
        print(f"failed to write child startup state: {exc}", file=sys.stderr)


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
                try:
                    write_child_startup_state_best_effort(
                        args.results_dir,
                        args.run_id,
                        "interrupted",
                        error,
                    )
                finally:
                    release_child_startup_lock(args.results_dir, args.run_id, args.lock_token)
                print(error, file=sys.stderr)
                return 130
            except Exception as exc:
                error = str(exc)
                try:
                    write_child_startup_state_best_effort(
                        args.results_dir,
                        args.run_id,
                        "failed",
                        error,
                    )
                finally:
                    release_child_startup_lock(args.results_dir, args.run_id, args.lock_token)
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
        log_path = select_log_path(
            args.results_dir / args.run_id,
            controller=args.controller,
        )
        return follow_file(log_path) if args.follow else print_log(log_path)
    if args.command == "resume":
        if args.child:
            try:
                install_signal_handlers()
                return resume_run(
                    args.results_dir,
                    args.run_id,
                    lock_token=args.lock_token,
                )
            except StopRequested as exc:
                error = str(exc) or "stop requested"
                try:
                    write_child_startup_state_best_effort(
                        args.results_dir,
                        args.run_id,
                        "interrupted",
                        error,
                    )
                finally:
                    release_child_startup_lock(args.results_dir, args.run_id, args.lock_token)
                print(error, file=sys.stderr)
                return 130
            except Exception as exc:
                error = str(exc)
                try:
                    write_child_startup_state_best_effort(
                        args.results_dir,
                        args.run_id,
                        "failed",
                        error,
                    )
                finally:
                    release_child_startup_lock(args.results_dir, args.run_id, args.lock_token)
                print(error, file=sys.stderr)
                return 1
        try:
            context = load_resume_context(args.results_dir, args.run_id)
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not context.pending_cases:
            print(f"nothing to resume: {args.run_id}")
            write_manifest(context.run_dir, context.initial_manifest)
            write_state(context.run_dir, finished_state(args.run_id, context.initial_manifest))
            return 0
        if args.detach:
            return start_detached(
                None,
                context.config,
                args.run_id,
                command_name="resume",
            )
        install_signal_handlers()
        return run_controller(
            context.config,
            run_id=args.run_id,
            initial_manifest=context.initial_manifest,
            cases_to_run=context.pending_cases,
        )
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
