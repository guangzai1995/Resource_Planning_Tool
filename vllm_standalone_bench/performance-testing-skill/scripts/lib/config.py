import json
from pathlib import Path


SUPPORTED_TARGET_TYPES = {
    "generic-http",
    "openai-chat",
    "openai-completion",
    "openai-asr",
}


class ConfigError(ValueError):
    """Raised when a benchmark config is missing required data."""


class Section:
    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)

    def to_dict(self):
        return dict(self.__dict__)


class BenchmarkConfig:
    def __init__(self, *, source_path, package_root, run, target, dataset):
        self.source_path = source_path
        self.package_root = package_root
        self.run = run
        self.target = target
        self.dataset = dataset

    @property
    def request_count(self):
        return int(getattr(self.run, "requests", 1))

    @property
    def concurrency(self):
        return int(getattr(self.run, "concurrency", 1))

    @property
    def warmup_requests(self):
        return int(getattr(self.run, "warmup_requests", 0))

    @property
    def timeout_sec(self):
        return float(getattr(self.run, "request_timeout_sec", 30))

    @property
    def output_dir(self):
        return getattr(self.run, "output_dir")


def load_config(path):
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc

    _require_mapping(raw, "config")
    run_raw = _required_section(raw, "run")
    target_raw = _required_section(raw, "target")
    dataset_raw = _required_section(raw, "dataset")

    package_root = _infer_package_root(config_path)
    run = _build_run(run_raw, package_root)
    target = _build_target(target_raw)
    dataset = _build_dataset(dataset_raw, package_root, config_path.parent)
    return BenchmarkConfig(
        source_path=config_path,
        package_root=package_root,
        run=run,
        target=target,
        dataset=dataset,
    )


def config_to_dict(config):
    return {
        "source_path": str(config.source_path),
        "package_root": str(config.package_root),
        "run": _json_safe(config.run.to_dict()),
        "target": config.target.to_dict(),
        "dataset": {
            **config.dataset.to_dict(),
            "path": str(config.dataset.path),
        },
    }


def _infer_package_root(config_path):
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _required_section(raw, name):
    if name not in raw:
        raise ConfigError(f"Missing required section: {name}")
    section = raw[name]
    _require_mapping(section, name)
    return section


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a JSON object")


def _build_run(raw, package_root):
    name = _required_text(raw, "name", "run")
    requests = _positive_int(raw.get("requests", 1), "run.requests")
    concurrency = _positive_int(raw.get("concurrency", 1), "run.concurrency")
    warmup = _non_negative_int(raw.get("warmup_requests", 0), "run.warmup_requests")
    timeout = _positive_float(raw.get("request_timeout_sec", 30), "run.request_timeout_sec")
    output_dir = raw.get("output_dir", "reports")
    output_path = _resolve_path(output_dir, package_root, package_root)
    return Section(
        name=name,
        requests=requests,
        concurrency=concurrency,
        warmup_requests=warmup,
        request_timeout_sec=timeout,
        output_dir=output_path,
    )


def _build_target(raw):
    target_type = _required_text(raw, "type", "target")
    if target_type not in SUPPORTED_TARGET_TYPES:
        raise ConfigError(
            f"target.type must be one of {sorted(SUPPORTED_TARGET_TYPES)}; got {target_type}"
        )
    base_url = _required_text(raw, "base_url", "target").rstrip("/")
    endpoint = raw.get("endpoint", "")
    if endpoint and not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    method = raw.get("method", "POST").upper()
    headers = raw.get("headers", {})
    _require_mapping(headers, "target.headers")
    body_template = raw.get("body_template")
    if body_template is not None and not isinstance(body_template, dict):
        raise ConfigError("target.body_template must be a JSON object")
    return Section(
        type=target_type,
        base_url=base_url,
        endpoint=endpoint,
        method=method,
        headers=dict(headers),
        model=raw.get("model"),
        body_template=body_template,
        extra_body=raw.get("extra_body", {}),
    )


def _build_dataset(raw, package_root, config_dir):
    dataset_type = raw.get("type", "jsonl")
    path_value = _required_text(raw, "path", "dataset")
    path = _resolve_path(path_value, package_root, config_dir)
    return Section(type=dataset_type, path=path)


def _resolve_path(value, package_root, config_dir):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    package_candidate = (package_root / path).resolve()
    if package_candidate.exists() or path.parts[:1] in [("datasets",), ("reports",)]:
        return package_candidate
    return (config_dir / path).resolve()


def _required_text(raw, key, section):
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section}.{key} must be a non-empty string")
    return value.strip()


def _positive_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return parsed


def _non_negative_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return parsed


def _positive_float(value, name):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive number") from exc
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive number")
    return parsed


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
