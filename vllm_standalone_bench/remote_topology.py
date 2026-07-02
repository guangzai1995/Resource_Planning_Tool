from __future__ import annotations

import os
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Mapping


@dataclass(frozen=True)
class RemoteAuth:
    type: str
    key_path: str | None = None
    env: str | None = None
    password: str | None = None


@dataclass(frozen=True)
class RemoteHost:
    name: str
    address: str
    ssh_user: str
    auth: RemoteAuth


@dataclass(frozen=True)
class TopologyNode:
    name: str
    host: str
    port: int
    bootstrap_port: int | None = None
    gpus: str = "all"
    args: tuple[str, ...] = field(default_factory=tuple)
    env: Mapping[str, str] = field(
        default_factory=lambda: types.MappingProxyType({})
    )
    volumes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TopologyFrontend:
    kind: str
    host: str
    port: int
    image: str | None = None
    command: tuple[str, ...] = field(default_factory=tuple)
    args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TopologyProfile:
    name: str
    engine: str
    mode: str
    provider: str
    hosts: Mapping[str, RemoteHost]
    prefill: tuple[TopologyNode, ...]
    decode: tuple[TopologyNode, ...]
    frontend: TopologyFrontend
    image: str | None = None
    router_image: str | None = None
    network: str = "host"
    transfer_backend: str | None = None
    disaggregation_ib_device: str | None = None
    env: Mapping[str, str] = field(
        default_factory=lambda: types.MappingProxyType({})
    )
    volumes: tuple[str, ...] = field(default_factory=tuple)


ErrorFactory = Callable[[str], Exception]
SafeName = Callable[[Any, str], str]


def parse_topology_profiles(
    data: Mapping[str, Any],
    *,
    error: ErrorFactory,
    safe_name: SafeName,
    supported_engines: Collection[str],
    environ: Mapping[str, str] | None = None,
) -> tuple[TopologyProfile, ...]:
    raw_profiles = data.get("topology_profiles")
    if raw_profiles is None:
        return ()
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise error("topology_profiles must be a non-empty list")

    env = os.environ if environ is None else environ
    parsed: list[TopologyProfile] = []
    for index, item in enumerate(raw_profiles):
        path = f"topology_profiles[{index}]"
        profile = _mapping(item, path, error)
        name = safe_name(_required(profile, "name", f"{path}.name", error), f"{path}.name")
        engine = _string(_required(profile, "engine", f"{path}.engine", error), f"{path}.engine", error)
        if engine not in supported_engines:
            raise error(
                f"{path}.engine must be one of {tuple(supported_engines)}, got {engine!r}"
            )
        mode = _string(_required(profile, "mode", f"{path}.mode", error), f"{path}.mode", error)
        if mode != "pd":
            raise error(f"{path}.mode only supports 'pd', got {mode!r}")
        provider = _string(
            _required(profile, "provider", f"{path}.provider", error),
            f"{path}.provider",
            error,
        )
        if provider != "ssh_docker":
            raise error(f"{path}.provider only supports 'ssh_docker', got {provider!r}")

        hosts = _parse_hosts(
            _required(profile, "hosts", f"{path}.hosts", error),
            f"{path}.hosts",
            error,
            safe_name,
            env,
        )
        prefill = _parse_nodes(
            _required(profile, "prefill", f"{path}.prefill", error),
            f"{path}.prefill",
            hosts,
            error,
            safe_name,
        )
        decode = _parse_nodes(
            _required(profile, "decode", f"{path}.decode", error),
            f"{path}.decode",
            hosts,
            error,
            safe_name,
        )
        frontend = _parse_frontend(
            _required(profile, "frontend", f"{path}.frontend", error),
            f"{path}.frontend",
            hosts,
            error,
        )
        parsed.append(TopologyProfile(
            name=name,
            engine=engine,
            mode=mode,
            provider=provider,
            hosts=types.MappingProxyType(hosts),
            prefill=prefill,
            decode=decode,
            frontend=frontend,
            image=_optional_string(profile.get("image"), f"{path}.image", error),
            router_image=_optional_string(
                profile.get("router_image"),
                f"{path}.router_image",
                error,
            ),
            network=_string(profile.get("network", "host"), f"{path}.network", error),
            transfer_backend=_optional_string(
                profile.get("transfer_backend"),
                f"{path}.transfer_backend",
                error,
            ),
            disaggregation_ib_device=_optional_string(
                profile.get("disaggregation_ib_device"),
                f"{path}.disaggregation_ib_device",
                error,
            ),
            env=_string_mapping(profile.get("env"), f"{path}.env", error),
            volumes=_string_tuple(profile.get("volumes", []), f"{path}.volumes", error),
        ))
    return tuple(parsed)


def _mapping(value: Any, path: str, error: ErrorFactory) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise error(f"{path} must be an object")
    return value


def _required(
    data: Mapping[str, Any],
    key: str,
    path: str,
    error: ErrorFactory,
) -> Any:
    if key not in data:
        raise error(f"{path} is required")
    return data[key]


def _string(value: Any, path: str, error: ErrorFactory) -> str:
    if not isinstance(value, str) or not value:
        raise error(f"{path} must be a non-empty string")
    return value


def _optional_string(value: Any, path: str, error: ErrorFactory) -> str | None:
    if value is None:
        return None
    return _string(value, path, error)


def _positive_int(value: Any, path: str, error: ErrorFactory) -> int:
    if type(value) is not int or value <= 0:
        raise error(f"{path} must be a positive integer")
    return value


def _optional_positive_int(value: Any, path: str, error: ErrorFactory) -> int | None:
    if value is None:
        return None
    return _positive_int(value, path, error)


def _string_tuple(value: Any, path: str, error: ErrorFactory) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise error(f"{path} must be a list of strings")
    return tuple(value)


def _string_mapping(value: Any, path: str, error: ErrorFactory) -> Mapping[str, str]:
    if value is None:
        return types.MappingProxyType({})
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        raise error(f"{path} must be an object mapping string to string")
    return types.MappingProxyType(dict(value))


def _parse_auth(
    value: Any,
    path: str,
    error: ErrorFactory,
    environ: Mapping[str, str],
) -> RemoteAuth:
    auth = _mapping(value, path, error)
    auth_type = _string(_required(auth, "type", f"{path}.type", error), f"{path}.type", error)
    if auth_type not in {"key", "password_env", "password"}:
        raise error(f"{path}.type must be one of key, password_env, password")
    key_path = _optional_string(auth.get("key_path"), f"{path}.key_path", error)
    env = _optional_string(auth.get("env"), f"{path}.env", error)
    password = _optional_string(auth.get("password"), f"{path}.password", error)
    if auth_type == "password_env":
        if env is None:
            raise error(f"{path}.env is required when auth.type is password_env")
        if env not in environ:
            raise error(f"{path}.env is not set in the local environment: {env}")
    if auth_type == "password" and password is None:
        raise error(f"{path}.password is required when auth.type is password")
    return RemoteAuth(
        type=auth_type,
        key_path=key_path,
        env=env,
        password=password,
    )


def _parse_hosts(
    value: Any,
    path: str,
    error: ErrorFactory,
    safe_name: SafeName,
    environ: Mapping[str, str],
) -> dict[str, RemoteHost]:
    raw_hosts = _mapping(value, path, error)
    if not raw_hosts:
        raise error(f"{path} must be a non-empty object")
    hosts: dict[str, RemoteHost] = {}
    for raw_name, raw_host in raw_hosts.items():
        name = safe_name(raw_name, f"{path} host name")
        host = _mapping(raw_host, f"{path}.{name}", error)
        hosts[name] = RemoteHost(
            name=name,
            address=_string(
                _required(host, "address", f"{path}.{name}.address", error),
                f"{path}.{name}.address",
                error,
            ),
            ssh_user=_string(
                _required(host, "ssh_user", f"{path}.{name}.ssh_user", error),
                f"{path}.{name}.ssh_user",
                error,
            ),
            auth=_parse_auth(
                _required(host, "auth", f"{path}.{name}.auth", error),
                f"{path}.{name}.auth",
                error,
                environ,
            ),
        )
    return hosts


def _check_host_ref(host: str, path: str, hosts: Mapping[str, RemoteHost],
                    error: ErrorFactory) -> str:
    if host not in hosts:
        raise error(f"{path}.host references unknown host: {host}")
    return host


def _parse_nodes(
    value: Any,
    path: str,
    hosts: Mapping[str, RemoteHost],
    error: ErrorFactory,
    safe_name: SafeName,
) -> tuple[TopologyNode, ...]:
    if not isinstance(value, list) or not value:
        raise error(f"{path} must be a non-empty list")
    parsed: list[TopologyNode] = []
    for index, item in enumerate(value):
        node_path = f"{path}[{index}]"
        node = _mapping(item, node_path, error)
        host = _string(_required(node, "host", f"{node_path}.host", error), f"{node_path}.host", error)
        parsed.append(TopologyNode(
            name=safe_name(_required(node, "name", f"{node_path}.name", error), f"{node_path}.name"),
            host=_check_host_ref(host, node_path, hosts, error),
            port=_positive_int(
                _required(node, "port", f"{node_path}.port", error),
                f"{node_path}.port",
                error,
            ),
            bootstrap_port=_optional_positive_int(
                node.get("bootstrap_port"),
                f"{node_path}.bootstrap_port",
                error,
            ),
            gpus=_string(node.get("gpus", "all"), f"{node_path}.gpus", error),
            args=_string_tuple(node.get("args", []), f"{node_path}.args", error),
            env=_string_mapping(node.get("env"), f"{node_path}.env", error),
            volumes=_string_tuple(node.get("volumes", []), f"{node_path}.volumes", error),
        ))
    return tuple(parsed)


def _parse_frontend(
    value: Any,
    path: str,
    hosts: Mapping[str, RemoteHost],
    error: ErrorFactory,
) -> TopologyFrontend:
    frontend = _mapping(value, path, error)
    host = _string(_required(frontend, "host", f"{path}.host", error), f"{path}.host", error)
    return TopologyFrontend(
        kind=_string(_required(frontend, "kind", f"{path}.kind", error), f"{path}.kind", error),
        host=_check_host_ref(host, path, hosts, error),
        port=_positive_int(
            _required(frontend, "port", f"{path}.port", error),
            f"{path}.port",
            error,
        ),
        image=_optional_string(frontend.get("image"), f"{path}.image", error),
        command=_string_tuple(frontend.get("command", []), f"{path}.command", error),
        args=_string_tuple(frontend.get("args", []), f"{path}.args", error),
    )
