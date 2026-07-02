from __future__ import annotations

import json
import os
import string
import types
from dataclasses import dataclass, field
from pathlib import Path
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
class RoleCommand:
    role_name: str
    host_name: str
    container_name: str
    argv: tuple[str, ...]
    masked_argv: tuple[str, ...]


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
    kv_transfer_config_template: Mapping[str, Any] = field(
        default_factory=lambda: types.MappingProxyType({})
    )
    disaggregation_ib_device: str | None = None
    env: Mapping[str, str] = field(
        default_factory=lambda: types.MappingProxyType({})
    )
    volumes: tuple[str, ...] = field(default_factory=tuple)

    def build_commands(
        self,
        config: Any,
        case: Any,
        run_dir: os.PathLike[str] | str,
    ) -> dict[str, RoleCommand]:
        if self.mode != "pd":
            raise _config_error(f"topology profile {self.name} only supports pd mode")
        if self.provider != "ssh_docker":
            raise _config_error(
                f"topology profile {self.name} only supports ssh_docker provider"
            )
        if self.engine == "sglang":
            return self._build_sglang_pd_commands(config, case, run_dir)
        if self.engine == "vllm":
            return self._build_vllm_pd_commands(config, case, run_dir)
        raise _config_error(
            f"topology profile {self.name} has unsupported engine: {self.engine}"
        )

    def _build_sglang_pd_commands(
        self,
        config: Any,
        case: Any,
        run_dir: os.PathLike[str] | str,
    ) -> dict[str, RoleCommand]:
        image = self._worker_image(config)
        commands: dict[str, RoleCommand] = {}
        for node in self.prefill:
            commands[node.name] = self._build_sglang_worker_command(
                config,
                case,
                run_dir,
                node,
                role="prefill",
                image=image,
            )
        for node in self.decode:
            commands[node.name] = self._build_sglang_worker_command(
                config,
                case,
                run_dir,
                node,
                role="decode",
                image=image,
            )
        commands[self.frontend.host] = self._build_sglang_router_command(
            case,
            run_dir,
        )
        return commands

    def _build_sglang_worker_command(
        self,
        config: Any,
        case: Any,
        run_dir: os.PathLike[str] | str,
        node: TopologyNode,
        *,
        role: str,
        image: str,
    ) -> RoleCommand:
        argv = self._docker_run_base(case, run_dir, node.name, node.host, role)
        argv.extend([
            "--gpus",
            _docker_gpus(node.gpus),
            "--network",
            self.network,
            "-v",
            f"{config.mounts.models}:/models:ro",
        ])
        self._append_env_and_volumes(argv, node)
        argv.extend([
            "--entrypoint",
            "python3",
            image,
            "-m",
            "sglang.launch_server",
            "--model-path",
            case.model.model_path,
            "--served-model-name",
            case.api_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(node.port),
            "--disaggregation-mode",
            role,
        ])
        if self.transfer_backend:
            argv.extend(["--disaggregation-transfer-backend", self.transfer_backend])
        if node.bootstrap_port is not None:
            argv.extend([
                "--disaggregation-bootstrap-port",
                str(node.bootstrap_port),
            ])
        if self.disaggregation_ib_device:
            argv.extend([
                "--disaggregation-ib-device",
                self.disaggregation_ib_device,
            ])
        argv.extend(node.args)
        return _role_command(
            role_name=node.name,
            host_name=node.host,
            container_name=_container_name(case, self.name, node.name),
            argv=argv,
        )

    def _build_sglang_router_command(
        self,
        case: Any,
        run_dir: os.PathLike[str] | str,
    ) -> RoleCommand:
        if self.frontend.kind != "router":
            raise _config_error(
                f"topology profile {self.name} frontend.kind must be router "
                "for sglang pd"
            )
        image = self.router_image or self._required_image()
        argv = self._docker_run_base(
            case,
            run_dir,
            self.frontend.host,
            self.frontend.host,
            "router",
        )
        argv.extend([
            "--network",
            self.network,
        ])
        self._append_env_and_volumes(argv, None)
        argv.extend([
            "--entrypoint",
            "python3",
            image,
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
        ])
        for node in self.prefill:
            host = self.hosts[node.host]
            argv.extend(["--prefill", f"http://{host.address}:{node.port}"])
        for node in self.decode:
            host = self.hosts[node.host]
            argv.extend(["--decode", f"http://{host.address}:{node.port}"])
        argv.extend([
            "--host",
            "0.0.0.0",
            "--port",
            str(self.frontend.port),
        ])
        argv.extend(self.frontend.args)
        return _role_command(
            role_name=self.frontend.host,
            host_name=self.frontend.host,
            container_name=_container_name(case, self.name, self.frontend.host),
            argv=argv,
        )

    def _build_vllm_pd_commands(
        self,
        config: Any,
        case: Any,
        run_dir: os.PathLike[str] | str,
    ) -> dict[str, RoleCommand]:
        image = self._worker_image(config)
        total_workers = len(self.prefill) + len(self.decode)
        commands: dict[str, RoleCommand] = {}
        rank = 0
        for node in self.prefill:
            commands[node.name] = self._build_vllm_worker_command(
                config,
                case,
                run_dir,
                node,
                kv_role="kv_producer",
                kv_rank=rank,
                kv_parallel_size=total_workers,
                image=image,
            )
            rank += 1
        for node in self.decode:
            commands[node.name] = self._build_vllm_worker_command(
                config,
                case,
                run_dir,
                node,
                kv_role="kv_consumer",
                kv_rank=rank,
                kv_parallel_size=total_workers,
                image=image,
            )
            rank += 1
        commands[self.frontend.host] = self._build_external_frontend_command(
            case,
            run_dir,
        )
        return commands

    def _build_vllm_worker_command(
        self,
        config: Any,
        case: Any,
        run_dir: os.PathLike[str] | str,
        node: TopologyNode,
        *,
        kv_role: str,
        kv_rank: int,
        kv_parallel_size: int,
        image: str,
    ) -> RoleCommand:
        argv = self._docker_run_base(
            case,
            run_dir,
            node.name,
            node.host,
            "prefill" if kv_role == "kv_producer" else "decode",
        )
        argv.extend([
            "--gpus",
            _docker_gpus(node.gpus),
            "--network",
            self.network,
            "-v",
            f"{config.mounts.models}:/models:ro",
        ])
        self._append_env_and_volumes(argv, node)
        argv.extend([
            "--entrypoint",
            "vllm",
            image,
            "serve",
            case.model.model_path,
            "--served-model-name",
            case.api_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(node.port),
            "--kv-transfer-config",
            self._render_kv_transfer_config(
                case,
                node,
                kv_role=kv_role,
                kv_rank=kv_rank,
                kv_parallel_size=kv_parallel_size,
            ),
        ])
        argv.extend(node.args)
        return _role_command(
            role_name=node.name,
            host_name=node.host,
            container_name=_container_name(case, self.name, node.name),
            argv=argv,
        )

    def _build_external_frontend_command(
        self,
        case: Any,
        run_dir: os.PathLike[str] | str,
    ) -> RoleCommand:
        if self.frontend.kind != "external":
            raise _config_error(
                f"topology profile {self.name} frontend.kind must be external "
                "for vllm pd"
            )
        if not self.frontend.image:
            raise _config_error(
                f"topology profile {self.name} frontend.image is required "
                "for external frontend"
            )
        if not self.frontend.command:
            raise _config_error(
                f"topology profile {self.name} frontend.command is required "
                "for external frontend"
            )
        argv = self._docker_run_base(
            case,
            run_dir,
            self.frontend.host,
            self.frontend.host,
            "frontend",
        )
        argv.extend([
            "--network",
            self.network,
        ])
        self._append_env_and_volumes(argv, None)
        substitutions = {
            "frontend_port": str(self.frontend.port),
            "run_id": case.run_id,
        }
        argv.append(self.frontend.image)
        argv.extend(
            _render_template_string(value, substitutions)
            for value in self.frontend.command
        )
        argv.extend(self.frontend.args)
        return _role_command(
            role_name=self.frontend.host,
            host_name=self.frontend.host,
            container_name=_container_name(case, self.name, self.frontend.host),
            argv=argv,
        )

    def _render_kv_transfer_config(
        self,
        case: Any,
        node: TopologyNode,
        *,
        kv_role: str,
        kv_rank: int,
        kv_parallel_size: int,
    ) -> str:
        if not self.kv_transfer_config_template:
            raise _config_error(
                f"topology profile {self.name} kv_transfer_config_template "
                "is required for vllm pd"
            )
        host = self.hosts[node.host]
        substitutions = {
            "kv_role": kv_role,
            "kv_rank": str(kv_rank),
            "kv_parallel_size": str(kv_parallel_size),
            "node_name": node.name,
            "node_address": host.address,
            "node_port": str(node.port),
            "run_id": case.run_id,
        }
        rendered = _render_template_value(
            self.kv_transfer_config_template,
            substitutions,
        )
        return json.dumps(rendered, separators=(",", ":"), ensure_ascii=True)

    def _worker_image(self, config: Any) -> str:
        if self.image:
            return self.image
        image = getattr(config.run, "images", {}).get(self.engine)
        if image:
            return image
        raise _config_error(
            f"topology profile {self.name} image is required for {self.engine}"
        )

    def _required_image(self) -> str:
        if not self.image:
            raise _config_error(f"topology profile {self.name} image is required")
        return self.image

    def _docker_run_base(
        self,
        case: Any,
        run_dir: os.PathLike[str] | str,
        role_name: str,
        host_name: str,
        role: str,
    ) -> list[str]:
        container_name = _container_name(case, self.name, role_name)
        return [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            *self._label_args(case, run_dir, role, role_name),
        ]

    def _label_args(
        self,
        case: Any,
        run_dir: os.PathLike[str] | str,
        role: str,
        role_name: str,
    ) -> list[str]:
        resolved_run_dir = Path(run_dir).resolve()
        labels = {
            "vllm_auto_bench.managed": "true",
            "vllm_auto_bench.run_id": case.run_id,
            "vllm_auto_bench.run_dir": str(resolved_run_dir),
            "vllm_auto_bench.model": case.model.name,
            "vllm_auto_bench.topology_profile": self.name,
            "vllm_auto_bench.role": role,
            "vllm_auto_bench.role_name": role_name,
        }
        args: list[str] = []
        for key, value in labels.items():
            args.extend(["--label", f"{key}={value}"])
        return args

    def _append_env_and_volumes(
        self,
        argv: list[str],
        node: TopologyNode | None,
    ) -> None:
        env = dict(self.env)
        if node is not None:
            env.update(node.env)
        for name, value in env.items():
            argv.extend(["-e", f"{name}={value}"])
        for volume in self.volumes:
            argv.extend(["-v", volume])
        if node is not None:
            for volume in node.volumes:
                argv.extend(["-v", volume])


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
            kv_transfer_config_template=_json_template_mapping(
                profile.get("kv_transfer_config_template"),
                f"{path}.kv_transfer_config_template",
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


def _json_template_mapping(
    value: Any,
    path: str,
    error: ErrorFactory,
) -> Mapping[str, Any]:
    if value is None:
        return types.MappingProxyType({})
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise error(f"{path} must be an object")
    _validate_json_template_value(value, path, error)
    return types.MappingProxyType(dict(value))


def _validate_json_template_value(
    value: Any,
    path: str,
    error: ErrorFactory,
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise error(f"{path} keys must be strings")
            _validate_json_template_value(item, f"{path}.{key}", error)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_template_value(item, f"{path}[{index}]", error)
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    raise error(f"{path} must contain JSON-compatible values")


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


def _container_name(case: Any, topology_name: str, role_name: str) -> str:
    return (
        f"bench-pd-{case.run_id}-{case.model.name}-{topology_name}-{role_name}"
    )


def _docker_gpus(value: str) -> str:
    if value == "all":
        return "all"
    return f"device={value}"


def _role_command(
    *,
    role_name: str,
    host_name: str,
    container_name: str,
    argv: list[str],
) -> RoleCommand:
    from remote_docker import mask_command

    return RoleCommand(
        role_name=role_name,
        host_name=host_name,
        container_name=container_name,
        argv=tuple(argv),
        masked_argv=tuple(mask_command(argv)),
    )


def _render_template_value(value: Any, substitutions: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _render_template_string(value, substitutions)
    if isinstance(value, dict):
        return {
            key: _render_template_value(item, substitutions)
            for key, item in value.items()
        }
    if isinstance(value, Mapping):
        return {
            str(key): _render_template_value(item, substitutions)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _render_template_value(item, substitutions)
            for item in value
        ]
    return value


def _render_template_string(value: str, substitutions: Mapping[str, str]) -> str:
    formatter = string.Formatter()
    for _literal, field_name, _format_spec, _conversion in formatter.parse(value):
        if field_name is None:
            continue
        if field_name not in substitutions:
            raise _config_error(
                f"unsupported template placeholder {{{field_name}}}"
            )
    return value.format(**substitutions)


def _config_error(message: str) -> Exception:
    from auto_bench import ConfigError

    return ConfigError(message)
