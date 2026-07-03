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
    kv_port: int | None = None
    side_channel_port: int | None = None
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
class VllmPdConfig:
    connector: str
    proxy_kind: str = "builtin"
    p2p_send_type: str = "PUT_ASYNC"
    nccl_num_channels: int | None = None


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
    vllm_pd: VllmPdConfig | None = None
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
        if self.frontend.kind not in {"router", "sglang_router"}:
            raise _config_error(
                f"topology profile {self.name} frontend.kind must be router "
                "or sglang_router for sglang pd"
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
            if node.bootstrap_port is None:
                raise _config_error(
                    f"topology profile {self.name} prefill node {node.name} "
                    "bootstrap_port is required for sglang pd"
                )
            argv.extend([
                "--prefill",
                f"http://{host.address}:{node.port}",
                str(node.bootstrap_port),
            ])
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
        if self.vllm_pd is None:
            return self._build_legacy_vllm_pd_commands(config, case, run_dir)
        return self._build_structured_vllm_pd_commands(config, case, run_dir)

    def _build_legacy_vllm_pd_commands(
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

    def _build_structured_vllm_pd_commands(
        self,
        config: Any,
        case: Any,
        run_dir: os.PathLike[str] | str,
    ) -> dict[str, RoleCommand]:
        image = self._worker_image(config)
        commands: dict[str, RoleCommand] = {}
        for node in self.prefill:
            commands[node.name] = self._build_vllm_worker_command(
                config,
                case,
                run_dir,
                node,
                kv_role="kv_producer",
                image=image,
                kv_config=self._structured_vllm_kv_config(
                    node,
                    kv_role="kv_producer",
                ),
                extra_env=self._structured_vllm_node_env(node),
            )
        for node in self.decode:
            commands[node.name] = self._build_vllm_worker_command(
                config,
                case,
                run_dir,
                node,
                kv_role="kv_consumer",
                image=image,
                kv_config=self._structured_vllm_kv_config(
                    node,
                    kv_role="kv_consumer",
                ),
                extra_env=self._structured_vllm_node_env(node),
            )
        commands[self.frontend.host] = self._build_builtin_vllm_pd_proxy_command(
            config,
            case,
            run_dir,
        )
        return commands

    def _structured_vllm_kv_config(
        self,
        node: TopologyNode,
        *,
        kv_role: str,
    ) -> dict[str, Any]:
        if self.vllm_pd is None:
            raise _config_error(
                f"topology profile {self.name} vllm_pd is required"
            )
        if self.vllm_pd.connector == "p2p_nccl":
            extra: dict[str, Any] = {
                "http_port": node.port,
                "send_type": self.vllm_pd.p2p_send_type,
            }
            if self.vllm_pd.nccl_num_channels is not None:
                extra["nccl_num_channels"] = self.vllm_pd.nccl_num_channels
            return {
                "kv_connector": "P2pNcclConnector",
                "kv_role": kv_role,
                "kv_port": node.kv_port,
                "kv_connector_extra_config": extra,
            }
        return {
            "kv_connector": "NixlConnector",
            "kv_role": kv_role,
        }

    def _structured_vllm_node_env(self, node: TopologyNode) -> Mapping[str, str]:
        if self.vllm_pd is None or self.vllm_pd.connector != "nixl":
            return types.MappingProxyType({})
        host = self.hosts[node.host]
        return {
            "VLLM_NIXL_SIDE_CHANNEL_HOST": host.address,
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(node.side_channel_port),
        }

    def _build_vllm_worker_command(
        self,
        config: Any,
        case: Any,
        run_dir: os.PathLike[str] | str,
        node: TopologyNode,
        *,
        kv_role: str,
        image: str,
        kv_rank: int | None = None,
        kv_parallel_size: int | None = None,
        kv_config: Mapping[str, Any] | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> RoleCommand:
        if kv_config is None:
            if kv_rank is None or kv_parallel_size is None:
                raise _config_error(
                    f"topology profile {self.name} kv_rank and "
                    "kv_parallel_size are required for legacy vllm pd"
                )
            kv_config_text = self._render_kv_transfer_config(
                case,
                node,
                kv_role=kv_role,
                kv_rank=kv_rank,
                kv_parallel_size=kv_parallel_size,
            )
        else:
            kv_config_text = json.dumps(
                kv_config,
                separators=(",", ":"),
                ensure_ascii=True,
            )
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
        self._append_env_and_volumes(argv, node, extra_env=extra_env)
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
            kv_config_text,
        ])
        argv.extend(node.args)
        return _role_command(
            role_name=node.name,
            host_name=node.host,
            container_name=_container_name(case, self.name, node.name),
            argv=argv,
        )

    def _build_builtin_vllm_pd_proxy_command(
        self,
        config: Any,
        case: Any,
        run_dir: os.PathLike[str] | str,
    ) -> RoleCommand:
        if self.frontend.kind != "builtin":
            raise _config_error(
                f"topology profile {self.name} frontend.kind must be builtin "
                "for structured vllm pd"
            )
        if self.vllm_pd is None:
            raise _config_error(
                f"topology profile {self.name} vllm_pd is required"
            )
        image = self.frontend.image or config.run.bench_image
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
        argv.extend([
            "--entrypoint",
            "python",
            image,
            "-m",
            "vllm_bench.pd_proxy",
            "--connector",
            self.vllm_pd.connector,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.frontend.port),
        ])
        for node in self.prefill:
            argv.extend(["--prefill", self._vllm_pd_proxy_endpoint_json(node)])
        for node in self.decode:
            argv.extend(["--decode", self._vllm_pd_proxy_endpoint_json(node)])
        argv.extend(self.frontend.args)
        return _role_command(
            role_name=self.frontend.host,
            host_name=self.frontend.host,
            container_name=_container_name(case, self.name, self.frontend.host),
            argv=argv,
        )

    def _vllm_pd_proxy_endpoint_json(self, node: TopologyNode) -> str:
        if self.vllm_pd is None:
            raise _config_error(
                f"topology profile {self.name} vllm_pd is required"
            )
        host = self.hosts[node.host]
        payload: dict[str, Any] = {
            "name": node.name,
            "url": f"http://{host.address}:{node.port}",
        }
        if self.vllm_pd.connector == "p2p_nccl":
            payload["kv_address"] = f"{host.address}:{node.kv_port}"
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)

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
        *,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        env = dict(self.env)
        if node is not None:
            env.update(node.env)
        if extra_env:
            env.update(extra_env)
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
        vllm_pd = _parse_vllm_pd_config(
            profile.get("vllm_pd"),
            f"{path}.vllm_pd",
            error,
        )
        _validate_role_names_unique(path, name, prefill, decode, frontend, error)
        _validate_topology_profile(
            path,
            name,
            engine,
            prefill,
            decode,
            frontend,
            vllm_pd,
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
            vllm_pd=vllm_pd,
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


def _parse_vllm_pd_config(
    value: Any,
    path: str,
    error: ErrorFactory,
) -> VllmPdConfig | None:
    if value is None:
        return None
    raw = _mapping(value, path, error)
    allowed = {
        "connector",
        "proxy",
        "proxy_kind",
        "p2p_send_type",
        "nccl_num_channels",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise error(f"{path} contains unsupported keys: {', '.join(unknown)}")
    connector = _string(
        _required(raw, "connector", f"{path}.connector", error),
        f"{path}.connector",
        error,
    )
    if connector not in {"p2p_nccl", "nixl"}:
        raise error(f"{path}.connector must be one of p2p_nccl, nixl")
    proxy = _mapping(raw.get("proxy", {"kind": "builtin"}), f"{path}.proxy", error)
    proxy_unknown = sorted(set(proxy) - {"kind"})
    if proxy_unknown:
        raise error(
            f"{path}.proxy contains unsupported keys: "
            + ", ".join(proxy_unknown)
        )
    proxy_kind = _string(proxy.get("kind", "builtin"), f"{path}.proxy.kind", error)
    if "proxy_kind" in raw:
        normalized_proxy_kind = _string(
            raw["proxy_kind"],
            f"{path}.proxy_kind",
            error,
        )
        if "proxy" in raw and normalized_proxy_kind != proxy_kind:
            raise error(f"{path}.proxy_kind must match {path}.proxy.kind")
        proxy_kind = normalized_proxy_kind
    if proxy_kind != "builtin":
        raise error(f"{path}.proxy.kind only supports builtin")
    p2p_send_type = _string(
        raw.get("p2p_send_type", "PUT_ASYNC"),
        f"{path}.p2p_send_type",
        error,
    )
    return VllmPdConfig(
        connector=connector,
        proxy_kind=proxy_kind,
        p2p_send_type=p2p_send_type,
        nccl_num_channels=_optional_positive_int(
            raw.get("nccl_num_channels"),
            f"{path}.nccl_num_channels",
            error,
        ),
    )


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


def _validate_role_names_unique(
    path: str,
    profile_name: str,
    prefill: tuple[TopologyNode, ...],
    decode: tuple[TopologyNode, ...],
    frontend: TopologyFrontend,
    error: ErrorFactory,
) -> None:
    role_names = [node.name for node in prefill]
    role_names.extend(node.name for node in decode)
    role_names.append(frontend.host)
    duplicates = sorted({
        name for name in role_names if role_names.count(name) > 1
    })
    if duplicates:
        raise error(
            f"topology profile {path} ({profile_name}) role names must be "
            "unique across prefill/decode node.name and frontend.host; "
            "duplicate role name: "
            + ", ".join(duplicates)
        )


def _validate_topology_profile(
    path: str,
    profile_name: str,
    engine: str,
    prefill: tuple[TopologyNode, ...],
    decode: tuple[TopologyNode, ...],
    frontend: TopologyFrontend,
    vllm_pd: VllmPdConfig | None,
    error: ErrorFactory,
) -> None:
    if engine == "sglang":
        for node in prefill:
            if node.bootstrap_port is None:
                raise error(
                    f"{path} ({profile_name}) prefill node {node.name} "
                    "bootstrap_port is required for sglang pd"
                )
        return
    if engine != "vllm" or vllm_pd is None:
        return
    if frontend.kind != "builtin":
        raise error(
            f"{path} ({profile_name}) frontend.kind must be builtin when "
            "structured vllm_pd is used"
        )
    if vllm_pd.connector == "p2p_nccl":
        for node in (*prefill, *decode):
            if node.kv_port is None:
                raise error(
                    f"{path} ({profile_name}) node {node.name} kv_port is "
                    "required for p2p_nccl"
                )
        return
    if vllm_pd.connector == "nixl":
        for node in (*prefill, *decode):
            if node.side_channel_port is None:
                raise error(
                    f"{path} ({profile_name}) node {node.name} "
                    "side_channel_port is required for nixl"
                )


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
            kv_port=_optional_positive_int(
                node.get("kv_port"),
                f"{node_path}.kv_port",
                error,
            ),
            side_channel_port=_optional_positive_int(
                node.get("side_channel_port"),
                f"{node_path}.side_channel_port",
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
