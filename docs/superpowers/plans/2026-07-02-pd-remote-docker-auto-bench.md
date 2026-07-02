# PD Remote Docker Auto Bench 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 `vllm_standalone_bench/auto_bench.py` 增加 SSH Docker 编排的 PD disaggregation benchmark，保留现有单机 `serve_profiles` 行为。

**架构：** 新增远程拓扑模块负责解析 `topology_profiles`、生成 SGLang/vLLM PD 命令并屏蔽敏感值；新增 SSH Docker runner 负责远程命令执行和 Docker 生命周期；`auto_bench.py` 只做矩阵展开、状态记录、bench-runner 调度和 legacy/topology 两条路径分派。远程资源监控复用 `resource_monitor.py` 的解析与汇总逻辑，通过 SSH 读取原始样本。

**技术栈：** Python dataclasses、subprocess `ssh`/`sshpass -e`、pytest fake runners、现有 Docker CLI 命令列表风格、现有 `ResourceMonitor`。

---

## 参考文档

- 规格：`docs/superpowers/specs/2026-07-02-pd-remote-docker-auto-bench-design.md`
- SGLang PD 文档：`https://docs.sglang.io/docs/advanced_features/pd_disaggregation`
- vLLM Disaggregated Prefilling 文档：`https://docs.vllm.ai/en/latest/features/disagg_prefill/`

## 文件结构

- 创建：`vllm_standalone_bench/remote_topology.py`
  - 拓扑 dataclass、配置解析、校验、敏感值 masking、SGLang/vLLM 命令生成。
- 创建：`vllm_standalone_bench/remote_docker.py`
  - `RemoteDockerRunner`、SSH 命令构造、远程 Docker helper、远程资源 reader。
- 修改：`vllm_standalone_bench/auto_bench.py`
  - `AutoBenchConfig`/`BenchmarkCase` 支持 `topology_profiles`，run loop 分派 legacy 和 topology 路径，manifest/status/resume/dry-run/bench command 兼容 topology case。
- 修改：`vllm_standalone_bench/resource_monitor.py`
  - 增加 host-prefixed resource summary merge，不改变现有无前缀 merge。
- 修改：`vllm_standalone_bench/bench_compare.py`
  - compare serving dimension 同时支持 `serve_profile` 和 `topology_profile`。
- 创建：`vllm_standalone_bench/configs/auto_bench.sglang_pd_remote.example.json`
  - 可 dry-run 的 SGLang 2P2D 远程拓扑示例。
- 测试：`vllm_standalone_bench/tests/test_remote_topology.py`
- 测试：`vllm_standalone_bench/tests/test_remote_docker.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_resource_monitor.py`
- 修改：`vllm_standalone_bench/tests/test_bench_compare.py`

## 任务 1：拓扑配置模型与 case 展开

**文件：**
- 创建：`vllm_standalone_bench/remote_topology.py`
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_remote_topology.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的配置解析测试**

在 `test_remote_topology.py` 添加：

```python
import json
from pathlib import Path

import pytest

import auto_bench as ab
from remote_topology import TopologyProfile
from test_auto_bench import minimal_config


def pd_topology_config(tmp_path):
    data = minimal_config(tmp_path)
    data.pop("serve_profiles")
    data["topology_profiles"] = [{
        "name": "sglang_pd_2p2d",
        "engine": "sglang",
        "mode": "pd",
        "provider": "ssh_docker",
        "transfer_backend": "mooncake",
        "network": "host",
        "image": "lmsysorg/sglang:latest",
        "router_image": "sglang-router:offline",
        "hosts": {
            "p1": {"address": "10.0.0.11", "ssh_user": "root", "auth": {"type": "key"}},
            "p2": {"address": "10.0.0.12", "ssh_user": "root", "auth": {"type": "key"}},
            "d1": {"address": "10.0.0.21", "ssh_user": "root", "auth": {"type": "key"}},
            "d2": {"address": "10.0.0.22", "ssh_user": "root", "auth": {"type": "key"}},
            "router": {"address": "10.0.0.30", "ssh_user": "root", "auth": {"type": "key"}},
        },
        "prefill": [
            {"name": "p1", "host": "p1", "port": 30000, "bootstrap_port": 12335, "gpus": "0,1"},
            {"name": "p2", "host": "p2", "port": 30000, "bootstrap_port": 12335, "gpus": "0,1"},
        ],
        "decode": [
            {"name": "d1", "host": "d1", "port": 30001, "bootstrap_port": 12335, "gpus": "0,1"},
            {"name": "d2", "host": "d2", "port": 30001, "bootstrap_port": 12335, "gpus": "0,1"},
        ],
        "frontend": {"kind": "sglang_router", "host": "router", "port": 8000},
    }]
    return data


def write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_config_accepts_topology_profiles_without_serve_profiles(tmp_path):
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    assert config.serve_profiles == ()
    assert len(config.topology_profiles) == 1
    topology = config.topology_profiles[0]
    assert isinstance(topology, TopologyProfile)
    assert topology.name == "sglang_pd_2p2d"
    assert [node.name for node in topology.prefill] == ["p1", "p2"]
    assert topology.frontend.host == "router"
```

在 `test_auto_bench.py` 添加：

```python
def test_expand_cases_uses_topology_profiles(tmp_path):
    from test_remote_topology import pd_topology_config, write_config
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    cases = ab.expand_cases(config, run_id="run123")
    assert len(cases) == 1
    assert cases[0].serve_profile is None
    assert cases[0].topology_profile.name == "sglang_pd_2p2d"
    assert cases[0].serving_name == "sglang_pd_2p2d"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_remote_topology.py::test_load_config_accepts_topology_profiles_without_serve_profiles vllm_standalone_bench/tests/test_auto_bench.py::test_expand_cases_uses_topology_profiles -q
```

预期：FAIL，报错包含 `topology_profiles` 或 `TopologyProfile` 未定义。

- [ ] **步骤 3：实现最小配置模型**

在 `remote_topology.py` 添加：

```python
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


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
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    volumes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopologyFrontend:
    kind: str
    host: str
    port: int
    image: str | None = None
    command: tuple[str, ...] = ()
    args: tuple[str, ...] = ()


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
    env: Mapping[str, str] = field(default_factory=dict)
    volumes: tuple[str, ...] = ()
```

在 `auto_bench.py` 更新 `AutoBenchConfig` 和 `BenchmarkCase`：

```python
from remote_topology import TopologyProfile, parse_topology_profiles


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

    @property
    def serving_name(self) -> str:
        if self.serve_profile is not None:
            return self.serve_profile.name
        if self.topology_profile is not None:
            return self.topology_profile.name
        raise ConfigError("benchmark case has no serving profile")
```

在 `_parse_serve_profiles()` 允许 `topology_profiles` 存在时 `serve_profiles` 为空；在 `load_config()` 调用 `parse_topology_profiles(config_data)`；在 `expand_cases()` 同时展开 legacy 和 topology cases。

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_remote_topology.py::test_load_config_accepts_topology_profiles_without_serve_profiles vllm_standalone_bench/tests/test_auto_bench.py::test_expand_cases_uses_topology_profiles -q
```

预期：PASS。

- [ ] **步骤 5：运行 legacy 回归并提交**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_load_config_applies_defaults_and_expands_cases vllm_standalone_bench/tests/test_auto_bench.py::test_build_serve_command_dispatches_vllm -q
```

预期：PASS。

提交：

```bash
git add vllm_standalone_bench/remote_topology.py vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_remote_topology.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat: parse remote topology profiles"
```

## 任务 2：SSH 认证、敏感值 masking、RemoteDockerRunner

**文件：**
- 创建：`vllm_standalone_bench/remote_docker.py`
- 修改：`vllm_standalone_bench/remote_topology.py`
- 测试：`vllm_standalone_bench/tests/test_remote_docker.py`
- 测试：`vllm_standalone_bench/tests/test_remote_topology.py`

- [ ] **步骤 1：编写失败的 SSH 与 masking 测试**

在 `test_remote_topology.py` 添加：

```python
def test_password_values_are_masked_in_resolved_topology(tmp_path, monkeypatch):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["hosts"]["p1"]["auth"] = {
        "type": "password",
        "password": "secret-pass",
    }
    config = ab.load_config(write_config(tmp_path, data))
    resolved = ab.config_to_dict(config)
    host = resolved["topology_profiles"][0]["hosts"]["p1"]
    assert host["auth"]["password"] == "***"
    assert "secret-pass" not in json.dumps(resolved)


def test_password_env_requires_existing_env(tmp_path, monkeypatch):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["hosts"]["p1"]["auth"] = {
        "type": "password_env",
        "env": "MISSING_PD_PASSWORD",
    }
    monkeypatch.delenv("MISSING_PD_PASSWORD", raising=False)
    with pytest.raises(ab.ConfigError, match="MISSING_PD_PASSWORD"):
        ab.load_config(write_config(tmp_path, data))
```

在 `test_remote_docker.py` 添加：

```python
import os

import pytest

from remote_docker import RemoteDockerRunner, build_ssh_base_command
from remote_topology import RemoteAuth, RemoteHost


def value_after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_key_auth_uses_plain_ssh_command():
    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("key", key_path="/keys/id_rsa"))
    cmd, env = build_ssh_base_command(host)
    assert cmd[0] == "ssh"
    assert value_after(cmd, "-i") == "/keys/id_rsa"
    assert "root@10.0.0.11" in cmd
    assert env == {}


def test_password_env_uses_sshpass_env_without_password_in_args(monkeypatch):
    monkeypatch.setenv("P1_PASSWORD", "secret-pass")
    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("password_env", env="P1_PASSWORD"))
    cmd, env = build_ssh_base_command(host, sshpass_path="/usr/bin/sshpass")
    assert cmd[:2] == ["/usr/bin/sshpass", "-e"]
    assert "secret-pass" not in " ".join(cmd)
    assert env["SSHPASS"] == "secret-pass"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_remote_topology.py::test_password_values_are_masked_in_resolved_topology vllm_standalone_bench/tests/test_remote_docker.py -q
```

预期：FAIL，报错包含 `remote_docker` 模块不存在或 auth masking 未实现。

- [ ] **步骤 3：实现认证和 runner**

在 `remote_topology.py` 添加：

```python
def mask_auth(auth: RemoteAuth) -> dict[str, str]:
    data = {"type": auth.type}
    if auth.key_path:
        data["key_path"] = auth.key_path
    if auth.env:
        data["env"] = auth.env
    if auth.password is not None:
        data["password"] = "***"
    return data


def topology_to_safe_dict(profile: TopologyProfile) -> dict[str, Any]:
    payload = dataclasses.asdict(profile)
    payload["hosts"] = {
        name: {
            "name": host.name,
            "address": host.address,
            "ssh_user": host.ssh_user,
            "auth": mask_auth(host.auth),
        }
        for name, host in profile.hosts.items()
    }
    return payload


def mask_command(argv: list[str] | tuple[str, ...]) -> list[str]:
    masked: list[str] = []
    redact_next = False
    for value in argv:
        if redact_next:
            masked.append("***")
            redact_next = False
            continue
        masked.append(value)
        if value in {"--password", "--api-key"}:
            redact_next = True
    return masked
```

在 `remote_docker.py` 添加：

```python
from __future__ import annotations

import os
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from auto_bench import Completed, ConfigError
from remote_topology import RemoteHost


def build_ssh_base_command(host: RemoteHost, sshpass_path: str | None = None):
    env: dict[str, str] = {}
    auth = host.auth
    ssh_cmd = ["ssh", "-o", "BatchMode=yes"]
    if auth.type == "key":
        if auth.key_path:
            ssh_cmd.extend(["-i", auth.key_path])
    elif auth.type in {"password_env", "password"}:
        resolved_sshpass = sshpass_path or shutil.which("sshpass")
        if not resolved_sshpass:
            raise ConfigError("password SSH auth requires sshpass in PATH")
        password = os.environ[auth.env] if auth.type == "password_env" else auth.password
        env["SSHPASS"] = password or ""
        ssh_cmd = [resolved_sshpass, "-e", "ssh", "-o", "BatchMode=no"]
    else:
        raise ConfigError(f"unsupported SSH auth type: {auth.type}")
    ssh_cmd.append(f"{host.ssh_user}@{host.address}")
    return ssh_cmd, env


@dataclass(frozen=True)
class RemoteDockerRunner:
    def run(self, host: RemoteHost, command: list[str], *, check: bool = False) -> Completed:
        ssh_cmd, extra_env = build_ssh_base_command(host)
        full_cmd = ssh_cmd + [shlex.join(command)]
        completed = subprocess.run(
            full_cmd,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, **extra_env},
        )
        result = Completed(full_cmd, completed.returncode, completed.stdout or "", completed.stderr or "")
        if check and result.returncode != 0:
            raise RuntimeError(f"remote command failed ({result.returncode}): {host.name}")
        return result

    def capture(self, host: RemoteHost, command: list[str]) -> str:
        return self.run(host, command, check=True).stdout

    def inspect_labels(self, host: RemoteHost, container_name: str) -> dict[str, str] | None:
        result = self.run(
            host,
            ["docker", "inspect", "--format", "{{json .Config.Labels}}", container_name],
            check=False,
        )
        if result.returncode != 0:
            return None
        payload = result.stdout.strip() or "null"
        labels = json.loads(payload)
        if not isinstance(labels, dict):
            return {}
        return {str(key): str(value) for key, value in labels.items()}
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_remote_docker.py vllm_standalone_bench/tests/test_remote_topology.py::test_password_values_are_masked_in_resolved_topology vllm_standalone_bench/tests/test_remote_topology.py::test_password_env_requires_existing_env -q
```

预期：PASS。

- [ ] **步骤 5：提交**

```bash
git add vllm_standalone_bench/remote_docker.py vllm_standalone_bench/remote_topology.py vllm_standalone_bench/tests/test_remote_docker.py vllm_standalone_bench/tests/test_remote_topology.py
git commit -m "feat: add remote docker ssh runner"
```

## 任务 3：SGLang 和 vLLM PD 命令生成

**文件：**
- 修改：`vllm_standalone_bench/remote_topology.py`
- 测试：`vllm_standalone_bench/tests/test_remote_topology.py`

- [ ] **步骤 1：编写失败的命令生成测试**

添加：

```python
def test_sglang_pd_commands_render_worker_and_router_flags(tmp_path):
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")
    p1 = commands["p1"].argv
    router = commands["router"].argv
    assert "-m" in p1 and "sglang.launch_server" in p1
    assert p1[p1.index("--disaggregation-mode") + 1] == "prefill"
    assert p1[p1.index("--disaggregation-bootstrap-port") + 1] == "12335"
    assert router[router.index("-m") + 1] == "sglang_router.launch_router"
    assert values_after(router, "--prefill") == ["http://10.0.0.11:30000", "http://10.0.0.12:30000"]
    assert values_after(router, "--decode") == ["http://10.0.0.21:30001", "http://10.0.0.22:30001"]
    assert "12335" not in values_after(router, "--prefill")


def test_vllm_pd_worker_command_renders_kv_template(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["kv_transfer_config_template"] = {
        "kv_connector": "NixlConnector",
        "kv_role": "{kv_role}",
        "kv_rank": "{kv_rank}",
        "kv_parallel_size": "{kv_parallel_size}",
        "kv_ip": "{node_address}",
        "kv_port": "{node_port}",
    }
    topology["frontend"] = {
        "kind": "external",
        "host": "router",
        "port": 8000,
        "image": "pd-proxy:offline",
        "command": ["python", "/opt/proxy.py", "--port", "{frontend_port}"],
    }
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")
    p1 = commands["p1"].argv
    kv_json = p1[p1.index("--kv-transfer-config") + 1]
    assert '"kv_connector":"NixlConnector"' in kv_json
    assert '"kv_role":"kv_producer"' in kv_json
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_remote_topology.py::test_sglang_pd_commands_render_worker_and_router_flags vllm_standalone_bench/tests/test_remote_topology.py::test_vllm_pd_worker_command_renders_kv_template -q
```

预期：FAIL，报错包含 `build_commands` 不存在。

- [ ] **步骤 3：实现命令生成**

在 `remote_topology.py` 添加：

```python
@dataclass(frozen=True)
class RoleCommand:
    role_name: str
    host_name: str
    container_name: str
    argv: tuple[str, ...]
    masked_argv: tuple[str, ...]


def _docker_base(container: str, labels: Mapping[str, str], network: str, gpus: str | None):
    cmd = ["docker", "run", "-d", "--name", container]
    for key, value in labels.items():
        cmd.extend(["--label", f"{key}={value}"])
    if gpus:
        cmd.extend(["--gpus", f"device={gpus}" if gpus != "all" else "all"])
    cmd.extend(["--network", network])
    return cmd


def _container_name(case, role_name: str) -> str:
    return f"bench-pd-{case.run_id}-{case.model.name}-{case.topology_profile.name}-{role_name}"


def _labels(case, role: str, role_name: str, run_dir) -> dict[str, str]:
    return {
        "vllm_auto_bench.managed": "true",
        "vllm_auto_bench.run_id": case.run_id,
        "vllm_auto_bench.run_dir": str(Path(run_dir).resolve()),
        "vllm_auto_bench.model": case.model.name,
        "vllm_auto_bench.topology_profile": case.topology_profile.name,
        "vllm_auto_bench.role": role,
        "vllm_auto_bench.role_name": role_name,
    }
```

实现 `build_sglang_pd_commands()`：

```python
def build_sglang_pd_commands(profile, config, case, run_dir):
    commands = {}
    for role, nodes in (("prefill", profile.prefill), ("decode", profile.decode)):
        for node in nodes:
            host = profile.hosts[node.host]
            container_name = _container_name(case, node.name)
            cmd = _docker_base(container_name, _labels(case, role, node.name, run_dir), profile.network, node.gpus)
            cmd.extend(["-v", f"{config.mounts.models}:/models:ro", "--entrypoint", "python3", profile.image])
            cmd.extend(["-m", "sglang.launch_server", "--model-path", case.model.model_path, "--served-model-name", case.api_model_name, "--host", "0.0.0.0", "--port", str(node.port)])
            cmd.extend(["--disaggregation-mode", role])
            if profile.transfer_backend:
                cmd.extend(["--disaggregation-transfer-backend", profile.transfer_backend])
            if node.bootstrap_port is not None:
                cmd.extend(["--disaggregation-bootstrap-port", str(node.bootstrap_port)])
            cmd.extend(node.args)
            commands[node.name] = RoleCommand(node.name, host.name, container_name, tuple(cmd), tuple(mask_command(cmd)))
    router = _build_sglang_router_command(profile, config, case, run_dir)
    commands["router"] = router
    return commands
```

实现 vLLM worker 使用 `json.dumps(rendered, separators=(",", ":"))` 输出 `--kv-transfer-config`；prefill roles 用 `kv_producer`，decode roles 用 `kv_consumer`，`kv_rank` 按 prefill+decode 顺序递增。

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_remote_topology.py::test_sglang_pd_commands_render_worker_and_router_flags vllm_standalone_bench/tests/test_remote_topology.py::test_vllm_pd_worker_command_renders_kv_template -q
```

预期：PASS。

- [ ] **步骤 5：提交**

```bash
git add vllm_standalone_bench/remote_topology.py vllm_standalone_bench/tests/test_remote_topology.py
git commit -m "feat: render pd topology commands"
```

## 任务 4：Dry-run、结果布局、bench-runner endpoint

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/remote_topology.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的 dry-run 与布局测试**

添加：

```python
def test_topology_layout_uses_topology_profile(tmp_path):
    from test_remote_topology import pd_topology_config, write_config
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    assert layout.serve_dir == tmp_path / "results" / "run123" / "qwen2_5_1_5b" / "sglang_pd_2p2d"
    assert layout.bench_dir == layout.serve_dir / "smoke"


def test_topology_bench_command_targets_frontend_endpoint(tmp_path):
    from test_remote_topology import pd_topology_config, write_config
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    cmd = ab.build_bench_run_command(config, case, tmp_path / "bench")
    assert value_after(cmd, "--base-url") == "http://10.0.0.30:8000/v1"
    assert "vllm_auto_bench.topology_profile=sglang_pd_2p2d" in " ".join(cmd)


def test_topology_dry_run_masks_passwords_and_prints_remote_commands(tmp_path, capsys, monkeypatch):
    from test_remote_topology import pd_topology_config, write_config
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["hosts"]["p1"]["auth"] = {"type": "password", "password": "secret-pass"}
    config = ab.load_config(write_config(tmp_path, data))
    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=True)
    out = capsys.readouterr().out
    assert result == 0
    assert "sglang.launch_server" in out
    assert "sglang_router.launch_router" in out
    assert "secret-pass" not in out
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_topology_layout_uses_topology_profile vllm_standalone_bench/tests/test_auto_bench.py::test_topology_bench_command_targets_frontend_endpoint vllm_standalone_bench/tests/test_auto_bench.py::test_topology_dry_run_masks_passwords_and_prints_remote_commands -q
```

预期：FAIL，layout 或 base-url 仍使用 legacy container name。

- [ ] **步骤 3：实现 layout、endpoint、dry-run 分派**

在 `auto_bench.py` 调整：

```python
CONTAINER_TOPOLOGY_PROFILE_LABEL = "vllm_auto_bench.topology_profile"
CONTAINER_ROLE_LABEL = "vllm_auto_bench.role"
CONTAINER_ROLE_NAME_LABEL = "vllm_auto_bench.role_name"


def case_serving_dimension(case: BenchmarkCase) -> tuple[str, str]:
    if case.serve_profile is not None:
        return "serve_profile", case.serve_profile.name
    return "topology_profile", case.topology_profile.name


def case_endpoint_base_url(config: AutoBenchConfig, case: BenchmarkCase) -> str:
    if case.topology_profile is None:
        assert case.container_name is not None
        return f"http://{case.container_name}:{config.run.container_port}/v1"
    host = case.topology_profile.hosts[case.topology_profile.frontend.host]
    return f"http://{host.address}:{case.topology_profile.frontend.port}/v1"
```

`build_layout()` 使用 `case.serving_name`。`build_bench_run_command()` 用 `case_endpoint_base_url()`，labels 对 topology case 写 `topology_profile` 并令 `serve_profile` label 缺省。`_run_controller_dry_run()` 对 topology group 打印每个 remote role 的 masked command，再打印 bench command。

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_topology_layout_uses_topology_profile vllm_standalone_bench/tests/test_auto_bench.py::test_topology_bench_command_targets_frontend_endpoint vllm_standalone_bench/tests/test_auto_bench.py::test_topology_dry_run_masks_passwords_and_prints_remote_commands -q
```

预期：PASS。

- [ ] **步骤 5：提交**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/remote_topology.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat: route bench cases to topology endpoints"
```

## 任务 5：远程 PD 生命周期编排与 artifacts

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/remote_docker.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的 fake integration 测试**

添加：

```python
class FakeRemoteDockerRunner:
    def __init__(self, failures=None):
        self.commands = []
        self.failures = failures or {}
        self.labels = {}

    def run(self, host, command, *, check=False):
        self.commands.append((host.name, list(command)))
        key = (host.name, " ".join(command[:3]))
        if key in self.failures:
            return ab.Completed(list(command), self.failures[key], "", "forced failure")
        if command[:3] == ["docker", "run", "-d"]:
            self.labels[value_after(command, "--name")] = dict(
                label.split("=", 1) for label in values_after(command, "--label") if "=" in label
            )
        if command[:2] == ["docker", "logs"]:
            return ab.Completed(list(command), 0, f"{host.name} log\n", "")
        if command[:2] == ["docker", "inspect"]:
            return ab.Completed(list(command), 0, "[]\n", "")
        return ab.Completed(list(command), 0, "ok\n", "")


def test_topology_run_starts_roles_then_bench_and_cleans_up(tmp_path, monkeypatch):
    from test_remote_topology import pd_topology_config, write_config
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    remote = FakeRemoteDockerRunner()
    local = FakeRunner()
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True)
    result = ab.run_controller(config, run_id="run123", runner=local)
    assert result == 0
    names = [cmd[1][cmd[1].index("--name") + 1] for cmd in remote.commands if cmd[1][:3] == ["docker", "run", "-d"]]
    assert names[:5] == [
        "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-p1",
        "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-p2",
        "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-d1",
        "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-d2",
        "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-router",
    ]
    assert len(bench_run_commands(local.commands)) == 1
    assert any(cmd[1][:2] == ["docker", "stop"] and "router" in cmd[1][2] for cmd in remote.commands)
```

补充失败路径测试：

```python
def test_topology_prefill_start_failure_cleans_started_roles(tmp_path, monkeypatch):
    from test_remote_topology import pd_topology_config, write_config
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    remote = FakeRemoteDockerRunner(failures={("p2", "docker run -d"): 1})
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote)
    result = ab.run_controller(config, run_id="run123", runner=FakeRunner())
    assert result == 1
    assert any(cmd[0] == "p1" and cmd[1][:2] == ["docker", "stop"] for cmd in remote.commands)
    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cases"][0]["status"] == "failed"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_topology_run_starts_roles_then_bench_and_cleans_up vllm_standalone_bench/tests/test_auto_bench.py::test_topology_prefill_start_failure_cleans_started_roles -q
```

预期：FAIL，topology run path 未实现。

- [ ] **步骤 3：实现 lifecycle helper**

在 `auto_bench.py` 添加：

```python
def _role_start_order(role_commands: Mapping[str, Any]) -> list[str]:
    names = list(role_commands)
    return [name for name in names if name != "router"] + [name for name in names if name == "router"]


def topology_role_ready_url(case: BenchmarkCase, role_name: str) -> str:
    topology = case.topology_profile
    assert topology is not None
    if role_name == "router":
        host = topology.hosts[topology.frontend.host]
        return f"http://{host.address}:{topology.frontend.port}/v1/models"
    for node in (*topology.prefill, *topology.decode):
        if node.name == role_name:
            host = topology.hosts[node.host]
            return (
                f"http://{host.address}:{node.port}/health"
                if topology.engine == "sglang"
                else f"http://{host.address}:{node.port}/v1/models"
            )
    raise ConfigError(f"unknown topology role: {role_name}")


def wait_for_remote_ready(config: AutoBenchConfig, case: BenchmarkCase, role_name: str) -> bool:
    return wait_for_http_ok(
        topology_role_ready_url(case, role_name),
        api_key=config.run.api_key if role_name == "router" else None,
        timeout_sec=config.run.ready_timeout_sec,
    )


def wait_for_http_ok(url: str, api_key: str | None, timeout_sec: int) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_sec
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5) as response:
                if response.status == 200:
                    return True
        except (OSError, TimeoutError, urllib.error.URLError):
            time.sleep(2)
    return False


def save_topology_artifacts_best_effort(config, case, layout, remote_runner, role_commands):
    topology = case.topology_profile
    assert topology is not None
    (layout.bench_dir / "commands").mkdir(parents=True, exist_ok=True)
    (layout.bench_dir / "logs").mkdir(parents=True, exist_ok=True)
    (layout.bench_dir / "inspect").mkdir(parents=True, exist_ok=True)
    write_json_atomic(layout.bench_dir / "topology.resolved.json", topology_to_safe_dict(topology))
    for role_name, role_command in role_commands.items():
        host = topology.hosts[role_command.host_name]
        container = role_command.container_name
        (layout.bench_dir / "commands" / f"{role_name}.txt").write_text(" ".join(role_command.masked_argv), encoding="utf-8")
        logs = remote_runner.run(host, ["docker", "logs", "--timestamps", container], check=False)
        (layout.bench_dir / "logs" / f"{role_name}.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
        inspect = remote_runner.run(host, ["docker", "inspect", container], check=False)
        (layout.bench_dir / "inspect" / f"{role_name}.json").write_text(inspect.stdout, encoding="utf-8")


def topology_role_labels_match(labels, case, role_command) -> bool:
    if labels is None:
        return False
    return (
        labels.get(NETWORK_MANAGED_LABEL) == "true"
        and labels.get(NETWORK_RUN_ID_LABEL) == case.run_id
        and labels.get(CONTAINER_MODEL_LABEL) == case.model.name
        and labels.get(CONTAINER_TOPOLOGY_PROFILE_LABEL) == case.topology_profile.name
        and labels.get(CONTAINER_ROLE_NAME_LABEL) == role_command.role_name
    )


def cleanup_topology_roles_best_effort(case, remote_runner, role_commands, started_role_names):
    topology = case.topology_profile
    assert topology is not None
    for role_name in started_role_names:
        role_command = role_commands[role_name]
        host = topology.hosts[role_command.host_name]
        labels = remote_runner.inspect_labels(host, role_command.container_name)
        if topology_role_labels_match(labels, case, role_command):
            remote_runner.run(host, ["docker", "stop", role_command.container_name], check=False)
            remote_runner.run(host, ["docker", "rm", "-f", role_command.container_name], check=False)


def run_bench_cases_for_started_service(config, run_id, all_cases, group_cases, manifest, completed, local_runner):
    group_exit = 0
    added = 0
    for case in group_cases:
        layout = build_layout(config, run_id, case)
        layout.bench_dir.mkdir(parents=True, exist_ok=True)
        write_state(run_dir=layout.run_dir, state=current_state(run_id, all_cases, completed + added, case, "running", manifest=manifest))
        bench_cmd = build_bench_run_command(config, case, layout.bench_dir)
        with (layout.bench_dir / "bench.log").open("w", encoding="utf-8") as log:
            result = local_runner.run(bench_cmd, check=False, capture=False, stdout=log, stderr=log)
        status = "passed" if result.returncode == 0 else "failed"
        error = None if status == "passed" else f"benchmark exited {result.returncode}"
        if status != "passed":
            group_exit = 1
        manifest.record(case, layout, status, error=error)
        write_json_atomic(layout.bench_dir / "status.json", {"status": status, "error": error})
        write_manifest(layout.run_dir, manifest)
        added += 1
    return added, group_exit


def run_topology_group(config, run_id, all_cases, group_cases, manifest, completed, local_runner, remote_runner):
    serve_case = group_cases[0]
    layout = build_layout(config, run_id, serve_case)
    role_commands = serve_case.topology_profile.build_commands(config, serve_case, layout.run_dir)
    started: list[str] = []
    try:
        for role_name in _role_start_order(role_commands):
            command = role_commands[role_name]
            host = serve_case.topology_profile.hosts[command.host_name]
            result = remote_runner.run(host, list(command.argv), check=False)
            if result.returncode != 0:
                raise RuntimeError(f"remote role failed to start: {role_name}")
            started.append(role_name)
        for role_name in started:
            if not wait_for_remote_ready(config, serve_case, role_name):
                raise RuntimeError(f"remote role ready timeout: {role_name}")
        return run_bench_cases_for_started_service(config, run_id, all_cases, group_cases, manifest, completed, local_runner)
    finally:
        save_topology_artifacts_best_effort(config, serve_case, layout, remote_runner, role_commands)
        cleanup_topology_roles_best_effort(config, serve_case, remote_runner, role_commands, reversed(started))
```

在 `run_controller()` 的 group loop 中：

```python
if serve_case.topology_profile is not None:
    completed_delta, group_exit = run_topology_group(
        config,
        run_id,
        all_cases,
        group_cases,
        manifest,
        completed,
        active_runner,
        RemoteDockerRunner(),
    )
    completed += completed_delta
    exit_code = max(exit_code, group_exit)
    continue
```

Artifacts 写入：

```text
commands/<role>.txt
logs/<role>.log
inspect/<role>.json
topology.resolved.json
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_topology_run_starts_roles_then_bench_and_cleans_up vllm_standalone_bench/tests/test_auto_bench.py::test_topology_prefill_start_failure_cleans_started_roles -q
```

预期：PASS。

- [ ] **步骤 5：提交**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/remote_docker.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat: orchestrate remote pd topology lifecycle"
```

## 任务 6：远程资源监控与 host-prefixed 结果列

**文件：**
- 修改：`vllm_standalone_bench/resource_monitor.py`
- 修改：`vllm_standalone_bench/remote_docker.py`
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_resource_monitor.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的 prefix merge 测试**

在 `test_resource_monitor.py` 添加：

```python
def test_append_prefixed_summaries_to_result_csv(tmp_path):
    result_csv = tmp_path / "result.csv"
    result_csv.write_text("model,throughput_tok_s\nm,12.5\n", encoding="utf-8-sig")
    summary = {
        "available": True,
        "sample_count": 2,
        "aggregate": {"cpu_util_avg_pct": 50.0, "gpu_mem_used_max_mb": 1234.0},
    }
    rm.append_prefixed_summaries_to_result_files(tmp_path, {"p1": summary, "router": summary})
    with result_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["p1_resource_monitor_available"] == "true"
    assert rows[0]["p1_cpu_util_avg_pct"] == "50.0"
    assert rows[0]["router_gpu_mem_used_max_mb"] == "1234.0"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py::test_append_prefixed_summaries_to_result_csv -q
```

预期：FAIL，函数不存在。

- [ ] **步骤 3：实现 prefix merge 与远程 readers**

在 `resource_monitor.py` 添加：

```python
def prefixed_resource_columns(prefix):
    return [f"{prefix}_{column}" for column in RESOURCE_RESULT_COLUMNS]


def flatten_prefixed_summaries(summaries):
    values = {}
    for prefix, summary in summaries.items():
        flat = flatten_summary_for_result(summary)
        for column, value in flat.items():
            values[f"{prefix}_{column}"] = value
    return values


def append_prefixed_summaries_to_result_files(output_dir, summaries):
    values = flatten_prefixed_summaries(summaries)
    append_dynamic_summary_to_csv(Path(output_dir) / "result.csv", values)
    append_dynamic_summary_to_xlsx(Path(output_dir) / "result.xlsx", values)


def append_dynamic_summary_to_csv(path, values):
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    merged = fieldnames + [column for column in values if column not in fieldnames]
    for row in rows:
        row.update({column: str(value).lower() if isinstance(value, bool) else value for column, value in values.items()})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged)
        writer.writeheader()
        writer.writerows(rows)


def append_dynamic_summary_to_xlsx(path, values):
    try:
        import openpyxl
    except ImportError:
        return
    path = Path(path)
    if not path.exists():
        return
    workbook = openpyxl.load_workbook(path)
    worksheet = workbook.active
    column_by_name = {
        worksheet.cell(row=1, column=column).value: column
        for column in range(1, worksheet.max_column + 1)
        if worksheet.cell(row=1, column=column).value is not None
    }
    next_column = worksheet.max_column + 1
    for column_name, value in values.items():
        column = column_by_name.get(column_name)
        if column is None:
            column = next_column
            next_column += 1
            worksheet.cell(row=1, column=column, value=column_name)
        for row in range(3 if worksheet.max_row >= 2 else 2, worksheet.max_row + 1):
            worksheet.cell(row=row, column=column, value=value)
    workbook.save(path)
```

在 `remote_docker.py` 添加：

```python
from resource_monitor import NVIDIA_SMI_QUERY


class RemoteResourceReaders:
    def __init__(self, runner, host):
        self.runner = runner
        self.host = host

    def proc_stat(self):
        return self.runner.capture(self.host, ["cat", "/proc/stat"])

    def meminfo(self):
        return self.runner.capture(self.host, ["cat", "/proc/meminfo"])

    def net_dev(self):
        return self.runner.capture(self.host, ["cat", "/proc/net/dev"])

    def diskstats(self):
        return self.runner.capture(self.host, ["cat", "/proc/diskstats"])

    def nvidia_smi(self):
        return self.runner.capture(self.host, NVIDIA_SMI_QUERY)
```

- [ ] **步骤 4：接入 topology run**

在 topology bench 前为每个参与 host 创建：

```python
ResourceMonitor(
    output_dir=layout.bench_dir / "resources" / host_name,
    interval_sec=config.run.resource_monitor.interval_sec,
    enabled=True,
    backend=config.run.resource_monitor.backend,
    readers=RemoteResourceReaders(remote_runner, host),
)
```

bench 结束后停止所有 monitor，将 summaries 传给 `append_prefixed_summaries_to_result_files(layout.bench_dir, summaries)`。monitor 失败只写 warning，不改 benchmark status。

- [ ] **步骤 5：运行测试验证通过并提交**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py::test_append_prefixed_summaries_to_result_csv vllm_standalone_bench/tests/test_auto_bench.py::test_topology_run_starts_roles_then_bench_and_cleans_up -q
```

预期：PASS。

提交：

```bash
git add vllm_standalone_bench/resource_monitor.py vllm_standalone_bench/remote_docker.py vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_resource_monitor.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat: collect remote topology resources"
```

## 任务 7：status、manifest、resume、compare 兼容 topology profile

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/bench_compare.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_bench_compare.py`

- [ ] **步骤 1：编写失败的 manifest/resume 测试**

添加：

```python
def test_topology_manifest_records_null_serve_profile(tmp_path, monkeypatch):
    from test_remote_topology import pd_topology_config, write_config
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: FakeRemoteDockerRunner())
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True)
    ab.run_controller(config, run_id="run123", runner=FakeRunner())
    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    row = manifest["cases"][0]
    assert row["serve_profile"] is None
    assert row["topology_profile"] == "sglang_pd_2p2d"


def test_plan_resume_cases_supports_topology_key(tmp_path):
    from test_remote_topology import pd_topology_config, write_config
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    cases = ab.expand_cases(config, run_id="run123")
    manifest_data = {
        "run_id": "run123",
        "cases": [{
            "model": "qwen2_5_1_5b",
            "serve_profile": None,
            "topology_profile": "sglang_pd_2p2d",
            "bench_profile": "smoke",
            "status": "passed",
        }],
    }
    initial, pending, unknown = ab.plan_resume_cases(run_id="run123", cases=cases, manifest_data=manifest_data)
    assert len(initial.cases) == 1
    assert pending == ()
    assert unknown == []
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_topology_manifest_records_null_serve_profile vllm_standalone_bench/tests/test_auto_bench.py::test_plan_resume_cases_supports_topology_key -q
```

预期：FAIL，manifest key 仍要求 string `serve_profile`。

- [ ] **步骤 3：实现 key helpers**

在 `auto_bench.py` 改造：

```python
def _case_key(case: BenchmarkCase) -> tuple[str, str | None, str | None, str]:
    return (
        case.model.name,
        case.serve_profile.name if case.serve_profile else None,
        case.topology_profile.name if case.topology_profile else None,
        case.bench_profile.name,
    )


def _manifest_row_key(row):
    model = row.get("model")
    serve_profile = row.get("serve_profile")
    topology_profile = row.get("topology_profile")
    bench_profile = row.get("bench_profile")
    if not isinstance(model, str) or not isinstance(bench_profile, str):
        return None
    if isinstance(serve_profile, str) and topology_profile is None:
        return (model, serve_profile, None, bench_profile)
    if serve_profile is None and isinstance(topology_profile, str):
        return (model, None, topology_profile, bench_profile)
    return None
```

`Manifest.record()` 写入两个字段：legacy case `serve_profile=<name>, topology_profile=None`；topology case `serve_profile=None, topology_profile=<name>`。

- [ ] **步骤 4：改造 compare 聚合**

在 `bench_compare.py` 将 serving dimension 抽象为：

```python
def _serving_profiles(config):
    for profile in config.serve_profiles:
        yield profile.name, profile.engine, "serve_profile"
    for profile in getattr(config, "topology_profiles", ()):
        yield profile.name, profile.engine, "topology_profile"
```

CSV 路径使用 `run_dir / model.name / serving_name / bench.name / "result.csv"`。输出列名保留 engine 前缀；同一 engine 多个 serving profile 时使用 serving name 作为列前缀，避免覆盖。

- [ ] **步骤 5：运行测试并提交**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_topology_manifest_records_null_serve_profile vllm_standalone_bench/tests/test_auto_bench.py::test_plan_resume_cases_supports_topology_key vllm_standalone_bench/tests/test_bench_compare.py -q
```

预期：PASS。

提交：

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/bench_compare.py vllm_standalone_bench/tests/test_auto_bench.py vllm_standalone_bench/tests/test_bench_compare.py
git commit -m "feat: support topology cases in status and resume"
```

## 任务 8：示例配置、文档 smoke、总验证

**文件：**
- 创建：`vllm_standalone_bench/configs/auto_bench.sglang_pd_remote.example.json`
- 修改：`vllm_standalone_bench/README.md`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的示例配置测试**

添加：

```python
def test_shipped_sglang_pd_remote_config_parses():
    path = CONFIG_DIR / "auto_bench.sglang_pd_remote.example.json"
    config = ab.load_config(path)
    assert config.serve_profiles == ()
    assert config.topology_profiles[0].engine == "sglang"
    assert config.topology_profiles[0].frontend.kind == "sglang_router"
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_shipped_sglang_pd_remote_config_parses -q
```

预期：FAIL，示例配置不存在。

- [ ] **步骤 3：添加示例配置**

创建 `auto_bench.sglang_pd_remote.example.json`，内容使用规格里的 2P2D 示例，模型路径指向 `/models/GLM-5.2-FP8`，所有 host 使用 RFC 5737 文档地址段：

```json
{
  "run": {
    "name": "sglang_pd_remote_bench",
    "results_dir": "vllm_standalone_bench/results",
    "bench_image": "vllm-bench-runner:offline",
    "api_key": "local-bench-key",
    "ready_timeout_sec": 1800
  },
  "mounts": {"models": "/models", "datasets": "/datasets"},
  "models": [{
    "name": "glm52",
    "model_path": "/models/GLM-5.2-FP8",
    "tokenizer_path": "/models/GLM-5.2-FP8",
    "served_model_name": "GLM-5.2"
  }],
  "topology_profiles": [{
    "name": "sglang_pd_2p2d",
    "engine": "sglang",
    "mode": "pd",
    "provider": "ssh_docker",
    "transfer_backend": "mooncake",
    "network": "host",
    "image": "lmsysorg/sglang:latest",
    "router_image": "sglang-router:offline",
    "hosts": {
      "p1": {"address": "192.0.2.11", "ssh_user": "root", "auth": {"type": "key"}},
      "p2": {"address": "192.0.2.12", "ssh_user": "root", "auth": {"type": "key"}},
      "d1": {"address": "192.0.2.21", "ssh_user": "root", "auth": {"type": "key"}},
      "d2": {"address": "192.0.2.22", "ssh_user": "root", "auth": {"type": "key"}},
      "router": {"address": "192.0.2.30", "ssh_user": "root", "auth": {"type": "key"}}
    },
    "prefill": [
      {"name": "p1", "host": "p1", "port": 30000, "bootstrap_port": 12335, "gpus": "0,1,2,3", "args": ["--tp-size", "4"]},
      {"name": "p2", "host": "p2", "port": 30000, "bootstrap_port": 12335, "gpus": "0,1,2,3", "args": ["--tp-size", "4"]}
    ],
    "decode": [
      {"name": "d1", "host": "d1", "port": 30001, "bootstrap_port": 12335, "gpus": "0,1,2,3", "args": ["--tp-size", "4"]},
      {"name": "d2", "host": "d2", "port": 30001, "bootstrap_port": 12335, "gpus": "0,1,2,3", "args": ["--tp-size", "4"]}
    ],
    "frontend": {"kind": "sglang_router", "host": "router", "port": 8000}
  }],
  "bench_profiles": [{
    "name": "latency_matrix",
    "backend": "openai-chat",
    "input_lens": [512],
    "output_lens": [256],
    "parallel_nums": [1],
    "epochs": 1
  }]
}
```

- [ ] **步骤 4：更新 README smoke 用法**

在 `vllm_standalone_bench/README.md` 增加远程 PD dry-run 命令：

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.sglang_pd_remote.example.json \
  --run-id pd_remote_dry_run_001 \
  --dry-run
```

说明：实际运行前需要替换 host 地址、镜像名、远程模型路径和 SSH auth。

- [ ] **步骤 5：运行总验证**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_remote_topology.py -q
python3 -m pytest vllm_standalone_bench/tests/test_remote_docker.py -q
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q
python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py -q
python3 -m pytest vllm_standalone_bench/tests/test_bench_compare.py -q
bash -n vllm_standalone_bench/run_auto_bench.sh
git diff --check
```

预期：全部 PASS，`bash -n` 无输出，`git diff --check` 无输出。

- [ ] **步骤 6：提交**

```bash
git add vllm_standalone_bench/configs/auto_bench.sglang_pd_remote.example.json vllm_standalone_bench/README.md vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "docs: add remote pd topology example"
```

## 规格覆盖自检

- 远程 Docker topology mode：任务 1、2、4、5。
- Legacy config unchanged：任务 1、4、8 的 legacy regression。
- SGLang PD first-class：任务 3、4、5、8。
- vLLM PD explicit proxy and KV config：任务 3。
- 本地 bench-runner 打远程 frontend：任务 4、5。
- artifacts under `results/<run_id>/`：任务 5。
- status/logs/stop/resume：任务 5、7。
- password masking：任务 2、4。
- remote resource monitoring：任务 6。
- compare aggregation serving dimension：任务 7。
- dry-run acceptance：任务 4、8。
