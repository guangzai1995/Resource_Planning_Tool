# vLLM PD Proxy and Topology 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为远程 Docker PD 拓扑补齐正确的 vLLM P2P/NIXL worker 配置、内置 vLLM PD proxy，以及 SGLang prefill bootstrap 端口修复。

**架构：** `remote_topology.py` 增加结构化 `vllm_pd` 配置、节点 KV/side-channel 端口和内置 proxy 命令生成；`vllm_bench.pd_proxy` 负责 OpenAI 兼容入口到 prefill/decode 两跳请求编排。旧的 external vLLM proxy 模板路径保留为兼容入口，结构化路径生成 vLLM 0.23.0 可接受的 KVTransferConfig。

**技术栈：** Python dataclass 配置解析、pytest、aiohttp.web、Docker command argv 渲染。

---

## 文件结构

- 修改：`vllm_standalone_bench/remote_topology.py`
  - 新增 `VllmPdConfig`
  - 扩展 `TopologyNode` 的 `kv_port`、`side_channel_port`
  - 扩展 vLLM structured PD 命令生成
  - 修复 SGLang router `--prefill URL BOOTSTRAP_PORT`
- 创建：`vllm_standalone_bench/vllm_bench/pd_proxy.py`
  - 内置 proxy CLI、aiohttp app、P2P request id/NIXL params helper
- 修改：`vllm_standalone_bench/tests/test_remote_topology.py`
  - 覆盖 SGLang bootstrap、vLLM P2P/NIXL 拓扑渲染、校验错误
- 创建：`vllm_standalone_bench/tests/test_pd_proxy.py`
  - 覆盖 proxy helper、请求编排、CLI endpoint 解析
- 修改：`vllm_standalone_bench/configs/auto_bench.sglang_pd_remote.example.json`
  - 确认示例 prefill bootstrap 端口显式存在
- 创建：`vllm_standalone_bench/configs/auto_bench.vllm_pd_p2p_remote.example.json`
  - 提供结构化 P2P/NCCL vLLM PD 示例
- 创建：`vllm_standalone_bench/configs/auto_bench.vllm_pd_nixl_remote.example.json`
  - 提供结构化 NIXL vLLM PD 示例
- 修改：`vllm_standalone_bench/README.md`
  - 简述 vLLM PD connector、端口字段和验证边界

## 任务 1：写拓扑层失败测试

**文件：**
- 修改：`vllm_standalone_bench/tests/test_remote_topology.py`

- [ ] **步骤 1：更新 SGLang fixture，显式配置不同 prefill bootstrap 端口**

在 `pd_topology_config()` 里把 prefill 节点改成：

```python
"prefill": [
    {"name": "p1", "host": "p1", "port": 30000, "bootstrap_port": 12335},
    {"name": "p2", "host": "p2", "port": 30000, "bootstrap_port": 12336},
],
```

- [ ] **步骤 2：更新 SGLang router 断言**

在 `test_sglang_pd_commands_render_worker_and_router_flags` 中删除对
`"12335" not in values_after(router, "--prefill")` 的断言，改成检查
`--prefill` 后面的两个参数：

```python
prefill_positions = [
    index for index, value in enumerate(router) if value == "--prefill"
]
assert [
    router[position + 1:position + 3]
    for position in prefill_positions
] == [
    ["http://10.0.0.11:30000", "12335"],
    ["http://10.0.0.12:30000", "12336"],
]
```

- [ ] **步骤 3：新增 SGLang prefill bootstrap 校验测试**

添加测试：

```python
def test_sglang_pd_rejects_prefill_without_bootstrap_port(tmp_path):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["prefill"][0].pop("bootstrap_port")

    with pytest.raises(ab.ConfigError, match="prefill.*bootstrap_port"):
        ab.load_config(write_config(tmp_path, data))
```

- [ ] **步骤 4：新增结构化 vLLM P2P 配置解析失败测试**

添加测试：

```python
def test_vllm_pd_p2p_rejects_missing_kv_port(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {"connector": "p2p_nccl", "proxy": {"kind": "builtin"}}
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}

    with pytest.raises(ab.ConfigError, match="kv_port"):
        ab.load_config(write_config(tmp_path, data))
```

- [ ] **步骤 5：新增结构化 vLLM NIXL 配置解析失败测试**

添加测试：

```python
def test_vllm_pd_nixl_rejects_missing_side_channel_port(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {"connector": "nixl", "proxy": {"kind": "builtin"}}
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}

    with pytest.raises(ab.ConfigError, match="side_channel_port"):
        ab.load_config(write_config(tmp_path, data))
```

- [ ] **步骤 6：新增未知 vLLM PD key 校验测试**

添加测试：

```python
def test_vllm_pd_rejects_unknown_structured_key(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {
        "connector": "p2p_nccl",
        "proxy": {"kind": "builtin"},
        "unknown": True,
    }
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}

    with pytest.raises(ab.ConfigError, match="unknown"):
        ab.load_config(write_config(tmp_path, data))
```

- [ ] **步骤 7：运行测试验证失败**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_remote_topology.py -q
```

预期：至少 SGLang router 断言失败，新增 vLLM 结构化校验测试失败，因为实现尚未解析 `vllm_pd`、`kv_port`、`side_channel_port`。

- [ ] **步骤 8：Commit 测试**

```bash
git add vllm_standalone_bench/tests/test_remote_topology.py
git commit -m "test: cover pd topology connector validation"
```

## 任务 2：实现拓扑数据模型和 SGLang bootstrap 修复

**文件：**
- 修改：`vllm_standalone_bench/remote_topology.py`
- 测试：`vllm_standalone_bench/tests/test_remote_topology.py`

- [ ] **步骤 1：增加 structured vLLM PD dataclass**

在 `TopologyFrontend` 后添加：

```python
@dataclass(frozen=True)
class VllmPdConfig:
    connector: str
    proxy_kind: str = "builtin"
    p2p_send_type: str = "PUT_ASYNC"
    nccl_num_channels: int | None = None
```

- [ ] **步骤 2：扩展 TopologyNode 和 TopologyProfile**

给 `TopologyNode` 添加字段：

```python
kv_port: int | None = None
side_channel_port: int | None = None
```

给 `TopologyProfile` 添加字段：

```python
vllm_pd: VllmPdConfig | None = None
```

- [ ] **步骤 3：解析节点新端口**

在 `_parse_nodes()` 创建 `TopologyNode` 时增加：

```python
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
```

- [ ] **步骤 4：解析 vllm_pd 对象**

新增函数：

```python
def _parse_vllm_pd_config(
    value: Any,
    path: str,
    error: ErrorFactory,
) -> VllmPdConfig | None:
    if value is None:
        return None
    raw = _mapping(value, path, error)
    allowed = {"connector", "proxy", "p2p_send_type", "nccl_num_channels"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise error(f"{path} contains unsupported keys: {', '.join(unknown)}")
    connector = _string(_required(raw, "connector", f"{path}.connector", error),
                        f"{path}.connector", error)
    if connector not in {"p2p_nccl", "nixl"}:
        raise error(f"{path}.connector must be one of p2p_nccl, nixl")
    proxy = _mapping(raw.get("proxy", {"kind": "builtin"}), f"{path}.proxy", error)
    proxy_kind = _string(proxy.get("kind", "builtin"), f"{path}.proxy.kind", error)
    if proxy_kind != "builtin":
        raise error(f"{path}.proxy.kind only supports builtin")
    send_type = _string(raw.get("p2p_send_type", "PUT_ASYNC"),
                        f"{path}.p2p_send_type", error)
    nccl_num_channels = _optional_positive_int(
        raw.get("nccl_num_channels"),
        f"{path}.nccl_num_channels",
        error,
    )
    return VllmPdConfig(
        connector=connector,
        proxy_kind=proxy_kind,
        p2p_send_type=send_type,
        nccl_num_channels=nccl_num_channels,
    )
```

- [ ] **步骤 5：把 vllm_pd 接入 parse_topology_profiles**

在 `TopologyProfile(...)` 构造参数中添加：

```python
vllm_pd=_parse_vllm_pd_config(
    profile.get("vllm_pd"),
    f"{path}.vllm_pd",
    error,
),
```

- [ ] **步骤 6：增加结构化配置校验**

新增函数并在 `_validate_role_names_unique(...)` 后调用：

```python
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
                    f"{path}.prefill node {node.name} bootstrap_port is required "
                    "for sglang pd"
                )
        return
    if engine != "vllm" or vllm_pd is None:
        return
    if frontend.kind != "builtin":
        raise error(
            f"{path}.frontend.kind must be builtin when structured vllm_pd is used"
        )
    if vllm_pd.connector == "p2p_nccl":
        for node in (*prefill, *decode):
            if node.kv_port is None:
                raise error(
                    f"{path} node {node.name} kv_port is required for p2p_nccl"
                )
    if vllm_pd.connector == "nixl":
        for node in (*prefill, *decode):
            if node.side_channel_port is None:
                raise error(
                    f"{path} node {node.name} side_channel_port is required for nixl"
                )
```

- [ ] **步骤 7：修复 SGLang router prefill 参数**

在 `_build_sglang_router_command()` 中改为：

```python
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
```

- [ ] **步骤 8：运行拓扑测试**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_remote_topology.py -q
```

预期：任务 1 的 SGLang 和解析校验测试通过；结构化 vLLM worker 命令测试尚未添加。

- [ ] **步骤 9：Commit 实现**

```bash
git add vllm_standalone_bench/remote_topology.py vllm_standalone_bench/tests/test_remote_topology.py
git commit -m "fix: validate pd topology connector ports"
```

## 任务 3：写并实现 vLLM structured worker 和内置 proxy 命令生成

**文件：**
- 修改：`vllm_standalone_bench/tests/test_remote_topology.py`
- 修改：`vllm_standalone_bench/remote_topology.py`

- [ ] **步骤 1：新增 P2P/NCCL 命令渲染测试**

添加 helper：

```python
def kv_config_after(argv):
    return json.loads(value_after(argv, "--kv-transfer-config"))
```

添加测试：

```python
def test_vllm_pd_p2p_commands_render_structured_builtin_proxy(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {
        "connector": "p2p_nccl",
        "proxy": {"kind": "builtin"},
        "p2p_send_type": "PUT_ASYNC",
        "nccl_num_channels": 16,
    }
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}
    topology["prefill"][0]["kv_port"] = 21001
    topology["prefill"][1]["kv_port"] = 21002
    topology["decode"][0]["kv_port"] = 22001
    topology["decode"][1]["kv_port"] = 22002

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    p1_config = kv_config_after(commands["p1"].argv)
    assert p1_config["kv_connector"] == "P2pNcclConnector"
    assert p1_config["kv_role"] == "kv_producer"
    assert p1_config["kv_port"] == 21001
    assert "kv_rank" not in p1_config
    assert "kv_parallel_size" not in p1_config
    assert p1_config["kv_connector_extra_config"] == {
        "http_port": 30000,
        "send_type": "PUT_ASYNC",
        "nccl_num_channels": 16,
    }

    d1_config = kv_config_after(commands["d1"].argv)
    assert d1_config["kv_connector"] == "P2pNcclConnector"
    assert d1_config["kv_role"] == "kv_consumer"
    assert d1_config["kv_port"] == 22001

    frontend = commands["router"].argv
    assert "vllm_bench.pd_proxy" in frontend
    assert value_after(frontend, "--connector") == "p2p_nccl"
    assert value_after(frontend, "--port") == "8000"
    assert len(values_after(frontend, "--prefill")) == 2
    assert len(values_after(frontend, "--decode")) == 2
    first_prefill = json.loads(values_after(frontend, "--prefill")[0])
    assert first_prefill == {
        "name": "p1",
        "url": "http://10.0.0.11:30000",
        "kv_address": "10.0.0.11:21001",
    }
```

- [ ] **步骤 2：新增 NIXL 命令渲染测试**

添加测试：

```python
def test_vllm_pd_nixl_commands_render_side_channel_env(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {"connector": "nixl", "proxy": {"kind": "builtin"}}
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}
    topology["prefill"][0]["side_channel_port"] = 5601
    topology["prefill"][1]["side_channel_port"] = 5602
    topology["decode"][0]["side_channel_port"] = 5701
    topology["decode"][1]["side_channel_port"] = 5702

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    p1_config = kv_config_after(commands["p1"].argv)
    assert p1_config == {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer",
    }
    p1_env = values_after(commands["p1"].argv, "-e")
    assert "VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.11" in p1_env
    assert "VLLM_NIXL_SIDE_CHANNEL_PORT=5601" in p1_env

    d1_config = kv_config_after(commands["d1"].argv)
    assert d1_config == {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer",
    }
    frontend = commands["router"].argv
    first_prefill = json.loads(values_after(frontend, "--prefill")[0])
    assert first_prefill == {
        "name": "p1",
        "url": "http://10.0.0.11:30000",
    }
```

- [ ] **步骤 3：运行新增测试验证失败**

运行：

```bash
python -m pytest \
  vllm_standalone_bench/tests/test_remote_topology.py::test_vllm_pd_p2p_commands_render_structured_builtin_proxy \
  vllm_standalone_bench/tests/test_remote_topology.py::test_vllm_pd_nixl_commands_render_side_channel_env \
  -q
```

预期：失败，原因是 `frontend.kind=builtin` 不受现有 vLLM 命令生成支持。

- [ ] **步骤 4：增加 extra env 支持**

把 `_append_env_and_volumes()` 签名改为：

```python
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
```

保留 volumes 处理逻辑不变。

- [ ] **步骤 5：拆分 vLLM command path**

把 `_build_vllm_pd_commands()` 改成：

```python
def _build_vllm_pd_commands(...):
    if self.vllm_pd is None:
        return self._build_legacy_vllm_pd_commands(config, case, run_dir)
    return self._build_structured_vllm_pd_commands(config, case, run_dir)
```

把当前 `_build_vllm_pd_commands()` 主体移动到
`_build_legacy_vllm_pd_commands()`，保持 external 模板兼容。

- [ ] **步骤 6：实现 structured vLLM worker config**

新增：

```python
def _structured_vllm_kv_config(
    self,
    node: TopologyNode,
    *,
    kv_role: str,
) -> dict[str, Any]:
    assert self.vllm_pd is not None
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
```

- [ ] **步骤 7：让 worker command 接收已渲染 config 和 extra_env**

把 `_build_vllm_worker_command()` 的 rank 参数替换为：

```python
kv_config: Mapping[str, Any] | None = None,
kv_role: str,
kv_rank: int | None = None,
kv_parallel_size: int | None = None,
extra_env: Mapping[str, str] | None = None,
```

如果 `kv_config` 为 `None`，调用旧 `_render_kv_transfer_config()`；否则：

```python
kv_config_text = json.dumps(kv_config, separators=(",", ":"), ensure_ascii=True)
```

调用 `_append_env_and_volumes(argv, node, extra_env=extra_env)`。

- [ ] **步骤 8：实现 structured commands 和 NIXL env**

新增：

```python
def _build_structured_vllm_pd_commands(...):
    image = self._worker_image(config)
    commands: dict[str, RoleCommand] = {}
    for node in self.prefill:
        commands[node.name] = self._build_vllm_worker_command(
            config,
            case,
            run_dir,
            node,
            kv_role="kv_producer",
            kv_config=self._structured_vllm_kv_config(node, kv_role="kv_producer"),
            extra_env=self._structured_vllm_node_env(node),
            image=image,
        )
    for node in self.decode:
        commands[node.name] = self._build_vllm_worker_command(
            config,
            case,
            run_dir,
            node,
            kv_role="kv_consumer",
            kv_config=self._structured_vllm_kv_config(node, kv_role="kv_consumer"),
            extra_env=self._structured_vllm_node_env(node),
            image=image,
        )
    commands[self.frontend.host] = self._build_builtin_vllm_pd_proxy_command(
        config,
        case,
        run_dir,
    )
    return commands
```

新增：

```python
def _structured_vllm_node_env(self, node: TopologyNode) -> Mapping[str, str]:
    assert self.vllm_pd is not None
    if self.vllm_pd.connector != "nixl":
        return types.MappingProxyType({})
    host = self.hosts[node.host]
    return {
        "VLLM_NIXL_SIDE_CHANNEL_HOST": host.address,
        "VLLM_NIXL_SIDE_CHANNEL_PORT": str(node.side_channel_port),
    }
```

- [ ] **步骤 9：实现内置 proxy 命令**

新增：

```python
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
    assert self.vllm_pd is not None
    image = self.frontend.image or config.run.bench_image
    argv = self._docker_run_base(
        case,
        run_dir,
        self.frontend.host,
        self.frontend.host,
        "frontend",
    )
    argv.extend(["--network", self.network])
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
```

新增：

```python
def _vllm_pd_proxy_endpoint_json(self, node: TopologyNode) -> str:
    assert self.vllm_pd is not None
    host = self.hosts[node.host]
    payload: dict[str, Any] = {
        "name": node.name,
        "url": f"http://{host.address}:{node.port}",
    }
    if self.vllm_pd.connector == "p2p_nccl":
        payload["kv_address"] = f"{host.address}:{node.kv_port}"
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
```

- [ ] **步骤 10：运行拓扑测试**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_remote_topology.py -q
```

预期：全部通过。

- [ ] **步骤 11：Commit**

```bash
git add vllm_standalone_bench/remote_topology.py vllm_standalone_bench/tests/test_remote_topology.py
git commit -m "feat: render structured vllm pd topology"
```

## 任务 4：写内置 proxy helper 和请求编排测试

**文件：**
- 创建：`vllm_standalone_bench/tests/test_pd_proxy.py`
- 创建：`vllm_standalone_bench/vllm_bench/pd_proxy.py`

- [ ] **步骤 1：创建 proxy helper 测试文件**

创建 `vllm_standalone_bench/tests/test_pd_proxy.py`：

```python
import asyncio
import json

from vllm_bench.pd_proxy import (
    Endpoint,
    build_nixl_prefill_body,
    build_p2p_prefill_body,
    build_p2p_request_id,
    inject_kv_transfer_params,
    parse_endpoint,
)


def test_parse_endpoint_accepts_json_with_kv_address():
    endpoint = parse_endpoint(json.dumps({
        "name": "p1",
        "url": "http://10.0.0.11:30000",
        "kv_address": "10.0.0.11:21001",
    }))

    assert endpoint == Endpoint(
        name="p1",
        url="http://10.0.0.11:30000",
        kv_address="10.0.0.11:21001",
    )


def test_build_p2p_request_id_matches_vllm_parser_format():
    request_id = build_p2p_request_id(
        "10.0.0.11:21001",
        "10.0.0.21:22001",
        request_uuid="abc",
    )

    assert request_id == (
        "___prefill_addr_10.0.0.11:21001"
        "___decode_addr_10.0.0.21:22001_abc"
    )


def test_build_p2p_prefill_body_forces_one_token_non_streaming():
    body = {"model": "m", "prompt": "hello", "max_tokens": 32, "stream": True}
    prefill_body = build_p2p_prefill_body(body, request_id="rid")

    assert prefill_body["max_tokens"] == 1
    assert prefill_body["stream"] is False
    assert prefill_body["request_id"] == "rid"
    assert body["max_tokens"] == 32


def test_build_nixl_prefill_body_injects_remote_decode_marker():
    body = {"model": "m", "messages": [], "max_tokens": 32, "stream": True}
    prefill_body = build_nixl_prefill_body(body)

    assert prefill_body["max_tokens"] == 1
    assert prefill_body["stream"] is False
    assert prefill_body["kv_transfer_params"] == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


def test_inject_kv_transfer_params_preserves_original_body():
    body = {"model": "m", "prompt": "hello"}
    params = {"remote_engine_id": "engine-a"}
    decode_body = inject_kv_transfer_params(body, params)

    assert decode_body == {
        "model": "m",
        "prompt": "hello",
        "kv_transfer_params": {"remote_engine_id": "engine-a"},
    }
    assert "kv_transfer_params" not in body
```

- [ ] **步骤 2：运行 helper 测试验证失败**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_pd_proxy.py -q
```

预期：导入失败，因为 `vllm_bench.pd_proxy` 尚不存在。

- [ ] **步骤 3：创建 pd_proxy.py helper 基础实现**

创建 `vllm_standalone_bench/vllm_bench/pd_proxy.py`，先写 helper：

```python
from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from aiohttp import ClientSession, web


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str
    kv_address: str | None = None


def parse_endpoint(value: str) -> Endpoint:
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError("endpoint must be a JSON object")
    name = raw.get("name")
    url = raw.get("url")
    kv_address = raw.get("kv_address")
    if not isinstance(name, str) or not name:
        raise argparse.ArgumentTypeError("endpoint.name must be a non-empty string")
    if not isinstance(url, str) or not url:
        raise argparse.ArgumentTypeError("endpoint.url must be a non-empty string")
    if kv_address is not None and not isinstance(kv_address, str):
        raise argparse.ArgumentTypeError("endpoint.kv_address must be a string")
    return Endpoint(name=name, url=url.rstrip("/"), kv_address=kv_address)


def build_p2p_request_id(
    prefill_kv_address: str,
    decode_kv_address: str,
    *,
    request_uuid: str | None = None,
) -> str:
    suffix = request_uuid or uuid.uuid4().hex
    return (
        f"___prefill_addr_{prefill_kv_address}"
        f"___decode_addr_{decode_kv_address}_{suffix}"
    )


def _one_token_body(body: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(body)
    copied["max_tokens"] = 1
    copied["stream"] = False
    return copied


def build_p2p_prefill_body(
    body: dict[str, Any],
    *,
    request_id: str,
) -> dict[str, Any]:
    copied = _one_token_body(body)
    copied["request_id"] = request_id
    return copied


def build_nixl_prefill_body(body: dict[str, Any]) -> dict[str, Any]:
    copied = _one_token_body(body)
    copied["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    return copied


def inject_kv_transfer_params(
    body: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    copied = copy.deepcopy(body)
    copied["kv_transfer_params"] = copy.deepcopy(params)
    return copied
```

- [ ] **步骤 4：运行 helper 测试确认通过**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_pd_proxy.py -q
```

预期：helper 测试通过。

- [ ] **步骤 5：增加 proxy 请求编排测试**

在 `test_pd_proxy.py` 追加 fake session：

```python
class FakeResponse:
    def __init__(self, payload, *, status=200, headers=None):
        self._payload = payload
        self.status = status
        self.headers = headers or {"content-type": "application/json"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode("utf-8")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)


def test_p2p_proxy_sends_prefill_then_decode():
    from vllm_bench.pd_proxy import PdProxy

    async def run_case():
        session = FakeSession([
            FakeResponse({"choices": []}),
            FakeResponse({"choices": [{"text": "ok"}]}),
        ])
        proxy = PdProxy(
            connector="p2p_nccl",
            prefill=[Endpoint("p1", "http://p1:30000", "p1:21001")],
            decode=[Endpoint("d1", "http://d1:31000", "d1:22001")],
            session=session,
        )

        status, headers, payload = await proxy.handle_json_completion(
            "/v1/completions",
            {"model": "m", "prompt": "hi", "max_tokens": 8},
        )
        return session, status, headers, payload

    session, status, headers, payload = asyncio.run(run_case())

    assert status == 200
    assert payload == b'{"choices": [{"text": "ok"}]}'
    assert session.calls[0][1] == "http://p1:30000/v1/completions"
    assert session.calls[1][1] == "http://d1:31000/v1/completions"
    prefill_json = session.calls[0][2]["json"]
    decode_json = session.calls[1][2]["json"]
    assert prefill_json["max_tokens"] == 1
    assert decode_json["request_id"] == prefill_json["request_id"]


def test_nixl_proxy_forwards_prefill_transfer_params_to_decode():
    from vllm_bench.pd_proxy import PdProxy

    async def run_case():
        params = {"remote_engine_id": "engine-a", "remote_block_ids": [1]}
        session = FakeSession([
            FakeResponse({"kv_transfer_params": params}),
            FakeResponse({"choices": [{"message": {"content": "ok"}}]}),
        ])
        proxy = PdProxy(
            connector="nixl",
            prefill=[Endpoint("p1", "http://p1:30000")],
            decode=[Endpoint("d1", "http://d1:31000")],
            session=session,
        )

        status, headers, payload = await proxy.handle_json_completion(
            "/v1/chat/completions",
            {"model": "m", "messages": [], "max_tokens": 8},
        )
        return params, session, status, headers, payload

    params, session, status, headers, payload = asyncio.run(run_case())

    assert status == 200
    assert payload == b'{"choices": [{"message": {"content": "ok"}}]}'
    assert session.calls[0][1] == "http://p1:30000/v1/chat/completions"
    assert session.calls[1][1] == "http://d1:31000/v1/chat/completions"
    assert session.calls[1][2]["json"]["kv_transfer_params"] == params
```

- [ ] **步骤 6：实现 PdProxy 类**

在 `pd_proxy.py` helper 后添加：

```python
class PdProxy:
    def __init__(
        self,
        *,
        connector: str,
        prefill: Iterable[Endpoint],
        decode: Iterable[Endpoint],
        session: Any,
    ) -> None:
        if connector not in {"p2p_nccl", "nixl"}:
            raise ValueError(f"unsupported connector: {connector}")
        self.connector = connector
        self.prefill = tuple(prefill)
        self.decode = tuple(decode)
        if not self.prefill or not self.decode:
            raise ValueError("prefill and decode endpoints are required")
        self._prefill_cycle = itertools.cycle(self.prefill)
        self._decode_cycle = itertools.cycle(self.decode)
        self.session = session

    async def handle_json_completion(
        self,
        path: str,
        body: dict[str, Any],
    ) -> tuple[int, dict[str, str], bytes]:
        prefill = next(self._prefill_cycle)
        decode = next(self._decode_cycle)
        if self.connector == "p2p_nccl":
            request_id = build_p2p_request_id(
                _required_kv_address(prefill),
                _required_kv_address(decode),
            )
            prefill_body = build_p2p_prefill_body(body, request_id=request_id)
            await self._post_json(prefill, path, prefill_body)
            decode_body = copy.deepcopy(body)
            decode_body["request_id"] = request_id
            return await self._post_json(decode, path, decode_body)

        prefill_body = build_nixl_prefill_body(body)
        _status, _headers, prefill_payload = await self._post_json(
            prefill,
            path,
            prefill_body,
        )
        prefill_json = json.loads(prefill_payload.decode("utf-8"))
        params = prefill_json.get("kv_transfer_params")
        if not isinstance(params, dict):
            raise web.HTTPBadGateway(
                reason="prefill response missing kv_transfer_params"
            )
        return await self._post_json(
            decode,
            path,
            inject_kv_transfer_params(body, params),
        )

    async def _post_json(
        self,
        endpoint: Endpoint,
        path: str,
        body: dict[str, Any],
    ) -> tuple[int, dict[str, str], bytes]:
        async with self.session.post(f"{endpoint.url}{path}", json=body) as response:
            payload = await response.read()
            return response.status, dict(response.headers), payload


def _required_kv_address(endpoint: Endpoint) -> str:
    if endpoint.kv_address is None:
        raise web.HTTPBadGateway(reason=f"{endpoint.name} missing kv_address")
    return endpoint.kv_address
```

- [ ] **步骤 7：运行 proxy 测试**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_pd_proxy.py -q
```

预期：全部通过。

- [ ] **步骤 8：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/pd_proxy.py vllm_standalone_bench/tests/test_pd_proxy.py
git commit -m "feat: add vllm pd proxy helpers"
```

## 任务 5：实现 proxy aiohttp CLI

**文件：**
- 修改：`vllm_standalone_bench/vllm_bench/pd_proxy.py`
- 修改：`vllm_standalone_bench/tests/test_pd_proxy.py`

- [ ] **步骤 1：新增 CLI parser 测试**

在 `test_pd_proxy.py` 添加：

```python
def test_parse_args_accepts_builtin_proxy_command():
    from vllm_bench.pd_proxy import parse_args

    args = parse_args([
        "--connector", "p2p_nccl",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--prefill", json.dumps({
            "name": "p1",
            "url": "http://p1:30000",
            "kv_address": "p1:21001",
        }),
        "--decode", json.dumps({
            "name": "d1",
            "url": "http://d1:31000",
            "kv_address": "d1:22001",
        }),
    ])

    assert args.connector == "p2p_nccl"
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.prefill[0].name == "p1"
```

- [ ] **步骤 2：实现 parse_args**

添加：

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM PD proxy")
    parser.add_argument("--connector", choices=("p2p_nccl", "nixl"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefill", action="append", type=parse_endpoint, required=True)
    parser.add_argument("--decode", action="append", type=parse_endpoint, required=True)
    return parser.parse_args(argv)
```

- [ ] **步骤 3：实现 aiohttp app**

添加：

```python
async def health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def root_v1(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app(proxy: PdProxy) -> web.Application:
    app = web.Application()
    app["proxy"] = proxy
    app.router.add_get("/health", health)
    app.router.add_get("/v1", root_v1)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/completions", handle_completion)
    app.router.add_post("/v1/chat/completions", handle_completion)
    return app
```

添加 handlers：

```python
async def handle_completion(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(reason="request body must be a JSON object")
    proxy: PdProxy = request.app["proxy"]
    status, headers, payload = await proxy.handle_json_completion(
        request.path,
        body,
    )
    content_type = headers.get("content-type", "application/json")
    return web.Response(body=payload, status=status, content_type=content_type)


async def handle_models(request: web.Request) -> web.StreamResponse:
    proxy: PdProxy = request.app["proxy"]
    endpoint = proxy.decode[0] if proxy.decode else proxy.prefill[0]
    async with proxy.session.get(f"{endpoint.url}/v1/models") as response:
        payload = await response.read()
        return web.Response(
            body=payload,
            status=response.status,
            content_type=response.headers.get("content-type", "application/json"),
        )
```

- [ ] **步骤 4：实现 main**

添加：

```python
async def _async_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    async with ClientSession() as session:
        proxy = PdProxy(
            connector=args.connector,
            prefill=args.prefill,
            decode=args.decode,
            session=session,
        )
        runner = web.AppRunner(create_app(proxy))
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.port)
        await site.start()
        await asyncio.Event().wait()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
```

- [ ] **步骤 5：运行 proxy 测试和 py_compile**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_pd_proxy.py -q
python -m py_compile vllm_standalone_bench/vllm_bench/pd_proxy.py
```

预期：测试和编译均通过。

- [ ] **步骤 6：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/pd_proxy.py vllm_standalone_bench/tests/test_pd_proxy.py
git commit -m "feat: expose builtin vllm pd proxy"
```

## 任务 6：补示例配置和文档

**文件：**
- 修改：`vllm_standalone_bench/README.md`
- 创建：`vllm_standalone_bench/configs/auto_bench.vllm_pd_p2p_remote.example.json`
- 创建：`vllm_standalone_bench/configs/auto_bench.vllm_pd_nixl_remote.example.json`
- 修改：`vllm_standalone_bench/tests/test_remote_topology.py`

- [ ] **步骤 1：新增示例配置可解析测试**

在 `test_remote_topology.py` 添加：

```python
@pytest.mark.parametrize("filename", [
    "auto_bench.vllm_pd_p2p_remote.example.json",
    "auto_bench.vllm_pd_nixl_remote.example.json",
])
def test_vllm_pd_example_configs_load(filename, monkeypatch):
    monkeypatch.setenv("P2_SSH_PASSWORD", "secret")
    config = ab.load_config(
        Path("vllm_standalone_bench/configs") / filename
    )

    assert len(config.topology_profiles) == 1
    assert config.topology_profiles[0].engine == "vllm"
    assert config.topology_profiles[0].vllm_pd is not None
```

确保文件顶部导入：

```python
from pathlib import Path
```

- [ ] **步骤 2：创建 P2P 示例配置**

以现有 SGLang PD 示例为模板创建
`vllm_standalone_bench/configs/auto_bench.vllm_pd_p2p_remote.example.json`。
关键 topology profile 必须包含：

```json
{
  "name": "vllm_pd_p2p_2p2d",
  "engine": "vllm",
  "mode": "pd",
  "provider": "ssh_docker",
  "network": "host",
  "image": "vllm/vllm-openai:0.23.0",
  "vllm_pd": {
    "connector": "p2p_nccl",
    "proxy": {"kind": "builtin"},
    "p2p_send_type": "PUT_ASYNC",
    "nccl_num_channels": 16
  },
  "prefill": [
    {"name": "p1", "host": "p1", "port": 30000, "kv_port": 21001, "gpus": "0"},
    {"name": "p2", "host": "p2", "port": 30000, "kv_port": 21002, "gpus": "0"}
  ],
  "decode": [
    {"name": "d1", "host": "d1", "port": 31000, "kv_port": 22001, "gpus": "0"},
    {"name": "d2", "host": "d2", "port": 31000, "kv_port": 22002, "gpus": "0"}
  ],
  "frontend": {"kind": "builtin", "host": "router", "port": 8000}
}
```

- [ ] **步骤 3：创建 NIXL 示例配置**

创建 `vllm_standalone_bench/configs/auto_bench.vllm_pd_nixl_remote.example.json`。
关键 topology profile 必须包含：

```json
{
  "name": "vllm_pd_nixl_2p2d",
  "engine": "vllm",
  "mode": "pd",
  "provider": "ssh_docker",
  "network": "host",
  "image": "vllm/vllm-openai:0.23.0",
  "vllm_pd": {
    "connector": "nixl",
    "proxy": {"kind": "builtin"}
  },
  "prefill": [
    {"name": "p1", "host": "p1", "port": 30000, "side_channel_port": 5601, "gpus": "0"},
    {"name": "p2", "host": "p2", "port": 30000, "side_channel_port": 5602, "gpus": "0"}
  ],
  "decode": [
    {"name": "d1", "host": "d1", "port": 31000, "side_channel_port": 5701, "gpus": "0"},
    {"name": "d2", "host": "d2", "port": 31000, "side_channel_port": 5702, "gpus": "0"}
  ],
  "frontend": {"kind": "builtin", "host": "router", "port": 8000}
}
```

- [ ] **步骤 4：更新 README**

在 remote PD 配置说明附近加入小节：

```markdown
### vLLM PD topology

vLLM PD supports structured `vllm_pd.connector` values `p2p_nccl` and `nixl`.
For `p2p_nccl`, every prefill/decode node needs `kv_port`; the built-in proxy
passes configured KV addresses through `request_id`. For `nixl`, every node
needs `side_channel_port`; the generated container env sets
`VLLM_NIXL_SIDE_CHANNEL_HOST` and `VLLM_NIXL_SIDE_CHANNEL_PORT`.

The built-in proxy runs from the bench-runner image through
`python -m vllm_bench.pd_proxy`. Local tests verify command rendering and proxy
request choreography; GPU/NIXL transport must be validated on the target hosts.
```

- [ ] **步骤 5：运行示例配置测试**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_remote_topology.py::test_vllm_pd_example_configs_load -q
```

预期：两个示例配置均可加载。

- [ ] **步骤 6：Commit**

```bash
git add \
  vllm_standalone_bench/README.md \
  vllm_standalone_bench/configs/auto_bench.vllm_pd_p2p_remote.example.json \
  vllm_standalone_bench/configs/auto_bench.vllm_pd_nixl_remote.example.json \
  vllm_standalone_bench/tests/test_remote_topology.py
git commit -m "docs: add vllm pd topology examples"
```

## 任务 7：最终验证和清理

**文件：**
- 修改：按测试反馈修正前面任务触达的文件

- [ ] **步骤 1：运行 targeted pytest**

运行：

```bash
python -m pytest vllm_standalone_bench/tests/test_remote_topology.py -q
python -m pytest vllm_standalone_bench/tests/test_pd_proxy.py -q
```

预期：全部通过。

- [ ] **步骤 2：运行编译检查**

运行：

```bash
python -m py_compile \
  vllm_standalone_bench/remote_topology.py \
  vllm_standalone_bench/vllm_bench/pd_proxy.py
```

预期：无输出，退出码 0。

- [ ] **步骤 3：运行空白检查**

运行：

```bash
git diff --check
```

预期：无输出，退出码 0。

- [ ] **步骤 4：检查工作树**

运行：

```bash
git status --short
```

预期：没有未提交代码变更；如果只有验证产物，删除或忽略验证产物后再检查。

- [ ] **步骤 5：记录验证边界**

最终回复中必须说明：

```text
已验证：拓扑命令渲染、结构化配置校验、proxy helper/请求编排、Python 编译、git diff --check。
未在本地验证：真实 GPU P2P/NCCL 和 NIXL 跨节点传输，需要目标机器运行示例配置验证。
```
