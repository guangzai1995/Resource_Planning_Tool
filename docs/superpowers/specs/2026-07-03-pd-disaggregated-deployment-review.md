# PD（Prefill-Decode 分离）部署能力审查报告

- **日期**：2026-07-03
- **审查对象**：`vllm_standalone_bench` 新增的 PD 分离部署功能（`feat/pd-remote-docker-auto-bench` 分支，合并于 `66e20af`）
- **参考实现**：`sglang-main`、`vllm-0.23.0`
- **审查重点**：命令渲染正确性（端到端**尚未跑通**，仅有代码 + 单测）
- **方法**：把 `remote_topology.py` / `auto_bench.py` 生成的每一条命令、每一个 flag/取值，逐一对照参考实现源码（逐字摘录）
- **结论状态**：本报告仅为诊断留档，**暂不修改代码**（用户决定）。C1 / H1 等问题待后续单独排期修复。

---

## 0. 结论速览

| 级别 | 引擎 | 编号 | 问题 | 现状 |
|---|---|---|---|---|
| 🔴 必须修复 | vLLM | C1 | kv-transfer-config 生成模型与 vLLM 0.23.0 实际机制根本不符 | **跑不起来** |
| 🟠 建议修改 | SGLang | H1 | router 没把 prefill 的 bootstrap 端口传给 router | 示例靠"端口全相同"侥幸绕过 |
| 🟡 建议修改 | vLLM | M2 | 缺 PD 请求路由层（proxy），责任全甩给用户 | 无法工作 |
| 🟡 建议修改 | SGLang | M1 | 给 decode 传 bootstrap 端口是依赖回退语义的隐式耦合 | 与 H1 同根因 |
| 🔵 仅供参考 | 通用 | L1 | 就绪检查不覆盖 KV bootstrap 握手 | 非阻塞 |
| 🔵 仅供参考 | vLLM | L2 | 模板含非标字段、缺必要字段 | 非阻塞 |
| 🔵 仅供参考 | 通用 | L3 | 不支持跨机 TP | 能力边界 |

**一句话**：SGLang PD 基本可跑（H1 是 latent bug，被示例配置碰巧绕过）；vLLM PD 按现状**无法端到端工作**（C1 + M2）。

---

## 1. 我们的实现现状（背景）

核心三文件：

- `remote_topology.py`：PD 拓扑数据模型 + docker 命令渲染
- `auto_bench.py`：PD 生命周期编排（启动顺序 prefill→decode→frontend、ready 轮询、清理）
- `remote_docker.py`：SSH 远程执行 + 密钥脱敏

两套引擎：

- **SGLang PD**（`remote_topology.py:104-242`）：worker 用 `--disaggregation-mode {prefill|decode}` + `--disaggregation-transfer-backend` + `--disaggregation-bootstrap-port` + `--disaggregation-ib-device`；router 用 `sglang_router.launch_router --pd-disaggregation --prefill URL... --decode URL...`。**有示例配置** `configs/auto_bench.sglang_pd_remote.example.json`（2P2D）。
- **vLLM PD**（`remote_topology.py:244-388`）：prefill=`kv_producer`(rank 从 0 递增)、decode=`kv_consumer`(rank 接续)，`kv_parallel_size = len(prefill)+len(decode)`，通过 `--kv-transfer-config` 模板注入；frontend 必须 `kind:external`（外部 proxy）。**只有单测、无示例配置**。

请求统一打到 frontend：`auto_bench.py:case_endpoint_base_url` 对 topology case 返回 `http://{frontend_host}:{frontend_port}/v1`。

---

## 2. 发现详解

### 🔴 C1 — vLLM PD 配置模型根本性错误（跑不起来）

**我们的实现**（`remote_topology.py:251-277`）：

```python
total_workers = len(self.prefill) + len(self.decode)   # 2P2D → 4
rank = 0
for node in self.prefill:   # kv_role=kv_producer, kv_rank=0,1
    ...
for node in self.decode:    # kv_role=kv_consumer, kv_rank=2,3
    ...
```

即对 2P2D 生成单个 `kv_parallel_size=4`、rank `0/1`(prefill) / `2/3`(decode) 的配置。测试 `tests/test_remote_topology.py:196-203` 把这个错误行为断言成期望：

```python
assert str(p1_kv_config["kv_parallel_size"]) == "4"
...
assert str(d1_kv_config["kv_rank"]) == "2"
```

**vLLM 0.23.0 的真实机制**（逐字证据）：

1. `vllm-0.23.0/vllm-0.23.0/vllm/config/kv_transfer.py:45-52` —— 字段语义写死了 1P1D：

```python
kv_rank: int | None = None
"""The rank of this vLLM instance in the KV cache transfer. Typical value:
0 for prefill instance, 1 for decode instance.
Currently only 1P1D is supported."""

kv_parallel_size: int = 1
"""The number of parallel instances for KV cache transfer. For
P2pNcclConnector, this should be 2."""
```

2. `__post_init__`（同文件 `:93-107`）**只校验 `kv_role`，完全不校验 `kv_rank` 范围或 `kv_parallel_size`** → 传 4/3 不报错，是**静默错误**：

```python
def __post_init__(self) -> None:
    if self.engine_id is None:
        self.engine_id = str(uuid.uuid4())
    if self.kv_role is not None and self.kv_role not in get_args(KVRole):
        raise ValueError(...)
    if self.kv_connector is not None and self.kv_role is None:
        raise ValueError(...)
```

3. v1 的 `P2pNcclConnector` / `NixlConnector` **完全不读取 `kv_rank` / `kv_parallel_size`**（整个 `vllm/distributed/kv_transfer/` 目录 grep 这两个字段零命中，只有 legacy `lmcache_mp_connector.py` 用）。角色与 rank 来自：

```python
# p2p_nccl_connector.py:87-90
self.is_producer = self._kv_transfer_config.is_kv_producer
self._rank = get_world_group().rank if role == KVConnectorRole.WORKER else 0
```

4. P2P NCCL communicator **硬编码 2-way**（`p2p_nccl_engine.py`）：

```python
# 发起侧 :222-224
rank = 0
comm: ncclComm_t = self.nccl.ncclCommInitRank(2, unique_id, rank)
# 监听侧 :381-385
rank = 1
comm: ncclComm_t = self.nccl.ncclCommInitRank(2, unique_id, rank)
```

收发用 `rank ^ 1`（即 0↔1 配对，`:533` 等）。连接按 `remote_address` 索引（`self.comms: dict`，`:169`），**不存在 4-way 组**。

5. 官方 XpYd 真实启动方式（`examples/disaggregated/p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh:163-212`）：**N 个独立 vLLM 进程**，各自独立 config + 独立 `kv_port`，**根本不传 `kv_rank` / `kv_parallel_size`**，靠 proxy（`disagg_proxy_p2p_nccl_xpyd.py:138-160`）做服务发现 + 请求级动态 1:1 配对（`count % len(list)` 轮询，把配对地址塞进 `request_id`）。

**后果**：我们生成的配置既违反字段语义、也不会产生正确的 NpNd 拓扑——"配错了但不报错"。叠加 M2（无 PD 路由层），**vLLM PD 按现状无法端到端工作**。

**修复方向（本次不实施，仅记录）**：
- **A（推荐）**：把 `engine=="vllm"` 在 `build_commands` 阶段降级为"实验性/未支持"，抛错或显眼告警，文档注明待实现。
- **B**：重写为"P2pNccl 1P1D 配对 + proxy"模式（成对独立 config，`kv_parallel_size=2`，独立 `kv_port`，提供 `disagg_proxy` 镜像）。
- **C**：改用 `NixlConnector`（原生支持 NpNd，靠每实例 `engine_id` 点对点握手，`nixl/scheduler.py:57-62`、`nixl/worker.py:399-409`）+ 提供 PD proxy + 正确 NIXL config。
- **必做**：同步改/删测试 `test_vllm_pd_worker_command_renders_kv_template` 对 `kv_parallel_size==4` / `kv_rank==2` 的断言。

---

### 🟠 H1 — SGLang router 未传 prefill 的 bootstrap 端口

**我们的实现**（`remote_topology.py:224-229`）：

```python
for node in self.prefill:
    host = self.hosts[node.host]
    argv.extend(["--prefill", f"http://{host.address}:{node.port}"])
for node in self.decode:
    host = self.hosts[node.host]
    argv.extend(["--decode", f"http://{host.address}:{node.port}"])
```

router 只收 `--prefill http://{addr}:{port}`，不带 bootstrap 端口。测试 `tests/test_remote_topology.py:113` 甚至断言 `"12335" not in values_after(router, "--prefill")`——把"不带"固化成期望。

**SGLang 的发现机制**（逐字证据，已纠正第一轮探索的误判）：

1. router 把 **prefill**（不是 decode）的 `bootstrap_host` / `bootstrap_port` 注入请求体（`sgl-model-gateway/src/routers/http/pd_router.rs:267-284`）：

```rust
obj.insert(BOOTSTRAP_HOST_KEY.to_string(), Value::from(prefill_worker.bootstrap_host()));
obj.insert(BOOTSTRAP_PORT_KEY.to_string(),
    match prefill_worker.bootstrap_port() {
        Some(v) => Value::from(v),
        None => Value::Null,
    });
obj.insert(BOOTSTRAP_ROOM_KEY.to_string(), Value::from(generate_room_id()));
```

2. `--prefill` 的 argparse（`sgl-model-gateway/bindings/python/src/sglang_router/router_args.py:386-393`）：

```python
pd_group.add_argument(
    f"--{prefix}prefill",
    nargs="+",
    action="append",
    help="... Format: --prefill URL [BOOTSTRAP_PORT]. ...",
)
```

`bootstrap_host` 从 URL 的 host 解析（`worker_builder.rs:136-157`，**不含端口**），`bootstrap_port` 来自可选第二参数（`worker_builder.rs:159-162`，未传则 `None`）。

3. decode 从请求体读这两个字段，拼成地址反查 prefill 拓扑（`python/sglang/srt/disaggregation/decode.py:100-102`）：

```python
def _bootstrap_addr(req: Req) -> str:
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()
```

```python
# common/conn.py:242-247
url = f"http://{bootstrap_addr}/route?prefill_dp_rank={-1}&..."
response = requests.get(url, timeout=5)
```

4. **不传 bootstrap port 时**：router 注入 `null`，decode 回退用自己的端口（`scheduler.py:1977-1979`）：

```python
if recv_req.bootstrap_port is None:
    # Use default bootstrap port
    recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port
```

**后果**：只在**所有实例 `bootstrap_port` 取值相同**时碰巧工作（decode 回退端口 == prefill 监听端口）。当前示例配置（p1/p2/d1/d2 全部 `12335`）恰好绕过。一旦各 prefill 用不同 bootstrap port（SGLang CI 标准做法 `9001..9004`，`scripts/ci/cuda/ci_start_disaggregation_servers.sh`），decode 会连到 `<prefill_host>:<错误端口>` → `try_ensure_parallel_info` 失败 → 请求 abort。是脆弱的隐式依赖。

**修复方向（本次不实施，仅记录）**：

- router 命令改为 `--prefill http://{addr}:{port} {bootstrap_port}`（每个 prefill 带 bootstrap 端口）。`node.bootstrap_port` 已有；若未配则用 sglang 默认 `8998`（`server_args.py:7370-7375`），或在 PD 模式下强制必填。
- 同步反转测试 `test_sglang_pd_commands_render_worker_and_router_flags:113` 的断言。
- 修了 H1 后 decode 不再需要 `--disaggregation-bootstrap-port`（顺带消除 M1）。

---

### 🟡 M1 — SGLang 给 decode 传 bootstrap 端口是隐式耦合

**我们的实现**（`remote_topology.py:175-179`）：prefill 和 decode 都传 `--disaggregation-bootstrap-port`。

**参考证据**（`sglang-main`）：

- bootstrap HTTP server **只在 prefill 启动**（`python/sglang/srt/managers/disagg_service.py:14-44`，`if disagg_mode == DisaggregationMode.PREFILL:`，注释 `# only start bootstrap server on prefill tm`）。
- decode 侧该参数不启 server，只作"请求未带 bootstrap_port 时的默认目标端口"（`scheduler.py:1979`）。
- `--disaggregation-bootstrap-port` 默认 `8998`，help 文本明写 "Bootstrap server port on the **prefill** server"（`server_args.py:7370-7375`）。

**后果**：decode 这个参数必须 == prefill 的监听端口才能让 H1 的回退生效。与 H1 同根因，修了 H1 后此参数对 decode 多余（但无害）。

---

### 🟡 M2 — vLLM PD 缺少 PD 请求路由层（proxy）

**我们的实现**：`frontend.kind="external"` 只跑用户提供的 `pd-proxy` 命令（`remote_topology.py:341-388`），框架不提供这个 proxy、也不校验它是否实现了 PD 路由。

**参考证据**：SGLang 有现成 `sglang_router --pd-disaggregation`；**vLLM 没有内置 PD router**，必须外挂 proxy 做"先 prefill（max_tokens=1）再 decode"两跳转发（`vllm-0.23.0/.../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py`、`examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py`）。没有它，即便 worker 命令对了请求也走不通 P→D。

**后果**：vLLM PD 即使修好 C1，仍需补一个正确的 proxy（或借用 vLLM 自带示例）才能跑通。

---

### 🔵 L1 — PD 就绪检查不覆盖 KV bootstrap 握手

**我们的实现**：所有 role 轮询 `http://{host}:{port}/v1`（`auto_bench.py:topology_role_ready_url` ≈ L1880-1890）。

**参考证据**：SGLang 的 bootstrap 发现是**惰性 per-request**的（首个请求才触发 decode↔prefill 握手，`scheduler.py` 在 `handle_generate_request` 里才组装 bootstrap 地址）；bootstrap server 在所有 rank 注册完成前返回 503（`common/conn.py:1353`）。

**后果**：`/v1` 通 ≠ PD 就绪。首请求会偏慢 / 触发 `try_ensure_parallel_info` 重试（5 次指数退避，`common/conn.py:235-278`）。非硬 bug，但首请求延迟 / 偶发失败需留意。

---

### 🔵 L2 — vLLM 模板含非标字段、缺必要字段

**我们的实现**（测试模板 `tests/test_remote_topology.py:166-175`）：

```python
topology["kv_transfer_config_template"] = {
    "kv_connector": "NixlConnector",
    "kv_role": "{kv_role}", "kv_rank": "{kv_rank}",
    "kv_parallel_size": "{kv_parallel_size}",
    "node_name": "{node_name}", "node_address": "{node_address}",
    "node_port": "{node_port}", "run_id": "{run_id}",
}
```

**问题**：

- `node_name` / `node_address` / `node_port` / `run_id` **不是 `KVTransferConfig` 字段**（`KVTransferConfig` 的真实字段见 `kv_transfer.py`）。`from_json` 是否严格拒绝未知字段**待确认**——若严格，vLLM PD 会启动即报错；若宽松，则被静默忽略（无意义）。
- P2pNccl 必需的 `kv_port`（ZMQ 端点，见 `disaggregated_prefill.sh:64`）和 `kv_connector_extra_config`（`proxy_ip`/`proxy_port`/`http_port`/`send_type` 等）模板里没有。

**后果**：即便走 1P1D，模板也不完整。

---

### 🔵 L3 — 不支持跨机 TP

**我们的实现**：每个 `TopologyNode` 绑定单个 `host`（`remote_topology.py:28-39`、`784-816`）。

**后果**：无法表达"一个 prefill 跨多机 TP"（需 `--dist-init-addr` / `--nnodes` / `--node-rank` 等）。当前只支持"每实例单机 TP"。对单机 TP 的 2P2D 场景够用，是能力边界而非 bug。

---

## 3. 做对的地方（避免误改）

- **argv→shell 用 `shlex.join`**（`remote_docker.py:130`：`script = f"exec {shlex.join(command)}\n"`，经 `ssh ... sh -s` stdin 执行）。`--kv-transfer-config '{"a":"b"}'`、`--gpus device=0,1,2,3` 等含特殊字符的参数都能正确转义。
- **密钥脱敏**（`mask_command` `remote_docker.py:79-107`）覆盖 `--api-key` / `password` / `secret` / `token` 及敏感 env（`API_KEY`/`PASSWORD`/`SECRET`/`TOKEN`），`config_to_dict` 也脱敏（测试 `test_config_to_dict_masks_inline_password` / `test_password_env_resolved_config_keeps_env_name_not_value`）。
- **容器归属校验 + 清理**：docker label（`auto_bench.py:55-57`、`remote_topology.py:461-469`）+ `remove_existing_topology_role_if_owned` / `cleanup_topology_roles_best_effort`，防 stale 容器冲突。
- **角色名全局唯一校验**（`_validate_role_names_unique` `remote_topology.py:761-781`）。
- **SGLang flag 合法性**：`--tp-size` 与 `--tensor-parallel-size` 同义（`server_args.py:5250-5256`）；`transfer_backend` 小写 `mooncake`/`nixl` 与 choices（`server_args.py:213-220`：`mooncake/nixl/ascend/fake/mori/mooncake_tcp`）匹配；`--disaggregation-mode` choices `null/prefill/decode`（`server_args.py:7356-7362`）；`sglang.launch_server` / `sglang_router.launch_router` 模块名均有效。
- **跨机独占整机 GPU 不需要 `--base-gpu-id`**（默认 0；仅同机混部多实例才需要，`server_args.py:5376-5381`，PD 路径未引用该参数）。

---

## 4. 修复优先级建议（本次不实施）

| 编号 | 引擎 | 建议优先级 | 工作量 | 备注 |
|---|---|---|---|---|
| H1 | SGLang | 高（但被示例绕过，非阻断） | 小（外科手术式） | 改 router 命令 + 反转测试断言 |
| C1 | vLLM | 高（功能不可用） | 大（重写或降级） | 建议先降级告警（A），再决定 B/C |
| M2 | vLLM | 高（C1 的前置） | 中 | 需引入 proxy |
| M1 | SGLang | 低（随 H1 一起解决） | 小 | — |
| L1/L2/L3 | 通用 | 低 | 小 | 非阻塞 |

---

## 5. 附录：关键源码位置速查

### 我们的代码（`vllm_standalone_bench/`）

| 用途 | 文件 | 行 |
|---|---|---|
| SGLang worker 命令 | `remote_topology.py` | `_build_sglang_worker_command` 136-191 |
| SGLang router 命令（**H1**） | `remote_topology.py` | `_build_sglang_router_command` 193-242；`--prefill/--decode` 224-229 |
| vLLM PD 命令（**C1**） | `remote_topology.py` | `_build_vllm_pd_commands` 244-282；`kv_parallel_size` 251 |
| vLLM worker 命令 | `remote_topology.py` | `_build_vllm_worker_command` 284-339 |
| vLLM kv 模板渲染 | `remote_topology.py` | `_render_kv_transfer_config` 390-418 |
| external frontend（**M2**） | `remote_topology.py` | `_build_external_frontend_command` 341-388 |
| argv→shell | `remote_docker.py` | `RemoteDockerRunner.run` 114-169；`shlex.join` 130 |
| 启动顺序 / ready URL | `auto_bench.py` | `_role_start_order` ≈1858-1877；`topology_role_ready_url` ≈1880-1890 |
| 请求 endpoint | `auto_bench.py` | `case_endpoint_base_url` ≈1081-1091 |
| PD 固化断言 | `tests/test_remote_topology.py` | 113、196-203 |

### 参考实现

| 用途 | 文件 |
|---|---|
| SGLang disaggregation 参数 | `sglang-main/python/sglang/srt/server_args.py` 7356-7386、213-220、5250-5256 |
| bootstrap server 仅 prefill | `sglang-main/python/sglang/srt/managers/disagg_service.py` 14-44 |
| decode 回退自身 bootstrap 端口 | `sglang-main/python/sglang/srt/managers/scheduler.py` 1977-1979 |
| decode 反查 prefill /route | `sglang-main/python/sglang/srt/disaggregation/{decode.py:100-102, common/conn.py:235-278}` |
| router 注入 bootstrap（prefill 来源） | `sglang-main/sgl-model-gateway/src/routers/http/pd_router.rs` 223-286 |
| router `--prefill [BOOTSTRAP_PORT]` | `sglang-main/sgl-model-gateway/bindings/python/src/sglang_router/router_args.py` 386-393；`worker_builder.rs` 136-162 |
| vLLM KVTransferConfig | `vllm-0.23.0/vllm-0.23.0/vllm/config/kv_transfer.py` 45-52、93-107 |
| vLLM P2P NCCL（硬编码 2-way） | `vllm-0.23.0/vllm-0.23.0/vllm/distributed/kv_transfer/kv_connector/v1/p2p/{p2p_nccl_connector.py:87-104, p2p_nccl_engine.py:222-224,381-385}` |
| vLLM NIXL（原生 NpNd） | `vllm-0.23.0/vllm-0.23.0/vllm/distributed/kv_transfer/kv_connector/v1/nixl/` |
| vLLM 1P1D 示例 | `vllm-0.23.0/vllm-0.23.0/examples/disaggregated/disaggregated_prefill.sh` 57-74 |
| vLLM XpYd 示例 + proxy | `vllm-0.23.0/vllm-0.23.0/examples/disaggregated/p2p_nccl_xpyd/{disagg_example_p2p_nccl_xpyd.sh:163-212, disagg_proxy_p2p_nccl_xpyd.py:138-160}` |

---

## 6. 待确认项

- **L2**：`KVTransferConfig.from_json` 是否严格拒绝 `node_name` / `node_address` / `node_port` / `run_id` 等非标字段（严格→启动即报错；宽松→静默忽略）。本次未读 `from_json` 实现，留待修复时确认。
- SGLang mooncake 传输在无 RDMA 环境下需 `MC_FORCE_TCP=1`（可通过 `node.env` / `profile.env` 注入，框架已支持，但示例配置未设置、也无文档提示）。
