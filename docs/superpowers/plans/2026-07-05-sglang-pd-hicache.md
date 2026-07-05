# SGLang PD HiCache 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `vllm_standalone_bench` 中为 SGLang PD 拓扑增加结构化 HiCache 配置、命令渲染、cache-source 指标落盘和远程示例配置。

**架构：** `remote_topology.py` 增加 `SglangHiCacheConfig`，解析 `topology_profiles[].sglang_hicache` 并在 SGLang worker Docker 命令中渲染源码支持的 CLI flags。`vllm_bench/serve.py` 扩展现有 `/metrics` 解析和 before/after counter 差分，`run_bench_multi.py` 负责把新指标写进固定 CSV/XLSX 表头。示例配置沿用现有 remote topology JSON 风格，不管理 Mooncake control plane。

**技术栈：** Python dataclass、pytest、SGLang CLI flags、Prometheus text metrics、Docker command rendering、JSON config。

---

## 规格来源

- 设计规格：`docs/superpowers/specs/2026-07-05-sglang-pd-hicache-design.md`
- SGLang 源码依据：
  - `/work/development-code/Resource_Planning_Tool/sglang-main/python/sglang/srt/server_args.py`
  - `/work/development-code/Resource_Planning_Tool/sglang-main/python/sglang/srt/environ.py`
  - `/work/development-code/Resource_Planning_Tool/sglang-main/python/sglang/srt/observability/metrics_collector.py`
  - `/work/development-code/Resource_Planning_Tool/sglang-main/docs/advanced_features/hicache_best_practices.md`

## 文件结构

- 修改：`vllm_standalone_bench/remote_topology.py`
  - 增加 `SglangHiCacheConfig` dataclass。
  - 增加 `sglang_hicache` 字段到 `TopologyProfile`。
  - 增加 `_parse_sglang_hicache_config` 和 SGLang `transfer_backend` 校验。
  - 增加 `_sglang_hicache_args(role)`，在 SGLang worker 命令中渲染 HiCache flags。
  - 将 `mount_infiniband` Docker flags 复用到 SGLang worker。
- 修改：`vllm_standalone_bench/tests/test_remote_topology.py`
  - 增加 SGLang HiCache 渲染、校验和 IB mount 测试。
- 修改：`vllm_standalone_bench/vllm_bench/serve.py`
  - 增加 `CacheSourceMetrics`。
  - 扩展 `RuntimeMetrics`、`parse_runtime_metrics_text`、benchmark before/after 差分和 result JSON。
- 修改：`vllm_standalone_bench/tests/test_serve_metrics.py`
  - 增加 SGLang cache-source Prometheus 解析与差分测试。
- 修改：`vllm_standalone_bench/run_bench_multi.py`
  - `_extract_row` 读取 cache-source 字段。
  - `CSV_HEADERS`、`CSV_HEADERS_ZH` 增加新列。
  - XLSX 说明表增加新指标解释。
- 修改：`vllm_standalone_bench/tests/test_extract_row.py`
  - 覆盖新字段提取和默认值。
- 修改：`vllm_standalone_bench/tests/test_result_csv_headers.py`
  - 更新固定中英文表头期望。
- 创建：`vllm_standalone_bench/configs/auto_bench.sglang_pd_hicache_remote.example.json`
  - dry-run-safe 示例，使用 RFC 5737 文档地址。
- 创建：`vllm_standalone_bench/configs/auto_bench.sglang_pd_hicache_remote_minimax.json`
  - Minimax 主机布局模板，2P2D full async offload + prefill-only isolation。
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`
  - 增加新示例配置 load 和 dry-run command 断言。
- 修改：`vllm_standalone_bench/README.md`
  - 增加 SGLang PD + HiCache 配置说明、Mooncake 前置条件和指标解释。

---

### 任务 1：拓扑层测试先行

**文件：**
- 修改：`vllm_standalone_bench/tests/test_remote_topology.py`

- [ ] **步骤 1：编写 prefill-only 命令渲染失败测试**

在 `test_sglang_pd_commands_render_disaggregation_ib_device` 后加入：

```python
def test_sglang_pd_hicache_prefill_only_renders_prefill_flags(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["image"] = "sglang:pd"
    topology["sglang_hicache"] = {
        "mode": "prefill_only",
        "page_size": 64,
        "ratio": 2.0,
        "size": 0,
        "write_policy": "write_through",
        "io_backend": "direct",
        "mem_layout": "page_first_direct",
        "storage_backend": "mooncake",
        "storage_prefetch_policy": "timeout",
        "storage_backend_extra_config": {"tp_lcm_size": 4},
        "enable_metrics": True,
        "enable_cache_report": True,
    }

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    p1 = commands["p1"].argv
    d1 = commands["d1"].argv
    assert value_after(p1, "--page-size") == "64"
    assert value_after(d1, "--page-size") == "64"
    assert "--enable-hierarchical-cache" in p1
    assert "--enable-hierarchical-cache" not in d1
    assert value_after(p1, "--hicache-storage-backend") == "mooncake"
    assert "--hicache-storage-backend" not in d1
    assert value_after(p1, "--hicache-storage-backend-extra-config") == '{"tp_lcm_size":4}'
    assert "--enable-metrics" in p1
    assert "--enable-cache-report" in p1
    assert "--enable-metrics" not in d1
    assert "--enable-cache-report" not in d1
```

- [ ] **步骤 2：编写 full-async-offload 命令渲染失败测试**

继续在同一文件加入：

```python
def test_sglang_pd_hicache_full_async_renders_decode_offload(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["image"] = "sglang:pd"
    topology["sglang_hicache"] = {
        "mode": "full_async_offload",
        "page_size": 64,
        "storage_backend": "mooncake",
    }

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    p1 = commands["p1"].argv
    d1 = commands["d1"].argv
    assert "--enable-hierarchical-cache" in p1
    assert "--disaggregation-decode-enable-offload-kvcache" not in p1
    assert "--enable-hierarchical-cache" not in d1
    assert "--disaggregation-decode-enable-offload-kvcache" in d1
    assert value_after(p1, "--hicache-storage-backend") == "mooncake"
    assert value_after(d1, "--hicache-storage-backend") == "mooncake"
    assert value_after(d1, "--hicache-storage-prefetch-policy") == "timeout"
```

- [ ] **步骤 3：编写 SGLang worker IB mount 失败测试**

继续在同一文件加入：

```python
def test_sglang_pd_mount_infiniband_adds_worker_flags(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["image"] = "sglang:pd"
    topology["mount_infiniband"] = True

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    p1 = commands["p1"].argv
    router = commands["router"].argv
    assert value_after(p1, "--device") == "/dev/infiniband"
    assert "IPC_LOCK" in p1
    assert "memlock=-1:-1" in p1
    assert "--device" not in router
```

- [ ] **步骤 4：编写配置校验失败测试**

继续在同一文件加入：

```python
@pytest.mark.parametrize("patch, match", [
    ({"mode": "bad"}, "mode"),
    ({"mode": "prefill_only", "write_policy": "bad"}, "write_policy"),
    ({"mode": "prefill_only", "io_backend": "bad"}, "io_backend"),
    ({"mode": "prefill_only", "mem_layout": "bad"}, "mem_layout"),
    ({"mode": "prefill_only", "storage_backend": "bad"}, "storage_backend"),
    ({"mode": "prefill_only", "storage_prefetch_policy": "bad"}, "storage_prefetch_policy"),
    ({"mode": "prefill_only", "page_size": 0}, "page_size"),
    ({"mode": "prefill_only", "ratio": 0}, "ratio"),
    ({"mode": "prefill_only", "size": -1}, "size"),
    ({"mode": "prefill_only", "storage_backend_extra_config": []}, "storage_backend_extra_config"),
    ({"mode": "prefill_only", "unknown": True}, "unsupported keys"),
])
def test_sglang_pd_hicache_rejects_invalid_config(tmp_path, patch, match):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["sglang_hicache"] = patch

    with pytest.raises(ab.ConfigError, match=match):
        ab.load_config(write_config(tmp_path, data))


def test_sglang_pd_hicache_full_async_requires_storage_backend(tmp_path):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["sglang_hicache"] = {"mode": "full_async_offload"}

    with pytest.raises(ab.ConfigError, match="storage_backend"):
        ab.load_config(write_config(tmp_path, data))


def test_sglang_pd_rejects_invalid_transfer_backend(tmp_path):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["transfer_backend"] = "bad"

    with pytest.raises(ab.ConfigError, match="transfer_backend"):
        ab.load_config(write_config(tmp_path, data))


def test_vllm_pd_rejects_sglang_hicache_config(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {"connector": "nixl", "proxy": {"kind": "builtin"}}
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}
    for i, node in enumerate(topology["prefill"] + topology["decode"]):
        node["side_channel_port"] = 5600 + i
    topology["sglang_hicache"] = {"mode": "prefill_only"}

    with pytest.raises(ab.ConfigError, match="sglang_hicache"):
        ab.load_config(write_config(tmp_path, data))
```

- [ ] **步骤 5：运行拓扑测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_remote_topology.py -q
```

预期：新增测试失败，报错包含 `sglang_hicache` 字段不存在、`--enable-hierarchical-cache` 未渲染或非法配置未被拒绝。

---

### 任务 2：实现 SGLang HiCache 拓扑配置和命令渲染

**文件：**
- 修改：`vllm_standalone_bench/remote_topology.py`

- [ ] **步骤 1：增加配置 dataclass**

在 `VllmPdConfig` 后加入：

```python
@dataclass(frozen=True)
class SglangHiCacheConfig:
    mode: str
    page_size: int | None = 64
    ratio: float | None = 2.0
    size: int | None = 0
    write_policy: str | None = "write_through"
    io_backend: str | None = "direct"
    mem_layout: str | None = "page_first_direct"
    storage_backend: str | None = None
    storage_prefetch_policy: str | None = "timeout"
    storage_backend_extra_config: Mapping[str, Any] | str | None = None
    enable_metrics: bool = True
    enable_cache_report: bool = True
```

在 `TopologyProfile` 字段列表中 `vllm_pd` 前加入：

```python
    sglang_hicache: SglangHiCacheConfig | None = None
```

- [ ] **步骤 2：增加 parser 常量和 helper**

在 `_parse_vllm_pd_config` 前加入：

```python
SGLANG_DISAGG_TRANSFER_BACKENDS = {
    "mooncake",
    "nixl",
    "ascend",
    "fake",
    "mori",
    "mooncake_tcp",
}
SGLANG_HICACHE_MODES = {"prefill_only", "full_async_offload"}
SGLANG_HICACHE_WRITE_POLICIES = {
    "write_back",
    "write_through",
    "write_through_selective",
}
SGLANG_HICACHE_IO_BACKENDS = {"direct", "kernel", "kernel_ascend"}
SGLANG_HICACHE_MEM_LAYOUTS = {
    "layer_first",
    "page_first",
    "page_first_direct",
    "page_first_kv_split",
    "page_head",
}
SGLANG_HICACHE_STORAGE_BACKENDS = {
    "file",
    "mooncake",
    "hf3fs",
    "nixl",
    "aibrix",
    "dynamic",
    "eic",
    "simm",
}
SGLANG_HICACHE_STORAGE_PREFETCH_POLICIES = {
    "best_effort",
    "wait_complete",
    "timeout",
}


def _optional_positive_float(value: Any, path: str, error: ErrorFactory) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise error(f"{path} must be a positive number")
    return float(value)


def _optional_non_negative_int(value: Any, path: str, error: ErrorFactory) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise error(f"{path} must be a non-negative integer")
    return value


def _optional_bool(value: Any, path: str, error: ErrorFactory, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise error(f"{path} must be a boolean")
    return value
```

- [ ] **步骤 3：增加 `_parse_sglang_hicache_config`**

在 `_parse_vllm_pd_config` 前加入：

```python
def _parse_sglang_hicache_config(
    value: Any,
    path: str,
    error: ErrorFactory,
) -> SglangHiCacheConfig | None:
    if value is None:
        return None
    raw = _mapping(value, path, error)
    allowed = {
        "mode",
        "page_size",
        "ratio",
        "size",
        "write_policy",
        "io_backend",
        "mem_layout",
        "storage_backend",
        "storage_prefetch_policy",
        "storage_backend_extra_config",
        "enable_metrics",
        "enable_cache_report",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise error(f"{path} contains unsupported keys: {', '.join(unknown)}")
    mode = _string(_required(raw, "mode", f"{path}.mode", error), f"{path}.mode", error)
    if mode not in SGLANG_HICACHE_MODES:
        raise error(f"{path}.mode must be one of {', '.join(sorted(SGLANG_HICACHE_MODES))}")

    def enum_value(key: str, choices: set[str]) -> str | None:
        parsed = _optional_string(raw.get(key), f"{path}.{key}", error)
        if parsed is not None and parsed not in choices:
            raise error(f"{path}.{key} must be one of {', '.join(sorted(choices))}")
        return parsed

    extra_config = raw.get("storage_backend_extra_config")
    if extra_config is not None:
        if isinstance(extra_config, Mapping):
            _validate_json_template_value(extra_config, f"{path}.storage_backend_extra_config", error)
            extra_config = types.MappingProxyType(dict(extra_config))
        elif not isinstance(extra_config, str):
            raise error(f"{path}.storage_backend_extra_config must be an object or string")

    storage_backend = enum_value("storage_backend", SGLANG_HICACHE_STORAGE_BACKENDS)
    if mode == "full_async_offload" and storage_backend is None:
        raise error(f"{path}.storage_backend is required when mode is full_async_offload")

    return SglangHiCacheConfig(
        mode=mode,
        page_size=_optional_positive_int(raw.get("page_size", 64), f"{path}.page_size", error),
        ratio=_optional_positive_float(raw.get("ratio", 2.0), f"{path}.ratio", error),
        size=_optional_non_negative_int(raw.get("size", 0), f"{path}.size", error),
        write_policy=enum_value("write_policy", SGLANG_HICACHE_WRITE_POLICIES) or "write_through",
        io_backend=enum_value("io_backend", SGLANG_HICACHE_IO_BACKENDS) or "direct",
        mem_layout=enum_value("mem_layout", SGLANG_HICACHE_MEM_LAYOUTS) or "page_first_direct",
        storage_backend=storage_backend,
        storage_prefetch_policy=(
            enum_value("storage_prefetch_policy", SGLANG_HICACHE_STORAGE_PREFETCH_POLICIES)
            or "timeout"
        ),
        storage_backend_extra_config=extra_config,
        enable_metrics=_optional_bool(raw.get("enable_metrics"), f"{path}.enable_metrics", error, True),
        enable_cache_report=_optional_bool(
            raw.get("enable_cache_report"), f"{path}.enable_cache_report", error, True
        ),
    )
```

- [ ] **步骤 4：接入 parser 和 validation**

在 `parse_topology_profiles` 中解析 `vllm_pd` 前加入：

```python
        sglang_hicache = _parse_sglang_hicache_config(
            profile.get("sglang_hicache"),
            f"{path}.sglang_hicache",
            error,
        )
```

在 `TopologyProfile(...)` 构造参数中加入：

```python
            sglang_hicache=sglang_hicache,
```

把 `_validate_topology_profile(...)` 调用和函数签名增加 `sglang_hicache` 参数，并在 SGLang 分支中加入：

```python
    if engine != "sglang" and sglang_hicache is not None:
        raise error(f"{path} ({profile_name}) sglang_hicache is only valid for sglang profiles")
    if engine == "sglang":
        if vllm_pd is not None:
            raise error(f"{path} ({profile_name}) vllm_pd is only valid for vllm profiles")
        if sglang_hicache is not None and not decode and sglang_hicache.mode == "full_async_offload":
            raise error(
                f"{path} ({profile_name}) decode workers are required when "
                "sglang_hicache.mode is full_async_offload"
            )
        for node in prefill:
            if node.bootstrap_port is None:
                raise error(
                    f"{path} ({profile_name}) prefill node {node.name} "
                    "bootstrap_port is required for sglang pd"
                )
        return
```

在读取 `transfer_backend` 时先保存变量：

```python
        transfer_backend = _optional_string(
            profile.get("transfer_backend"),
            f"{path}.transfer_backend",
            error,
        )
        if engine == "sglang" and transfer_backend is not None:
            if transfer_backend not in SGLANG_DISAGG_TRANSFER_BACKENDS:
                raise error(
                    f"{path}.transfer_backend must be one of "
                    + ", ".join(sorted(SGLANG_DISAGG_TRANSFER_BACKENDS))
                )
```

然后构造 `TopologyProfile` 时使用 `transfer_backend=transfer_backend`。

- [ ] **步骤 5：增加 HiCache 命令渲染 helper**

在 `_build_sglang_worker_command` 前加入：

```python
    def _sglang_hicache_args(self, role: str) -> list[str]:
        config = self.sglang_hicache
        if config is None:
            return []
        argv: list[str] = []
        if config.page_size is not None:
            argv.extend(["--page-size", str(config.page_size)])
        active_hicache = role == "prefill" or config.mode == "full_async_offload"
        if not active_hicache:
            return argv
        if role == "prefill":
            argv.append("--enable-hierarchical-cache")
        if config.enable_metrics:
            argv.append("--enable-metrics")
        if config.enable_cache_report:
            argv.append("--enable-cache-report")
        if config.ratio is not None:
            argv.extend(["--hicache-ratio", _format_number(config.ratio)])
        if config.size is not None:
            argv.extend(["--hicache-size", str(config.size)])
        if config.write_policy:
            argv.extend(["--hicache-write-policy", config.write_policy])
        if config.io_backend:
            argv.extend(["--hicache-io-backend", config.io_backend])
        if config.mem_layout:
            argv.extend(["--hicache-mem-layout", config.mem_layout])
        if config.storage_backend:
            argv.extend(["--hicache-storage-backend", config.storage_backend])
        if config.storage_prefetch_policy:
            argv.extend([
                "--hicache-storage-prefetch-policy",
                config.storage_prefetch_policy,
            ])
        if config.storage_backend_extra_config is not None:
            if isinstance(config.storage_backend_extra_config, str):
                extra_config = config.storage_backend_extra_config
            else:
                extra_config = json.dumps(
                    dict(config.storage_backend_extra_config),
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            argv.extend(["--hicache-storage-backend-extra-config", extra_config])
        if role == "decode" and config.mode == "full_async_offload":
            argv.append("--disaggregation-decode-enable-offload-kvcache")
        return argv
```

在模块级 helper 区域加入：

```python
def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)
```

- [ ] **步骤 6：接入 SGLang worker Docker flags**

在 `_build_sglang_worker_command` 中，`self._append_env_and_volumes(argv, node)` 后加入：

```python
        if self.mount_infiniband:
            argv.extend([
                "--device", "/dev/infiniband",
                "--cap-add", "IPC_LOCK",
                "--ulimit", "memlock=-1:-1",
            ])
```

在 `if self.disaggregation_ib_device:` 块后、`argv.extend(node.args)` 前加入：

```python
        argv.extend(self._sglang_hicache_args(role))
```

- [ ] **步骤 7：运行拓扑测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_remote_topology.py -q
```

预期：全部通过，包含新增 SGLang HiCache 测试。

- [ ] **步骤 8：提交拓扑实现**

运行：

```bash
git add vllm_standalone_bench/remote_topology.py vllm_standalone_bench/tests/test_remote_topology.py
git commit -m "feat: 支持 SGLang PD HiCache 拓扑配置"
```

---

### 任务 3：示例配置和配置加载测试

**文件：**
- 创建：`vllm_standalone_bench/configs/auto_bench.sglang_pd_hicache_remote.example.json`
- 创建：`vllm_standalone_bench/configs/auto_bench.sglang_pd_hicache_remote_minimax.json`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写示例配置 load 失败测试**

在 `test_shipped_sglang_pd_remote_config_parses` 后加入：

```python
def test_shipped_sglang_pd_hicache_remote_config_parses(tmp_path):
    path = CONFIG_DIR / "auto_bench.sglang_pd_hicache_remote.example.json"
    config = ab.load_config(path)
    assert config.serve_profiles == ()
    profile = config.topology_profiles[0]
    assert profile.engine == "sglang"
    assert profile.sglang_hicache is not None
    assert profile.sglang_hicache.mode == "full_async_offload"

    case = ab.expand_cases(config, run_id="dryrun")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "dryrun")
    p1 = commands["p1"].argv
    d1 = commands["d1"].argv
    router = commands["router"].argv
    assert "--enable-hierarchical-cache" in p1
    assert "--disaggregation-decode-enable-offload-kvcache" in d1
    assert value_after(p1, "--hicache-storage-backend") == "mooncake"
    assert value_after(d1, "--hicache-storage-backend") == "mooncake"
    assert value_after(p1, "--device") == "/dev/infiniband"
    assert "--device" not in router
```

继续加入 Minimax 模板测试：

```python
def test_shipped_sglang_pd_hicache_minimax_config_parses():
    path = CONFIG_DIR / "auto_bench.sglang_pd_hicache_remote_minimax.json"
    config = ab.load_config(path)
    assert [profile.name for profile in config.topology_profiles] == [
        "sglang_pd_hicache_minimax_m27_2p2d",
        "sglang_pd_hicache_minimax_m27_prefill_only_2p2d",
    ]
    assert all(profile.engine == "sglang" for profile in config.topology_profiles)
    assert all(profile.sglang_hicache is not None for profile in config.topology_profiles)
```

- [ ] **步骤 2：运行配置测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_shipped_sglang_pd_hicache_remote_config_parses vllm_standalone_bench/tests/test_auto_bench.py::test_shipped_sglang_pd_hicache_minimax_config_parses -q
```

预期：失败，报错包含新 JSON 文件不存在。

- [ ] **步骤 3：创建 dry-run-safe 示例配置**

创建 `vllm_standalone_bench/configs/auto_bench.sglang_pd_hicache_remote.example.json`，内容结构沿用 `auto_bench.sglang_pd_remote.example.json`，核心差异为：

```json
{
  "run": {
    "name": "sglang_pd_hicache_remote_bench",
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
    "name": "sglang_pd_hicache_2p2d",
    "engine": "sglang",
    "mode": "pd",
    "provider": "ssh_docker",
    "transfer_backend": "mooncake",
    "disaggregation_ib_device": "mlx5_0",
    "mount_infiniband": true,
    "network": "host",
    "image": "lmsysorg/sglang:latest",
    "router_image": "sglang-router:offline",
    "sglang_hicache": {
      "mode": "full_async_offload",
      "page_size": 64,
      "storage_backend": "mooncake",
      "storage_backend_extra_config": {"tp_lcm_size": 4}
    },
    "env": {
      "MOONCAKE_MASTER": "192.0.2.10:50051",
      "MOONCAKE_TE_META_DATA_SERVER": "http://192.0.2.10:8080/metadata",
      "MOONCAKE_PROTOCOL": "rdma",
      "MOONCAKE_DEVICE": "mlx5_0",
      "MOONCAKE_GLOBAL_SEGMENT_SIZE": "64gb"
    },
    "hosts": {
      "p1": {"address": "192.0.2.11", "ssh_user": "root", "auth": {"type": "key"}},
      "p2": {"address": "192.0.2.12", "ssh_user": "root", "auth": {"type": "key"}},
      "d1": {"address": "192.0.2.21", "ssh_user": "root", "auth": {"type": "key"}},
      "d2": {"address": "192.0.2.22", "ssh_user": "root", "auth": {"type": "key"}},
      "router": {"address": "192.0.2.30", "ssh_user": "root", "auth": {"type": "key"}}
    },
    "prefill": [
      {"name": "p1", "host": "p1", "port": 30000, "bootstrap_port": 12335, "gpus": "0,1,2,3", "args": ["--tp-size", "4"]},
      {"name": "p2", "host": "p2", "port": 30000, "bootstrap_port": 12336, "gpus": "0,1,2,3", "args": ["--tp-size", "4"]}
    ],
    "decode": [
      {"name": "d1", "host": "d1", "port": 30001, "gpus": "0,1,2,3", "args": ["--tp-size", "4"]},
      {"name": "d2", "host": "d2", "port": 30001, "gpus": "0,1,2,3", "args": ["--tp-size", "4"]}
    ],
    "frontend": {
      "kind": "sglang_router",
      "host": "router",
      "port": 8000,
      "args": ["--prefill-policy", "cache_aware", "--decode-policy", "power_of_two"]
    }
  }],
  "bench_profiles": [{
    "name": "hicache_reuse_matrix",
    "backend": "openai-chat",
    "input_lens": [4096],
    "output_lens": [256],
    "parallel_nums": [16],
    "epochs": 2,
    "prefix_ratios": [0.0, 0.6, 0.8]
  }]
}
```

- [ ] **步骤 4：创建 Minimax 模板配置**

创建 `vllm_standalone_bench/configs/auto_bench.sglang_pd_hicache_remote_minimax.json`。使用现有 Minimax vLLM NIXL 配置中的主机、端口、模型路径和节点参数作为基线，生成两个 profile：

```json
[
  {
    "name": "sglang_pd_hicache_minimax_m27_2p2d",
    "sglang_hicache": {
      "mode": "full_async_offload",
      "page_size": 64,
      "storage_backend": "mooncake",
      "storage_backend_extra_config": {"tp_lcm_size": 4}
    }
  },
  {
    "name": "sglang_pd_hicache_minimax_m27_prefill_only_2p2d",
    "sglang_hicache": {
      "mode": "prefill_only",
      "page_size": 64,
      "storage_backend": "mooncake",
      "storage_backend_extra_config": {"tp_lcm_size": 4}
    }
  }
]
```

每个 profile 必须包含：

```json
"transfer_backend": "mooncake",
"disaggregation_ib_device": "mlx5_0",
"mount_infiniband": true,
"env": {
  "MOONCAKE_MASTER": "10.200.1.10:50051",
  "MOONCAKE_TE_META_DATA_SERVER": "http://10.200.1.10:8080/metadata",
  "MOONCAKE_PROTOCOL": "rdma",
  "MOONCAKE_DEVICE": "mlx5_0",
  "MOONCAKE_GLOBAL_SEGMENT_SIZE": "64gb"
}
```

如果 Minimax 的实际 Mooncake control-plane 地址不同，保留显式占位地址并在 README 中说明必须按集群改值。不要在配置里启动 Mooncake 服务。

- [ ] **步骤 5：运行配置测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_shipped_sglang_pd_hicache_remote_config_parses vllm_standalone_bench/tests/test_auto_bench.py::test_shipped_sglang_pd_hicache_minimax_config_parses -q
```

预期：2 个测试通过。

- [ ] **步骤 6：提交示例配置**

运行：

```bash
git add vllm_standalone_bench/configs/auto_bench.sglang_pd_hicache_remote.example.json vllm_standalone_bench/configs/auto_bench.sglang_pd_hicache_remote_minimax.json vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "test: 覆盖 SGLang PD HiCache 示例配置"
```

---

### 任务 4：Prometheus cache-source metrics 测试先行

**文件：**
- 修改：`vllm_standalone_bench/tests/test_serve_metrics.py`

- [ ] **步骤 1：编写 metrics text 解析失败测试**

在 `test_parse_runtime_metrics_keeps_spec_decode_metrics` 后加入：

```python
def test_parse_runtime_metrics_reads_sglang_cache_source_counters():
    metrics = serve.parse_runtime_metrics_text(
        "\n".join([
            'sglang:prompt_tokens_total{model_name="m"} 1000',
            'sglang:cached_tokens_total{model_name="m",cache_source="device"} 100',
            'sglang:cached_tokens_total{model_name="m",cache_source="host"} 50',
            'sglang:cached_tokens_total{model_name="m",cache_source="storage_mooncake"} 25',
        ])
    )

    assert metrics.cache_source is not None
    assert metrics.cache_source.prompt_tokens == 1000
    assert metrics.cache_source.cached_tokens_by_source == {
        "device": 100,
        "host": 50,
        "storage_mooncake": 25,
    }
```

- [ ] **步骤 2：编写 before/after 差分失败测试**

在同一文件加入：

```python
def test_cache_source_metrics_delta_calculates_hit_rate():
    before = serve.CacheSourceMetrics(
        prompt_tokens=1000,
        cached_tokens_by_source={"device": 100, "host": 50},
    )
    after = serve.CacheSourceMetrics(
        prompt_tokens=1600,
        cached_tokens_by_source={
            "device": 160,
            "host": 95,
            "storage_mooncake": 45,
        },
    )

    stats = serve.calculate_cache_source_stats(before, after)

    assert stats == {
        "cache_hit_rate_metrics": 25.0,
        "cache_hit_tokens_device": 60,
        "cache_hit_tokens_host": 45,
        "cache_hit_tokens_storage": 45,
        "cache_hit_tokens_storage_mooncake": 45,
    }
```

- [ ] **步骤 3：运行 serve metrics 测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_serve_metrics.py::test_parse_runtime_metrics_reads_sglang_cache_source_counters vllm_standalone_bench/tests/test_serve_metrics.py::test_cache_source_metrics_delta_calculates_hit_rate -q
```

预期：失败，报错包含 `RuntimeMetrics` 没有 `cache_source` 或 `CacheSourceMetrics` 未定义。

---

### 任务 5：实现 cache-source metrics 解析和 result JSON

**文件：**
- 修改：`vllm_standalone_bench/vllm_bench/serve.py`
- 修改：`vllm_standalone_bench/tests/test_serve_metrics.py`

- [ ] **步骤 1：扩展 dataclass**

把 import 改成：

```python
from dataclasses import dataclass, field
```

在 `SpecDecodeMetrics` 后加入：

```python
@dataclass
class CacheSourceMetrics:
    """SGLang cached-token counters from Prometheus."""

    prompt_tokens: int = 0
    cached_tokens_by_source: dict[str, int] = field(default_factory=dict)
```

把 `RuntimeMetrics` 改成：

```python
@dataclass
class RuntimeMetrics:
    """Runtime metrics parsed from the server's Prometheus endpoint."""

    spec_decode: SpecDecodeMetrics | None = None
    gpu_kv_cache_usage: float | None = None
    cache_source: CacheSourceMetrics | None = None
```

- [ ] **步骤 2：增加 Prometheus label parser**

在 `_metric_value_from_prometheus_line` 后加入：

```python
def _metric_labels_from_prometheus_line(line: str) -> dict[str, str]:
    if "{" not in line or "}" not in line:
        return {}
    label_text = line.split("{", 1)[1].split("}", 1)[0]
    labels: dict[str, str] = {}
    for item in label_text.split(","):
        key, separator, raw_value = item.partition("=")
        if not separator:
            continue
        labels[key.strip()] = raw_value.strip().strip('"')
    return labels
```

- [ ] **步骤 3：解析 SGLang cache counters**

在 `parse_runtime_metrics_text` 的本地变量中加入：

```python
    prompt_tokens_total = 0
    cached_tokens_by_source: dict[str, int] = {}
```

在主循环中，`if metric_name in GPU_KV_CACHE_USAGE_METRICS:` 后加入：

```python
        if metric_name == "sglang:prompt_tokens_total":
            prompt_tokens_total += int(value)
            continue

        if metric_name == "sglang:cached_tokens_total":
            labels = _metric_labels_from_prometheus_line(line)
            source = labels.get("cache_source")
            if source:
                cached_tokens_by_source[source] = (
                    cached_tokens_by_source.get(source, 0) + int(value)
                )
            continue
```

在 return 前构造：

```python
    cache_source = None
    if prompt_tokens_total > 0 or cached_tokens_by_source:
        cache_source = CacheSourceMetrics(
            prompt_tokens=prompt_tokens_total,
            cached_tokens_by_source=cached_tokens_by_source,
        )
```

并在 `RuntimeMetrics(...)` 中传入 `cache_source=cache_source`。

- [ ] **步骤 4：增加差分函数**

在 `fetch_spec_decode_metrics` 后加入：

```python
def calculate_cache_source_stats(
    before: CacheSourceMetrics,
    after: CacheSourceMetrics,
) -> dict[str, Any]:
    prompt_delta = max(after.prompt_tokens - before.prompt_tokens, 0)
    source_deltas: dict[str, int] = {}
    sources = set(before.cached_tokens_by_source) | set(after.cached_tokens_by_source)
    for source in sources:
        delta = after.cached_tokens_by_source.get(source, 0) - before.cached_tokens_by_source.get(source, 0)
        source_deltas[source] = max(delta, 0)

    device = source_deltas.get("device", 0)
    host = source_deltas.get("host", 0)
    storage = sum(value for source, value in source_deltas.items() if source.startswith("storage_"))
    total_cached = sum(source_deltas.values())
    hit_rate = round(total_cached / prompt_delta * 100, 4) if prompt_delta > 0 else 0.0

    stats: dict[str, Any] = {
        "cache_hit_rate_metrics": hit_rate,
        "cache_hit_tokens_device": device,
        "cache_hit_tokens_host": host,
        "cache_hit_tokens_storage": storage,
    }
    for source, value in sorted(source_deltas.items()):
        if source.startswith("storage_"):
            stats[f"cache_hit_tokens_{source}"] = value
    return stats
```

- [ ] **步骤 5：把 before/after 接入 benchmark result**

在 `main_async` 里，把 benchmark 前的 `spec_decode_metrics_before = await fetch_spec_decode_metrics(...)` 替换成：

```python
    runtime_metrics_before = await fetch_runtime_metrics(
        base_url, session, extra_headers
    )
    spec_decode_metrics_before = (
        runtime_metrics_before.spec_decode if runtime_metrics_before is not None else None
    )
    cache_source_metrics_before = (
        runtime_metrics_before.cache_source if runtime_metrics_before is not None else None
    )
```

把 benchmark 后的 `spec_decode_metrics_after = await fetch_spec_decode_metrics(...)` 替换成：

```python
    runtime_metrics_after = await fetch_runtime_metrics(
        base_url, session, extra_headers
    )
    spec_decode_metrics_after = (
        runtime_metrics_after.spec_decode if runtime_metrics_after is not None else None
    )
    cache_source_metrics_after = (
        runtime_metrics_after.cache_source if runtime_metrics_after is not None else None
    )
```

在 `add_runtime_metrics_to_result(result, runtime_metrics_summary)` 后加入：

```python
    if cache_source_metrics_before is not None and cache_source_metrics_after is not None:
        result.update(
            calculate_cache_source_stats(
                cache_source_metrics_before,
                cache_source_metrics_after,
            )
        )
```

- [ ] **步骤 6：运行 serve metrics 测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_serve_metrics.py -q
```

预期：全部通过。

- [ ] **步骤 7：提交 metrics 实现**

运行：

```bash
git add vllm_standalone_bench/vllm_bench/serve.py vllm_standalone_bench/tests/test_serve_metrics.py
git commit -m "feat: 采集 SGLang HiCache cache-source 指标"
```

---

### 任务 6：结果提取、CSV/XLSX 表头和测试

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py`
- 修改：`vllm_standalone_bench/tests/test_extract_row.py`
- 修改：`vllm_standalone_bench/tests/test_result_csv_headers.py`

- [ ] **步骤 1：编写 `_extract_row` 失败测试**

在 `test_extract_row_includes_runtime_metrics` 后加入：

```python
def test_extract_row_includes_sglang_cache_source_metrics():
    result = _result(
        total_in=1000,
        total_out=100,
        total_cached=200,
        completed=2,
    )
    result.update({
        "cache_hit_rate_metrics": 25.0,
        "cache_hit_tokens_device": 60,
        "cache_hit_tokens_host": 45,
        "cache_hit_tokens_storage": 45,
        "cache_hit_tokens_storage_mooncake": 45,
    })

    row = m._extract_row(result, 500, 50, 2, 1, "m", "sglang", has_tokenizer=True)

    assert row["cache_hit_rate_metrics"] == 25.0
    assert row["cache_hit_tokens_device"] == 60
    assert row["cache_hit_tokens_host"] == 45
    assert row["cache_hit_tokens_storage"] == 45
    assert row["cache_hit_tokens_storage_mooncake"] == 45
```

在默认值测试区域加入：

```python
def test_extract_row_sglang_cache_source_metrics_default_zero():
    row = m._extract_row(_result(), 100, 10, 1, 1, "m", "sglang")

    assert row["cache_hit_rate_metrics"] == 0.0
    assert row["cache_hit_tokens_device"] == 0
    assert row["cache_hit_tokens_host"] == 0
    assert row["cache_hit_tokens_storage"] == 0
    assert row["cache_hit_tokens_storage_mooncake"] == 0
```

- [ ] **步骤 2：更新固定表头失败测试**

在 `tests/test_result_csv_headers.py` 的 `EXPECTED_HEADERS` 中，把新列放在 `cache_hit_rate` 后：

```python
"cache_hit_rate_metrics",
"cache_hit_tokens_device",
"cache_hit_tokens_host",
"cache_hit_tokens_storage",
"cache_hit_tokens_storage_mooncake",
```

在 `EXPECTED_HEADERS_ZH` 对应位置加入：

```python
"Metrics缓存命中率(%)",
"Device缓存命中tokens",
"Host缓存命中tokens",
"Storage缓存命中tokens",
"Mooncake缓存命中tokens",
```

把长度断言从 `52` 改为 `57`。

- [ ] **步骤 3：运行结果测试验证失败**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_extract_row.py vllm_standalone_bench/tests/test_result_csv_headers.py -q
```

预期：失败，报错包含 `_extract_row` 缺少新字段或表头不一致。

- [ ] **步骤 4：实现 row 字段**

在 `_extract_row` return dict 中，`cache_hit_rate` 后加入：

```python
        'cache_hit_rate_metrics': _f('cache_hit_rate_metrics'),
        'cache_hit_tokens_device': _i('cache_hit_tokens_device'),
        'cache_hit_tokens_host': _i('cache_hit_tokens_host'),
        'cache_hit_tokens_storage': _i('cache_hit_tokens_storage'),
        'cache_hit_tokens_storage_mooncake': _i('cache_hit_tokens_storage_mooncake'),
```

- [ ] **步骤 5：实现 CSV/XLSX 表头和说明**

在 `CSV_HEADERS` 的 `'avg_cached_tokens', 'cache_hit_rate',` 后加入：

```python
    'cache_hit_rate_metrics',
    'cache_hit_tokens_device', 'cache_hit_tokens_host',
    'cache_hit_tokens_storage', 'cache_hit_tokens_storage_mooncake',
```

在 `CSV_HEADERS_ZH` 对应位置加入：

```python
    'Metrics缓存命中率(%)',
    'Device缓存命中tokens', 'Host缓存命中tokens',
    'Storage缓存命中tokens', 'Mooncake缓存命中tokens',
```

在 XLSX metric explanation list 中 `cache_hit_rate` 后加入：

```python
        ('cache_hit_rate_metrics', 'Metrics缓存命中率(%)', 'SGLang /metrics 中 cached_tokens_total 差分 ÷ prompt_tokens_total 差分 × 100'),
        ('cache_hit_tokens_device', 'Device缓存命中tokens', 'SGLang /metrics 中 cache_source=device 的 cached_tokens_total 差分'),
        ('cache_hit_tokens_host', 'Host缓存命中tokens', 'SGLang /metrics 中 cache_source=host 的 cached_tokens_total 差分'),
        ('cache_hit_tokens_storage', 'Storage缓存命中tokens', '所有 cache_source=storage_* 的 cached_tokens_total 差分之和'),
        ('cache_hit_tokens_storage_mooncake', 'Mooncake缓存命中tokens', 'SGLang /metrics 中 cache_source=storage_mooncake 的 cached_tokens_total 差分'),
```

- [ ] **步骤 6：运行结果测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_extract_row.py vllm_standalone_bench/tests/test_result_csv_headers.py -q
```

预期：全部通过。

- [ ] **步骤 7：提交结果链路实现**

运行：

```bash
git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/tests/test_extract_row.py vllm_standalone_bench/tests/test_result_csv_headers.py
git commit -m "feat: 输出 SGLang HiCache cache-source 指标"
```

---

### 任务 7：README 使用说明

**文件：**
- 修改：`vllm_standalone_bench/README.md`

- [ ] **步骤 1：定位 PD 或远程拓扑说明位置**

运行：

```bash
rg -n "PD|topology_profiles|remote|cache_hit_rate|Mooncake|HiCache" vllm_standalone_bench/README.md
```

预期：找到远程拓扑或指标说明章节。

- [ ] **步骤 2：增加 SGLang PD + HiCache 配置说明**

在远程 PD 配置说明附近加入一节：

```markdown
### SGLang PD + HiCache

SGLang PD 拓扑可以通过 `topology_profiles[].sglang_hicache` 渲染
`sglang.launch_server` 支持的 HiCache 参数。当前支持两种模式：

- `prefill_only`：只在 prefill worker 开启 `--enable-hierarchical-cache`，
  用于验证 shared prefix reuse。
- `full_async_offload`：prefill worker 开启 HiCache，decode worker 增加
  `--disaggregation-decode-enable-offload-kvcache`，用于验证 decode KV
  offload 后被 prefill 复用。

`page_size` 是 SGLang worker 全局 KV page size，不是 HiCache 专属参数。
配置后会同时渲染到 prefill 和 decode worker，避免 PD 两侧 page size 不一致。

Mooncake master、metadata server 和 store service 需要在 benchmark 外部准备。
`auto_bench` 只把 `MOONCAKE_*` 环境变量传入 worker 容器，不负责启动或发现
Mooncake control plane。
```

- [ ] **步骤 3：增加指标说明**

在指标表附近加入：

```markdown
| **cache_hit_rate_metrics** | Metrics缓存命中率 | SGLang `/metrics` 中 `sglang:cached_tokens_total` 差分除以 `sglang:prompt_tokens_total` 差分 |
| **cache_hit_tokens_device** | Device缓存命中tokens | `cache_source="device"` 的 cached tokens 差分 |
| **cache_hit_tokens_host** | Host缓存命中tokens | `cache_source="host"` 的 cached tokens 差分 |
| **cache_hit_tokens_storage** | Storage缓存命中tokens | 所有 `cache_source="storage_*"` 的 cached tokens 差分之和 |
| **cache_hit_tokens_storage_mooncake** | Mooncake缓存命中tokens | `cache_source="storage_mooncake"` 的 cached tokens 差分 |
```

- [ ] **步骤 4：运行文档和相关测试**

运行：

```bash
git diff --check
python3 -m pytest vllm_standalone_bench/tests/test_remote_topology.py vllm_standalone_bench/tests/test_auto_bench.py::test_shipped_sglang_pd_hicache_remote_config_parses vllm_standalone_bench/tests/test_auto_bench.py::test_shipped_sglang_pd_hicache_minimax_config_parses -q
```

预期：`git diff --check` 无输出，pytest 通过。

- [ ] **步骤 5：提交 README**

运行：

```bash
git add vllm_standalone_bench/README.md
git commit -m "docs: 说明 SGLang PD HiCache 配置"
```

---

### 任务 8：最终验证和收尾审查

**文件：**
- 检查：全部本分支变更

- [ ] **步骤 1：运行核心测试集**

运行：

```bash
python3 -m pytest \
  vllm_standalone_bench/tests/test_remote_topology.py \
  vllm_standalone_bench/tests/test_auto_bench.py \
  vllm_standalone_bench/tests/test_serve_metrics.py \
  vllm_standalone_bench/tests/test_extract_row.py \
  vllm_standalone_bench/tests/test_result_csv_headers.py \
  -q
```

预期：全部通过。

- [ ] **步骤 2：运行格式检查**

运行：

```bash
git diff --check
```

预期：无输出，exit code 0。

- [ ] **步骤 3：检查提交历史和工作区**

运行：

```bash
git status --short --branch
git log --oneline --decorate main..HEAD
```

预期：工作区干净，提交包含 spec、plan 和 4-5 个实现提交。

- [ ] **步骤 4：执行代码审查**

使用 `requesting-code-review` 或手工 review 模式检查：

```bash
git diff --stat main..HEAD
git diff main..HEAD -- vllm_standalone_bench/remote_topology.py
git diff main..HEAD -- vllm_standalone_bench/vllm_bench/serve.py
git diff main..HEAD -- vllm_standalone_bench/run_bench_multi.py
```

重点核对：

- `sglang_hicache` 只对 SGLang PD 生效。
- `node.args` 仍在结构化 flags 后追加，保留覆盖能力。
- Router 不挂 `/dev/infiniband`。
- Missing `/metrics` 不影响 benchmark 完成。
- 新 result 字段进入 CSV/XLSX 固定表头。

- [ ] **步骤 5：最终提交状态检查**

如果步骤 4 发现小修订，完成修订并运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_remote_topology.py vllm_standalone_bench/tests/test_serve_metrics.py vllm_standalone_bench/tests/test_extract_row.py vllm_standalone_bench/tests/test_result_csv_headers.py -q
git diff --check
```

预期：pytest 通过，diff check 通过。
