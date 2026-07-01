# vLLM 编译/JIT Cache 持久化实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 `vllm_standalone_bench/auto_bench.py` 增加 vLLM serving 容器编译/JIT cache 持久化能力，让 GLM5.2 自动化 benchmark 第一次运行填充 cache，后续相同 profile 复用 cache 降低反复启停总耗时。

**架构：** 在 `run.vllm_cache` 中配置宿主机 cache 根目录和容器内路径，在 `serve_profiles[].cache_key` 中可选指定稳定 key。`auto_bench.py` 解析配置、创建 cache 目录、为 vLLM docker run 注入 cache mount/env，并在 serve 目录写入 `vllm_cache.json` 记录实际使用的 cache 信息。默认不启用 cache，现有命令和配置行为保持不变。

**技术栈：** Python 3 标准库、dataclasses、pathlib、Docker CLI 命令拼装、pytest。

---

## 当前执行状态（2026-07-01）

- Worktree：`/Resource_Planning_Tool/.worktrees/vllm-cache-persistence`
- 分支：`feat/vllm-cache-persistence`
- 已完成并通过两阶段审查：任务 1、任务 2、任务 3。
- 已完成实现、尚未审查：任务 4（commit `1b6eecd`，`feat(bench): create vllm cache directories`）。
- 下一步：对任务 4 执行规格合规审查和代码质量审查；通过后进入任务 5。
- 当前 targeted 测试状态：`PYTHONPATH=vllm_standalone_bench pytest -q vllm_standalone_bench/tests/test_auto_bench.py` 为 `168 passed`（任务 4 worker 报告）。
- 已知完整基线问题：`pytest -q` 仍有与本任务无关的既有失败，`tests/test_inference_token_factory_report.py` 依赖缺失的 `outputs/context_analysis_20260609_034248/01_overview.json`。
- 详细 handoff：`docs/superpowers/plans/2026-07-01-vllm-cache-persistence-handoff.md`

---

## 文件结构

- 修改：`vllm_standalone_bench/auto_bench.py`
  - 新增 `VllmCacheConfig` dataclass。
  - 扩展 `RunConfig` 和 `ServeProfile`。
  - 解析 `run.vllm_cache`、`serve_profiles[].cache_key`。
  - 生成 per-case cache key、host cache dir、cache env。
  - 创建 cache 目录。
  - 为 vLLM docker run 注入 `-v` 和 `-e`。
  - 写入 `vllm_cache.json` artifact。
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`
  - 覆盖默认关闭 cache。
  - 覆盖配置解析、非法配置、cache 目录创建、docker run 命令注入、SGLang 不注入、metadata artifact。
- 修改：`vllm_standalone_bench/README.md`
  - 增加 warm-cache benchmark 配置说明和 GLM5.2 示例片段。
- 修改：`.gitignore`
  - 增加 `.cache/`，防止 vLLM 编译/JIT 产物进入 git。

## 约束与已知基线

- 当前 worktree 完整 `pytest -q` 已知存在与本任务无关的既有失败：`tests/test_inference_token_factory_report.py` 依赖缺失的 `outputs/context_analysis_20260609_034248/01_overview.json`。
- 本计划的主要验证命令使用 targeted test：

```bash
PYTHONPATH=vllm_standalone_bench pytest -q vllm_standalone_bench/tests/test_auto_bench.py
```

---

### 任务 1：配置 schema 与解析

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的解析测试**

在 `vllm_standalone_bench/tests/test_auto_bench.py` 中追加以下测试，放在现有 `test_load_config_warmup_opts_default_none` 附近：

```python
def enable_vllm_cache(data, root):
    data["run"]["vllm_cache"] = {
        "enabled": True,
        "root": str(root),
        "container_path": "/vllm-cache",
        "set_default_env": True,
    }
    return data


def test_load_config_vllm_cache_defaults_disabled(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))

    assert config.run.vllm_cache.enabled is False
    assert config.run.vllm_cache.root is None
    assert config.run.vllm_cache.container_path == "/vllm-cache"
    assert config.run.vllm_cache.set_default_env is True
    assert config.serve_profiles[0].cache_key is None


def test_load_config_parses_enabled_vllm_cache(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"

    config = ab.load_config(write_config(tmp_path, data))

    assert config.run.vllm_cache.enabled is True
    assert config.run.vllm_cache.root == (tmp_path / "cache").resolve()
    assert config.run.vllm_cache.container_path == "/vllm-cache"
    assert config.run.vllm_cache.set_default_env is True
    assert config.serve_profiles[0].cache_key == "glm52-fp8-tp8-h20-o2"


def test_load_config_resolves_relative_vllm_cache_root_from_config_dir(tmp_path):
    config_dir = tmp_path / "configs"
    data = enable_vllm_cache(minimal_config(tmp_path), "relative-cache")

    config = ab.load_config(write_config_at(config_dir / "config.json", data))

    assert config.run.vllm_cache.root == (config_dir / "relative-cache").resolve()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
PYTHONPATH=vllm_standalone_bench pytest -q \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_config_vllm_cache_defaults_disabled \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_config_parses_enabled_vllm_cache \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_config_resolves_relative_vllm_cache_root_from_config_dir
```

预期：FAIL，报错包含 `RunConfig` 没有 `vllm_cache` 或 `ServeProfile` 没有 `cache_key`。

- [ ] **步骤 3：实现配置 dataclass 与解析**

在 `vllm_standalone_bench/auto_bench.py` 中做以下修改。

在 `RunConfig` 前新增：

```python
@dataclass(frozen=True)
class VllmCacheConfig:
    enabled: bool = False
    root: Path | None = None
    container_path: str = "/vllm-cache"
    set_default_env: bool = True
```

扩展 `RunConfig`：

```python
    vllm_cache: VllmCacheConfig = field(default_factory=VllmCacheConfig)
```

扩展 `ServeProfile`：

```python
    cache_key: str | None = None
```

新增 container path 校验函数，放在 `_container_path_to_host()` 前后均可：

```python
def _container_abs_path(value: Any, field_name: str) -> str:
    path_text = _string(value, field_name)
    container_path = PurePosixPath(path_text)
    if not container_path.is_absolute():
        raise ConfigError(f"{field_name} must be absolute inside the container: {path_text}")
    if ".." in container_path.parts:
        raise ConfigError(f"{field_name} must not contain '..': {path_text}")
    return str(container_path)
```

新增 cache 解析函数：

```python
def _parse_vllm_cache(run: dict[str, Any], config_dir: Path) -> VllmCacheConfig:
    raw = run.get("vllm_cache")
    if raw is None:
        return VllmCacheConfig()
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
    root_raw = cache.get("root")
    if not enabled:
        return VllmCacheConfig(
            enabled=False,
            root=None,
            container_path=container_path,
            set_default_env=set_default_env,
        )
    if root_raw is None:
        raise ConfigError("run.vllm_cache.root is required when enabled=true")
    root = Path(_string(root_raw, "run.vllm_cache.root"))
    if not root.is_absolute():
        root = config_dir / root
    return VllmCacheConfig(
        enabled=True,
        root=root.resolve(),
        container_path=container_path,
        set_default_env=set_default_env,
    )
```

修改 `_parse_run` 签名和返回值：

```python
def _parse_run(data: dict[str, Any], config_dir: Path) -> RunConfig:
    run = _require_mapping(data.get("run"), "run")
    ...
        images=images,
        vllm_cache=_parse_vllm_cache(run, config_dir),
    )
```

修改 `load_config()` 中的调用：

```python
    run = _parse_run(config_data, config_path.parent)
```

修改 `_parse_serve_profiles()` 中 `ServeProfile(...)`：

```python
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
PYTHONPATH=vllm_standalone_bench pytest -q \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_config_vllm_cache_defaults_disabled \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_config_parses_enabled_vllm_cache \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_config_resolves_relative_vllm_cache_root_from_config_dir
```

预期：3 passed。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): parse vllm cache config"
```

---

### 任务 2：非法配置校验

**文件：**
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`
- 修改：`vllm_standalone_bench/auto_bench.py`

- [ ] **步骤 1：编写失败的非法配置测试**

在 `vllm_standalone_bench/tests/test_auto_bench.py` 中追加：

```python
def test_vllm_cache_enabled_requires_root(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["vllm_cache"] = {"enabled": True}

    with pytest.raises(ab.ConfigError, match="vllm_cache.root"):
        ab.load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("container_path", ["relative/cache", "/cache/../bad"])
def test_vllm_cache_container_path_must_be_absolute_and_safe(tmp_path, container_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["run"]["vllm_cache"]["container_path"] = container_path

    with pytest.raises(ab.ConfigError, match="container_path|absolute|contain"):
        ab.load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("cache_key", ["bad/name", ".", ".."])
def test_serve_profile_cache_key_must_be_safe(tmp_path, cache_key):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = cache_key

    with pytest.raises(ab.ConfigError, match="cache_key|safe filename"):
        ab.load_config(write_config(tmp_path, data))
```

- [ ] **步骤 2：运行测试验证失败或通过现有实现**

运行：

```bash
PYTHONPATH=vllm_standalone_bench pytest -q \
  vllm_standalone_bench/tests/test_auto_bench.py::test_vllm_cache_enabled_requires_root \
  vllm_standalone_bench/tests/test_auto_bench.py::test_vllm_cache_container_path_must_be_absolute_and_safe \
  vllm_standalone_bench/tests/test_auto_bench.py::test_serve_profile_cache_key_must_be_safe
```

预期：如果任务 1 实现已覆盖全部校验，则 5 passed；否则按失败信息补齐 `_parse_vllm_cache()` 或 `_parse_serve_profiles()`。

- [ ] **步骤 3：补齐最少实现代码**

如果失败，确保以下行为存在：

```python
if root_raw is None:
    raise ConfigError("run.vllm_cache.root is required when enabled=true")
```

以及：

```python
if ".." in container_path.parts:
    raise ConfigError(f"{field_name} must not contain '..': {path_text}")
```

以及 `cache_key` 使用 `_safe_name(cache_key, "serve_profile.cache_key")`。

- [ ] **步骤 4：运行测试验证通过**

运行同步骤 2 命令。

预期：5 passed。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "test(bench): cover invalid vllm cache config"
```

---

### 任务 3：cache key、cache dir 与 env helper

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的 helper 测试**

在 `vllm_standalone_bench/tests/test_auto_bench.py` 中追加：

```python
def test_resolve_vllm_cache_dir_uses_explicit_cache_key(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    assert ab.resolve_vllm_cache_dir(config, case) == (
        tmp_path / "cache" / "glm52-fp8-tp8-h20-o2"
    ).resolve()


def test_resolve_vllm_cache_dir_uses_stable_default_key(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    cache_dir = ab.resolve_vllm_cache_dir(config, case)

    assert cache_dir is not None
    assert cache_dir.parent == (tmp_path / "cache").resolve()
    assert case.model.name in cache_dir.name
    assert case.serve_profile.name in cache_dir.name
    assert ab.resolve_vllm_cache_dir(config, case) == cache_dir


def test_build_vllm_cache_env_defaults(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    config = ab.load_config(write_config(tmp_path, data))

    assert ab.build_vllm_cache_env(config) == {
        "VLLM_CACHE_ROOT": "/vllm-cache",
        "DG_JIT_CACHE_DIR": "/vllm-cache/deep_gemm",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "/vllm-cache/flashinfer_autotune",
    }


def test_build_vllm_cache_env_can_be_disabled(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["run"]["vllm_cache"]["set_default_env"] = False
    config = ab.load_config(write_config(tmp_path, data))

    assert ab.build_vllm_cache_env(config) == {}
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
PYTHONPATH=vllm_standalone_bench pytest -q \
  vllm_standalone_bench/tests/test_auto_bench.py::test_resolve_vllm_cache_dir_uses_explicit_cache_key \
  vllm_standalone_bench/tests/test_auto_bench.py::test_resolve_vllm_cache_dir_uses_stable_default_key \
  vllm_standalone_bench/tests/test_auto_bench.py::test_build_vllm_cache_env_defaults \
  vllm_standalone_bench/tests/test_auto_bench.py::test_build_vllm_cache_env_can_be_disabled
```

预期：FAIL，报错包含 `resolve_vllm_cache_dir` 或 `build_vllm_cache_env` 未定义。

- [ ] **步骤 3：实现 helper**

在 `vllm_standalone_bench/auto_bench.py` 顶部 import 增加：

```python
import hashlib
```

在 `expand_cases()` 前后新增：

```python
def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def default_vllm_cache_key(config: AutoBenchConfig, case: BenchmarkCase) -> str:
    image = config.run.images["vllm"]
    return f"{case.model.name}__{case.serve_profile.name}__{_short_hash(image)}"


def vllm_cache_key(config: AutoBenchConfig, case: BenchmarkCase) -> str | None:
    if case.serve_profile.engine != "vllm" or not config.run.vllm_cache.enabled:
        return None
    return case.serve_profile.cache_key or default_vllm_cache_key(config, case)


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
    root = cache.container_path.rstrip("/")
    return {
        "VLLM_CACHE_ROOT": root,
        "DG_JIT_CACHE_DIR": f"{root}/deep_gemm",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": f"{root}/flashinfer_autotune",
    }
```

- [ ] **步骤 4：运行测试验证通过**

运行同步骤 2 命令。

预期：4 passed。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): resolve vllm cache paths"
```

---

### 任务 4：创建 cache 目录

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的目录创建测试**

在 `vllm_standalone_bench/tests/test_auto_bench.py` 中追加：

```python
def test_validate_local_paths_creates_vllm_cache_dirs(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    cache_dir = ab.resolve_vllm_cache_dir(config, case)

    assert cache_dir is not None
    assert not cache_dir.exists()

    ab.validate_local_paths(config)

    assert cache_dir.is_dir()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
PYTHONPATH=vllm_standalone_bench pytest -q \
  vllm_standalone_bench/tests/test_auto_bench.py::test_validate_local_paths_creates_vllm_cache_dirs
```

预期：FAIL，`cache_dir.is_dir()` 为 false。

- [ ] **步骤 3：实现目录创建**

在 `vllm_standalone_bench/auto_bench.py` 中新增：

```python
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
```

修改 `validate_local_paths()` 末尾：

```python
    ensure_vllm_cache_dirs(config)
```

- [ ] **步骤 4：运行测试验证通过**

运行同步骤 2 命令。

预期：1 passed。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): create vllm cache directories"
```

---

### 任务 5：vLLM docker run 注入 cache mount/env

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的命令测试**

在 `vllm_standalone_bench/tests/test_auto_bench.py` 中追加：

```python
def test_build_vllm_command_omits_cache_when_disabled(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_vllm_run_command(config, case, tmp_path / "results" / "run123")

    assert "/vllm-cache" not in " ".join(cmd)
    assert "VLLM_CACHE_ROOT=/vllm-cache" not in cmd


def test_build_vllm_command_includes_cache_mount_and_env(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_vllm_run_command(config, case, tmp_path / "results" / "run123")

    mounts = values_after(cmd, "-v")
    envs = values_after(cmd, "-e")
    assert f"{(tmp_path / 'cache' / 'glm52-fp8-tp8-h20-o2').resolve()}:/vllm-cache:rw" in mounts
    assert "VLLM_CACHE_ROOT=/vllm-cache" in envs
    assert "DG_JIT_CACHE_DIR=/vllm-cache/deep_gemm" in envs
    assert "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/vllm-cache/flashinfer_autotune" in envs


def test_build_sglang_command_omits_vllm_cache_mount_and_env(tmp_path):
    data = enable_vllm_cache(sglang_config(tmp_path), tmp_path / "cache")
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_serve_run_command(config, case, tmp_path / "results" / "run123")

    assert "/vllm-cache" not in " ".join(cmd)
    assert "VLLM_CACHE_ROOT=/vllm-cache" not in cmd
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
PYTHONPATH=vllm_standalone_bench pytest -q \
  vllm_standalone_bench/tests/test_auto_bench.py::test_build_vllm_command_omits_cache_when_disabled \
  vllm_standalone_bench/tests/test_auto_bench.py::test_build_vllm_command_includes_cache_mount_and_env \
  vllm_standalone_bench/tests/test_auto_bench.py::test_build_sglang_command_omits_vllm_cache_mount_and_env
```

预期：第二个测试 FAIL，命令缺少 cache mount/env；另外两个应通过或随着实现通过。

- [ ] **步骤 3：实现命令注入**

在 `build_vllm_run_command()` 中，模型挂载之后、`--entrypoint` 之前加入：

```python
    cache_dir = resolve_vllm_cache_dir(config, case)
    if cache_dir is not None:
        cmd.extend(["-v", f"{cache_dir}:{config.run.vllm_cache.container_path}:rw"])
        for name, value in build_vllm_cache_env(config).items():
            cmd.extend(["-e", f"{name}={value}"])
```

保持 `_build_sglang_run_command()` 不变。

- [ ] **步骤 4：运行测试验证通过**

运行同步骤 2 命令。

预期：3 passed。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): mount persistent vllm cache"
```

---

### 任务 6：写入 `vllm_cache.json` artifact

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的 metadata 测试**

在 `vllm_standalone_bench/tests/test_auto_bench.py` 中追加：

```python
def test_vllm_cache_metadata_payload(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    payload = ab.vllm_cache_metadata(config, case)

    assert payload == {
        "enabled": True,
        "cache_key": "glm52-fp8-tp8-h20-o2",
        "host_dir": str((tmp_path / "cache" / "glm52-fp8-tp8-h20-o2").resolve()),
        "container_path": "/vllm-cache",
        "env": {
            "VLLM_CACHE_ROOT": "/vllm-cache",
            "DG_JIT_CACHE_DIR": "/vllm-cache/deep_gemm",
            "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "/vllm-cache/flashinfer_autotune",
        },
    }


def test_run_controller_writes_vllm_cache_metadata(tmp_path, monkeypatch):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"
    config = ab.load_config(write_config(tmp_path, data))
    monkeypatch.setattr(ab, "wait_for_ready", lambda *a, **k: True)
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner)

    metadata_path = (
        tmp_path / "results" / "run123" / "qwen2_5_1_5b" / "bf16_default" / "vllm_cache.json"
    )
    assert result == 0
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["cache_key"] == (
        "glm52-fp8-tp8-h20-o2"
    )
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
PYTHONPATH=vllm_standalone_bench pytest -q \
  vllm_standalone_bench/tests/test_auto_bench.py::test_vllm_cache_metadata_payload \
  vllm_standalone_bench/tests/test_auto_bench.py::test_run_controller_writes_vllm_cache_metadata
```

预期：FAIL，`vllm_cache_metadata` 未定义或 `vllm_cache.json` 不存在。

- [ ] **步骤 3：实现 metadata helper 与控制器写入**

在 `vllm_standalone_bench/auto_bench.py` 中新增：

```python
def vllm_cache_metadata(config: AutoBenchConfig, case: BenchmarkCase) -> dict[str, Any] | None:
    cache_dir = resolve_vllm_cache_dir(config, case)
    key = vllm_cache_key(config, case)
    if cache_dir is None or key is None:
        return None
    return {
        "enabled": True,
        "cache_key": key,
        "host_dir": str(cache_dir),
        "container_path": config.run.vllm_cache.container_path,
        "env": build_vllm_cache_env(config),
    }


def write_vllm_cache_metadata(config: AutoBenchConfig, case: BenchmarkCase,
                              layout: CaseLayout) -> None:
    payload = vllm_cache_metadata(config, case)
    if payload is None:
        return
    write_json_atomic(layout.serve_dir / "vllm_cache.json", payload)
```

在 `run_controller()` 中，`serve_layout` 计算后、`serve_cmd` 构建前写入：

```python
            serve_layout = build_layout(config, run_id, serve_case)
            if not dry_run:
                write_vllm_cache_metadata(config, serve_case, serve_layout)
            serve_cmd = build_serve_run_command(config, serve_case, serve_layout.run_dir)
```

不要改 `_run_controller_dry_run()` 的文件输出行为；dry-run 仍只写 `config.resolved.json`。

- [ ] **步骤 4：运行测试验证通过**

运行同步骤 2 命令。

预期：2 passed。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): record vllm cache metadata"
```

---

### 任务 7：文档与 `.gitignore`

**文件：**
- 修改：`vllm_standalone_bench/README.md`
- 修改：`.gitignore`
- 修改：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的文档/ignore 测试**

在 `vllm_standalone_bench/tests/test_auto_bench.py` 中追加：

```python
def test_readme_documents_vllm_cache_persistence():
    readme = (Path(ab.__file__).resolve().parent / "README.md").read_text(encoding="utf-8")

    assert "vllm_cache" in readme
    assert "VLLM_CACHE_ROOT" in readme
    assert "DG_JIT_CACHE_DIR" in readme
    assert "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" in readme


def test_gitignore_ignores_local_cache_dir():
    gitignore = (Path(ab.__file__).resolve().parents[1] / ".gitignore").read_text(
        encoding="utf-8"
    )

    assert ".cache/" in gitignore.splitlines()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
PYTHONPATH=vllm_standalone_bench pytest -q \
  vllm_standalone_bench/tests/test_auto_bench.py::test_readme_documents_vllm_cache_persistence \
  vllm_standalone_bench/tests/test_auto_bench.py::test_gitignore_ignores_local_cache_dir
```

预期：FAIL，README 或 `.gitignore` 缺少对应内容。

- [ ] **步骤 3：更新 README**

在 `vllm_standalone_bench/README.md` 的 auto_bench 章节、fixed warmup 说明之后加入：

````markdown
### vLLM 编译/JIT cache 持久化

GLM5.2 这类 DSA/MoE 模型首次启动会触发 torch.compile、AOT、Triton/Inductor、
DeepGEMM JIT 和 FlashInfer autotune。默认情况下这些缓存位于 vLLM 容器内，
auto_bench 停止并删除容器后会丢失。需要反复运行同一套 benchmark 时，可以启用
`run.vllm_cache`：

```json
"run": {
  "vllm_cache": {
    "enabled": true,
    "root": "/Resource_Planning_Tool/.cache/vllm_auto_bench",
    "container_path": "/vllm-cache",
    "set_default_env": true
  }
}
```

启用后，vLLM serving 容器会挂载 `<root>/<cache_key>:/vllm-cache:rw`，并自动设置：

- `VLLM_CACHE_ROOT=/vllm-cache`
- `DG_JIT_CACHE_DIR=/vllm-cache/deep_gemm`
- `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/vllm-cache/flashinfer_autotune`

建议为 GLM5.2 正式 profile 显式配置稳定 `cache_key`：

```json
"serve_profiles": [{
  "name": "glm52_fp8_tp8_o2",
  "engine": "vllm",
  "cache_key": "glm52-fp8-tp8-h20-o2",
  "gpus": "all",
  "args": ["--tensor-parallel-size", "8", "--kv-cache-dtype", "fp8"]
}]
```

第一次运行仍会完整编译和 JIT；后续相同镜像、模型、GPU 架构、TP、dtype 和 serve args
应复用 cache。不要让不同硬件或不同 serve 参数共享同一个 `cache_key`。cache 目录不会随
run 清理，可手动删除 `.cache/vllm_auto_bench/<cache_key>` 释放磁盘空间。
````

- [ ] **步骤 4：更新 `.gitignore`**

在 `.gitignore` 的 Outputs / temp 或 Local agent/tooling 附近加入：

```gitignore
.cache/
```

- [ ] **步骤 5：运行测试验证通过**

运行同步骤 2 命令。

预期：2 passed。

- [ ] **步骤 6：Commit**

```bash
git add .gitignore vllm_standalone_bench/README.md vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "docs(bench): document vllm cache persistence"
```

---

### 任务 8：回归验证与收尾

**文件：**
- 只读验证：`vllm_standalone_bench/auto_bench.py`
- 只读验证：`vllm_standalone_bench/tests/test_auto_bench.py`
- 只读验证：`vllm_standalone_bench/README.md`
- 只读验证：`.gitignore`

- [ ] **步骤 1：运行 auto_bench targeted tests**

运行：

```bash
PYTHONPATH=vllm_standalone_bench pytest -q vllm_standalone_bench/tests/test_auto_bench.py
```

预期：所有 `test_auto_bench.py` 测试通过。

- [ ] **步骤 2：运行 dry-run 验证命令拼装**

创建临时配置：

```bash
python3 - <<'PY'
import json
from pathlib import Path

src = Path("vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json")
dst = Path("/tmp/auto_bench_vllm_cache_smoke.json")
data = json.loads(src.read_text(encoding="utf-8"))
data["run"]["name"] = "cache_dry_run"
data["run"]["vllm_cache"] = {
    "enabled": True,
    "root": "/Resource_Planning_Tool/.cache/vllm_auto_bench",
    "container_path": "/vllm-cache",
    "set_default_env": True,
}
data["serve_profiles"][0]["cache_key"] = "qwen15b-bf16-default-cache-dry-run"
data["serve_profiles"] = data["serve_profiles"][:1]
dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(dst)
PY
```

运行 dry-run：

```bash
python3 vllm_standalone_bench/auto_bench.py run \
  --config /tmp/auto_bench_vllm_cache_smoke.json \
  --run-id cache_dry_run \
  --dry-run
```

预期输出包含：

```text
-v /Resource_Planning_Tool/.cache/vllm_auto_bench/qwen15b-bf16-default-cache-dry-run:/vllm-cache:rw
-e VLLM_CACHE_ROOT=/vllm-cache
-e DG_JIT_CACHE_DIR=/vllm-cache/deep_gemm
-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/vllm-cache/flashinfer_autotune
```

- [ ] **步骤 3：检查 `.cache/` 被 git 忽略**

运行：

```bash
git check-ignore -q .cache
```

预期：exit 0。

- [ ] **步骤 4：运行 whitespace 检查**

运行：

```bash
git diff --check
```

预期：exit 0，无输出。

- [ ] **步骤 5：记录完整 pytest 基线状态**

运行：

```bash
pytest -q
```

预期：允许仍出现既有失败 `tests/test_inference_token_factory_report.py` 缺少 `outputs/context_analysis_20260609_034248/01_overview.json`。最终汇报必须明确区分该既有失败和本次 targeted tests。

- [ ] **步骤 6：最终 Commit**

如果任务 1-7 已经分步 commit，本步骤只确认没有未提交变更：

```bash
git status --short
```

预期：无未提交变更。

如果执行者没有分步 commit，则一次性提交：

```bash
git add .gitignore vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py vllm_standalone_bench/README.md
git commit -m "feat(bench): persist vllm compile cache"
```
