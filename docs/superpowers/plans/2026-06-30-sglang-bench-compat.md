# vLLM / SGLang 同台对比基准 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让 `vllm_standalone_bench` 在同一次 run 内可分别启动 vLLM 与 SGLang 服务，对同一压测矩阵产出可对照的性能数据（`compare.csv`/`compare.xlsx` + 图表），原始 per-profile 结果保留。

**架构：** 方案 1 最小侵入——`ServeProfile` 加 `engine` 字段、`RunConfig` 加 `images` 映射（向后兼容 `vllm_image`）；`build_serve_run_command` 按 `engine` 分派（vLLM 走原逻辑，SGLang 生成 `launch_server` 命令）；新增 `bench_compare` 聚合层在 `run_controller` 末尾产出对比表与图表。压测层（`run_bench_multi.py`）0 改动。

**技术栈：** Python 3.11、aiohttp、numpy、matplotlib（新增，绘图）、openpyxl、pytest、Docker。

**规格来源：** `docs/superpowers/specs/2026-06-30-sglang-bench-compat-design.md`

**worktree：** `.worktrees/sglang-bench-compat` ↔ 分支 `feat/sglang-bench-compat`。所有命令在 `vllm_standalone_bench/` 目录下运行（`conftest.py` 已将该目录加入 `sys.path`，故 `import auto_bench`、`from bench_compare import aggregate_compare` 均可工作）。

---

## 范围检查

规格覆盖单一子系统（多引擎对比扩展），无需拆分为多个计划。六个任务按依赖顺序串联，每个任务独立可测、可 commit。

## 文件结构

- **改** `vllm_standalone_bench/auto_bench.py` — config schema（`RunConfig`/`ServeProfile`）、解析函数、启动命令分派、`run_controller` 接入聚合、顶部 import
- **新** `vllm_standalone_bench/bench_compare.py` — 多引擎结果对比聚合（对齐 / `compare.csv`+`.xlsx` / 绘图），只读原始 csv
- **改** `vllm_standalone_bench/requirements.txt` — 新增 `matplotlib`
- **新** `vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.sglang_compare.json` — 两引擎并存样例
- **改** `vllm_standalone_bench/README.md` — SGLang 镜像离线搬运、参数等价、对比用法
- **改** `vllm_standalone_bench/tests/test_auto_bench.py` — config / 命令分派 / 聚合接入 / sglang dry-run 测试
- **新** `vllm_standalone_bench/tests/test_bench_compare.py` — 聚合对齐 / 缺失引擎 / 原数据保留测试

---

## 任务 1：配置 schema 扩展（engine + images + 校验）

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`（`RunConfig` dataclass、`ServeProfile` dataclass、`_parse_run`、`_parse_serve_profiles`、`load_config`）
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`（追加）

- [ ] **步骤 1：编写失败的测试**

追加到 `tests/test_auto_bench.py` 末尾：

```python
def test_serve_profile_engine_defaults_to_vllm(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    assert config.serve_profiles[0].engine == "vllm"


def test_invalid_engine_rejected(tmp_path):
    data = minimal_config(tmp_path)
    data["serve_profiles"][0]["engine"] = "trtllm"
    with pytest.raises(ab.ConfigError, match="engine"):
        ab.load_config(write_config(tmp_path, data))


def test_images_falls_back_to_vllm_image(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    assert config.run.images == {"vllm": "009e4cb46541"}


def test_images_missing_engine_rejected(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["images"] = {"vllm": data["run"]["vllm_image"]}
    data["serve_profiles"][0]["engine"] = "sglang"
    with pytest.raises(ab.ConfigError, match="missing image"):
        ab.load_config(write_config(tmp_path, data))


def test_images_without_vllm_image_supported(tmp_path):
    data = minimal_config(tmp_path)
    del data["run"]["vllm_image"]
    data["run"]["images"] = {"sglang": "sglang:latest"}
    data["serve_profiles"][0]["engine"] = "sglang"
    config = ab.load_config(write_config(tmp_path, data))
    assert config.run.images == {"sglang": "sglang:latest"}
    assert config.run.vllm_image is None
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_auto_bench.py -k "engine or images" -v`
预期：FAIL（`ServeProfile` 无 `engine` 字段 / `RunConfig` 无 `images` 字段）。

- [ ] **步骤 3：编写实现代码**

3a. 在 `auto_bench.py` 顶部常量区（`class ConfigError` 定义之后）添加：

```python
SUPPORTED_ENGINES = ("vllm", "sglang")
```

3b. 将 `RunConfig`（约 41-54 行）改为（`vllm_image` 改可选、新增 `images`）：

```python
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
    images: dict[str, str] = field(default_factory=dict)
```

3c. 将 `ServeProfile`（约 72-76 行）改为（新增 `engine`，置于 `name` 之后）：

```python
@dataclass(frozen=True)
class ServeProfile:
    name: str
    engine: str = "vllm"
    gpus: str = "all"
    args: tuple[str, ...] = field(default_factory=tuple)
```

3d. 将 `_parse_run`（约 282-304 行）改为（`vllm_image` 改用 `_optional_string`，新增 `images` 解析）：

```python
def _parse_run(data: dict[str, Any]) -> RunConfig:
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
    )


def _parse_engine_images(run: dict[str, Any], vllm_image: str | None) -> dict[str, str]:
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
    if not images:
        raise ConfigError("run.images (or run.vllm_image) must define at least one engine image")
    return images
```

3e. 在 `_parse_serve_profiles`（约 347-364 行）的 `parsed.append(ServeProfile(...))` 之前加入 engine 解析与校验，并把 `engine=engine` 传入 `ServeProfile`：

```python
        engine = _string(profile.get("engine", "vllm"), "serve_profile.engine")
        if engine not in SUPPORTED_ENGINES:
            raise ConfigError(
                f"serve_profile.engine must be one of {SUPPORTED_ENGINES}, got {engine!r}"
            )
        parsed.append(ServeProfile(
            name=_safe_name(_required(profile, "name", "serve_profile.name"),
                            "serve_profile.name"),
            engine=engine,
            gpus=_string(profile.get("gpus", "all"), "serve_profile.gpus"),
            args=tuple(args),
        ))
```

3f. 将 `load_config`（约 411-425 行）末尾改为（新增跨字段校验）：

```python
    config = AutoBenchConfig(run, mounts, models, serve_profiles, bench_profiles)
    _validate_images_cover_engines(config)
    return config


def _validate_images_cover_engines(config: AutoBenchConfig) -> None:
    engines = {profile.engine for profile in config.serve_profiles}
    missing = sorted(engine for engine in engines if engine not in config.run.images)
    if missing:
        raise ConfigError(
            f"run.images missing image for engine(s): {', '.join(missing)}"
        )
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_auto_bench.py -v`
预期：PASS（新增 5 个测试 + 原有测试全绿；`minimal_config` 含 `vllm_image`，向后兼容）。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): ServeProfile.engine + RunConfig.images 配置与校验"
```

---

## 任务 2：启动命令分派（SGLang launch_server）

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`（`build_vllm_run_command` 镜像源、新增 `build_serve_run_command` 与 `_build_sglang_run_command`、`run_controller` 切换调用）
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`（追加）

- [ ] **步骤 1：编写失败的测试**

追加到 `tests/test_auto_bench.py` 末尾（`sglang_config` 复用 `minimal_config` 构造单 sglang profile）：

```python
def sglang_config(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["images"] = {"vllm": data["run"]["vllm_image"], "sglang": "sglang:latest"}
    profile = {
        "name": "sglang_bf16",
        "engine": "sglang",
        "gpus": "all",
        "args": ["--dtype", "bfloat16", "--mem-fraction-static", "0.70"],
    }
    data["serve_profiles"] = [profile]
    return data


def test_build_sglang_command_uses_launch_server(tmp_path):
    config = ab.load_config(write_config(tmp_path, sglang_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_serve_run_command(config, case, tmp_path / "results" / "run123")

    assert value_after(cmd, "--entrypoint") == "python3"
    assert "sglang:latest" in cmd
    assert value_after(cmd, "-m") == "sglang.launch_server"
    assert value_after(cmd, "--model-path") == "/models/Qwen2.5-1.5B-Instruct"
    assert value_after(cmd, "--host") == "0.0.0.0"
    assert value_after(cmd, "--port") == "8000"
    assert value_after(cmd, "--served-model-name") == "qwen2_5_1_5b"
    assert value_after(cmd, "--api-key") == "local-bench-key"
    assert value_after(cmd, "--mem-fraction-static") == "0.70"
    assert value_after(cmd, "--gpus") == "all"


def test_build_serve_command_dispatches_vllm(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_serve_run_command(config, case, tmp_path / "results" / "run123")

    assert value_after(cmd, "--entrypoint") == "vllm"
    assert "serve" in cmd
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_auto_bench.py -k "sglang_command or dispatches_vllm" -v`
预期：FAIL（`build_serve_run_command` 未定义）。

- [ ] **步骤 3：编写实现代码**

3a. 在 `build_vllm_run_command`（约 466-499 行）中，将镜像来源由 `config.run.vllm_image` 改为 `config.run.images["vllm"]`（`minimal_config` 经回落后 `images["vllm"] == vllm_image`，现有断言 `cmd.index(config.run.vllm_image)` 仍通过）：

```python
    cmd.extend([
        config.run.images["vllm"],
        "serve", case.model.model_path,
        "--served-model-name", case.api_model_name,
        "--host", "0.0.0.0",
        "--port", str(config.run.container_port),
    ])
```

3b. 在 `build_vllm_run_command` 之后、`_append_many` 之前（约 499 行后）新增分派函数与 sglang 命令构造：

```python
def build_serve_run_command(config: AutoBenchConfig, case: BenchmarkCase,
                            run_dir: Path) -> list[str]:
    """按 serve_profile.engine 分派服务启动命令。args 原样透传，不做参数翻译。"""
    if case.serve_profile.engine == "sglang":
        return _build_sglang_run_command(config, case, run_dir)
    return build_vllm_run_command(config, case, run_dir)


def _build_sglang_run_command(config: AutoBenchConfig, case: BenchmarkCase,
                              run_dir: Path) -> list[str]:
    resolved_run_dir = Path(run_dir).resolve()
    cmd = [
        "docker", "run", "-d",
        "--name", case.container_name,
        "--label", f"{NETWORK_MANAGED_LABEL}=true",
        "--label", f"{NETWORK_RUN_ID_LABEL}={case.run_id}",
        "--label", f"{CONTAINER_RUN_DIR_LABEL}={resolved_run_dir}",
        "--label", f"{CONTAINER_MODEL_LABEL}={case.model.name}",
        "--label", f"{CONTAINER_SERVE_PROFILE_LABEL}={case.serve_profile.name}",
        "--gpus", case.serve_profile.gpus,
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
    cmd.extend(case.serve_profile.args)
    return cmd
```

3c. 在 `run_controller` 中（约 1571 行）将服务启动命令构造切换到分派函数，并把后续变量名 `vllm_cmd` 改为 `serve_cmd`（涉及约 1571、1576、1585 三处）：

```python
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_auto_bench.py -v`
预期：PASS（新增 sglang/dispatch 测试 + 原有 `build_vllm_run_command` 测试仍绿）。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): build_serve_run_command 按 engine 分派 vllm/sglang"
```

---

## 任务 3：对比聚合模块（bench_compare）

**文件：**
- 创建：`vllm_standalone_bench/bench_compare.py`
- 测试：`vllm_standalone_bench/tests/test_bench_compare.py`（新）

- [ ] **步骤 1：编写失败的测试**

新建 `tests/test_bench_compare.py`：

```python
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import bench_compare as bc


CSV_HEADER = (
    "model,backend,input_len,output_len,parallel_num,epochs,num_prompts,n_success,"
    "n_failed,avg_input_tokens,avg_output_tokens,throughput_req_s,throughput_tok_s,"
    "ttft_mean_ms,ttft_p50_ms,ttft_p90_ms,ttft_p99_ms,tpot_mean_ms,tpot_p50_ms,"
    "tpot_p90_ms,tpot_p99_ms,e2el_mean_ms,e2el_p50_ms,e2el_p90_ms,e2el_p99_ms,duration_s"
)


def _write_result_csv(path: Path, parallel: int, ttft_p50: int, tput: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        CSV_HEADER + "\n"
        f"m,openai-chat,64,32,{parallel},1,1,1,0,64,32,1.0,{tput},10,{ttft_p50},"
        f"20,30,5,4,6,8,50,40,60,70,10\n",
        encoding="utf-8",
    )


def _fake_config():
    serve = [
        SimpleNamespace(name="vllm_bf16", engine="vllm", gpus="all", args=()),
        SimpleNamespace(name="sglang_bf16", engine="sglang", gpus="all", args=()),
    ]
    return SimpleNamespace(
        serve_profiles=serve,
        models=[SimpleNamespace(name="qwen")],
        bench_profiles=[SimpleNamespace(name="smoke")],
    )


def test_aggregate_aligns_two_engines_and_preserves_originals(tmp_path):
    config = _fake_config()
    run_dir = tmp_path / "run1"
    vllm_csv = run_dir / "qwen" / "vllm_bf16" / "smoke" / "result.csv"
    sglang_csv = run_dir / "qwen" / "sglang_bf16" / "smoke" / "result.csv"
    _write_result_csv(vllm_csv, parallel=1, ttft_p50=11, tput=100)
    _write_result_csv(sglang_csv, parallel=1, ttft_p50=22, tput=200)
    before_vllm = hashlib.sha256(vllm_csv.read_bytes()).hexdigest()
    before_sglang = hashlib.sha256(sglang_csv.read_bytes()).hexdigest()

    out = bc.aggregate_compare(config, run_dir)

    assert out == run_dir / "compare.csv"
    assert (run_dir / "compare.xlsx").exists()
    plots = list((run_dir / "plots").glob("*.png"))
    assert plots, "应至少生成一张图表"
    rows = list(csv.DictReader((run_dir / "compare.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 1
    row = rows[0]
    assert row["vllm__throughput_tok_s"] == "100"
    assert row["sglang__throughput_tok_s"] == "200"
    assert row["vllm__ttft_p50_ms"] == "11"
    assert row["sglang__ttft_p50_ms"] == "22"
    # 原始 result.csv 必须未被修改
    assert hashlib.sha256(vllm_csv.read_bytes()).hexdigest() == before_vllm
    assert hashlib.sha256(sglang_csv.read_bytes()).hexdigest() == before_sglang


def test_aggregate_missing_engine_fills_na(tmp_path):
    config = _fake_config()
    run_dir = tmp_path / "run1"
    _write_result_csv(run_dir / "qwen" / "vllm_bf16" / "smoke" / "result.csv",
                      parallel=1, ttft_p50=11, tput=100)

    out = bc.aggregate_compare(config, run_dir)

    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert rows[0]["vllm__throughput_tok_s"] == "100"
    assert rows[0]["sglang__throughput_tok_s"] == "N/A"


def test_aggregate_no_results_returns_none(tmp_path):
    config = _fake_config()

    assert bc.aggregate_compare(config, tmp_path / "empty") is None
    assert not (tmp_path / "empty" / "compare.csv").exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_bench_compare.py -v`
预期：FAIL（`No module named 'bench_compare'`）。

- [ ] **步骤 3：编写实现代码**

新建 `vllm_standalone_bench/bench_compare.py`（matplotlib / openpyxl 在函数内 lazy import，避免模块导入即触发重依赖）：

```python
"""多引擎结果对比聚合。

读取各 serve_profile 的 result.csv，按 (bench_profile, input_len, output_len,
parallel_num) 对齐多引擎，产出 compare.csv / compare.xlsx 与图表。

铁律：原始 result.csv 只读，本模块永不修改或删除它们。
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 参与对比的指标（须为 result.csv 真实列名）
COMPARE_METRICS = ("throughput_tok_s", "ttft_p50_ms", "ttft_p90_ms", "tpot_p50_ms")
PLOT_METRICS = ("throughput_tok_s", "ttft_p50_ms")
_PLOT_YLABEL = {
    "throughput_tok_s": "输出吞吐 (tok/s)",
    "ttft_p50_ms": "TTFT p50 (ms)",
}


def _engine_by_serve_profile(config: Any) -> dict[str, str]:
    return {profile.name: profile.engine for profile in config.serve_profiles}


def _read_result_rows(path: Path, bench_profile: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["bench_profile"] = bench_profile  # csv 内无此列，由目录名注入
    return rows


def _collect_aligned(
    config: Any, run_dir: Path
) -> dict[tuple, dict[str, dict[str, str]]]:
    aligned: dict[tuple, dict[str, dict[str, str]]] = {}
    for serve_name, engine in _engine_by_serve_profile(config).items():
        for model in config.models:
            for bench in config.bench_profiles:
                csv_path = run_dir / model.name / serve_name / bench.name / "result.csv"
                if not csv_path.exists():
                    logger.warning("对比缺失结果文件，跳过：%s", csv_path)
                    continue
                for row in _read_result_rows(csv_path, bench.name):
                    key = (
                        bench.name,
                        int(row["input_len"]),
                        int(row["output_len"]),
                        int(row["parallel_num"]),
                    )
                    aligned.setdefault(key, {})[engine] = row
    return aligned


def _ordered_engines(
    aligned: dict[tuple, dict[str, dict[str, str]]]
) -> list[str]:
    seen: list[str] = []
    for engine_map in aligned.values():
        for engine in engine_map:
            if engine not in seen:
                seen.append(engine)
    return seen


def _compare_fieldnames(engines: list[str]) -> list[str]:
    cols = ["bench_profile", "input_len", "output_len", "parallel_num"]
    for engine in engines:
        for metric in COMPARE_METRICS:
            cols.append(f"{engine}__{metric}")
    return cols


def _build_compare_rows(
    aligned: dict[tuple, dict[str, dict[str, str]]], engines: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(aligned):
        bench_profile, in_len, out_len, parallel = key
        row: dict[str, Any] = {
            "bench_profile": bench_profile,
            "input_len": in_len,
            "output_len": out_len,
            "parallel_num": parallel,
        }
        engine_map = aligned[key]
        for engine in engines:
            present = engine in engine_map
            for metric in COMPARE_METRICS:
                col = f"{engine}__{metric}"
                row[col] = engine_map[engine].get(metric, "") if present else "N/A"
        rows.append(row)
    return rows


def _write_compare_csv(
    rows: list[dict[str, Any]], engines: list[str], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_compare_fieldnames(engines), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_compare_xlsx(
    rows: list[dict[str, Any]], engines: list[str], path: Path
) -> None:
    try:
        from openpyxl import Workbook
    except ImportError:
        logger.warning("openpyxl 不可用，跳过 compare.xlsx")
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "compare"
    fieldnames = _compare_fieldnames(engines)
    ws.append(fieldnames)
    for row in rows:
        ws.append([row.get(col, "") for col in fieldnames])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _plot(
    run_dir: Path,
    aligned: dict[tuple, dict[str, dict[str, str]]],
    engines: list[str],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 不可用，跳过绘图")
        return
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    # 按 (bench_profile, input_len, output_len) 聚点
    grouped: dict[tuple, dict[str, list[tuple[int, float]]]] = {}
    for key, engine_map in aligned.items():
        bench_profile, in_len, out_len, parallel = key
        gkey = (bench_profile, in_len, out_len)
        series = grouped.setdefault(gkey, {})
        for engine, row in engine_map.items():
            for metric in PLOT_METRICS:
                series.setdefault(f"{engine}__{metric}", []).append(
                    (parallel, _to_float(row.get(metric)))
                )
    for gkey, series in grouped.items():
        bench_profile, in_len, out_len = gkey
        for metric in PLOT_METRICS:
            plt.figure(figsize=(7, 4))
            for label, points in sorted(series.items()):
                if not label.endswith(f"__{metric}"):
                    continue
                points_sorted = sorted(points, key=lambda p: p[0])
                xs = [p[0] for p in points_sorted]
                ys = [p[1] for p in points_sorted]
                engine = label.split("__", 1)[0]
                plt.plot(xs, ys, marker="o", label=engine)
            plt.xlabel("并发数 (parallel_num)")
            plt.ylabel(_PLOT_YLABEL[metric])
            plt.title(f"{bench_profile} in={in_len} out={out_len} · {metric}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{bench_profile}__{in_len}x{out_len}__{metric}.png")
            plt.close()


def aggregate_compare(config: Any, run_dir: Path) -> Path | None:
    """聚合多引擎结果。返回 compare.csv 路径；无任何结果时返回 None。"""
    run_dir = Path(run_dir)
    aligned = _collect_aligned(config, run_dir)
    if not aligned:
        logger.info("无可对比结果，跳过聚合")
        return None
    engines = _ordered_engines(aligned)
    rows = _build_compare_rows(aligned, engines)
    compare_csv = run_dir / "compare.csv"
    _write_compare_csv(rows, engines, compare_csv)
    _write_compare_xlsx(rows, engines, run_dir / "compare.xlsx")
    _plot(run_dir, aligned, engines)
    logger.info("对比聚合完成：%s（引擎：%s）", compare_csv, engines)
    return compare_csv
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_bench_compare.py -v`
预期：PASS（对齐、缺失填 N/A、原数据哈希不变、图表生成）。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/bench_compare.py vllm_standalone_bench/tests/test_bench_compare.py
git commit -m "feat(bench): 新增 bench_compare 多引擎结果对比聚合与绘图"
```

---

## 任务 4：run_controller 接入聚合

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`（顶部 import、`run_controller` 末尾插入聚合调用）
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`（追加）

- [ ] **步骤 1：编写失败的测试**

追加到 `tests/test_auto_bench.py` 末尾：

```python
def test_controller_invokes_aggregate_after_groups(tmp_path, monkeypatch):
    data = minimal_config(tmp_path)
    data["run"]["images"] = {"vllm": data["run"]["vllm_image"], "sglang": "sglang:latest"}
    sglang_profile = {
        "name": "sglang_bf16",
        "engine": "sglang",
        "gpus": "all",
        "args": ["--dtype", "bfloat16"],
    }
    data["serve_profiles"].append(sglang_profile)
    config = ab.load_config(write_config(tmp_path, data))
    calls = []
    monkeypatch.setattr(ab, "aggregate_compare", lambda c, rd: calls.append(Path(rd)) or None)
    monkeypatch.setattr(ab, "wait_for_ready", lambda *a, **k: True)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    assert result == 0
    assert len(calls) == 1
    assert calls[0].name == "run123"


def test_controller_dry_run_skips_aggregate(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    calls = []
    monkeypatch.setattr(ab, "aggregate_compare", lambda c, rd: calls.append(rd))

    ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=True)

    assert calls == []
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_auto_bench.py -k "invokes_aggregate or dry_run_skips_aggregate" -v`
预期：FAIL（`ab.aggregate_compare` 不存在 / 未被调用）。

- [ ] **步骤 3：编写实现代码**

3a. 在 `auto_bench.py` 顶部 import 区添加（若模块尚未定义 logger，一并补上）：

```python
import logging
from bench_compare import aggregate_compare

logger = logging.getLogger("auto_bench")
```

> 若 `import logging` 或 `logger` 已存在则不重复添加；`from bench_compare import aggregate_compare` 必须新增（`bench_compare` 与 `auto_bench` 同目录，`conftest` 已把该目录加入 `sys.path`，导入可用；`bench_compare` 内部 lazy import 重依赖，不会在导入时拉起 matplotlib）。

3b. 在 `run_controller` 中，group 循环 `if interrupted: break`（约 1734-1735 行）之后、`write_state(run_dir, finished_state(run_id, manifest))`（约 1737 行）之前插入聚合调用（best-effort，失败不阻断主流程）：

```python
            if interrupted:
                break

        if not dry_run and not interrupted:
            try:
                aggregate_compare(config, run_dir)
            except Exception as exc:
                logger.warning("结果对比聚合失败：%s", exc)

        write_state(run_dir, finished_state(run_id, manifest))
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_auto_bench.py -v`
预期：PASS（聚合在多引擎 run 后被调用一次；dry-run 不调用）。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): run_controller 末尾接入 bench_compare 聚合"
```

---

## 任务 5：样例配置 + requirements + README

**文件：**
- 创建：`vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.sglang_compare.json`
- 修改：`vllm_standalone_bench/requirements.txt`
- 修改：`vllm_standalone_bench/README.md`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`（追加 shipped config 解析测试）

- [ ] **步骤 1：编写失败的测试**

追加到 `tests/test_auto_bench.py` 末尾：

```python
def test_shipped_sglang_compare_config_parses():
    path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "auto_bench.qwen2_5_1_5b.sglang_compare.json"
    )
    config = ab.load_config(path)
    engines = {profile.engine for profile in config.serve_profiles}
    assert engines == {"vllm", "sglang"}
    assert "vllm" in config.run.images
    assert "sglang" in config.run.images
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_auto_bench.py -k "sglang_compare_config_parses" -v`
预期：FAIL（配置文件不存在）。

- [ ] **步骤 3：编写实现代码**

3a. 在 `requirements.txt` 的 `transformers>=4.36.0` 之后追加：

```
matplotlib>=3.7.0
```

3b. 新建 `configs/auto_bench.qwen2_5_1_5b.sglang_compare.json`（基于现有 `auto_bench.qwen2_5_1_5b.smoke.json`，`run` 段用 `images`，两个 serve_profile）：

```json
{
  "run": {
    "name": "qwen2_5_1_5b_sglang_compare",
    "results_dir": "vllm_standalone_bench/results",
    "images": { "vllm": "009e4cb46541", "sglang": "lmsysorg/sglang:latest" },
    "bench_image": "vllm-bench-runner:offline",
    "network": "vllm-bench-net",
    "create_network": true,
    "cleanup_network": true,
    "container_port": 8888,
    "publish_host_port": false,
    "host_port": 18000,
    "api_key": "local-bench-key",
    "ready_timeout_sec": 1800,
    "cooldown_sec": 5
  },
  "mounts": {
    "models": "/Resource_Planning_Tool/model"
  },
  "models": [
    {
      "name": "qwen2_5_1_5b",
      "model_path": "/models/Qwen2.5-1.5B-Instruct",
      "tokenizer_path": "/models/Qwen2.5-1.5B-Instruct",
      "served_model_name": "qwen2_5_1_5b"
    }
  ],
  "serve_profiles": [
    {
      "name": "vllm_bf16",
      "engine": "vllm",
      "gpus": "all",
      "args": ["--dtype", "bfloat16", "--gpu-memory-utilization", "0.70"]
    },
    {
      "name": "sglang_bf16",
      "engine": "sglang",
      "gpus": "all",
      "args": ["--dtype", "bfloat16", "--mem-fraction-static", "0.70"]
    }
  ],
  "bench_profiles": [
    {
      "name": "smoke",
      "backend": "openai-chat",
      "input_lens": [1024, 2048, 4096],
      "output_lens": [256, 256, 256],
      "parallel_nums": [1, 4, 8, 16],
      "epochs": 3,
      "prefix_ratio": 0.0,
      "warmup_requests": 1,
      "cross_product": false
    }
  ]
}
```

3c. 在 `README.md` 的「离线双镜像自动化压测」章节之后追加一节：

```markdown
## vLLM / SGLang 同台对比

通过 `serve_profiles` 的 `engine` 字段与 `run.images` 映射，可在同一次 run 内分别启动 vLLM 与 SGLang 做对比。样例：`configs/auto_bench.qwen2_5_1_5b.sglang_compare.json`。

### SGLang 镜像离线搬运

```bash
# 联网机
docker pull lmsysorg/sglang:latest
docker save lmsysorg/sglang:latest -o sglang.offline.tar
# 离线机
docker load -i sglang.offline.tar
```

### 常用启动参数等价（仅文档参考，配置里 `args` 原样透传，不自动翻译）

| 用途 | vLLM | SGLang |
|---|---|---|
| 显存占用比例 | `--gpu-memory-utilization` | `--mem-fraction-static` |
| 张量并行 | `--tensor-parallel-size` | `--tp-size` |
| 最大上下文 | `--max-model-len` | `--context-length` |

run 结束后在 `results/<run_id>/` 产出 `compare.csv`、`compare.xlsx` 与 `plots/*.png`，各引擎原始 `result.csv` 保留在 `<model>/<serve_profile>/<bench_profile>/` 子目录。
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_auto_bench.py -k "sglang_compare_config_parses" -v`
预期：PASS（shipped 配置可解析，含双引擎与 images）。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.sglang_compare.json \
        vllm_standalone_bench/requirements.txt \
        vllm_standalone_bench/README.md \
        vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): sglang 对比样例配置 + matplotlib 依赖 + README"
```

---

## 任务 6：SGLang dry-run 集成测试 + 全量验收

**文件：**
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`（追加）

- [ ] **步骤 1：编写失败的测试**

追加到 `tests/test_auto_bench.py` 末尾：

```python
def test_controller_dry_run_prints_sglang_command(tmp_path, capsys):
    data = minimal_config(tmp_path)
    data["run"]["images"] = {"vllm": data["run"]["vllm_image"], "sglang": "sglang:latest"}
    data["serve_profiles"][0]["engine"] = "sglang"
    config = ab.load_config(write_config(tmp_path, data))

    ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=True)

    out = capsys.readouterr().out
    assert "sglang.launch_server" in out
    assert "sglang:latest" in out
    assert "--model-path" in out
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_auto_bench.py -k "dry_run_prints_sglang_command" -v`
预期：FAIL（dry-run 仍打印 vllm 命令，不含 `sglang.launch_server`；本任务依赖任务 2 的分派已生效——若任务 2 已完成则本测试应直接 PASS，此时步骤 2 仅作回归确认）。

- [ ] **步骤 3：编写实现代码**

无新实现代码（本任务为集成回归与全量验收）。若步骤 2 失败，回查任务 2 的 `run_controller` 分派是否已切换到 `build_serve_run_command`。

- [ ] **步骤 4：运行全量测试验收**

运行：
```bash
python -m pytest -q
```
预期：PASS（全部测试绿，含新增 config / 命令分派 / 聚合 / dry-run 用例）。

并手动 dry-run 验证命令形态：
```bash
python3 auto_bench.py run \
  --config configs/auto_bench.qwen2_5_1_5b.sglang_compare.json \
  --run-id plan_dry_run --dry-run
```
预期：打印的命令中 vLLM group 含 `vllm serve`、SGLang group 含 `sglang.launch_server`，两镜像分别取自 `images.vllm` / `images.sglang`。

- [ ] **步骤 5：Commit（仅当步骤 1 新增了测试）**

```bash
git add vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "test(bench): sglang dry-run 集成回归"
```

---

## 人工验收（需 GPU + 镜像，非自动化）

完成上述任务后，在具备 GPU 与离线镜像的环境执行真实 smoke（非本计划强制步骤）：

```bash
python3 auto_bench.py run \
  --config configs/auto_bench.qwen2_5_1_5b.sglang_compare.json
```

确认：① vLLM 与 SGLang 各自起服务、压测落盘；② `ignore_eos=true` + `max_tokens` 生效，SGLang 输出到达指定长度；③ `results/<run_id>/compare.csv`、`compare.xlsx`、`plots/*.png` 生成；④ 各引擎原始 `result.csv` 完整保留。

---

## 自检

**1. 规格覆盖度：**
- 同台对比 → 任务 1（schema）+ 任务 2（分派）+ 任务 4（聚合接入）+ 任务 5（样例）。
- 输出到指定长度 → 现有 `run_bench_multi.py` 已强制 `ignore_eos=True`（规格第 2 节确认 SGLang 支持），无需新任务；任务 6 人工验收核验。
- 自动脚本兼容 → 任务 2 分派后 `run_controller` 对 sglang 自动生效；任务 6 dry-run 回归。
- 原数据保留 → 任务 3 测试用哈希断言原始 csv 不变。
- 错误处理（非法 engine / 缺镜像 / 聚合 best-effort）→ 任务 1（ConfigError）+ 任务 4（try/except）。
- 依赖与前置（matplotlib / 镜像搬运 / 参数等价）→ 任务 5。
- 无遗漏。

**2. 占位符扫描：** 无 TODO / "适当处理" / "类似任务 N"。每个代码步骤含完整可粘贴代码与精确命令。

**3. 类型一致性：** `engine`（任务 1 的 `ServeProfile.engine`）→ 任务 2 `case.serve_profile.engine`、任务 3 `_engine_by_serve_profile` 读取 `profile.engine`；`images`（任务 1 `RunConfig.images: dict[str,str]`）→ 任务 2 `config.run.images["vllm"]`/`["sglang"]`；`aggregate_compare(config, run_dir)`（任务 3）→ 任务 4 调用签名一致。命名统一。
