# Warmup 固定并发预热 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 `vllm_standalone_bench` 增加「固定并发预热」能力，消除小并发档（4 并发）TTFT 首批冷启动尖峰。

**架构：** 解耦 warmup 的并发度与输出长度。`serve.py` 提取纯函数 `resolve_warmup_config` + 加两个可选参数；`run_bench_multi.py` 加两个 CLI 透传；`auto_bench.py` 的 `BenchProfile` 加两个可选字段并透传到命令。全局仅在首个配置预热一次：固定并发 4 × 首个配置输入 × 输出 128。所有新参数默认 `None`，行为完全等同现状（向后兼容）。

**技术栈：** Python 3 + asyncio + aiohttp；pytest；无 vllm/torch 依赖（走 `run_bench_serve` shim）。

**规格：** `docs/superpowers/specs/2026-06-30-warmup-fixed-concurrency-design.md`

---

## 文件结构

| 文件 | 职责 | 动作 |
|---|---|---|
| `vllm_standalone_bench/vllm_bench/serve.py` | warmup 决策纯函数 + benchmark_async 参数 + main_async 透传 | 修改 |
| `vllm_standalone_bench/run_bench_multi.py` | CLI 参数 + _build_base_args 透传 + 首次预热请求数 | 修改 |
| `vllm_standalone_bench/auto_bench.py` | BenchProfile 字段 + 解析 + 命令构建 | 修改 |
| `vllm_standalone_bench/tests/test_serve_metrics.py` | resolve_warmup_config 单测 | 修改 |
| `vllm_standalone_bench/tests/test_shell_scripts.py` 或新建 `test_warmup_config.py` | run_bench_multi / auto_bench 透传单测 | 修改/新建 |
| `vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.sglang_compare.json` | 默认启用固定并发预热 | 修改 |
| `vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json` | 默认启用固定并发预热 | 修改 |
| `vllm_standalone_bench/README.md` | 补 warmup 说明 | 修改 |

所有路径相对仓库根。worktree 根为 `.worktrees/warmup-fixed-concurrency/`。

---

## 任务 1：serve.py — 提取 warmup 决策纯函数并解耦

**文件：**
- 修改：`vllm_standalone_bench/vllm_bench/serve.py`（模块级新增 `resolve_warmup_config`；改 `benchmark_async` 签名 line 666 附近、test_input line 706-718、warmup_semaphore line 751-752、main_async 透传 line 1920 附近）
- 测试：`vllm_standalone_bench/tests/test_serve_metrics.py`

- [ ] **步骤 1：编写失败的测试**

追加到 `tests/test_serve_metrics.py` 末尾：

```python
def test_resolve_warmup_config_defaults_to_profile_values():
    # 均为 None → 回退到该档并发度与输出长度（向后兼容）
    cc, ol = serve.resolve_warmup_config(
        max_concurrency=8, warmup_concurrency=None,
        output_len=1024, warmup_output_len=None)
    assert cc == 8
    assert ol == 1024


def test_resolve_warmup_config_overrides_when_set():
    # 非 None → 用 warmup 专用值（固定并发 + 短输出）
    cc, ol = serve.resolve_warmup_config(
        max_concurrency=8, warmup_concurrency=4,
        output_len=1024, warmup_output_len=128)
    assert cc == 4
    assert ol == 128


def test_resolve_warmup_config_concurrency_none_falls_back_even_when_output_set():
    cc, ol = serve.resolve_warmup_config(
        max_concurrency=16, warmup_concurrency=None,
        output_len=1024, warmup_output_len=128)
    assert cc == 16   # 并发回退该档
    assert ol == 128  # 输出仍用短值
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_serve_metrics.py::test_resolve_warmup_config_defaults_to_profile_values tests/test_serve_metrics.py::test_resolve_warmup_config_overrides_when_set tests/test_serve_metrics.py::test_resolve_warmup_config_concurrency_none_falls_back_even_when_output_set -v`
预期：FAIL，`AttributeError: module has no attribute 'resolve_warmup_config'`

- [ ] **步骤 3：实现 resolve_warmup_config 纯函数**

在 `vllm_bench/serve.py` 中 `calculate_metrics` 函数定义之前（模块级），加入：

```python
def resolve_warmup_config(
    *,
    max_concurrency: int | None,
    warmup_concurrency: int | None,
    output_len: int,
    warmup_output_len: int | None,
) -> tuple[int | None, int]:
    """决定 warmup 阶段的并发度与输出长度。

    warmup_concurrency 为 None 时回退到 max_concurrency（向后兼容）；
    warmup_output_len 为 None 时回退到该档 output_len。
    """
    cc = warmup_concurrency if warmup_concurrency else max_concurrency
    ol = warmup_output_len if warmup_output_len else output_len
    return cc, ol
```

- [ ] **步骤 4：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_serve_metrics.py -k resolve_warmup_config -v`
预期：3 passed

- [ ] **步骤 5：benchmark_async 签名加两个可选参数**

在 `benchmark_async` 签名（`lora_assignment: Literal["random", "round-robin"] = "random",` 这一行之后、`ramp_up_strategy` 之前）插入两行：

```python
    warmup_concurrency: int | None = None,
    warmup_output_len: int | None = None,
```

- [ ] **步骤 6：test_input 用 warmup 输出长度**

定位 `test_prompt, test_prompt_len, test_output_len, test_mm_content = (...)`（约 line 691）之后、`test_input = RequestFuncInput(...)`（约 line 706）之前，插入决策调用：

```python
    # 决定 warmup 用的并发度与输出长度（None 时回退到该档值，向后兼容）
    warmup_cc, warmup_ol = resolve_warmup_config(
        max_concurrency=max_concurrency,
        warmup_concurrency=warmup_concurrency,
        output_len=test_output_len,
        warmup_output_len=warmup_output_len,
    )
```

并把 `test_input = RequestFuncInput(...)` 中的 `output_len=test_output_len,` 改为 `output_len=warmup_ol,`。

- [ ] **步骤 7：warmup_semaphore 用 warmup 并发度**

定位 warmup 块（`if num_warmups > 0:` 内）：

```python
        warmup_semaphore = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency
            else contextlib.nullcontext()
        )
```

改为：

```python
        warmup_semaphore = (
            asyncio.Semaphore(warmup_cc)
            if warmup_cc
            else contextlib.nullcontext()
        )
```

- [ ] **步骤 8：main_async 透传新参数**

定位 `main_async` 中对 `benchmark_async(` 的调用（约 line 1905-1925），在 `max_concurrency=args.max_concurrency,` 之后加两行：

```python
        warmup_concurrency=getattr(args, "warmup_concurrency", None),
        warmup_output_len=getattr(args, "warmup_output_len", None),
```

- [ ] **步骤 9：全量回归**

运行：`cd vllm_standalone_bench && python -m pytest tests/ -v`
预期：全部 PASS（新测试通过，旧测试不破坏）

- [ ] **步骤 10：Commit**

```bash
git add vllm_standalone_bench/vllm_bench/serve.py vllm_standalone_bench/tests/test_serve_metrics.py
git commit -m "feat(bench): serve.py 解耦 warmup 并发与输出长度

新增 resolve_warmup_config 纯函数，benchmark_async 加 warmup_concurrency/
warmup_output_len 可选参数（默认 None 向后兼容），main_async 透传。"
```

---

## 任务 2：run_bench_multi.py — CLI 与透传

**文件：**
- 修改：`vllm_standalone_bench/run_bench_multi.py`（argparse 约 line 790；_build_base_args 约 line 161；is_first_run 约 line 614-620）
- 测试：`vllm_standalone_bench/tests/test_shell_scripts.py`（或新建 `tests/test_warmup_config.py`）

- [ ] **步骤 1：编写失败的测试**

新建 `vllm_standalone_bench/tests/test_warmup_config.py`：

```python
"""warmup 固定并发预热的参数透传测试。"""
import run_bench_multi


def test_build_arg_parser_has_warmup_opts():
    args = run_bench_multi.build_arg_parser().parse_args([
        "--model", "m", "--backend", "openai",
        "--host", "127.0.0.1", "--port", "8000",
        "--warmup-concurrency", "4", "--warmup-output-len", "128",
    ])
    assert args.warmup_concurrency == 4
    assert args.warmup_output_len == 128


def test_build_arg_parser_warmup_opts_default_none():
    args = run_bench_multi.build_arg_parser().parse_args([
        "--model", "m", "--backend", "openai",
        "--host", "127.0.0.1", "--port", "8000",
    ])
    assert args.warmup_concurrency is None
    assert args.warmup_output_len is None


def test_build_base_args_passes_warmup_opts():
    args = run_bench_multi.build_arg_parser().parse_args([
        "--model", "m", "--backend", "openai",
        "--host", "127.0.0.1", "--port", "8000",
        "--warmup-concurrency", "4", "--warmup-output-len", "128",
    ])
    base = run_bench_multi._build_base_args(args)
    assert base.warmup_concurrency == 4
    assert base.warmup_output_len == 128
```

> 注：若 `run_bench_multi` 当前没有 `build_arg_parser()`，步骤 3 会先提取它。

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_warmup_config.py -v`
预期：FAIL（`build_arg_parser` 不存在或属性缺失）

- [ ] **步骤 3：提取 build_arg_parser**

`run_bench_multi.py` 当前 CLI 解析内联在 `_parse_args()`（约 line 723）：`p = argparse.ArgumentParser(...)` 后接所有 `add_argument` / `add_argument_group`。将其提取为模块级函数：

```python
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(...),                          # 原 _parse_args 中的描述原样搬入
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=...,
    )
    # 原 _parse_args 中 p.add_argument_group / add_argument 调用全部原样搬入
    return p
```

`_parse_args()` 改为 `p = build_arg_parser()`，保留其后的 `args = p.parse_args()` 与任何后续校验逻辑不动。`main()` 仍调 `_parse_args()`，行为不变。

- [ ] **步骤 4：CLI 加两个参数**

在 `--warmup-requests` 的 `add_argument` 之后加入：

```python
    bench.add_argument('--warmup-concurrency', type=int, default=None,
                       help='warmup 固定并发数（仅首次预热生效；默认 None=跟随该档并发）')
    bench.add_argument('--warmup-output-len', type=int, default=None,
                       help='warmup 请求输出长度（默认 None=跟随该档输出；设短值省 decode 时间）')
```

- [ ] **步骤 5：_build_base_args 透传**

在 `_build_base_args` 中 `base.num_warmups = 0`（约 line 161）附近加入：

```python
    base.warmup_concurrency = our_args.warmup_concurrency
    base.warmup_output_len = our_args.warmup_output_len
```

- [ ] **步骤 6：is_first_run 用 warmup_concurrency 作预热请求数**

定位（约 line 614-620）：

```python
            if is_first_run:
                cfg.ready_check_timeout_sec = 600
                cfg.num_warmups = our_args.warmup_requests
                is_first_run = False
```

改为：

```python
            if is_first_run:
                cfg.ready_check_timeout_sec = 600
                # 固定并发预热：warmup_concurrency 设了则凑齐一波满并发，否则沿用 warmup_requests
                cfg.num_warmups = our_args.warmup_concurrency or our_args.warmup_requests
                is_first_run = False
```

- [ ] **步骤 7：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_warmup_config.py -v`
预期：2 passed

- [ ] **步骤 8：全量回归**

运行：`cd vllm_standalone_bench && python -m pytest tests/ -v`
预期：全部 PASS

- [ ] **步骤 9：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/tests/test_warmup_config.py
git commit -m "feat(bench): run_bench_multi 透传 warmup 固定并发参数

新增 --warmup-concurrency / --warmup-output-len CLI（默认 None 向后兼容）；
首次预热请求数随 warmup_concurrency 凑齐一波满并发。"
```

---

## 任务 3：auto_bench.py — 配置字段与命令构建

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`（BenchProfile 约 line 93-101；解析约 line 434-435；build_bench_run_command 约 line 624）
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

追加到 `tests/test_auto_bench.py`（沿用该文件现有构造 config/case 的辅助；若名称不同，改用实际辅助）：

```python
def test_load_config_parses_warmup_opts(tmp_path):
    data = minimal_config(tmp_path)
    data["bench_profiles"][0]["warmup_concurrency"] = 4
    data["bench_profiles"][0]["warmup_output_len"] = 128
    config = ab.load_config(write_config(tmp_path, data))
    bp = config.bench_profiles[0]
    assert bp.warmup_concurrency == 4
    assert bp.warmup_output_len == 128


def test_load_config_warmup_opts_default_none(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    bp = config.bench_profiles[0]
    assert bp.warmup_concurrency is None
    assert bp.warmup_output_len is None


def test_build_bench_command_includes_warmup_opts(tmp_path):
    data = minimal_config(tmp_path)
    data["bench_profiles"][0]["warmup_concurrency"] = 4
    data["bench_profiles"][0]["warmup_output_len"] = 128
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config)[0]
    cmd = ab.build_bench_run_command(config, case, tmp_path / "bench")
    assert value_after(cmd, "--warmup-concurrency") == "4"
    assert value_after(cmd, "--warmup-output-len") == "128"


def test_build_bench_command_omits_warmup_opts_when_none(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config)[0]
    cmd = ab.build_bench_run_command(config, case, tmp_path / "bench")
    assert "--warmup-concurrency" not in cmd
    assert "--warmup-output-len" not in cmd
```

> 复用 `tests/test_auto_bench.py` 顶部已有的 `write_config`、`minimal_config`、`value_after` 辅助。

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py -k warmup -v`
预期：FAIL（`BenchProfile` 无该属性 / 解析报未知字段）

- [ ] **步骤 3：BenchProfile 加字段**

在 `auto_bench.py` 的 `class BenchProfile` 中 `warmup_requests: int = 1` 之后加：

```python
    warmup_concurrency: int | None = None
    warmup_output_len: int | None = None
```

- [ ] **步骤 4：加 _optional_positive_int 解析辅助**

在 `auto_bench.py` 现有 `_non_negative_int` / `_positive_int` 辅助附近加：

```python
def _optional_positive_int(value, field: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{field} must be a positive int or null, got {value!r}")
    return value
```

- [ ] **步骤 5：解析配置字段**

在 BenchProfile 解析处（`warmup_requests=_non_negative_int(...)` 之后）加：

```python
            warmup_concurrency=_optional_positive_int(
                profile.get("warmup_concurrency"), "bench_profile.warmup_concurrency"),
            warmup_output_len=_optional_positive_int(
                profile.get("warmup_output_len"), "bench_profile.warmup_output_len"),
```

- [ ] **步骤 6：命令构建透传**

在 `build_bench_run_command` 中 `cmd.extend(["--warmup-requests", str(bench.warmup_requests)])` 之后加：

```python
    if bench.warmup_concurrency is not None:
        cmd.extend(["--warmup-concurrency", str(bench.warmup_concurrency)])
    if bench.warmup_output_len is not None:
        cmd.extend(["--warmup-output-len", str(bench.warmup_output_len)])
```

- [ ] **步骤 7：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py -k warmup -v`
预期：4 passed

- [ ] **步骤 8：全量回归**

运行：`cd vllm_standalone_bench && python -m pytest tests/ -v`
预期：全部 PASS

- [ ] **步骤 9：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): auto_bench BenchProfile 支持 warmup 固定并发字段

bench_profile 新增可选 warmup_concurrency / warmup_output_len，
非 None 时透传到 run_bench_multi 命令；缺省 None 向后兼容。"
```

---

## 任务 4：默认配置启用固定并发预热

**文件：**
- 修改：`vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.sglang_compare.json`
- 修改：`vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json`

- [ ] **步骤 1：编写失败的测试（配置校验）**

追加到 `tests/test_auto_bench.py`：

```python
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_sglang_compare_config_enables_fixed_warmup():
    cfg = json.loads((CONFIG_DIR / "auto_bench.qwen2_5_1_5b.sglang_compare.json").read_text())
    bp = cfg["bench_profiles"][0]
    assert bp["warmup_concurrency"] == 4
    assert bp["warmup_output_len"] == 128


def test_smoke_config_enables_fixed_warmup():
    cfg = json.loads((CONFIG_DIR / "auto_bench.qwen2_5_1_5b.smoke.json").read_text())
    bp = cfg["bench_profiles"][0]
    assert bp["warmup_concurrency"] == 4
    assert bp["warmup_output_len"] == 128
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py -k fixed_warmup -v`
预期：FAIL（`KeyError: 'warmup_concurrency'`）

- [ ] **步骤 3：更新 sglang_compare 配置**

在 `bench_profiles[0]` 中加入（与 `prefix_ratio`、`warmup_requests` 同级）：

```json
    "warmup_concurrency": 4,
    "warmup_output_len": 128,
```

- [ ] **步骤 4：更新 smoke 配置**

同样在 `bench_profiles[0]` 加入相同两字段。

- [ ] **步骤 5：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py -k fixed_warmup -v`
预期：2 passed

- [ ] **步骤 6：Commit**

```bash
git add vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.sglang_compare.json \
        vllm_standalone_bench/configs/auto_bench.qwen2_5_1_5b.smoke.json \
        vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): smoke/sglang_compare 配置默认启用固定并发预热

bench_profile 默认 warmup_concurrency=4、warmup_output_len=128。"
```

---

## 任务 5：README 说明 + dry-run 验证

**文件：**
- 修改：`vllm_standalone_bench/README.md`

- [ ] **步骤 1：README 补一段**

在 README「vLLM / SGLang 同台对比」章节后（或「离线双镜像自动化压测」内合适位置）加小节：

```markdown
### 固定并发预热（消除小并发档 TTFT 首批尖峰）

`bench_profile` 可选字段：
- `warmup_concurrency`：warmup 固定并发数（默认 `null`=跟随该档并发）
- `warmup_output_len`：warmup 请求输出长度（默认 `null`=跟随该档输出）

整个测试仅在首个配置前预热一次：用固定并发 × 首个配置输入 × 指定输出长度，
凑齐一波满并发，热起 vLLM 多请求调度路径。`smoke` / `sglang_compare` 配置
默认 `warmup_concurrency=4`、`warmup_output_len=128`。CLI 直跑可用
`--warmup-concurrency 4 --warmup-output-len 128`。
```

- [ ] **步骤 2：dry-run 验证命令拼装**

运行：
```bash
cd vllm_standalone_bench
python3 auto_bench.py run \
  --config configs/auto_bench.qwen2_5_1_5b.sglang_compare.json \
  --run-id warmup_dryrun --dry-run 2>&1 | grep -E "warmup|launch_server"
```
预期：bench-runner 命令中含 `--warmup-concurrency 4 --warmup-output-len 128`。

清理：`rm -rf results/warmup_dryrun`

- [ ] **步骤 3：Commit**

```bash
git add vllm_standalone_bench/README.md
git commit -m "docs(bench): README 补充固定并发预热说明"
```

---

## 收尾

- 全量回归：`cd vllm_standalone_bench && python -m pytest tests/ -v` 全绿。
- `git diff --check` 无空白错误。
- 按 AGENTS.md 用 `finishing-a-branch` 收尾：回到 `main` 合并 `feat/warmup-fixed-concurrency`。
