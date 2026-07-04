# 编排器日志记录与显示优化 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 `auto_bench.py` 编排器建立零新增依赖的结构化彩色日志门面 `bench_log.py`，让每条日志带时间戳+级别+case/阶段上下文，解决 controller.log 经常为空、进度与报错混杂、status/logs 显示不友好等问题。

**架构：** 新建 `bench_log.py` 作为唯一日志门面（标准库 `logging` + ANSI 转义），提供 `setup_logging()`（FileHandler 始终写 controller.log + ConsoleHandler 仅 isatty 时挂彩色）、`case_scope()`（用 `contextvars` 注入 `[case idx/total][phase]` / `[phase][label]` 前缀）、两个 Formatter。`auto_bench.py` 在 `run_controller` 开头接入门面，在 topology/legacy 双路径用 `case_scope` 包裹 serve（per-group）与 bench（per-case）阶段，把现有裸 `print` 按语义替换为 `logger.info/warning/error`，并美化 `status`/`logs` 子命令。

**技术栈：** Python 3.11（主机原生仅 python2，须用项目 miniconda py311_24.5.0）、标准库 `logging`/`contextvars`、pytest。

**规格：** `docs/superpowers/specs/2026-07-04-orchestrator-logging-design.md`

**相对 spec 的一处实现期微调（更 DRY）：** `case_scope` 去掉未使用的 `logger` 参数（上下文走 ContextVar，logger 不需要传入），签名变为 `case_scope(*, total, phase, idx=None, label=None)`。

---

## 文件结构

- **创建** `vllm_standalone_bench/bench_log.py`：日志门面。职责：颜色常量、`_CaseContext`（ContextVar）、`case_scope`、`ConsoleFormatter`/`FileFormatter`、`setup_logging`/`get_logger`。单一职责、可独立单测。
- **创建** `vllm_standalone_bench/tests/test_bench_log.py`：门面纯函数单测。
- **修改** `vllm_standalone_bench/auto_bench.py`：接入门面、`case_scope` 注入双路径、`print`→`logger` 替换、`status`/`logs` 美化。
- **修改** `vllm_standalone_bench/tests/test_auto_bench.py`：集成测试断言更新（controller.log 内容、status 输出格式）。

---

## 任务 1：bench_log.py — 颜色常量与 Formatter

**文件：**
- 创建：`vllm_standalone_bench/bench_log.py`
- 测试：`vllm_standalone_bench/tests/test_bench_log.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_bench_log.py`：

```python
"""bench_log 门面单测。"""
import logging
import sys

import bench_log


def _make_record(msg="hello", level=logging.INFO, name="auto_bench"):
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )


def test_file_formatter_has_timestamp_level_message():
    fmt = bench_log.FileFormatter()
    line = fmt.format(_make_record("starting serve"))
    # 纯文本、无 ANSI、含级别与消息；无 case 上下文时前缀为空
    assert "\033[" not in line
    assert "INFO" in line
    assert "starting serve" in line
    # 日期格式：YYYY-MM-DD HH:MM:SS
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line)


def test_console_formatter_color_when_enabled():
    fmt = bench_log.ConsoleFormatter(color=True)
    line = fmt.format(_make_record("boom", logging.ERROR))
    assert "\033[31m" in line  # ERROR 红
    assert "\033[0m" in line   # RESET


def test_console_formatter_no_color_when_disabled():
    fmt = bench_log.ConsoleFormatter(color=False)
    line = fmt.format(_make_record("boom", logging.ERROR))
    assert "\033[" not in line
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_bench_log.py -v`
预期：FAIL，`ModuleNotFoundError: No module named 'bench_log'`（或 ImportError）

- [ ] **步骤 3：编写最少实现代码**

创建 `vllm_standalone_bench/bench_log.py`：

```python
"""编排器统一日志门面：零第三方依赖的结构化彩色日志。

设计见 docs/superpowers/specs/2026-07-04-orchestrator-logging-design.md。
- FileHandler 始终写 controller.log（纯文本 + 完整时间戳，结构化全量）
- ConsoleHandler 仅 sys.stderr.isatty() 时挂（彩色）
- case_scope 注入 [case idx/total][phase] 或 [phase][label] 前缀（contextvars）
"""
from __future__ import annotations

import logging


class _Color:
    RESET = "\033[0m"
    GREY = "\033[90m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREEN = "\033[32m"


LEVEL_COLOR = {
    "DEBUG": _Color.GREY,
    "INFO": _Color.GREY,
    "WARNING": _Color.YELLOW,
    "ERROR": _Color.RED,
    "CRITICAL": _Color.RED,
}


def _current_prefix() -> str:
    """读取当前 case 上下文前缀；无上下文时返回空串。"""
    ctx = _CTX.get()
    if not ctx:
        return ""
    idx, total, phase, label = ctx
    if idx is not None:
        return f"[case {idx}/{total}][{phase}]"
    return f"[{phase}][{label}]"


class _BaseFormatter(logging.Formatter):
    """把当前 case 前缀注入 record.case_prefix 后再按 fmt 渲染。"""

    def format(self, record: logging.LogRecord) -> str:
        record.case_prefix = _current_prefix()
        return super().format(record)


class FileFormatter(_BaseFormatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-5s %(case_prefix)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class ConsoleFormatter(_BaseFormatter):
    def __init__(self, color: bool = True) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-5s %(case_prefix)s%(message)s",
            datefmt="%H:%M:%S",
        )
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not self.color:
            return msg
        c = LEVEL_COLOR.get(record.levelname, "")
        return f"{c}{msg}{_Color.RESET}" if c else msg
```

> 注：`_CTX`、`case_scope`、`setup_logging`、`get_logger` 在后续任务加入。本任务先把 `_current_prefix` 依赖的 `_CTX` 占位定义在文件里（见步骤 3 补充），否则 import 报错。

补充：在 `bench_log.py` 顶部 `import logging` 之后、`class _Color` 之前，加入：

```python
import contextvars

_CTX: contextvars.ContextVar[tuple | None] = contextvars.ContextVar(
    "bench_log_case_ctx", default=None,
)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_bench_log.py -v`
预期：3 个测试 PASS

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/bench_log.py vllm_standalone_bench/tests/test_bench_log.py
git commit -m "feat(bench_log): add color constants and File/Console formatters"
```

---

## 任务 2：bench_log.py — CaseContext 与 case_scope

**文件：**
- 修改：`vllm_standalone_bench/bench_log.py`
- 测试：`vllm_standalone_bench/tests/test_bench_log.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_bench_log.py` 追加：

```python
def test_case_scope_bench_prefix():
    rec = _make_record("bench input=4096")
    with bench_log.case_scope(total=8, phase="bench", idx=3):
        line = bench_log.FileFormatter().format(rec)
    assert "[case 3/8][bench]" in line


def test_case_scope_serve_prefix_no_idx():
    rec = _make_record("starting serve")
    with bench_log.case_scope(total=8, phase="serve", label="glm52_fp8"):
        line = bench_log.FileFormatter().format(rec)
    assert "[serve][glm52_fp8]" in line
    assert "[case" not in line


def test_case_scope_resets_after_block():
    rec = _make_record("outside")
    with bench_log.case_scope(total=8, phase="bench", idx=1):
        pass
    line = bench_log.FileFormatter().format(rec)
    assert "[case" not in line  # 退出 with 后前缀复位


def test_case_scope_propagates_across_function_call():
    """同线程内跨函数调用应继承当前前缀。"""
    captured = {}

    def inner():
        captured["line"] = bench_log.FileFormatter().format(_make_record("deep"))

    with bench_log.case_scope(total=4, phase="bench", idx=2):
        inner()
    assert "[case 2/4][bench]" in captured["line"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_bench_log.py -k case_scope -v`
预期：FAIL，`AttributeError: module 'bench_log' has no attribute 'case_scope'`

- [ ] **步骤 3：编写最少实现代码**

在 `bench_log.py` 的 `_current_prefix` 函数之后加入：

```python
import contextlib


@contextlib.contextmanager
def case_scope(*, total: int, phase: str, idx: int | None = None,
               label: str | None = None):
    """with 块内所有日志自动带前缀。

    idx 非 None（bench per-case）→ ``[case idx/total][phase]``；
    idx 为 None（serve per-group）→ ``[phase][label]``。
    基于 contextvars，同线程内跨函数自动传播；退出后复位。
    """
    token = _CTX.set((idx, total, phase, label))
    try:
        yield
    finally:
        _CTX.reset(token)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_bench_log.py -k case_scope -v`
预期：4 个测试 PASS

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/bench_log.py vllm_standalone_bench/tests/test_bench_log.py
git commit -m "feat(bench_log): add case_scope context prefix via contextvars"
```

---

## 任务 3：bench_log.py — setup_logging 双 handler 分流

**文件：**
- 修改：`vllm_standalone_bench/bench_log.py`
- 测试：`vllm_standalone_bench/tests/test_bench_log.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_bench_log.py` 追加（注意：`setup_logging` 操作 root logger，每个测试后须清理 handler，用 fixture）：

```python
import pytest


@pytest.fixture
def clean_root():
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    root.handlers.clear()
    yield root
    root.handlers.clear()
    for h in saved:
        root.addHandler(h)
    root.setLevel(saved_level)


def test_setup_logging_always_attaches_file_handler(tmp_path, clean_root):
    bench_log.setup_logging(tmp_path, color=False)
    kinds = {type(h).__name__ for h in clean_root.handlers}
    assert "FileHandler" in kinds
    assert "StreamHandler" in kinds  # color=False 时 StreamHandler 仍挂但无色


def test_setup_logging_color_false_uses_plain_stream(tmp_path, clean_root):
    bench_log.setup_logging(tmp_path, color=False)
    sh = [h for h in clean_root.handlers if type(h).__name__ == "StreamHandler"]
    assert sh and not sh[0].formatter.color


def test_setup_logging_level_applied(tmp_path, clean_root):
    bench_log.setup_logging(tmp_path, level="WARNING", color=False)
    assert clean_root.level == logging.WARNING


def test_setup_logging_writes_to_controller_log(tmp_path, clean_root):
    bench_log.setup_logging(tmp_path, color=False)
    logging.getLogger("auto_bench").warning("boom-city")
    log_text = (tmp_path / "controller.log").read_text(encoding="utf-8")
    assert "boom-city" in log_text
    assert "WARNING" in log_text
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_bench_log.py -k setup -v`
预期：FAIL，`AttributeError: module 'bench_log' has no attribute 'setup_logging'`

- [ ] **步骤 3：编写最少实现代码**

在 `bench_log.py` 末尾加入：

```python
import sys
from pathlib import Path


def setup_logging(run_dir, level: str = "INFO", *, color: bool | None = None,
                  log_file=None) -> None:
    """初始化 root logger：FileHandler（始终）+ ConsoleHandler（仅彩色时）。

    color=None 时按 sys.stderr.isatty() 自动决定；isatty 检测失败回退 False。
    detach 模式 stderr 被重定向（非 tty）→ color=False → 不挂彩色
    StreamHandler 写 stderr，避免与 FileHandler 双写 controller.log。
    """
    logging.raiseExceptions = False
    if color is None:
        try:
            color = sys.stderr.isatty()
        except Exception:
            color = False
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    target = Path(log_file) if log_file else (Path(run_dir) / "controller.log")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(target, mode="a", encoding="utf-8")
        fh.setFormatter(FileFormatter())
        fh.setLevel(level)
        root.addHandler(fh)
    except OSError:
        fh = None
    if color:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(ConsoleFormatter(color=True))
        ch.setLevel(level)
        root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_bench_log.py -v`
预期：全部测试 PASS

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/bench_log.py vllm_standalone_bench/tests/test_bench_log.py
git commit -m "feat(bench_log): add setup_logging dual-handler routing"
```

---

## 任务 4：auto_bench.py 接入门面（run_controller 调 setup_logging）

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py:32`（logger 定义）
- 修改：`vllm_standalone_bench/auto_bench.py:3556` 附近（run_controller，setup_logging 注入点）
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_auto_bench.py` 追加（复用现有 `minimal_config`/`write_config` helper）：

```python
def test_run_controller_dry_run_logs_to_controller_log(tmp_path):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_id = "log_dry_run_001"
    rc = ab.run_controller(config, run_id, dry_run=True)
    assert rc == 0
    log_path = config.run.results_dir / run_id / "controller.log"
    assert log_path.exists(), "controller.log should be created"
    text = log_path.read_text(encoding="utf-8")
    assert "INFO" in text or "WARNING" in text  # 至少有一条结构化日志
```

> 若 `load_config` 名称不同，执行者用文件里实际的配置加载函数（grep `def load_config` 确认）。dry_run 路径也需挂 FileHandler（见步骤 3 对 `_run_controller_dry_run` 的处理）。

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py::test_run_controller_dry_run_logs_to_controller_log -v`
预期：FAIL——`controller.log` 不存在或不含结构化日志（现状无 logging 配置）

- [ ] **步骤 3：编写实现代码**

3a. 在 `auto_bench.py` 顶部 import 区（约第 7 行 `import logging` 附近）加入：

```python
import bench_log
```

3b. 把第 32 行 `logger = logging.getLogger("auto_bench")` 改为：

```python
logger = bench_log.get_logger("auto_bench")
```

3c. 在 `run_controller`（3545）函数体里，`all_cases = expand_cases(config, run_id=run_id)`（3556）之前加入 setup_logging；dry_run 分支也注入。把：

```python
    active_runner: Runner = runner or DockerRunner()
    if dry_run:
        return _run_controller_dry_run(config, run_id)

    all_cases = expand_cases(config, run_id=run_id)
```

改为：

```python
    active_runner: Runner = runner or DockerRunner()
    run_dir_preview = config.run.results_dir / run_id
    bench_log.setup_logging(run_dir_preview)
    logger.info("controller started: run_id=%s dry_run=%s", run_id, dry_run)
    if dry_run:
        return _run_controller_dry_run(config, run_id)

    all_cases = expand_cases(config, run_id=run_id)
```

> dry_run 提前 return 前已 `setup_logging` 且打了 `controller started`，保证 controller.log 至少有一条结构化日志（任务 4 测试断言据此通过）。`_run_controller_dry_run` 内的 `print_cmd` 走 stdout（命令预览，保留 print）；serve/bench 阶段的进度 logger 在任务 5 补。`setup_logging` 内部对 run_dir 做 `mkdir(parents=True)`，确保目录存在。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py::test_run_controller_dry_run_logs_to_controller_log tests/test_auto_bench.py -v`
预期：新测试 PASS，且现有 `test_auto_bench.py` 不回归

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(auto_bench): wire bench_log setup_logging into run_controller"
```

---

## 任务 5：auto_bench.py — case_scope 注入双路径 + case_index

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`（新增 `_case_index_map` helper，置于 `_case_key`/`expand_cases` 附近）
- 修改：`vllm_standalone_bench/auto_bench.py:3505-3540`（**dry_run 路径** `_run_controller_dry_run`，最易测，注入点之首）
- 修改：`vllm_standalone_bench/auto_bench.py:3556-3560`（`run_controller` 建 case_index/total）
- 修改：`vllm_standalone_bench/auto_bench.py:3338-3410`（topology 实际路径 `run_topology_group`）
- 修改：`vllm_standalone_bench/auto_bench.py:3603-3695`（legacy 实际路径）
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_auto_bench.py` 追加：

```python
def test_controller_log_contains_case_prefix(tmp_path):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_id = "log_prefix_001"
    rc = ab.run_controller(config, run_id, dry_run=True)
    assert rc == 0
    text = (config.run.results_dir / run_id / "controller.log").read_text(encoding="utf-8")
    # dry-run 也会走 serve/bench 阶段日志，应出现 case 前缀或 serve 前缀
    assert ("[case " in text) or ("[serve]" in text), text
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py::test_controller_log_contains_case_prefix -v`
预期：FAIL——controller.log 不含 `[case`/`[serve]` 前缀

- [ ] **步骤 3：编写实现代码**

3a. 新增 case_index helper（置于 `_case_key` 之后或 `expand_cases` 之后）：

```python
def _case_index_map(cases):
    """全量 cases 的 1-based 序号映射，key=_case_key(case)。

    resume（pending 子集）或 skip 下，剩余 case 的 idx 仍映射回全量口径，不会错位。
    """
    return {_case_key(c): i + 1 for i, c in enumerate(cases)}
```

3b. **dry_run 路径** `_run_controller_dry_run`（3505，最易测，先改）——它内部 `cases = expand_cases(...)` 后有 serve per-group + bench per-case 双层循环（用 `print_cmd` 预览命令）。在 `cases = expand_cases(...)` 之后建映射，并把两个循环用 case_scope 包裹（`print_cmd` 保留作命令预览，logger 是额外进度）：

```python
    cases = expand_cases(config, run_id=run_id)
    case_index = _case_index_map(cases)
    total = len(cases)
    run_dir = config.run.results_dir / run_id
    ...
    for group_cases in _group_cases_by_serve(cases).values():
        serve_case = group_cases[0]
        with bench_log.case_scope(total=total, phase="serve",
                                  label=serve_case.serving_name):
            logger.info("serve commands for %s", serve_case.serving_name)
            ...   # 原有 build_commands / print_cmd 逻辑保持不变
        for case in group_cases:
            with bench_log.case_scope(total=total, phase="bench",
                                      idx=case_index[_case_key(case)]):
                logger.info("bench command")
                ...   # 原有 print_cmd(build_bench_run_command) 保持不变
```

> `serving_name` 是 `BenchmarkCase` 已有 property（返回 serve_profile.name 或 topology_profile.name）。

3c. **`run_controller`（3556）建映射** + **topology 实际路径** `run_topology_group`（3338）：
- `run_controller` 里 `all_cases = expand_cases(...)` 之后加 `case_index = _case_index_map(all_cases); total = len(all_cases)`。
- 给 `run_topology_group` 加 `case_index` 与 `total` 参数，serve（`serve_case = group_cases[0]` 处）与 bench（`for case in group_cases:` 3403）循环按 3b 同款 case_scope 包裹。更新所有调用点：`grep -n "run_topology_group(" auto_bench.py`（约 3461/3471），补传 `case_index=case_index, total=total`。

3d. **legacy 实际路径**（run_controller 内 3603-3695）：用 `run_controller` 已建的 `case_index`/`total`。serve（`serve_cmd = build_serve_run_command(...)` 3636 前）注入 `case_scope(phase="serve", label=serve_case.serving_name)`；bench（`for case in group_cases:` 3675）注入 `case_scope(phase="bench", idx=case_index[_case_key(case)])`。

> 三处循环（dry_run / topology / legacy）均保留原有 `print_cmd` / `_run_bench_case` 等业务逻辑不变，仅在外层套 `with case_scope(...)` 并补 `logger.info`。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py::test_controller_log_contains_case_prefix tests/test_auto_bench.py -v`
预期：新测试 PASS，现有测试不回归

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(auto_bench): inject case_scope into topology/legacy paths"
```

---

## 任务 6：auto_bench.py — print→logger 替换 + 补充进度节点

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`（全文件裸 print 按语义替换；主循环补 logger.info）
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_auto_bench.py` 追加（dry_run 路径，断言任务 6 新加的结束节点）：

```python
def test_controller_log_has_run_finished_node(tmp_path):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_id = "log_progress_001"
    rc = ab.run_controller(config, run_id, dry_run=True)
    assert rc == 0
    text = (config.run.results_dir / run_id / "controller.log").read_text(encoding="utf-8")
    assert "run finished" in text
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py::test_controller_log_has_run_finished_node -v`
预期：FAIL——controller.log 不含 "run finished"（任务 5 只加了 serve/bench 阶段 logger，未加结束节点）

- [ ] **步骤 3：编写实现代码**

3a. **结束/启动节点**：
- `_run_controller_dry_run`（3505）的 `return 0`（try 块内）之前加：

```python
    logger.info("run finished: run_id=%s cases=%d (dry-run)", run_id, total)
```

- `run_controller` 实际路径主循环开始前加 `logger.info("run started: %s cases=%d", run_id, total)`；主循环结束（return 前）加 `logger.info("run finished: %s", _format_counts(...))`。

3b. **裸 print 分类替换**（保留 CLI 展示类 print，替换运行期 print）：
- 运行进度类 print（如 `print(f"run_id: {run_id}")` 在 detach 启动处 3970-3972 是给前台调用方的，**保留**）
- `file=sys.stderr` 的 warning（约 41 处）→ `logger.warning(...)`：典型如 1625/1647/1674（network cleanup warning）、2222/2241/2287/2302（topology cleanup warning）、2579/2608/2611/2642/2647（run lock warning）等。逐个把 `print("warning: ...", file=sys.stderr)` 改为 `logger.warning("...")`。
- failed/error/not found → `logger.error(...)`
- **保留为 print** 的：`status`/`logs`/`stop`/`resume` 子命令的 CLI 输出（`print_status`/`follow_file`/`print_log` 等内的 `print`，约 4003-4305），这些是命令展示，不该进 controller.log。

> 执行指引：用 `grep -n 'file=sys.stderr' auto_bench.py` 与 `grep -n 'print("warning' auto_bench.py` 列出全部替换点，逐个改为 `logger.warning`/`logger.error`，保持消息文本不变。每改一批跑一次 `python -m pytest tests/test_auto_bench.py -q` 确认不回归。

3c. 在 serve/bench case_scope with 块内补充节点（任务 5 已埋点，这里补全消息）：
- serve：`logger.info("starting serve tp=%s", tp)`（若有）、`logger.info("endpoint ready")`
- bench：`logger.info("bench input=%s out=%s conc=%s", ...)`、case 结束 `logger.info("case %s", status)` 或失败 `logger.error("case failed: %s", error)`

- [ ] **步骤 4：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py -v`
预期：全部 PASS（含新进度节点测试）

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(auto_bench): replace prints with logger calls and add progress nodes"
```

---

## 任务 7：auto_bench.py — status/logs 展示层美化 + logs --level

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py:3981-3992`（`_format_current`/`_format_counts`）
- 修改：`vllm_standalone_bench/auto_bench.py:4003`（`print_status`）
- 修改：`vllm_standalone_bench/auto_bench.py:4053-4083`（`follow_file`/`print_log`）
- 修改：`vllm_standalone_bench/auto_bench.py`（`logs` 子命令 argparse，加 `--level`）
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_auto_bench.py` 追加：

```python
def test_print_status_renders_colored_summary(tmp_path, capsys, monkeypatch):
    run_dir = tmp_path / "rd"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(json.dumps({
        "run_id": "rd", "status": "running",
        "current": {"model": "m", "serve_profile": "sp", "bench_profile": "bp"},
        "counts": {"passed": 2, "failed": 1, "running": 1, "total": 4},
    }), encoding="utf-8")
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    rc = ab.print_status(run_dir)
    out = capsys.readouterr().out
    assert rc == 0
    assert "passed=2" in out
    assert "failed=1" in out


def test_follow_file_level_filter(tmp_path, capsys, monkeypatch):
    log = tmp_path / "controller.log"
    log.write_text(
        "2026-07-04 12:00:00 INFO  [case 1/2][bench] ok\n"
        "2026-07-04 12:00:01 ERROR [case 1/2][bench] boom\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ab.time, "sleep", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    rc = ab.follow_file(log, level="ERROR")
    out = capsys.readouterr().out
    assert "boom" in out
    assert "ok" not in out  # INFO 行被过滤
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py -k "print_status or follow_file" -v`
预期：FAIL——`print_status` 输出可能已含 counts（现有 `_format_counts` 已有），但 `follow_file` 无 `level` 参数（`TypeError`）

- [ ] **步骤 3：编写实现代码**

3a. 把 `follow_file(path: Path)` 改为 `follow_file(path: Path, level: str | None = None)`，在 `sys.stdout.write(line)` 前按级别过滤：

```python
def follow_file(path: Path, level: str | None = None) -> int:
    if not path.exists():
        print(f"log file not found: {path}", file=sys.stderr)
        return 1
    level_upper = level.upper() if level else None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            while True:
                line = handle.readline()
                if line:
                    if level_upper and not _line_matches_level(line, level_upper):
                        continue
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(1)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"failed to read log file: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
```

加入级别匹配 helper（仅匹配带级别前缀的单行）：

```python
_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def _line_matches_level(line: str, level_upper: str) -> bool:
    """仅过滤带级别字段的行；无级别前缀的行（第三方/docker 回显）不过滤命中。

    语义：行级别 >= 指定级别则保留（WARNING 会包含 ERROR）。
    """
    threshold = _LEVEL_ORDER.get(level_upper, 0)
    for name, val in _LEVEL_ORDER.items():
        if val >= threshold and f" {name} " in line:
            return True
    return False
```

> 注：`_line_matches_level` 用 `f" {name} "` 匹配级别字段（formatter 输出 ` INFO  ` 形式）。多行 traceback 续行、无级别前缀的第三方回显不命中——按 spec，`--level` 仅过滤带级别单行。

3b. 给 `logs` 子命令的 argparse 加 `--level`（grep `add_parser("logs"` 定位），把值透传给 `follow_file`/`print_log`。`print_log(path)` 同样加 `level` 参数过滤（一次性读全部后按行过滤输出）。

3c. `print_status`（4003）保持现有 `print` 输出（这是 CLI 展示，不进 controller.log），但可选用 `bench_log` 颜色常量给 passed/failed/running 上色（仅 isatty 时）。最小改动：保持文本格式，可选加色。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_auto_bench.py -v`
预期：全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(auto_bench): add logs --level filter and status coloring"
```

---

## 任务 8：全量回归 + 收尾

**文件：**
- 验证：全测试套件 + shell 脚本

- [ ] **步骤 1：运行全量测试**

```bash
cd vllm_standalone_bench && python -m pytest -q
```
预期：全绿。若有 `test_shell_scripts.py` 失败（`run_auto_bench.sh` 引用日志路径），同步修复。

- [ ] **步骤 2：smoke 验证（若主机有 docker + 镜像）**

```bash
cd vllm_standalone_bench && python auto_bench.py run \
  --config configs/auto_bench.qwen2_5_1_5b.smoke.json --dry-run
# 检查 results/<run_id>/controller.log 含结构化日志与 [case]/[serve] 前缀
```
若主机无 docker/镜像，跳过 smoke 并注明（dry-run 已在测试覆盖）。

- [ ] **步骤 3：确认 git diff 干净 + worktree 状态**

```bash
git status
git diff --check
git log --oneline main..HEAD
```

- [ ] **步骤 4：Commit（如有收尾改动）**

```bash
git add -A
git commit -m "test: full regression for orchestrator logging"
```

- [ ] **步骤 5：移交 finishing-a-development-branch 收尾合并**

按 AGENTS.md：回到本地 `main`，`git merge --no-ff feat/orchestrator-logging`，清理 worktree。

---

## 自检

**1. 规格覆盖度：**

| spec 章节 | 对应任务 |
|---|---|
| `bench_log.py` 公开接口（setup_logging/get_logger/case_scope） | 任务 1-3 |
| 双 handler 分流（FileHandler 始终 + ConsoleHandler 仅 isatty） | 任务 3 |
| CaseContext 注入点（topology/legacy 双路径，serve per-group→bench per-case） | 任务 5 |
| idx/total 语义（case_index 全局序号、total=manifest.total） | 任务 5 |
| `print`→`logger` 分类映射 | 任务 6 |
| controller 主循环补充日志节点 | 任务 6 |
| `status`/`logs` 展示层 + `logs --level` | 任务 7 |
| 错误处理（raiseExceptions=False、isatty 回退、FileHandler 失败降级、flush、ContextVar 澄清） | 任务 3（setup_logging 内）+ 任务 6 |
| 测试策略（bench_log 单测 + 集成 + 回归） | 任务 1-3 + 4-7 + 8 |

无遗漏。

**2. 占位符扫描：** 无 TODO/待定；任务 5/6 的 print 替换用 grep 列举 + 逐个替换指引（非占位符，是确定的批量操作）。`load_config`/`run_topology_group` 调用点已给出 grep 定位命令。

**3. 类型一致性：**
- `case_scope(*, total, phase, idx=None, label=None)` —— 任务 2 定义、任务 5 调用，签名一致。
- `setup_logging(run_dir, level, *, color, log_file)` —— 任务 3 定义、任务 4 调用 `setup_logging(run_dir_preview)`，一致。
- `follow_file(path, level=None)` / `print_log(path, level=None)` —— 任务 7 定义与调用一致。
- `_case_key`/`serving_name`/`_format_counts` 均为代码库既有符号，已核实（2960/193/3992）。
- `case_index` 在任务 5 定义后，topology（需传入 `run_topology_group`）与 legacy 路径均引用同一变量。

一致。
