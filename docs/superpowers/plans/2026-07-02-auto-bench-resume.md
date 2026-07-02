# Auto Bench Resume 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 `vllm_standalone_bench` 自动压测控制器新增显式 `resume` 能力：同一 `RUN_ID` 下跳过已 `passed` 的 case，只补跑未成功 case。

**架构：** 在 `auto_bench.py` 中新增一组小的恢复规划函数，负责读取 `state.json`、`manifest.json`、`config.resolved.json` 并选择 pending cases。改造 `run_controller`，让它可接收初始 manifest 和待运行 case 列表；普通 run 继续使用完整 case 列表，resume 使用旧 passed rows + pending cases。CLI 增加 `resume`，后台恢复复用现有 lock、controller metadata、signal 和 Docker ownership 保护。

**技术栈：** Python 3 标准库、Docker CLI 编排、pytest、Bash wrapper。

**规格：** `docs/superpowers/specs/2026-07-02-auto-bench-resume-design.md`

---

## 文件结构

| 文件 | 职责 | 动作 |
|---|---|---|
| `vllm_standalone_bench/auto_bench.py` | resume 状态读取、case 选择、controller 参数化、CLI、后台子进程命令 | 修改 |
| `vllm_standalone_bench/run_auto_bench.sh` | shell wrapper 增加 `resume` 子命令并透传 `DETACH` | 修改 |
| `vllm_standalone_bench/README.md` | 记录 stop/resume 语义与使用方式 | 修改 |
| `vllm_standalone_bench/tests/test_auto_bench.py` | 恢复规划、controller resume、CLI/后台行为单测 | 修改 |
| `vllm_standalone_bench/tests/test_shell_scripts.py` | wrapper 支持 `resume` 的单测 | 修改 |

所有路径相对仓库根。worktree 根为 `.worktrees/auto-bench-resume/`。

---

## 任务 1：恢复规划纯函数

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

在 `test_manifest_status_matrix` 后追加这些 helper 和测试：

```python
def two_bench_config(tmp_path):
    data = minimal_config(tmp_path)
    second_profile = dict(data["bench_profiles"][0])
    second_profile["name"] = "smoke2"
    data["bench_profiles"].append(second_profile)
    return data


def write_resolved_config_for_resume(tmp_path, data=None):
    data = data or two_bench_config(tmp_path)
    config = ab.load_config(write_config(tmp_path, data))
    run_dir = tmp_path / "results" / "run123"
    ab.write_json_atomic(run_dir / "config.resolved.json", ab.config_to_dict(config))
    return config, run_dir


def test_plan_resume_cases_keeps_passed_and_selects_pending(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    first_layout = ab.build_layout(config, "run123", cases[0])
    old_manifest = {
        "run_id": "run123",
        "status": "interrupted",
        "cases": [
            {
                "model": cases[0].model.name,
                "serve_profile": cases[0].serve_profile.name,
                "bench_profile": cases[0].bench_profile.name,
                "status": "passed",
                "csv": str((first_layout.bench_dir / "result.csv").relative_to(first_layout.run_dir)),
                "xlsx": str((first_layout.bench_dir / "result.xlsx").relative_to(first_layout.run_dir)),
            },
            {
                "model": cases[1].model.name,
                "serve_profile": cases[1].serve_profile.name,
                "bench_profile": cases[1].bench_profile.name,
                "status": "interrupted",
                "csv": "old.csv",
                "xlsx": "old.xlsx",
            },
        ],
    }

    initial_manifest, pending, unknown = ab.plan_resume_cases(
        run_id="run123",
        cases=cases,
        manifest_data=old_manifest,
    )

    assert unknown == []
    assert [row["bench_profile"] for row in initial_manifest.cases] == ["smoke"]
    assert initial_manifest.cases[0]["status"] == "passed"
    assert [case.bench_profile.name for case in pending] == ["smoke2"]
    assert initial_manifest.total == 2


def test_plan_resume_cases_reruns_failed_skipped_and_missing(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    manifest_data = {
        "run_id": "run123",
        "status": "completed_with_failures",
        "cases": [
            {
                "model": cases[0].model.name,
                "serve_profile": cases[0].serve_profile.name,
                "bench_profile": cases[0].bench_profile.name,
                "status": "failed",
                "csv": "old.csv",
                "xlsx": "old.xlsx",
            }
        ],
    }

    initial_manifest, pending, unknown = ab.plan_resume_cases(
        run_id="run123",
        cases=cases,
        manifest_data=manifest_data,
    )

    assert unknown == []
    assert initial_manifest.cases == []
    assert [case.bench_profile.name for case in pending] == ["smoke", "smoke2"]


def test_plan_resume_cases_reports_manifest_rows_not_in_config(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    manifest_data = {
        "run_id": "run123",
        "status": "interrupted",
        "cases": [
            {
                "model": "old_model",
                "serve_profile": "old_serve",
                "bench_profile": "old_bench",
                "status": "passed",
                "csv": "old.csv",
                "xlsx": "old.xlsx",
            }
        ],
    }

    initial_manifest, pending, unknown = ab.plan_resume_cases(
        run_id="run123",
        cases=cases,
        manifest_data=manifest_data,
    )

    assert initial_manifest.cases == []
    assert [case.bench_profile.name for case in pending] == ["smoke", "smoke2"]
    assert unknown == [("old_model", "old_serve", "old_bench")]
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest \
  vllm_standalone_bench/tests/test_auto_bench.py::test_plan_resume_cases_keeps_passed_and_selects_pending \
  vllm_standalone_bench/tests/test_auto_bench.py::test_plan_resume_cases_reruns_failed_skipped_and_missing \
  vllm_standalone_bench/tests/test_auto_bench.py::test_plan_resume_cases_reports_manifest_rows_not_in_config \
  -q
```

预期：失败，报错包含 `AttributeError: module 'auto_bench' has no attribute 'plan_resume_cases'`。

- [ ] **步骤 3：实现恢复规划函数**

在 `auto_bench.py` 顶部 import 改为：

```python
from dataclasses import dataclass, field, replace
```

在 `TERMINAL_RUN_STATUSES` 下方新增：

```python
RESUMABLE_RUN_STATUSES = frozenset({
    "interrupted",
    "failed",
    "completed_with_failures",
})
```

在 `_manifest_case_keys` 下方新增：

```python
def _manifest_row_key(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    model = row.get("model")
    serve_profile = row.get("serve_profile")
    bench_profile = row.get("bench_profile")
    if not all(isinstance(value, str) for value in (model, serve_profile, bench_profile)):
        return None
    return (model, serve_profile, bench_profile)


def _copy_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(key, str) and isinstance(value, (str, int, float, bool, type(None))):
            copied[key] = value
    return copied


def plan_resume_cases(
    *,
    run_id: str,
    cases: tuple[BenchmarkCase, ...],
    manifest_data: Mapping[str, Any],
) -> tuple[Manifest, tuple[BenchmarkCase, ...], list[tuple[str, str, str]]]:
    rows = manifest_data.get("cases")
    if not isinstance(rows, list):
        raise ConfigError("manifest cases must be a list")
    if manifest_data.get("run_id") not in (None, run_id):
        raise ConfigError(
            f"manifest run_id mismatch: expected {run_id}, got {manifest_data.get('run_id')}"
        )

    full_keys = {_case_key(case) for case in cases}
    passed_rows: list[dict[str, Any]] = []
    passed_keys: set[tuple[str, str, str]] = set()
    unknown_keys: list[tuple[str, str, str]] = []

    for row in rows:
        if not isinstance(row, dict):
            raise ConfigError("manifest cases must contain objects")
        key = _manifest_row_key(row)
        if key is None:
            raise ConfigError("manifest case row is missing model/serve_profile/bench_profile")
        if key not in full_keys:
            unknown_keys.append(key)
            continue
        if row.get("status") == "passed":
            passed_keys.add(key)
            passed_rows.append(_copy_manifest_row(row))

    initial_manifest = Manifest(run_id=run_id, total=len(cases), cases=passed_rows)
    pending = tuple(case for case in cases if _case_key(case) not in passed_keys)
    return initial_manifest, pending, unknown_keys
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest \
  vllm_standalone_bench/tests/test_auto_bench.py::test_plan_resume_cases_keeps_passed_and_selects_pending \
  vllm_standalone_bench/tests/test_auto_bench.py::test_plan_resume_cases_reruns_failed_skipped_and_missing \
  vllm_standalone_bench/tests/test_auto_bench.py::test_plan_resume_cases_reports_manifest_rows_not_in_config \
  -q
```

预期：`3 passed`。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): plan resumable auto bench cases"
```

---

## 任务 2：读取恢复上下文并校验状态

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

在任务 1 的恢复规划测试后追加：

```python
def write_resume_state(run_dir, status="interrupted"):
    ab.write_state(run_dir, {
        "run_id": run_dir.name,
        "status": status,
        "current": None,
        "counts": {"passed": 1, "failed": 0, "skipped": 0, "running": 0, "total": 2},
    })


def write_resume_manifest(run_dir, cases, config, statuses):
    rows = []
    for case, status in zip(cases, statuses):
        layout = ab.build_layout(config, run_dir.name, case)
        rows.append({
            "model": case.model.name,
            "serve_profile": case.serve_profile.name,
            "bench_profile": case.bench_profile.name,
            "status": status,
            "csv": str((layout.bench_dir / "result.csv").relative_to(layout.run_dir)),
            "xlsx": str((layout.bench_dir / "result.xlsx").relative_to(layout.run_dir)),
        })
    ab.write_json_atomic(run_dir / "manifest.json", {
        "run_id": run_dir.name,
        "status": "interrupted",
        "cases": rows,
    })


def test_load_resume_context_uses_cli_results_dir_over_resolved_config(tmp_path):
    original_data = two_bench_config(tmp_path)
    config, original_run_dir = write_resolved_config_for_resume(tmp_path, original_data)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(original_run_dir)
    write_resume_manifest(original_run_dir, cases[:1], config, ["passed"])

    other_results = tmp_path / "other-results"
    moved_run_dir = other_results / "run123"
    moved_run_dir.parent.mkdir()
    original_run_dir.rename(moved_run_dir)

    context = ab.load_resume_context(other_results, "run123")

    assert context.config.run.results_dir == other_results
    assert [case.bench_profile.name for case in context.pending_cases] == ["smoke2"]
    assert [row["bench_profile"] for row in context.initial_manifest.cases] == ["smoke"]


def test_load_resume_context_rejects_active_state(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir, status="running")
    write_resume_manifest(run_dir, cases[:1], config, ["passed"])

    with pytest.raises(ab.ConfigError, match="active|running"):
        ab.load_resume_context(tmp_path / "results", "run123")


def test_load_resume_context_completed_with_no_pending_is_empty(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir, status="completed")
    write_resume_manifest(run_dir, cases, config, ["passed", "passed"])

    context = ab.load_resume_context(tmp_path / "results", "run123")

    assert context.pending_cases == ()
    assert [row["status"] for row in context.initial_manifest.cases] == ["passed", "passed"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_resume_context_uses_cli_results_dir_over_resolved_config \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_resume_context_rejects_active_state \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_resume_context_completed_with_no_pending_is_empty \
  -q
```

预期：失败，报错包含 `AttributeError: module 'auto_bench' has no attribute 'load_resume_context'`。

- [ ] **步骤 3：实现 ResumeContext 和读取函数**

在 `RunLock` dataclass 后新增：

```python
@dataclass(frozen=True)
class ResumeContext:
    config: AutoBenchConfig
    run_id: str
    run_dir: Path
    initial_manifest: Manifest
    pending_cases: tuple[BenchmarkCase, ...]
    unknown_manifest_cases: tuple[tuple[str, str, str], ...]
```

在 `_read_run_state` 后新增 JSON 读取 helper：

```python
def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{label} invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{label} must be a JSON object: {path}")
    return payload
```

在 `reject_active_run` 后新增：

```python
def _config_with_results_dir(config: AutoBenchConfig, results_dir: Path) -> AutoBenchConfig:
    return replace(config, run=replace(config.run, results_dir=results_dir))


def load_resume_context(results_dir: Path, run_id: str) -> ResumeContext:
    _safe_name(run_id, "run_id")
    resolved_results_dir = Path(results_dir)
    run_dir = resolved_results_dir / run_id
    state = _read_json_object(run_dir / "state.json", "state")
    status = state.get("status")
    if state.get("run_id") not in (None, run_id):
        raise ConfigError(f"state run_id mismatch: expected {run_id}, got {state.get('run_id')}")
    if status in ("starting", "running"):
        raise ConfigError(f"run is active and cannot be resumed: {status}")

    config = load_config(run_dir / "config.resolved.json")
    config = _config_with_results_dir(config, resolved_results_dir)
    cases = expand_cases(config, run_id=run_id)
    manifest_data = _read_json_object(run_dir / "manifest.json", "manifest")
    initial_manifest, pending, unknown = plan_resume_cases(
        run_id=run_id,
        cases=cases,
        manifest_data=manifest_data,
    )
    if status not in RESUMABLE_RUN_STATUSES and pending:
        raise ConfigError(f"run status cannot be resumed: {status}")
    return ResumeContext(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        initial_manifest=initial_manifest,
        pending_cases=pending,
        unknown_manifest_cases=tuple(unknown),
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
python3 -m pytest \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_resume_context_uses_cli_results_dir_over_resolved_config \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_resume_context_rejects_active_state \
  vllm_standalone_bench/tests/test_auto_bench.py::test_load_resume_context_completed_with_no_pending_is_empty \
  -q
```

预期：`3 passed`。

- [ ] **步骤 5：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): load auto bench resume context"
```

---

## 任务 3：让 controller 可执行 pending case

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

在 controller 相关测试后追加：

```python
def test_run_controller_with_initial_manifest_skips_passed_case(tmp_path, monkeypatch):
    data = two_bench_config(tmp_path)
    config = ab.load_config(write_config(tmp_path, data))
    all_cases = ab.expand_cases(config, run_id="run123")
    first_layout = ab.build_layout(config, "run123", all_cases[0])
    initial = ab.Manifest(run_id="run123", total=2)
    initial.record(all_cases[0], first_layout, "passed")
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True, raising=False)

    result = ab.run_controller(
        config,
        run_id="run123",
        runner=runner,
        initial_manifest=initial,
        cases_to_run=(all_cases[1],),
    )

    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    bench_commands = bench_run_commands(runner.commands)
    assert result == 0
    assert [row["bench_profile"] for row in manifest["cases"]] == ["smoke", "smoke2"]
    assert [row["status"] for row in manifest["cases"]] == ["passed", "passed"]
    assert len(bench_commands) == 1
    assert "smoke2" in " ".join(bench_commands[0])
    assert "smoke-run123" not in " ".join(bench_commands[0])


def test_run_controller_pending_empty_writes_finished_state_without_docker(tmp_path):
    config = ab.load_config(write_config(tmp_path, two_bench_config(tmp_path)))
    all_cases = ab.expand_cases(config, run_id="run123")
    initial = ab.Manifest(run_id="run123", total=2)
    for case in all_cases:
        initial.record(case, ab.build_layout(config, "run123", case), "passed")
    runner = FakeRunner()

    result = ab.run_controller(
        config,
        run_id="run123",
        runner=runner,
        initial_manifest=initial,
        cases_to_run=(),
    )

    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert result == 0
    assert state["status"] == "completed"
    assert state["counts"]["passed"] == 2
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_run_controller_resume_stop_preserves_old_passed_row(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, two_bench_config(tmp_path)))
    all_cases = ab.expand_cases(config, run_id="run123")
    initial = ab.Manifest(run_id="run123", total=2)
    initial.record(all_cases[0], ab.build_layout(config, "run123", all_cases[0]), "passed")

    def stop_ready(*args, **kwargs):
        raise ab.StopRequested("resume stopped")

    monkeypatch.setattr(ab, "wait_for_container_ready", stop_ready, raising=False)
    result = ab.run_controller(
        config,
        run_id="run123",
        runner=FakeRunner(),
        initial_manifest=initial,
        cases_to_run=(all_cases[1],),
    )

    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    assert result == 130
    assert manifest["status"] == "interrupted"
    assert [row["bench_profile"] for row in manifest["cases"]] == ["smoke", "smoke2"]
    assert [row["status"] for row in manifest["cases"]] == ["passed", "interrupted"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest \
  vllm_standalone_bench/tests/test_auto_bench.py::test_run_controller_with_initial_manifest_skips_passed_case \
  vllm_standalone_bench/tests/test_auto_bench.py::test_run_controller_pending_empty_writes_finished_state_without_docker \
  vllm_standalone_bench/tests/test_auto_bench.py::test_run_controller_resume_stop_preserves_old_passed_row \
  -q
```

预期：失败，报错包含 `TypeError: run_controller() got an unexpected keyword argument 'initial_manifest'`。

- [ ] **步骤 3：扩展 run_controller 签名和初始化**

把 `run_controller` 签名改为：

```python
def run_controller(config: AutoBenchConfig, run_id: str,
                   runner: Runner | None = None,
                   dry_run: bool = False,
                   lock_token: str | None = None,
                   initial_manifest: Manifest | None = None,
                   cases_to_run: tuple[BenchmarkCase, ...] | None = None) -> int:
```

把函数开头的 case 和 manifest 初始化改成：

```python
    all_cases = expand_cases(config, run_id=run_id)
    cases = all_cases if cases_to_run is None else cases_to_run
    run_dir = config.run.results_dir / run_id
```

把原来的：

```python
    manifest = Manifest(run_id=run_id, total=len(cases))
```

替换为：

```python
    manifest = initial_manifest or Manifest(run_id=run_id, total=len(all_cases))
```

把 `completed = 0` 替换为：

```python
    completed = len(manifest.cases)
```

- [ ] **步骤 4：处理 pending 为空和 state total**

在 `validate_local_paths(config)` 后、Docker network 检查前插入：

```python
        if not cases:
            write_manifest(run_dir, manifest)
            write_state(run_dir, finished_state(run_id, manifest))
            return 0
```

把 `current_state(... cases, ...)` 调用中的第二个参数改为 `all_cases`：

```python
                            all_cases,
```

把外层 `except StopRequested` 中 `_record_interrupted_group(... list(cases), ...)` 保持为 pending `cases`，不要改成 `all_cases`。

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
python3 -m pytest \
  vllm_standalone_bench/tests/test_auto_bench.py::test_run_controller_with_initial_manifest_skips_passed_case \
  vllm_standalone_bench/tests/test_auto_bench.py::test_run_controller_pending_empty_writes_finished_state_without_docker \
  vllm_standalone_bench/tests/test_auto_bench.py::test_run_controller_resume_stop_preserves_old_passed_row \
  -q
```

预期：`3 passed`。

- [ ] **步骤 6：运行现有 controller 回归测试**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q
```

预期：全部通过。

- [ ] **步骤 7：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): run auto bench pending resume cases"
```

---

## 任务 4：新增 resume CLI 和后台子进程

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

在 main / detach 相关测试后追加：

```python
def test_controller_command_matches_resume_child(tmp_path):
    cmd = [
        sys.executable,
        str(Path(ab.__file__).resolve()),
        "resume",
        "--run-id",
        "run123",
        "--child",
        "--results-dir",
        str(tmp_path / "results"),
    ]

    assert ab.controller_command_matches(cmd, "run123", tmp_path / "results") is True


def test_main_resume_foreground_runs_pending_cases(tmp_path, monkeypatch):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir)
    write_resume_manifest(run_dir, cases[:1], config, ["passed"])
    calls = []

    def fake_run_controller(config_arg, run_id, runner=None, dry_run=False, lock_token=None,
                            initial_manifest=None, cases_to_run=None):
        calls.append((run_id, tuple(case.bench_profile.name for case in cases_to_run), len(initial_manifest.cases)))
        return 0

    monkeypatch.setattr(ab, "install_signal_handlers", lambda: calls.append("signals"), raising=False)
    monkeypatch.setattr(ab, "run_controller", fake_run_controller)

    exit_code = ab.main([
        "resume",
        "--results-dir", str(tmp_path / "results"),
        "--run-id", "run123",
    ])

    assert exit_code == 0
    assert calls == ["signals", ("run123", ("smoke2",), 1)]


def test_main_resume_detach_starts_resume_child(tmp_path, monkeypatch):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir)
    write_resume_manifest(run_dir, cases[:1], config, ["passed"])
    popen_calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        popen_calls.append(command)
        return FakeProcess()

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)

    exit_code = ab.main([
        "resume",
        "--results-dir", str(tmp_path / "results"),
        "--run-id", "run123",
        "--detach",
    ])

    controller = json.loads((run_dir / "controller.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert popen_calls[0][2] == "resume"
    assert "--child" in popen_calls[0]
    assert controller["command"][2] == "resume"


def test_main_resume_empty_pending_does_not_start_controller(tmp_path, monkeypatch, capsys):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir, status="completed")
    write_resume_manifest(run_dir, cases, config, ["passed", "passed"])

    def fail_run_controller(*args, **kwargs):
        raise AssertionError("empty resume should not start controller")

    monkeypatch.setattr(ab, "run_controller", fail_run_controller)

    exit_code = ab.main([
        "resume",
        "--results-dir", str(tmp_path / "results"),
        "--run-id", "run123",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "no pending" in captured.out.lower() or "nothing to resume" in captured.out.lower()


def test_main_resume_child_load_context_failure_releases_lock(tmp_path, monkeypatch, capsys):
    results_dir = tmp_path / "results"
    run_dir = results_dir / "run123"
    run_dir.mkdir(parents=True)
    (run_dir / ".run.lock").write_text(
        json.dumps({"pid": os.getpid(), "token": "tok", "created_at": 1.0}),
        encoding="utf-8",
    )

    def fail_context(*args, **kwargs):
        raise ab.ConfigError("bad resume context")

    monkeypatch.setattr(ab, "install_signal_handlers", lambda: None, raising=False)
    monkeypatch.setattr(ab, "load_resume_context", fail_context)

    exit_code = ab.main([
        "resume",
        "--results-dir", str(results_dir),
        "--run-id", "run123",
        "--child",
        "--lock-token", "tok",
    ])

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert state["status"] == "failed"
    assert "bad resume context" in state["error"]
    assert "bad resume context" in captured.err
    assert not (run_dir / ".run.lock").exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest \
  vllm_standalone_bench/tests/test_auto_bench.py::test_controller_command_matches_resume_child \
  vllm_standalone_bench/tests/test_auto_bench.py::test_main_resume_foreground_runs_pending_cases \
  vllm_standalone_bench/tests/test_auto_bench.py::test_main_resume_detach_starts_resume_child \
  vllm_standalone_bench/tests/test_auto_bench.py::test_main_resume_empty_pending_does_not_start_controller \
  vllm_standalone_bench/tests/test_auto_bench.py::test_main_resume_child_load_context_failure_releases_lock \
  -q
```

预期：失败，至少包含 parser 不认识 `resume` 或 `controller_command_matches` 返回 False。

- [ ] **步骤 3：让 controller metadata 接受 resume child**

把 `controller_command_matches` 中：

```python
    if command[2] != "run":
        return False
```

替换为：

```python
    if command[2] not in {"run", "resume"}:
        return False
```

- [ ] **步骤 4：扩展 detached command**

把 `build_detach_command` 签名改为：

```python
def build_detach_command(config_path: Path | None, run_id: str,
                         results_dir: Path,
                         lock_token: str | None = None,
                         command_name: str = "run") -> list[str]:
```

替换函数体开头为：

```python
    cmd = [sys.executable, str(Path(__file__).resolve()), command_name]
    if command_name == "run":
        if config_path is None:
            raise ValueError("run detached command requires config_path")
        cmd.extend(["--config", str(config_path)])
    elif command_name != "resume":
        raise ValueError(f"unsupported detached command: {command_name}")
    cmd.extend([
        "--run-id",
        run_id,
        "--child",
        "--results-dir",
        str(results_dir),
    ])
```

保留尾部：

```python
    if lock_token is not None:
        cmd.extend(["--lock-token", lock_token])
    return cmd
```

把 `start_detached` 签名改为：

```python
def start_detached(config_path: Path | None, config: AutoBenchConfig, run_id: str,
                   command_name: str = "run") -> int:
```

调用 `build_detach_command` 时传入 `command_name=command_name`，并把 `controller.json`
中的 `config_path` 改为：

```python
            "config_path": str(config_path) if config_path is not None else None,
```

- [ ] **步骤 5：实现 resume_run 和 CLI parser**

在 `stop_run` 前新增：

```python
def resume_run(results_dir: Path, run_id: str, *,
               runner: Runner | None = None,
               lock_token: str | None = None) -> int:
    context = load_resume_context(results_dir, run_id)
    if context.unknown_manifest_cases:
        print(
            f"warning: ignoring manifest cases not in resolved config: {context.unknown_manifest_cases}",
            file=sys.stderr,
        )
    if not context.pending_cases:
        print(f"nothing to resume: {run_id}")
        write_manifest(context.run_dir, context.initial_manifest)
        write_state(context.run_dir, finished_state(run_id, context.initial_manifest))
        return 0
    return run_controller(
        context.config,
        run_id=run_id,
        runner=runner,
        lock_token=lock_token,
        initial_manifest=context.initial_manifest,
        cases_to_run=context.pending_cases,
    )
```

在 `parse_args` 中 `stop_parser` 后新增：

```python
    resume_parser = subparsers.add_parser("resume", help="resume interrupted benchmark cases")
    resume_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--detach", action="store_true")
    resume_parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    resume_parser.add_argument("--lock-token", help=argparse.SUPPRESS)
```

在 `main` 中 `if args.command == "stop":` 前新增：

```python
    if args.command == "resume":
        if args.child:
            try:
                install_signal_handlers()
                return resume_run(
                    args.results_dir,
                    args.run_id,
                    lock_token=args.lock_token,
                )
            except StopRequested as exc:
                error = str(exc) or "stop requested"
                try:
                    write_child_startup_state_best_effort(
                        args.results_dir,
                        args.run_id,
                        "interrupted",
                        error,
                    )
                finally:
                    release_child_startup_lock(args.results_dir, args.run_id, args.lock_token)
                print(error, file=sys.stderr)
                return 130
            except Exception as exc:
                error = str(exc)
                try:
                    write_child_startup_state_best_effort(
                        args.results_dir,
                        args.run_id,
                        "failed",
                        error,
                    )
                finally:
                    release_child_startup_lock(args.results_dir, args.run_id, args.lock_token)
                print(error, file=sys.stderr)
                return 1
        try:
            context = load_resume_context(args.results_dir, args.run_id)
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not context.pending_cases:
            print(f"nothing to resume: {args.run_id}")
            write_manifest(context.run_dir, context.initial_manifest)
            write_state(context.run_dir, finished_state(args.run_id, context.initial_manifest))
            return 0
        if args.detach:
            return start_detached(
                None,
                context.config,
                args.run_id,
                command_name="resume",
            )
        install_signal_handlers()
        return run_controller(
            context.config,
            run_id=args.run_id,
            initial_manifest=context.initial_manifest,
            cases_to_run=context.pending_cases,
        )
```

- [ ] **步骤 6：运行测试验证通过**

运行：

```bash
python3 -m pytest \
  vllm_standalone_bench/tests/test_auto_bench.py::test_controller_command_matches_resume_child \
  vllm_standalone_bench/tests/test_auto_bench.py::test_main_resume_foreground_runs_pending_cases \
  vllm_standalone_bench/tests/test_auto_bench.py::test_main_resume_detach_starts_resume_child \
  vllm_standalone_bench/tests/test_auto_bench.py::test_main_resume_empty_pending_does_not_start_controller \
  vllm_standalone_bench/tests/test_auto_bench.py::test_main_resume_child_load_context_failure_releases_lock \
  -q
```

预期：`5 passed`。

- [ ] **步骤 7：运行 stop 安全回归**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -k 'stop_run or controller_command_matches or start_detached or main_resume' -q
```

预期：全部通过。

- [ ] **步骤 8：Commit**

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
git commit -m "feat(bench): add auto bench resume command"
```

---

## 任务 5：更新 shell wrapper 和 README

**文件：**
- 修改：`vllm_standalone_bench/run_auto_bench.sh`
- 修改：`vllm_standalone_bench/README.md`
- 测试：`vllm_standalone_bench/tests/test_shell_scripts.py`

- [ ] **步骤 1：编写失败的 wrapper 测试**

在 `test_run_auto_bench_uses_project_root_as_working_directory` 后追加：

```python
def test_run_auto_bench_resume_forwards_detach(tmp_path):
    capture = tmp_path / "capture.txt"
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"printf '%s\\n' \"$@\" > {capture}",
        ]),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": str(fake_python),
        "RUN_ID": "resume_check",
        "DETACH": "true",
    })

    subprocess.run(
        [str(SCRIPTS_DIR / "run_auto_bench.sh"), "resume"],
        cwd=SCRIPTS_DIR,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    lines = capture.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [str(SCRIPTS_DIR / "auto_bench.py"), "resume"]
    assert "--results-dir" in lines
    assert "--run-id" in lines
    assert "resume_check" in lines
    assert "--detach" in lines
```

并把 `test_helper_shell_scripts_are_ready_to_use` 中 `run_auto_bench.sh` 的
`expected_fragments` 增加 `"resume"`。

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
python3 -m pytest \
  vllm_standalone_bench/tests/test_shell_scripts.py::test_run_auto_bench_resume_forwards_detach \
  vllm_standalone_bench/tests/test_shell_scripts.py::test_helper_shell_scripts_are_ready_to_use \
  -q
```

预期：失败，`resume` 尚未被 wrapper 接受。

- [ ] **步骤 3：修改 run_auto_bench.sh**

在顶部使用说明中添加：

```bash
#   RUN_ID=qwen_smoke_001 ./run_auto_bench.sh resume
#   RUN_ID=qwen_smoke_001 DETACH=false ./run_auto_bench.sh resume
```

把子命令 case 从：

```bash
        run|status|logs|stop)
```

改为：

```bash
        run|status|logs|stop|resume)
```

把命令拼装 case 中 `status|stop)` 分支改为：

```bash
    status|stop)
        CMD=("${PYTHON_BIN}" "${AUTO_BENCH}" "${COMMAND}" --results-dir "${RESULTS_DIR}" --run-id "${RUN_ID}")
        ;;
    resume)
        CMD=("${PYTHON_BIN}" "${AUTO_BENCH}" resume --results-dir "${RESULTS_DIR}" --run-id "${RUN_ID}")
        if [[ "${DETACH}" == "true" ]]; then
            CMD+=(--detach)
        fi
        ;;
```

在摘要显示逻辑的 `else` 分支里增加后台状态：

```bash
    if [[ "${COMMAND}" == "resume" ]]; then
        printf "║  后台运行  : %-48s║\n" "${DETACH}"
    fi
```

- [ ] **步骤 4：更新 README**

在后台控制示例的 `stop` 后追加：

```markdown
python3 vllm_standalone_bench/auto_bench.py resume \
  --results-dir vllm_standalone_bench/results \
  --run-id <run_id> \
  --detach
```

把说明段改为：

```markdown
默认使用 Docker bridge network `vllm-bench-net`，不使用 `--network host`，也不暴露主机端口。控制器只会清理本次自动创建并带有本次运行标签或元数据的资源，包括 vLLM 容器和 Docker network；`stop` 会请求后台控制器优雅退出并执行这些清理。`stop` 不是容器暂停：中止后当前容器会被删除。需要继续同一 `run_id` 时，使用 `resume`，它会跳过 `manifest.json` 中已 `passed` 的 case，只补跑未成功或未记录的 case。
```

- [ ] **步骤 5：运行 wrapper 测试验证通过**

运行：

```bash
python3 -m pytest vllm_standalone_bench/tests/test_shell_scripts.py -q
```

预期：全部通过。

- [ ] **步骤 6：Commit**

```bash
git add vllm_standalone_bench/run_auto_bench.sh vllm_standalone_bench/README.md vllm_standalone_bench/tests/test_shell_scripts.py
git commit -m "docs(bench): document auto bench resume wrapper"
```

---

## 任务 6：最终验证和边界检查

**文件：**
- 修改：无新功能文件，验证整个改动集。

- [ ] **步骤 1：运行 auto bench 单测**

```bash
python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q
```

预期：全部通过。

- [ ] **步骤 2：运行 shell wrapper 单测**

```bash
python3 -m pytest vllm_standalone_bench/tests/test_shell_scripts.py -q
```

预期：全部通过。

- [ ] **步骤 3：运行 Bash 语法检查**

```bash
bash -n vllm_standalone_bench/run_auto_bench.sh
```

预期：退出码 0，无输出。

- [ ] **步骤 4：运行 diff 空白检查**

```bash
git diff --check main...HEAD
```

预期：退出码 0，无输出。

- [ ] **步骤 5：人工核对规格覆盖**

逐项检查 `docs/superpowers/specs/2026-07-02-auto-bench-resume-design.md`：

```text
resume CLI：任务 4、任务 5 覆盖
跳过 passed：任务 1、任务 3 覆盖
failed/skipped/interrupted/未记录重跑：任务 1 覆盖，任务 3 执行
active run 拒绝：任务 2 覆盖，复用 reject_active_run
resolved config + CLI results_dir：任务 2 覆盖
后台恢复 + stop 可识别 resume child：任务 4 覆盖
README stop 不是暂停：任务 5 覆盖
```

- [ ] **步骤 6：最终状态检查**

```bash
git status --short --branch
```

预期：在 `feat/auto-bench-resume` 分支，仅允许没有未提交变更。若有生成缓存文件，先确认是否为测试缓存；不要提交 `.pytest_cache` 或结果目录。

- [ ] **步骤 7：记录最终验证结果**

如果步骤 1-6 发现需要修正实现、测试或文档，先完成修正，再按涉及文件精确执行：

```bash
git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/run_auto_bench.sh vllm_standalone_bench/README.md vllm_standalone_bench/tests/test_auto_bench.py vllm_standalone_bench/tests/test_shell_scripts.py
git commit -m "test(bench): verify auto bench resume flow"
```

如果步骤 1-6 没有产生文件改动，不创建空提交；在最终回复中报告各验证命令的输出摘要。
