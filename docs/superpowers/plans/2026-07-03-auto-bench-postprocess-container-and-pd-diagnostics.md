# Auto Bench 后处理容器化与 PD 诊断实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 auto bench 后处理放进 bench-runner 容器执行，并在 PD 远程角色启动失败时保存可诊断的 Docker 输出。

**架构：** `auto_bench.py` 新增 postprocess 子命令和容器启动命令构造函数。控制器完成后通过 Docker 运行同一脚本的 postprocess 子命令；容器内逻辑复用现有 `resource_monitor` 和 `bench_compare` 函数。远程 role 启动失败时把 Docker stdout/stderr 写入 case 目录。

**技术栈：** Python、pytest、Docker CLI、现有 auto bench 模块。

---

### 任务 1：后处理命令与容器编排

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

新增测试：

```python
def test_build_postprocess_container_command_uses_bench_image_and_host_user(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    monkeypatch.setattr(ab.os, "getuid", lambda: 1001)
    monkeypatch.setattr(ab.os, "getgid", lambda: 1002)

    cmd = ab.build_postprocess_container_command(
        config,
        config_path=tmp_path / "config.json",
        run_id="run123",
    )

    assert cmd[:3] == ["docker", "run", "--rm"]
    assert value_after(cmd, "--user") == "1001:1002"
    assert value_after(cmd, "-w") == "/workspace"
    assert "vllm-bench-runner:offline" in cmd
    assert "postprocess" in cmd
    assert "--run-id" in cmd
    assert "run123" in cmd
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_build_postprocess_container_command_uses_bench_image_and_host_user -q`

预期：FAIL，提示 `build_postprocess_container_command` 不存在。

- [ ] **步骤 3：实现最少代码**

在 `auto_bench.py` 中添加 `build_postprocess_container_command()` 和 `run_postprocess_container()`，使用 `run.bench_image`、`--user os.getuid():os.getgid()`、repo root bind mount、`postprocess` 子命令。

- [ ] **步骤 4：运行测试验证通过**

运行同一步骤 2 的 pytest，预期 PASS。

### 任务 2：postprocess 子命令

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

新增测试：

```python
def test_postprocess_run_merges_resource_summaries_and_aggregates(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_dir = config.run.results_dir / "run123"
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    layout.bench_dir.mkdir(parents=True)
    (layout.bench_dir / "result.csv").write_text("model,throughput_tok_s\nm,1\n", encoding="utf-8-sig")
    (layout.bench_dir / "resource_summary.json").write_text('{"available": true, "sample_count": 1, "aggregate": {"cpu_util_avg_pct": 12.5}}\\n', encoding="utf-8")
    calls = []
    monkeypatch.setattr(ab, "aggregate_compare", lambda c, rd: calls.append(Path(rd)) or None)

    assert ab.run_postprocess(config, "run123") == 0

    assert calls == [run_dir]
    with (layout.bench_dir / "result.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["resource_monitor_available"] == "true"
    assert rows[0]["cpu_util_avg_pct"] == "12.5"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_postprocess_run_merges_resource_summaries_and_aggregates -q`

预期：FAIL，提示 `run_postprocess` 不存在。

- [ ] **步骤 3：实现最少代码**

实现 `run_postprocess()`：遍历 `expand_cases(config, run_id)`，读取每个 case 的 `resource_summary.json` 和 `resources/<role>/resource_summary.json`，调用现有合并函数；最后调用 `aggregate_compare(config, run_dir)`。

- [ ] **步骤 4：运行测试验证通过**

运行同一步骤 2 的 pytest，预期 PASS。

### 任务 3：控制器收尾调用容器化后处理

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

调整 `test_controller_invokes_aggregate_after_groups` 为验证 Docker postprocess 命令被 runner 调用，并新增 dry-run 跳过测试保持不变。

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_controller_invokes_postprocess_container_after_groups -q`

预期：FAIL，因为控制器仍直接调用 `aggregate_compare`。

- [ ] **步骤 3：实现最少代码**

把 `run_controller()` 中完成后的 `aggregate_compare(config, run_dir)` 替换成 `run_postprocess_container(config, config_path, run_id, runner)`。为此给 `run_controller()` 增加可选 `config_path` 参数，并在 CLI run/resume 调用处传入。

- [ ] **步骤 4：运行测试验证通过**

运行同一步骤 2 的 pytest，预期 PASS。

### 任务 3.5：benchmark runner 使用宿主机用户写结果

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

在 `test_build_bench_command_targets_container_dns` 中断言 benchmark Docker 命令包含：

```python
assert "--user" in cmd
assert value_after(cmd, "--user") == f"{os.getuid()}:{os.getgid()}"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_build_bench_command_targets_container_dns -q`

预期：FAIL，命令中缺少 `--user`。

- [ ] **步骤 3：编写最少实现代码**

新增 `host_user_spec()`，并在 `build_bench_run_command()` 的 `docker run --rm` 参数中加入 `--user host_user_spec()`。

- [ ] **步骤 4：运行测试验证通过**

运行同一步骤 2 的 pytest，预期 PASS。

### 任务 4：PD 远程启动失败诊断落盘

**文件：**
- 修改：`vllm_standalone_bench/auto_bench.py`
- 测试：`vllm_standalone_bench/tests/test_auto_bench.py`

- [ ] **步骤 1：编写失败的测试**

新增测试：

```python
def test_topology_start_failure_writes_start_log_and_status_error(tmp_path, monkeypatch):
    from test_remote_topology import pd_topology_config, write_config
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["image"] = "sglang:pd"
    config = ab.load_config(write_config(tmp_path, data))
    remote = FakeRemoteDockerRunner(failures={("p1", "docker run -d"): 125})
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), config_path=tmp_path / "config.json")

    assert result == 1
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    start_log = (layout.bench_dir / "logs" / "p1.start.log").read_text(encoding="utf-8")
    assert "returncode: 125" in start_log
    assert "forced failure" in start_log
    status = json.loads((layout.bench_dir / "status.json").read_text(encoding="utf-8"))
    assert "forced failure" in status["error"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py::test_topology_start_failure_writes_start_log_and_status_error -q`

预期：FAIL，`p1.start.log` 不存在。

- [ ] **步骤 3：实现最少代码**

新增 `_write_topology_start_failure_artifact()` 和 `_topology_start_error()`。`run_topology_group()` 在 `docker run -d` return code 非 0 时写日志，并把 stderr 摘要放进异常文本。

- [ ] **步骤 4：运行测试验证通过**

运行同一步骤 2 的 pytest，预期 PASS。

### 任务 5：验证

**文件：**
- 修改：无
- 测试：相关 pytest

- [ ] 运行 `python3 -m pytest vllm_standalone_bench/tests/test_auto_bench.py -q`
- [ ] 运行 `python3 -m pytest vllm_standalone_bench/tests/test_resource_monitor.py vllm_standalone_bench/tests/test_bench_compare.py -q`
- [ ] 运行 `python3 -m pytest vllm_standalone_bench/tests -q`
