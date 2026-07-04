# 编排器日志记录与显示优化设计

## 背景

`vllm_standalone_bench/auto_bench.py`（4680 行）是离线自动压测的编排器。当前
编排器日志存在以下问题（已通过代码与历史 run 产物核实）：

- **没有 logging 配置**：文件里只有 `logger = logging.getLogger("auto_bench")`，
  全文件无 `basicConfig` / handler / formatter。Python logging 默认行为下，
  `logger.info` / `logger.debug` 完全不输出，仅 14 处 `logger.warning` 会按默认
  格式（`WARNING:auto_bench:...`）吐到 stderr。logger 形同虚设。
- **信息全靠 ~80 处裸 `print()`**：无时间戳、无日志级别、无 case/阶段标识
  （如 `[3/8] [serve]`）。正常进度与报错混在一起，无法区分、无法 grep。
- **controller.log 经常为空**：检查 5 个历史 run，4 个 `controller.log` 为 0 行，
  唯一非空的 `pd-test-0703-1` 只有 6 行无结构报错。根因是执行循环
  （topology `_run_topology_group` 与 legacy 路径的 `for case in group_cases`，
  见下文「CaseContext 注入点」）只通过 `write_state()` 更新 `current/counts`，
  **关键进度事件根本没有输出到日志流**。
- **detach 重定向串色风险**：detach 模式把整个 stdout/stderr 重定向到
  `controller.log`（`~3948` 行），若直接给 logger 挂彩色 ConsoleHandler 写 stderr，
  会与 FileHandler 双写同一文件。
- **`status` / `logs` 子命令**只是 `tail -f` + 裸 print，无级别筛选、无颜色、
  无统一格式。
- **host 侧无统一日志配置**：`auto_bench.py` 既无 `basicConfig` 也无 handler
  （`run_bench_multi.py` 自带的 `basicConfig` 属容器内范畴，见非目标，本次不动）；
  `resource_monitor.py` / `remote_docker.py` 完全没有日志（0 处）。

## 目标

1. 为编排器建立统一的、零新增第三方依赖的结构化彩色日志门面
   （仅用标准库 `logging` + 已有 `tqdm` + ANSI 转义）。
2. 每条日志带 **时间戳 + 级别 + case/阶段上下文**前缀，终端彩色、文件纯文本。
3. 解决 controller.log 信息缺失：在主循环关键节点补充 `logger.info`，
   使 `controller.log` 成为可事后排查的全量结构化日志。
4. 终端彩色与文件纯文本自动分流：前台实时彩色、后台重定向不串色、零双写、
   `controller.log` 路径不变（向后兼容现有 `status` / `logs` 命令与文档）。
5. `status` / `logs` 子命令输出带色摘要，`logs` 支持 `--level` 过滤。

## 非目标

- **不引入第三方日志库**（`rich` / `loguru` / `structlog`）。离线 CentOS 7 主机
  新增依赖搬运成本高，标准库 + ANSI 足够覆盖需求。
- **不做结构化 JSON 日志**。用户未要求机器可读日志，YAGNI。
- **不纳入容器内执行的日志**：`vllm_bench/serve.py`、`vllm_bench/pd_proxy.py`、
  **`run_bench_multi.py`** 三者均由 `build_bench_run_command`（`auto_bench.py:1188`）
  以 `docker run ... <bench_image> python /opt/vllm_standalone_bench/run_bench_multi.py`
  在 bench-runner 容器内执行（容器内路径 `/opt/vllm_standalone_bench/`）。host 侧
  改这些源码不进镜像、容器内也无 host 的 run_dir / controller.log、门面不可达，
  属独立日志范畴，本次一律不改。`run_bench_multi.py` 现有的 `basicConfig` 保持原样。
- **不改 benchmark 指标计算逻辑**，不改 case 调度 / resume / manifest 语义。
- `resource_monitor.py` / `remote_docker.py` 当前无日志，本次不强接（仅在需要
  报错时调用主 logger，作为可选小改）。

## 设计

### 新增模块 `bench_log.py`

作为唯一日志门面，职责单一，可独立单测。公开接口：

```python
def setup_logging(
    run_dir: Path,
    level: str = "INFO",
    *,
    color: bool | None = None,
    log_file: Path | None = None,
) -> None:
    """初始化 root logger：挂 FileHandler（始终）+ ConsoleHandler（仅 isatty）。
    color=None 时按 sys.stderr.isatty() 自动决定终端是否启用 ANSI。"""

def get_logger(name: str) -> logging.Logger:
    """返回已配置好的 logger，调用方直接 .info/.warning/.error。"""

@contextmanager
def case_scope(
    logger: logging.Logger,
    *,
    total: int,
    phase: str,
    idx: int | None = None,
    label: str | None = None,
):
    """with 块内所有日志自动带前缀。
    idx 非 None（bench per-case 阶段）→ 前缀 ``[case idx/total][phase]``；
    idx 为 None（serve per-group 阶段）→ 前缀 ``[phase][label]``（label 取
    serve_profile / topology_profile 名）。"""
```

内部组件：

- `ConsoleFormatter`：格式 `%H:%M:%S LEVEL [ctx] msg`，按级别上色
  （INFO 灰 / WARN 黄 / ERROR 红 / 自定义 OK 绿）；非 tty 自动去 ANSI。
- `FileFormatter`：格式 `%Y-%m-%d %H:%M:%S LEVEL [ctx] msg`，永远纯文本。
- `_CaseContext`：基于 `contextvars.ContextVar` 存当前 `idx/total/phase`，
  同线程内跨函数自动传播（formatter 读取并拼前缀）。

### 双 handler 分流（核心，解决空日志与串色）

- **FileHandler**：始终挂，写 `controller.log`（路径不变），append，纯文本 +
  完整时间戳。这是 controller.log 的结构化全量来源。
- **ConsoleHandler**：仅在 `sys.stderr.isatty()` 为真时挂，写 stderr，彩色。
- **isatty 自然区分前后台，零双写**：
  - 前台运行（stderr 是 tty）：FileHandler 写 `controller.log`，
    ConsoleHandler 写终端彩色——两条独立路径，不双写。
  - detach 运行（stderr 被重定向到文件，非 tty）：只挂 FileHandler，
    ConsoleHandler 不挂。logger 输出只走 FileHandler → `controller.log`。
    detach 的 stdout/stderr 重定向继续兜底捕获子进程 / 第三方库的非 logger
    输出（如 docker 命令回显）到 `controller.log`，与结构化日志共存于同一文件。
- 因此 `controller.log` 路径、`status` / `logs` 命令指向均不变，向后兼容。

### CaseContext 注入点（topology / legacy 双路径，serve per-group → bench per-case）

controller 执行分两条独立路径，每条都是「serve per-group → bench per-case」结构：

- **topology 路径** `_run_topology_group`（`auto_bench.py ~3338`）：
  `serve_case = group_cases[0]` 启动整组 role（~3360-3401），ready 后
  `for case in group_cases`（~3403）对每个 case 调 `_run_bench_case`。
- **legacy 路径**（~3630-3695）：`build_serve_run_command` 启动 `serve_case`
  （per-group，~3636），ready 后 `for case in group_cases`（~3675）对每个 case
  调 `_run_bench_case`。

> 注：`auto_bench.py ~3269` 的 `for case in expand_cases` 是 `run_postprocess`
> 后处理（合并资源摘要 + `aggregate_compare`），**不是执行循环**，不在此注入。

两条路径统一套用下面的注入模式：

```python
total = len(all_cases)                      # = manifest.total，固定
# serve 阶段（per-group）：一个 serve 服务整组，不带 case idx
with case_scope(logger, total=total, phase="serve",
                label=serve_profile.name):  # 或 topology_profile.name
    logger.info("starting serve tp=%s gpu=%s", tp, gpus)
    ...                                      # 启动 role / 容器 + wait_for_ready
# bench 阶段（per-case）：idx 为该 case 在全量 all_cases 中的 1-based 序号
for case in group_cases:
    with case_scope(logger, total=total, phase="bench", idx=case.global_index):
        logger.info("bench input=%s out=%s conc=%s", in_len, out_len, conc)
        status, error = _run_bench_case(...)  # 被调函数内的 logger 也自动带前缀
```

**idx / total 语义（关键，保证 resume/skip 下口径稳定）**：

- `total = manifest.total = len(all_cases)`，是全量 case 数，**固定不变**。
- `idx = case.global_index`，是 case 在全量 `all_cases` 中的 1-based 序号，
  **在 `expand_cases` 后预计算并绑定到 case 对象上**。这样 resume（`pending` 是
  `all_cases` 子集，`auto_bench.py:3055`）只补跑部分 case、或 skip（`:3673 continue`）
  时，剩余 case 的 `idx/total` 仍映射回全量口径，不会因跳过而错位。
- serve 阶段无单一 case idx（一个 serve 服务整组），前缀用 `[serve][<profile>]`。

formatter 输出示例：

```text
12:31:05 INFO [serve][glm52_fp8_tp8] starting serve tp=8 gpu=all
12:31:42 INFO [case 3/8][bench] bench input=4096 out=128 conc=8
12:40:02 ERROR [case 7/8][bench] endpoint timeout, retrying (2/3)
```

`case_scope` 基于 `contextvars.ContextVar`，**同线程内跨函数自动传播**——
`_run_bench_case`（`auto_bench.py:1813`）等被调函数内的 `logger` 调用无需额外
传参即可继承当前 case 前缀。

### `print` → `logger` 分类映射

| 现有 print | 改为 | 说明 |
|---|---|---|
| 进度（starting / done / ready / run started）| `logger.info` | 现在几乎缺失，需新增 |
| `file=sys.stderr` 的 warning（41 处）| `logger.warning` | 保留语义 |
| failed / error / not found | `logger.error` | |
| `status` / `logs` 命令的 `run_id:` / `counts:` 等 CLI 展示 | **保留 print，抽到 `render_status()` 函数** | 这是命令输出，不该进 controller.log |
| detach 启动提示（`run_id:` / `controller_log:`）| 保留 print | 给前台调用方看 |

### controller 主循环补充的日志节点（现在全缺，是 controller.log 变「有料」的关键）

- run 启动：`run started: <run_id> cases=<total>`
- 每个 case：serve 启动 → endpoint ready → bench 参数 → bench 完成（带
  ttft / tpot 关键指标）→ 资源汇总 → case 结束（`passed` / `failed: <reason>`）
- run 结束：`run finished: passed=X failed=Y skipped=Z`

### `status` / `logs` 展示层

- `print_status()`：复用 `bench_log` 的颜色常量，把 `status / counts / current`
  渲染成带色摘要（passed 绿 / failed 红 / running 黄 / skipped 灰）。
- `logs` 子命令：仍 tail `controller.log`，新增 `--level {INFO,WARNING,ERROR}`
  过滤。**语义明确**：`--level` 仅过滤**带级别前缀的单行**（`follow_file`
  流式读取时按行级别字段匹配）；多行日志的续行（如 traceback 后续行）与
  无级别前缀的第三方 / docker 回显（经 detach 重定向进 controller.log）不在
  过滤范围内——未指定 `--level` 时全部原样输出，指定时未命中级别的行不输出。
  不追求多行块完整匹配，避免实现复杂化。

### 错误处理

- 颜色：`isatty` 检测包 try/except，失败回退纯文本。
- `logging.raiseExceptions = False`：日志故障不拖垮 benchmark。
- FileHandler 写失败（如磁盘满）：捕获，降级只留 ConsoleHandler，benchmark 继续。
- formatter 读取 ContextVar 为空时前缀留空，不抛异常。
- **FileHandler 缓冲策略**：detach 模式下 FileHandler 与 detach 继承的进程 stdout
  句柄并发 append 同一 `controller.log`，两者缓冲策略不同可能导致行序交错、
  影响 `logs --level` 行级 grep。FileHandler 在关键节点（case 边界、serve 启动/
  结束、error）显式 `flush()`，保证结构化日志行及时落盘。注：现状全是 `print`
  块缓冲同样会交错，本改造不让其恶化，但显式 flush 可改善可读性。
- **ContextVar 作用域澄清**：`case_scope` 用 `contextvars` 是为**同线程内跨函数**
  传播前缀（主线程 → `_run_bench_case` 等被调函数），不是为了线程隔离。
  `resource_monitor` 采样在子线程但不输出日志（错误回主线程 `logger.warning`），
  故无需为监控线程做 ContextVar 跨线程复制兼容。

### 范围边界

- **纳入**：`auto_bench.py`（主改造，host 侧编排器）、`bench_log.py`（新增门面）。
- **不纳入（容器内执行，见非目标）**：`vllm_bench/serve.py`、
  `vllm_bench/pd_proxy.py`、`run_bench_multi.py`。
- **不强接**：`resource_monitor.py` / `remote_docker.py` 当前 0 日志；
  `resource_monitor` 采样错误已在**主线程**经 `auto_bench.py:1853/1878` 的
  `logger.warning` 捕获，子线程不直接输出日志，无需为它做跨线程前缀兼容。
  如后续需要这两个模块的报错也走结构化日志，作为独立小改另议，不并入本次范围。

## 测试策略

- **`bench_log.py` 纯函数单测**（新增 `tests/test_bench_log.py`）：
  - `ConsoleFormatter` / `FileFormatter` 输出格式符合预期（含时间戳、级别、
    `[case idx/total][phase]` 前缀）。
  - `case_scope` 前缀拼接正确：idx 非 None 输出 `[case idx/total][phase]`、
    idx=None（serve per-group）输出 `[phase][label]`；with 块退出后 ContextVar
    前缀复位；同线程内跨函数调用（模拟 `_run_bench_case` 被调）能继承当前前缀。
  - `isatty` 为 False 时 ConsoleFormatter 去 ANSI（mock `sys.stderr.isatty`）。
  - `color=False` 显式禁用颜色。
  - `setup_logging` 同时挂 FileHandler +（仅 tty 时）ConsoleHandler，
    非 tty 时不双写。
- **集成测试**（扩展 `tests/test_auto_bench.py`）：
  - 跑最小 smoke（dry-run 或单 case）后断言 `controller.log` 非空，且包含
    `[case 1/N]`、时间戳、级别字段。
  - `status` / `logs` 子命令输出格式变化的断言同步更新。
- **回归**：现有 `tests/test_auto_bench.py`、`tests/test_shell_scripts.py` 全绿
  （`run_auto_bench.sh` 若引用日志路径需同步）。
