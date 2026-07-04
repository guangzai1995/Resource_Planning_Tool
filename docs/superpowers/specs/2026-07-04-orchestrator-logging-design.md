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
  唯一非空的 `pd-test-0703-1` 只有 6 行无结构报错。根因是 controller 主 case 循环
  （`~3269 for case in expand_cases`）只通过 `write_state()` 更新 `current/counts`，
  **关键进度事件根本没有输出到日志流**。
- **detach 重定向串色风险**：detach 模式把整个 stdout/stderr 重定向到
  `controller.log`（`~3948` 行），若直接给 logger 挂彩色 ConsoleHandler 写 stderr，
  会与 FileHandler 双写同一文件。
- **`status` / `logs` 子命令**只是 `tail -f` + 裸 print，无级别筛选、无颜色、
  无统一格式。
- **风格不统一**：`run_bench_multi.py` 自己有一套 `basicConfig`（`%(asctime)s
  %(levelname)s`），与 `auto_bench.py` 不一致；`resource_monitor.py` /
  `remote_docker.py` 完全没有日志（0 处）。

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
- **不纳入容器内执行的日志**：`vllm_bench/serve.py`、`vllm_bench/pd_proxy.py`
  跑在 bench-runner / vLLM 容器内，是独立日志范畴，本次不改。
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
def case_scope(logger: logging.Logger, idx: int, total: int, phase: str):
    """with 块内所有日志自动带 [case idx/total][phase] 前缀。"""
```

内部组件：

- `ConsoleFormatter`：格式 `%H:%M:%S LEVEL [ctx] msg`，按级别上色
  （INFO 灰 / WARN 黄 / ERROR 红 / 自定义 OK 绿）；非 tty 自动去 ANSI。
- `FileFormatter`：格式 `%Y-%m-%d %H:%M:%S LEVEL [ctx] msg`，永远纯文本。
- `_CaseContext`：基于 `contextvars.ContextVar` 存当前 `idx/total/phase`
  （线程安全，兼容 resource_monitor 后台线程）。formatter 读取并拼前缀。

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

### CaseContext 注入点

controller 主 case 循环（`auto_bench.py ~3269`）用 `case_scope` 包裹每个 case
的各阶段：

```python
for idx, case in enumerate(cases, 1):
    with case_scope(logger, idx, total, "serve"):
        logger.info("starting vLLM container tp=%s gpu=%s", tp, gpus)
        ...
    with case_scope(logger, idx, total, "bench"):
        logger.info("bench input=%s out=%s conc=%s", in_len, out_len, conc)
        ...
```

formatter 输出示例：`12:31:42 INFO [case 3/8][bench] bench input=4096 out=128 conc=8`。

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
  过滤（按级别字段 grep 行）。

### 错误处理

- 颜色：`isatty` 检测包 try/except，失败回退纯文本。
- `logging.raiseExceptions = False`：日志故障不拖垮 benchmark。
- FileHandler 写失败（如磁盘满）：捕获，降级只留 ConsoleHandler，benchmark 继续。
- formatter 读取 ContextVar 为空时前缀留空，不抛异常。

### 范围边界

- **纳入**：`auto_bench.py`（主改造）、`bench_log.py`（新增）、
  `run_bench_multi.py`（替换其 `basicConfig`，统一到门面）。
- **可选小改**：`resource_monitor.py` / `remote_docker.py` 报错时调用主 logger。
- **不纳入**：`vllm_bench/serve.py`、`vllm_bench/pd_proxy.py`（容器内）。

## 测试策略

- **`bench_log.py` 纯函数单测**（新增 `tests/test_bench_log.py`）：
  - `ConsoleFormatter` / `FileFormatter` 输出格式符合预期（含时间戳、级别、
    `[case idx/total][phase]` 前缀）。
  - `case_scope` 前缀拼接正确，with 块退出后前缀复位。
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
