# result 列顺序重排 + 绘图中文修复 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。
>
> **实现前必读：** 按 `CLAUDE.md` 强制要求，代码实现必须在专用 worktree 中进行（不在 `main` 上直接改）。规格见 `docs/superpowers/specs/2026-07-05-result-column-and-plot-font-design.md`。

**目标：** 把 result.csv/xlsx 的吞吐量与 TTFT 指标列前移到基本信息之后；修复 plots 中文方框（CJK 字体打进 bench-runner 镜像 + matplotlib 设字体）。

**架构：** Part 1 仅重排 `run_bench_multi.py` 的 `CSV_HEADERS` / `CSV_HEADERS_ZH` 两个并行列表（消费方全按键取值，顺序安全）。Part 2 在 `Dockerfile.bench-runner` 装 `fonts-wqy-microhei`，并在 `bench_compare.py::_plot()` 用 rcParams 指定中文字体，删掉原先压制 Glyph 告警的代码。

**技术栈：** Python 3.11、csv.DictWriter、openpyxl、matplotlib (Agg)、Debian apt（python:3.11-slim 基镜像）、pytest。

**关键安全结论（已核查）：**
- `CSV_HEADERS` 的所有消费方（`save_csv:505`、`save_xlsx:533/538/555/562`、`resource_monitor.flatten_summary_for_result`）均按键取值，与列顺序无关。
- 现有 `tests/test_integration.py:305-318`、`tests/test_extract_row.py:296-322` 对 `CSV_HEADERS` 的断言全是顺序无关的（len 相等、前 4 列、`num_prompts` 紧跟 `seed`、成员关系），本重排保留全部这些不变量 → 这些测试继续绿，无需改。

---

## 文件结构

| 文件 | 职责 | 动作 |
|---|---|---|
| `vllm_standalone_bench/run_bench_multi.py` | result.csv/xlsx 列顺序定义 | 修改 `CSV_HEADERS`（457-477）+ `CSV_HEADERS_ZH`（479-498），仅重排 |
| `vllm_standalone_bench/bench_compare.py` | 跨配置绘图 | 加 `_CJK_FONT_SANS_SERIF` 常量 + `_apply_cjk_font()` 辅助；`_plot()` 调用它，删 warnings 压制 |
| `vllm_standalone_bench/Dockerfile.bench-runner` | bench-runner 镜像 | apt-get 加 `fonts-wqy-microhei` |
| `vllm_standalone_bench/tests/test_result_csv_headers.py` | Part 1 列序 + Part 2 字体配置断言 | 新建 |

---

## 任务 1：Part 1 — 重排 CSV_HEADERS / CSV_HEADERS_ZH（TDD）

**文件：**
- 测试：`vllm_standalone_bench/tests/test_result_csv_headers.py`（新建）
- 修改：`vllm_standalone_bench/run_bench_multi.py:457-498`

- [ ] **步骤 1：编写失败的测试**

新建 `vllm_standalone_bench/tests/test_result_csv_headers.py`：

```python
"""result.csv/xlsx 列顺序与中文字体配置的断言。

Part 1：CSV_HEADERS / CSV_HEADERS_ZH 的顺序——基本信息在前，吞吐+TTFT 紧随，
其余在后；列集合与中英表头一一对应。
Part 2：bench_compare 的 CJK 字体 fallback 配置。
"""
import pytest

import run_bench_multi as m
import bench_compare as bc


# 期望的新顺序（52 列）：基本信息(17) → 吞吐(5) → TTFT(4) → 其他(26)
EXPECTED_HEADERS = [
    # 基本信息
    "model", "backend", "dataset_name", "language",
    "input_len", "output_len", "total_input_len", "prefix_ratio", "prefix_tokens",
    "parallel_num", "epochs", "num_prompts", "seed",
    "n_success", "n_failed",
    "avg_input_tokens", "avg_output_tokens",
    # 吞吐量（前移）
    "throughput_req_s", "throughput_tok_s", "input_throughput_tok_s",
    "prefill_effective_tok_s", "decode_effective_tok_s",
    # TTFT（前移）
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
    # 其他
    "input_compliance", "output_compliance",
    "finish_reason_length_pct", "token_source",
    "avg_cached_tokens", "cache_hit_rate",
    "avg_gpu_kv_cache_usage", "peak_gpu_kv_cache_usage",
    "spec_decode_acceptance_rate", "spec_decode_system_efficiency",
    "spec_decode_num_drafts", "spec_decode_num_accepted_tokens",
    "spec_decode_num_draft_tokens", "spec_decode_per_position_acceptance_rates",
    "tpot_mean_ms", "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms",
    "e2el_mean_ms", "e2el_p50_ms", "e2el_p90_ms", "e2el_p99_ms",
    "audio_duration_s_total", "audio_duration_s_avg", "rtfx",
    "duration_s",
]

EXPECTED_HEADERS_ZH = [
    # 基本信息
    "模型", "接口类型", "数据集", "语言",
    "输入长度(token)", "输出长度(token)", "总输入长度(token)", "前缀比例", "前缀tokens数",
    "并发数", "测试轮数", "总请求数", "随机种子",
    "成功请求数", "失败请求数",
    "平均实际输入tokens", "平均实际输出tokens",
    # 吞吐量
    "请求吞吐(req/s)", "输出Token系统吞吐(tok/s)", "输入Token系统吞吐(tok/s)",
    "Prefill有效速率(tok/s)", "Decode有效速率(tok/s)",
    # TTFT
    "TTFT均值(ms)", "TTFT_P50(ms)", "TTFT_P90(ms)", "TTFT_P99(ms)",
    # 其他
    "输入长度合规(%)", "输出长度合规(%)",
    "length停止占比(%)", "token来源",
    "平均缓存命中tokens", "缓存命中率(%)",
    "平均GPU KV缓存占用率(%)", "峰值GPU KV缓存占用率(%)",
    "SpecDecode接受率(%)", "SpecDecode系统效率",
    "SpecDecode草稿轮数", "SpecDecode接受tokens数",
    "SpecDecode草稿tokens数", "SpecDecode分位置接受率(%)",
    "TPOT均值(ms)", "TPOT_P50(ms)", "TPOT_P90(ms)", "TPOT_P99(ms)",
    "E2EL均值(ms)", "E2EL_P50(ms)", "E2EL_P90(ms)", "E2EL_P99(ms)",
    "音频总时长(s)", "平均音频时长(s)", "RTFx",
    "测试耗时(s)",
]

THROUGHPUT_COLS = [
    "throughput_req_s", "throughput_tok_s", "input_throughput_tok_s",
    "prefill_effective_tok_s", "decode_effective_tok_s",
]
TTFT_COLS = ["ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms"]


def test_csv_headers_exact_order():
    assert m.CSV_HEADERS == EXPECTED_HEADERS


def test_csv_headers_zh_exact_order():
    assert m.CSV_HEADERS_ZH == EXPECTED_HEADERS_ZH


def test_csv_headers_count_is_52():
    assert len(m.CSV_HEADERS) == 52
    assert len(m.CSV_HEADERS_ZH) == 52


def test_zh_pairs_one_to_one():
    """中英表头按位一一对应，数量一致。"""
    assert len(m.CSV_HEADERS) == len(m.CSV_HEADERS_ZH)


def test_throughput_and_ttft_block_after_basic_info():
    """吞吐块紧跟基本信息块、在 TTFT 块之前；二者都在所有'其他'列之前。"""
    idx = {h: i for i, h in enumerate(m.CSV_HEADERS)}
    basic_last = idx["avg_output_tokens"]
    tp_indices = [idx[c] for c in THROUGHPUT_COLS]
    ttft_indices = [idx[c] for c in TTFT_COLS]
    # 吞吐块紧跟基本信息
    assert min(tp_indices) == basic_last + 1
    assert max(tp_indices) == basic_last + len(THROUGHPUT_COLS)
    # TTFT 块紧跟吞吐块
    assert min(ttft_indices) == max(tp_indices) + 1
    assert max(ttft_indices) == min(ttft_indices) + len(TTFT_COLS) - 1
    # 所有'其他'列都在 TTFT 块之后
    other_cols = [
        "input_compliance", "avg_cached_tokens", "cache_hit_rate",
        "spec_decode_acceptance_rate", "tpot_mean_ms", "e2el_mean_ms",
        "audio_duration_s_total", "duration_s",
    ]
    for c in other_cols:
        assert idx[c] > max(ttft_indices), f"{c} 应在 TTFT 块之后"


def test_column_set_unchanged():
    """重排不增不减列：集合与期望完全相同。"""
    assert set(m.CSV_HEADERS) == set(EXPECTED_HEADERS)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_result_csv_headers.py -v`
预期：FAIL——`test_csv_headers_exact_order` / `test_csv_headers_zh_exact_order` / `test_throughput_and_ttft_block_after_basic_info` 失败（当前顺序里吞吐在 32-36、TTFT 在 37-40，不在基本信息之后）。`test_csv_headers_count_is_52` / `test_zh_pairs_one_to_one` / `test_column_set_unchanged` 当前即通过。

- [ ] **步骤 3：重排 `CSV_HEADERS`（run_bench_multi.py:457-477）**

把现有 `CSV_HEADERS = [...]` 整段替换为（仅顺序变化，列与内容不变）：

```python
CSV_HEADERS = [
    # ── 基本信息 ──────────────────────────────────
    'model', 'backend', 'dataset_name', 'language',
    'input_len', 'output_len', 'total_input_len', 'prefix_ratio', 'prefix_tokens',
    'parallel_num', 'epochs', 'num_prompts', 'seed',
    'n_success', 'n_failed',
    'avg_input_tokens', 'avg_output_tokens',
    # ── 吞吐量（前移）──────────────────────────────
    'throughput_req_s', 'throughput_tok_s', 'input_throughput_tok_s',
    'prefill_effective_tok_s', 'decode_effective_tok_s',
    # ── TTFT（前移）────────────────────────────────
    'ttft_mean_ms', 'ttft_p50_ms', 'ttft_p90_ms', 'ttft_p99_ms',
    # ── 其他 ──────────────────────────────────────
    'input_compliance', 'output_compliance',
    'finish_reason_length_pct', 'token_source',
    'avg_cached_tokens', 'cache_hit_rate',
    'avg_gpu_kv_cache_usage', 'peak_gpu_kv_cache_usage',
    'spec_decode_acceptance_rate', 'spec_decode_system_efficiency',
    'spec_decode_num_drafts', 'spec_decode_num_accepted_tokens',
    'spec_decode_num_draft_tokens', 'spec_decode_per_position_acceptance_rates',
    'tpot_mean_ms', 'tpot_p50_ms', 'tpot_p90_ms', 'tpot_p99_ms',
    'e2el_mean_ms', 'e2el_p50_ms', 'e2el_p90_ms', 'e2el_p99_ms',
    'audio_duration_s_total', 'audio_duration_s_avg', 'rtfx',
    'duration_s',
]
```

- [ ] **步骤 4：重排 `CSV_HEADERS_ZH`（run_bench_multi.py:479-498）**

把现有 `CSV_HEADERS_ZH = [...]` 整段替换为（与上面 EN 列表按位对应）：

```python
CSV_HEADERS_ZH = [
    # ── 基本信息 ──
    '模型', '接口类型', '数据集', '语言',
    '输入长度(token)', '输出长度(token)', '总输入长度(token)', '前缀比例', '前缀tokens数',
    '并发数', '测试轮数', '总请求数', '随机种子',
    '成功请求数', '失败请求数',
    '平均实际输入tokens', '平均实际输出tokens',
    # ── 吞吐量（前移）──
    '请求吞吐(req/s)', '输出Token系统吞吐(tok/s)', '输入Token系统吞吐(tok/s)',
    'Prefill有效速率(tok/s)', 'Decode有效速率(tok/s)',
    # ── TTFT（前移）──
    'TTFT均值(ms)', 'TTFT_P50(ms)', 'TTFT_P90(ms)', 'TTFT_P99(ms)',
    # ── 其他 ──
    '输入长度合规(%)', '输出长度合规(%)',
    'length停止占比(%)', 'token来源',
    '平均缓存命中tokens', '缓存命中率(%)',
    '平均GPU KV缓存占用率(%)', '峰值GPU KV缓存占用率(%)',
    'SpecDecode接受率(%)', 'SpecDecode系统效率',
    'SpecDecode草稿轮数', 'SpecDecode接受tokens数',
    'SpecDecode草稿tokens数', 'SpecDecode分位置接受率(%)',
    'TPOT均值(ms)', 'TPOT_P50(ms)', 'TPOT_P90(ms)', 'TPOT_P99(ms)',
    'E2EL均值(ms)', 'E2EL_P50(ms)', 'E2EL_P90(ms)', 'E2EL_P99(ms)',
    '音频总时长(s)', '平均音频时长(s)', 'RTFx',
    '测试耗时(s)',
]
```

- [ ] **步骤 5：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_result_csv_headers.py -v`
预期：PASS（全部 6 个测试通过）。

- [ ] **步骤 6：跑现有回归测试确认不破**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_integration.py tests/test_extract_row.py tests/test_bench_compare.py -v`
预期：PASS。重点确认 `test_integration.py` 的 CSV_HEADERS 相关断言（line 305-318）与 `test_extract_row.py`（line 296-322）仍绿——它们检查的不变量（前 4 列、num_prompts→seed 相邻、len 相等、成员关系）本重排均保留。

- [ ] **步骤 7：Commit**

```bash
git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/tests/test_result_csv_headers.py
git commit -m "refactor: result.csv/xlsx 吞吐+TTFT 列前移到基本信息之后

CSV_HEADERS / CSV_HEADERS_ZH 重排：基本信息(1-17)不动，吞吐块+TTFT块
从原 32-40 位前移到 18-26 位，其余顺延。列集合/内容不变，仅顺序。
"
```

---

## 任务 2：Part 2 — 绘图中文修复（TDD：字体配置 + Dockerfile）

**文件：**
- 测试：`vllm_standalone_bench/tests/test_result_csv_headers.py`（追加 Part 2 断言）
- 修改：`vllm_standalone_bench/bench_compare.py`（加常量+辅助，改 `_plot()`）
- 修改：`vllm_standalone_bench/Dockerfile.bench-runner:11`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_result_csv_headers.py` 末尾追加：

```python
def test_cjk_font_first_choice_is_wqy_microhei():
    """bench-runner 镜像装的是 fonts-wqy-microhei，应作为首选字体。"""
    assert bc._CJK_FONT_SANS_SERIF[0] == "WenQuanYi Micro Hei"
    assert "DejaVu Sans" in bc._CJK_FONT_SANS_SERIF  # 兜底，保证不崩


def test_apply_cjk_font_sets_rcparams():
    """_apply_cjk_font 应把首选 CJK 字体写进 matplotlib rcParams。"""
    matplotlib = pytest.importorskip("matplotlib")  # 无 matplotlib 的环境跳过
    bc._apply_cjk_font()
    assert matplotlib.rcParams["font.sans-serif"][0] == "WenQuanYi Micro Hei"
    assert not matplotlib.rcParams["axes.unicode_minus"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_result_csv_headers.py::test_cjk_font_first_choice_is_wqy_microhei tests/test_result_csv_headers.py::test_apply_cjk_font_sets_rcparams -v`
预期：FAIL——`AttributeError: module 'bench_compare' has no attribute '_CJK_FONT_SANS_SERIF'`（常量与辅助函数尚未定义）。

- [ ] **步骤 3：在 bench_compare.py 加常量与辅助函数**

在 `bench_compare.py` 现有 `PLOT_METRICS` / `_PLOT_YLABEL` 常量定义之后（约 line 25 附近，`COMPARE_METRICS` 在 line 19、`_PLOT_YLABEL` 在 21-24）追加：

```python
# CJK 字体 fallback：bench-runner 镜像装 fonts-wqy-microhei 命中首个；
# 兼容装了 Noto Sans CJK 的环境；都没有则退回 DejaVu Sans（不崩，但中文仍为方框）
_CJK_FONT_SANS_SERIF = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']


def _apply_cjk_font():
    """配置 matplotlib 中文字体，避免中文标签（如"并发数""输出吞吐"）渲染成方框。"""
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = _CJK_FONT_SANS_SERIF
    matplotlib.rcParams['axes.unicode_minus'] = False
```

- [ ] **步骤 4：改 `_plot()`，删 warnings 压制、改为调 `_apply_cjk_font()`**

`bench_compare.py:194-198` 现有：

```python
    # 默认 DejaVu Sans 缺中文字形会刷屏 UserWarning，静默该已知告警
    import warnings
    warnings.filterwarnings(
        "ignore", message="Glyph .* missing from font", category=UserWarning
    )
```

替换为：

```python
    # 设中文字体（镜像内 fonts-wqy-microhei 命中），不再压制缺字告警——
    # 设了真字体后 Glyph missing 本就不会触发；保留压制反而掩盖真实缺字问题
    _apply_cjk_font()
```

注意：此段位于 `_plot()` 内 `import matplotlib.pyplot as plt`（line 190）之后，调用时 matplotlib 已可用。

- [ ] **步骤 5：运行测试验证通过**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_result_csv_headers.py -v`
预期：PASS（Part 1 的 6 个 + Part 2 的 2 个，共 8 个全过；无 matplotlib 时 Part 2 第二个跳过）。

- [ ] **步骤 6：Dockerfile 加字体包**

`vllm_standalone_bench/Dockerfile.bench-runner:10-13` 现有：

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt openpyxl modelscope
```

改为（仅 apt-get 行加 `fonts-wqy-microhei`）：

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt openpyxl modelscope
```

- [ ] **步骤 7：跑 bench_compare 相关回归**

运行：`cd vllm_standalone_bench && python -m pytest tests/test_bench_compare.py -v`
预期：PASS（`_plot()` 改动不影响聚合/对比逻辑；现有测试不涉及字体）。

- [ ] **步骤 8：Commit**

```bash
git add vllm_standalone_bench/bench_compare.py vllm_standalone_bench/Dockerfile.bench-runner vllm_standalone_bench/tests/test_result_csv_headers.py
git commit -m "fix: plots 中文方框 — 镜像装 fonts-wqy-microhei + _plot() 设 rcParams

bench-runner 镜像 apt 加 fonts-wqy-microhei（~5MB，离线可分发）；
bench_compare 新增 _CJK_FONT_SANS_SERIF 与 _apply_cjk_font()，_plot() 调用
它替代原先压制 Glyph 告警的 warnings.filterwarnings。重 build 镜像后生效。
"
```

---

## 任务 3：端到端验证 + 镜像重 build（手动）

**说明：** 单测覆盖代码层；本任务验证字体真在镜像里、plot 真出汉字。需有网机器重 build。

- [ ] **步骤 1：重 build 镜像**

在有网机器上运行：
```bash
cd vllm_standalone_bench && ./build_bench_runner.sh
```
预期：构建成功，apt 安装 `fonts-wqy-microhei` 无报错。

- [ ] **步骤 2：验证镜像内有 CJK 字体**

```bash
docker run --rm vllm-bench-runner:offline fc-list :lang=zh
```
预期：输出含 `WenQuanYi Micro Hei`（非空）。

- [ ] **步骤 3：跑一次 postprocess，检查 plot**

用现有配置跑一次 auto_bench（或直接对已有 results 目录跑 postprocess 容器），生成 `plots/*.png`，打开任一 `throughput_tok_s.png`：
预期：xlabel "并发数 (parallel_num)"、ylabel "输出吞吐 (tok/s)" 为正常汉字，**非方框**。

- [ ] **步骤 4：导出离线镜像包并搬到测试主机（如需）**

```bash
SAVE_TAR=/tmp/vllm-bench-runner.offline.tar ./build_bench_runner.sh
# 搬到离线机后：docker load -i /tmp/vllm-bench-runner.offline.tar
```

- [ ] **步骤 5：全量回归**

```bash
cd vllm_standalone_bench && python -m pytest tests/ -v
```
预期：全绿（无新增失败）。

---

## 自检

**1. 规格覆盖度：**
- 规格 §3（Part 1 列重排）→ 任务 1 ✓
- 规格 §4（Part 2 字体：Dockerfile + rcParams + 删 warnings）→ 任务 2 ✓
- 规格 §6 验证（CSV_HEADERS 断言、rcParams 断言、端到端重 build 看 plot）→ 任务 1/2/3 ✓
- 规格 §5 受影响文件（run_bench_multi.py、bench_compare.py、Dockerfile.bench-runner、tests/）→ 全覆盖 ✓

**2. 占位符扫描：** 无 TODO/待定/"适当处理"；每个代码步骤都给了完整代码块；Dockerfile/apt/test 均为可执行的具体内容。✓

**3. 类型/命名一致性：**
- `_CJK_FONT_SANS_SERIF`、`_apply_cjk_font()` 在任务 2 步骤 3 定义，步骤 4 在 `_plot()` 调用，测试步骤 1 引用——名称一致 ✓
- `EXPECTED_HEADERS` / `EXPECTED_HEADERS_ZH` 的 52 项与任务 1 步骤 3/4 写入 `CSV_HEADERS`/`CSV_HEADERS_ZH` 的列表严格一致 ✓
- `axes.unicode_minus`（非 `font.unicode_minus`）——matplotlib 正确键名 ✓
