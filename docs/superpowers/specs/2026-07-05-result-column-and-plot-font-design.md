# result 报告列顺序重排 + 绘图中文修复 · 设计规格

- 日期：2026-07-05
- 范围：`vllm_standalone_bench`
- 关联代码：`run_bench_multi.py`、`bench_compare.py`、`Dockerfile.bench-runner`

## 1. 背景与问题

基准跑完产出两类"最终显示"：

1. **`result.csv` / `result.xlsx`**（由 `run_bench_multi.py` 产出）：单次 run 的原始指标表。
2. **`plots/*.png`**（由 `bench_compare.py::_plot()` 产出，在 postprocess 容器内运行）：跨配置对比折线图。

当前两个痛点：

- **列顺序不友好**：吞吐量块在第 32–36 列、TTFT 块在第 37–40 列，被合规、cache、spec_decode 等次要列挡在后面。看核心性能指标要横向滚动很远。
- **绘图中文成方框**：`_plot()` 用 matplotlib 默认 DejaVu Sans，无中文字形；而 xlabel `"并发数 (parallel_num)"`、ylabel `"输出吞吐 (tok/s)"` 含中文。代码用 `warnings.filterwarnings("ignore", "Glyph .* missing")` 把缺字告警静默了——问题是真被压掉了，不是修好了。根因：bench-runner 镜像（`python:3.11-slim`）里没装任何 CJK 字体，且离线测试主机也没有，运行时无法补装。

## 2. 目标 / 非目标

### 目标

- result.csv / result.xlsx 中，吞吐量与 TTFT 相关列前移到"基本信息"之后、其他次要列之前；列集合与单元格值完全不变，仅顺序调整。
- plots 中文正常显示，不出现豆腐块；方案需在离线环境下生效。

### 非目标

- 不新增、不删除、不重命名任何指标列。
- 不改 `compare.csv` / `compare.xlsx`（聚合报告）的列顺序。
- 不改终端每行摘要打印（`run_bench_multi.py:799-812`，已是 吞吐→TTFT→TPOT→E2EL 序）。
- 不改 `resource_monitor` 追加列的逻辑与位置（本就追加在末尾）。
- 不改 auto_bench 编排、serve、remote_docker 等无关模块。

## 3. Part 1 · result 列顺序重排

### 3.1 机制

`run_bench_multi.py` 中：

- `CSV_HEADERS`（EN，约 line 457-477）：`save_csv` 用 `csv.DictWriter(fieldnames=CSV_HEADERS, extrasaction='ignore')` 消费；`save_xlsx` 的列宽循环也按 `CSV_HEADERS` 顺序遍历。
- `CSV_HEADERS_ZH`（约 line 479-498）：xlsx 的中文表头，与 `CSV_HEADERS` 是**一一对应的并行数组**（`save_xlsx` line 538 `for col, h in enumerate(CSV_HEADERS_ZH, 1)`）。

因此**只需重排这两个列表（保持对应）**，csv 与 xlsx 的列顺序即同步生效。单元格值由 `DictWriter`/`openpyxl` 按 key 取，与顺序无关，内容不变。

### 3.2 新顺序（共 52 列 + 末尾 resource 列）

基本信息块（1–17）**完全不动**；吞吐块、TTFT 块从原 32–40 位前移到 18–26 位；被挤掉的中间列（原 18–31）与尾部（原 41–52）按原相对序顺延。

| 新位 | 列名 | 原位 | 块 |
|---|---|---|---|
| 1 | model | 1 | 基本信息（不动） |
| 2 | backend | 2 | |
| 3 | dataset_name | 3 | |
| 4 | language | 4 | |
| 5 | input_len | 5 | 上下文长度 |
| 6 | output_len | 6 | 上下文长度 |
| 7 | total_input_len | 7 | |
| 8 | prefix_ratio | 8 | |
| 9 | prefix_tokens | 9 | |
| 10 | parallel_num | 10 | 并发 |
| 11 | epochs | 11 | |
| 12 | num_prompts | 12 | |
| 13 | seed | 13 | |
| 14 | n_success | 14 | |
| 15 | n_failed | 15 | |
| 16 | avg_input_tokens | 16 | 实际上下文长度 |
| 17 | avg_output_tokens | 17 | 实际上下文长度 |
| 18 | throughput_req_s | 32 | **吞吐（前移）** |
| 19 | throughput_tok_s | 33 | |
| 20 | input_throughput_tok_s | 34 | |
| 21 | prefill_effective_tok_s | 35 | |
| 22 | decode_effective_tok_s | 36 | |
| 23 | ttft_mean_ms | 37 | **TTFT（前移）** |
| 24 | ttft_p50_ms | 38 | |
| 25 | ttft_p90_ms | 39 | |
| 26 | ttft_p99_ms | 40 | |
| 27 | input_compliance | 18 | 其他（顺延） |
| 28 | output_compliance | 19 | |
| 29 | finish_reason_length_pct | 20 | |
| 30 | token_source | 21 | |
| 31 | avg_cached_tokens | 22 | |
| 32 | cache_hit_rate | 23 | |
| 33 | avg_gpu_kv_cache_usage | 24 | |
| 34 | peak_gpu_kv_cache_usage | 25 | |
| 35 | spec_decode_acceptance_rate | 26 | |
| 36 | spec_decode_system_efficiency | 27 | |
| 37 | spec_decode_num_drafts | 28 | |
| 38 | spec_decode_num_accepted_tokens | 29 | |
| 39 | spec_decode_num_draft_tokens | 30 | |
| 40 | spec_decode_per_position_acceptance_rates | 31 | |
| 41 | tpot_mean_ms | 41 | 尾部（不动相对序） |
| 42 | tpot_p50_ms | 42 | |
| 43 | tpot_p90_ms | 43 | |
| 44 | tpot_p99_ms | 44 | |
| 45 | e2el_mean_ms | 45 | |
| 46 | e2el_p50_ms | 46 | |
| 47 | e2el_p90_ms | 47 | |
| 48 | e2el_p99_ms | 48 | |
| 49 | audio_duration_s_total | 49 | |
| 50 | audio_duration_s_avg | 50 | |
| 51 | rtfx | 51 | |
| 52 | duration_s | 52 | |
| — | resource_monitor_*（追加） | — | 末尾，不动 |

`CSV_HEADERS_ZH` 按上表同一新顺序重排（每个 EN 列对应既有的 ZH 表头不变，仅随位置一起移动）。

### 3.3 不变项

- 列集合：52 列一个不少。
- 列内容：每行每列的值不变（`DictWriter` 按 key 写）。
- `resource_monitor` 追加逻辑、追加位置（末尾）不变。

## 4. Part 2 · 绘图中文修复（字体打进 bench-runner 镜像）

### 4.1 运行位置确认

postprocess（含 `bench_compare.aggregate_compare` → `_plot()`）在容器内运行：`auto_bench.py:build_postprocess_container_command` 用 `config.run.bench_image`（即 bench-runner 镜像）执行 `python auto_bench.py postprocess`。matplotlib 用的是镜像内安装的字体。因此**把 CJK 字体装进 `Dockerfile.bench-runner` 即可修**，git 仓库零增量，离线机只需重 save/load 一次镜像。

### 4.2 改动

**a. `Dockerfile.bench-runner`（line 10-13 的 apt-get）加一个字体包：**

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt openpyxl modelscope
```

选 `fonts-wqy-microhei`（文泉驿微米黑）：apt 安装约 5MB，覆盖简体中文，slim 镜像可接受。不选 `fonts-noto-cjk`（~300MB，过大）。matplotlib 通过 fontconfig 自动发现该字体，family 名为 `WenQuanYi Micro Hei`。

**b. `bench_compare.py::_plot()`（约 line 194-198）把"压制告警"换成"设字体"：**

删除：

```python
import warnings
warnings.filterwarnings(
    "ignore", message="Glyph .* missing from font", category=UserWarning
)
```

替换为：

```python
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
```

- fallback 链：镜像内命中 `WenQuanYi Micro Hei`；其他环境若装了 Noto 也能用；都没有则退回 DejaVu Sans（不崩，但中文仍是方框——属可接受的降级，主路径是容器）。
- `axes.unicode_minus = False`：避免负号渲染异常。
- 删掉 `warnings.filterwarnings`：设了真字体后缺字告警本就不会触发；保留它反而会掩盖将来真实的缺字问题。

**c. 生效前提：** 在有网机器上 `./build_bench_runner.sh` 重 build 镜像，再 `docker save` / `docker load` 搬到离线测试主机（即 `build_bench_runner.sh` 注释里描述的现有离线搬运流程）。

## 5. 受影响文件

| 文件 | 改动 |
|---|---|
| `vllm_standalone_bench/run_bench_multi.py` | 重排 `CSV_HEADERS` + `CSV_HEADERS_ZH`（仅顺序） |
| `vllm_standalone_bench/bench_compare.py` | `_plot()` 设 rcParams 字体，删 warnings 压制 |
| `vllm_standalone_bench/Dockerfile.bench-runner` | apt-get 加 `fonts-wqy-microhei` |
| `vllm_standalone_bench/tests/`（新增或扩充） | Part 1 列序断言 + Part 2 rcParams 断言 |

不碰：`auto_bench.py`、`bench_compare.py` 的聚合/对比逻辑、`resource_monitor.py`、`remote_docker.py`、compare 报告。

## 6. 验证

### Part 1

- 单测（新增 `tests/test_result_csv_headers.py`，目前仓库无 `CSV_HEADERS` 相关测试）：
  - 断言 `CSV_HEADERS` 在 `n_failed` 之后紧接全部 5 个 `throughput_*` 列、再全部 4 个 `ttft_*` 列；
  - 断言 `len(CSV_HEADERS) == len(CSV_HEADERS_ZH)` 且两者按位一一对应（防 ZH 漂移）；
  - 断言列集合（`set`）与改前完全相同（用一份写死的预期 52 列集合比对），证明只是重排。
- 回归：跑现有 `tests/test_bench_compare.py`、`tests/test_integration.py`、`tests/test_extract_row.py` 确认 `save_csv`/`save_xlsx` 与行抽取逻辑不破。

### Part 2

- 代码层断言：`rcParams['font.sans-serif'][0] == 'WenQuanYi Micro Hei'` 且 `axes.unicode_minus is False`。
- 端到端：重 build 镜像后跑一次 postprocess，肉眼确认 `plots/*.png` 中"并发数""输出吞吐"为正常汉字、非方框。

## 7. 风险

- **镜像体积**：`fonts-wqy-microhei` 约 +5MB，可接受。
- **旧镜像不生效**：已部署的离线机若不重 build + save/load，绘图仍是方框——需在交付时提示重 build。
- **ZH 表头漂移**：手动维护两列表易错，靠单测的等长+对应断言兜底。
