# KV Cache Usage CSV 设计

## 背景

MTP benchmark 日志中会出现 vLLM runtime 行：

```text
Engine 000: Avg prompt throughput: 4379.7 tokens/s, Avg generation throughput: 14.3 tokens/s, Running: 32 reqs, Waiting: 0 reqs, GPU KV cache usage: 9.8%, Prefix cache hit rate: 62.7%
```

当前自动化结果 CSV 已有 `cache_hit_rate`，但它表示 prefix cache token 命中率，来源是 OpenAI usage 中的 `cached_tokens`，不是 `GPU KV cache usage`。因此需要新增单独的 GPU KV cache usage 指标列，避免把 prefix cache 命中率和 GPU KV cache 占用率混用。

本次目标是把每轮 benchmark 期间的 GPU KV cache usage 写入 CSV/XLSX，并同时修正 SpecDecode accepted/draft token 字段名兼容问题。

## 目标

- CSV 和 XLSX 新增两列：
  - `avg_gpu_kv_cache_usage`
  - `peak_gpu_kv_cache_usage`
- 两列单位为百分比，范围按 `0-100` 输出。例如日志中的 `GPU KV cache usage: 9.8%` 写为 `9.8`。
- 每轮 benchmark 独立采样和汇总，不把不同 input/output/parallel 配置混在一起。
- metrics 不可用、鉴权失败、没有 GPU KV cache 指标时不让 benchmark 失败，结果列默认 `0.0`。
- 继续保留现有 `avg_cached_tokens` 和 `cache_hit_rate`，语义不变。
- 修正 SpecDecode 字段别名兼容：
  - `serve.py` 当前返回 `spec_decode_accepted_tokens` 和 `spec_decode_draft_tokens`
  - `run_bench_multi.py` 当前 CSV 提取 `spec_decode_num_accepted_tokens` 和 `spec_decode_num_draft_tokens`
  - CSV 提取应兼容两组字段名，避免 accepted/draft token 列落 `0`

## 非目标

- 不解析 vLLM 容器日志来生成 CSV。
- 不改变 benchmark 调度、并发模型、warmup 行为。
- 不改变现有 prefix cache hit rate 的计算方式。
- 不接入 Prometheus 服务端或外部监控系统。

## 方案选择

### 方案 A：解析 vLLM 日志

优点是与用户肉眼看到的日志行完全一致。

缺点是 bench runner 当前不直接消费 vLLM log；同一个 `vllm.log` 中混有 warmup、多组 benchmark 和 cooldown，按时间切分容易错。Docker log 时间戳、容器 clock、bench runner clock 之间也会引入边界误差。

### 方案 B：benchmark 前后各拉一次 `/metrics`

优点是改动小。

缺点是 GPU KV cache usage 是瞬时 gauge，前后各一次无法代表运行期间的峰值，也无法得出可靠平均值。高并发短时峰值容易被漏掉。

### 方案 C：benchmark 期间后台轮询 `/metrics`

推荐方案。它复用现有 `/metrics` 鉴权修复和 base URL 归一逻辑，在每轮 benchmark 生命周期内后台采样。每次 scrape 解析 GPU KV cache usage gauge，单次采样取所有 engine/worker 的最大值；一轮结束后计算平均值和峰值。

该方案直接反映当前 benchmark 期间服务端暴露的 runtime metrics，边界清晰，且不依赖日志文件位置。

## 数据来源

`serve.py` 已经通过 `fetch_spec_decode_metrics(base_url, session, extra_headers)` 拉取 `/metrics` 并解析 `vllm:spec_decode*`。本次在同一 HTTP scrape 基础上解析 GPU KV cache usage。

支持以下 Prometheus metric 名：

- `vllm:gpu_cache_usage_perc`
- `vllm:kv_cache_usage_perc`
- `ray_vllm_kv_cache_usage_perc`

解析规则：

- 忽略注释行和空行。
- metric 名取 labels 前的部分，例如 `vllm:gpu_cache_usage_perc{engine="0"} 9.8` 的 metric 名是 `vllm:gpu_cache_usage_perc`。
- 只接受有限数值。
- 单次 scrape 如果有多个 engine/worker 值，取最大值作为该时刻的 GPU KV cache usage。
- 如果只看到 `0.098` 这类比例值，输出前转换为 `9.8`；如果看到 `9.8`，按百分比直接输出。

## 采样生命周期

在 `benchmark()` 内：

1. 创建 `RuntimeMetricsSampler`。
2. benchmark 请求发出前启动 sampler。
3. sampler 立即 scrape 一次，然后每 `1.0s` scrape 一次。
4. `asyncio.gather(*tasks)` 完成后停止 sampler。
5. 停止时再 scrape 一次，覆盖很短的 benchmark。
6. sampler 返回：
   - `avg_gpu_kv_cache_usage`
   - `peak_gpu_kv_cache_usage`

采样失败处理：

- 单次 scrape 失败只忽略该样本。
- 全部 scrape 都失败时，avg 和 peak 都是 `0.0`。
- `/metrics` 返回非 200 时不抛错。
- 继续使用 `extra_headers`，确保开启 API key 的服务不会对 `/metrics` 报 401。

## 输出字段

`serve.py` result dict 新增：

```python
{
    "avg_gpu_kv_cache_usage": 9.1,
    "peak_gpu_kv_cache_usage": 12.7,
}
```

`run_bench_multi.py` CSV/XLSX 新增英文表头：

```python
"avg_gpu_kv_cache_usage",
"peak_gpu_kv_cache_usage",
```

中文表头：

```python
"平均GPU KV缓存占用率(%)",
"峰值GPU KV缓存占用率(%)",
```

列位置放在现有缓存列后面：

```text
avg_cached_tokens, cache_hit_rate, avg_gpu_kv_cache_usage,
peak_gpu_kv_cache_usage, spec_decode_acceptance_rate,
spec_decode_system_efficiency, spec_decode_num_drafts,
spec_decode_num_accepted_tokens, spec_decode_num_draft_tokens,
spec_decode_per_position_acceptance_rates
```

XLSX “指标说明”页增加这两项说明。

## SpecDecode 字段别名修正

当前 `serve.py` 返回：

```python
result["spec_decode_draft_tokens"]
result["spec_decode_accepted_tokens"]
```

当前 CSV 提取读取：

```python
spec_decode_num_draft_tokens
spec_decode_num_accepted_tokens
```

设计为在 `_extract_row()` 中增加兼容 helper：

```python
def _i_any(*keys: str, default: int = 0) -> int:
    for key in keys:
        if result.get(key) is not None:
            return int(result.get(key) or default)
    return default
```

CSV 输出列名保持不变：

- `spec_decode_num_accepted_tokens`
- `spec_decode_num_draft_tokens`

取值兼容：

- accepted 优先读 `spec_decode_num_accepted_tokens`，再读 `spec_decode_accepted_tokens`
- draft 优先读 `spec_decode_num_draft_tokens`，再读 `spec_decode_draft_tokens`

## 测试计划

- `test_serve_metrics.py`
  - fake `/metrics` 文本包含 `vllm:gpu_cache_usage_perc 9.8` 时能解析为 `9.8`
  - 多个 engine/worker 值时单次 scrape 取最大值
  - 比例值 `0.098` 转换为 `9.8`
  - metrics 缺失时返回空样本，不影响 SpecDecode 解析
  - sampler 汇总多个样本得到 avg 和 peak

- `test_extract_row.py`
  - `_extract_row()` 输出 `avg_gpu_kv_cache_usage` 和 `peak_gpu_kv_cache_usage`
  - 缺少 GPU KV cache 字段时两列默认 `0.0`
  - CSV 英文表头和中文表头都包含新增列
  - `spec_decode_accepted_tokens` 能填充 `spec_decode_num_accepted_tokens`
  - `spec_decode_draft_tokens` 能填充 `spec_decode_num_draft_tokens`

- 全量验证：
  - `pytest -q vllm_standalone_bench/tests`
  - `git diff --check`

## 兼容性

- 旧 CSV 消费方仍可读取原有列；新增列只追加，不删除旧列。
- 没有 `/metrics` 或没有 GPU KV cache 指标的后端会输出 `0.0`，不会失败。
- `cache_hit_rate` 继续表示 prefix cache hit rate，不改名、不改语义。
- 新增 sampler 只在 benchmark 期间运行，不改变请求路径和请求负载。
