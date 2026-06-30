# Warmup 固定并发预热设计（Fixed-Concurrency Warmup）

## Status

Ready for implementation planning.

## Context

`vllm_standalone_bench` 在 4 并发小样本档观测到 TTFT 异常尖峰：例如 4096 / 4 并发档 TTFT 均值 3999ms、P90 10198ms，反而高于 8 / 16 并发档；而同一组里 E2EL、req/s、TPOT 全部正常单调。大输入长度（16384+）的 4 并发档则完全正常。

根因诊断（详见 memory `bench-ttft-low-concurrency-spike`）：

1. **inf + Semaphore 固定并发模型**：`run_bench_multi.py:143` 强制 `request_rate = float('inf')`，每组一次性 `create_task` 全部 `parallel × epochs` 个任务，靠 `Semaphore(max_concurrency)` 限并发。首批 N 个请求在 t≈0 同时到达 vLLM。
2. **首批冷启动**：4 并发是该输入组里首个"多请求重叠"档，vLLM 的 continuous-batching 多请求 prefill 合并、KV cache 多 block 分配、新 batch 形状下 CUDA graph 首次捕获等一次性开销全部压在首批请求上。
3. **warmup 失效**：`run_bench_multi.py:614-620` 仅第一个配置 warmup，且 `num_warmups` 默认 1——无论目标并发多大只发 1 个请求，对并发档的多请求重叠路径毫无预热作用。
4. **小样本放大**：4 并发 × 3 epochs = 12 个样本，P90(12)≈第 11 大值，1-2 个慢请求即把 P90 / 均值拉爆。

## Goals

- 消除小并发档（尤其 4 并发）的 TTFT 首批冷启动尖峰。
- 以恒定、可控的成本预热，**不随目标并发度或输入长度爆炸**（长上下文大并发档成本不能失控）。
- 向后兼容：现有配置与 CLI 默认行为不变。
- 保留"按 `parallel_num` 精确控并发"的测量语义（不退回 Poisson 速率模型）。

## Non-Goals

- 不改请求到达模型（保持 inf + Semaphore，不走 Poisson）。
- 不解决小样本统计噪声本身（不强制改 epochs / num_prompts，留用户自调）。
- 不做"每输入组 / 每档预热"（YAGNI，留作未来 opt-in；当前数据表明全局一次已覆盖问题档）。
- 不改 prefix-caching 启用时 warmup 单一 prompt 命中缓存的问题（当前对比配置未启用 prefix cache）。

## Design

### 核心思路

**固定并发预热**：整个测试仅在第一个配置前 warmup 一次，用 **固定并发 W（默认 4）× 首个配置的输入长度 × 输出长度 L（默认 128）**，发 W 个请求凑齐"一波满并发"。

为什么固定 W=4 有效：

- 首批抖动只在 4 并发这种小样本档显形；8 / 16 / 24 / 32 并发档样本多（24 / 48 / 72 / 96），首批 1-2 个慢请求被稀释，本来就正常。
- 多请求并发重叠路径（调度器、显存池、CUDA graph）用任意 ≥2 并发即可触发，W=4 足够，**无需复刻目标并发度 N**。
- 第一个输入组（通常 4096）恰好是出问题档，全局 warmup 用的就是它的输入，精准覆盖。

### 为什么必须解耦 warmup 的并发与输出

全局 warmup 发生在第一个配置（通常是 `parallel=1`）。现有 `serve.py`：

- `warmup_semaphore = asyncio.Semaphore(max_concurrency)`（`serve.py:742`）→ 此时 `max_concurrency=1`，warmup 被卡成 1 并发，固定 W 失效。
- warmup 的 `test_input.output_len = input_requests[0].expected_output_len`（`serve.py:706-712`）→ 该档输出（如 1024），warmup 主要时间耗在 decode；而首批抖动是 **prefill 阶段**的事，decode 无需预热。

故必须让 warmup 的并发与输出独立于该档配置。

## Changes

### 1. `vllm_bench/serve.py`（两处小改）

`main_async` 新增两个可选参数：

- `warmup_concurrency: int | None = None` — warmup 信号量大小。`None` 时用 `max_concurrency`（向后兼容）。
- `warmup_output_len: int | None = None` — warmup 请求的输出长度。`None` 时用该档 `output_len`（向后兼容）。

改动点：

- `serve.py:741-742`：`warmup_semaphore = Semaphore(warmup_concurrency if warmup_concurrency else max_concurrency)`。
- `serve.py:706-718`：构造 warmup 的 `test_input` 时，`output_len = warmup_output_len if warmup_output_len else test_output_len`。

warmup 仍用 `input_requests[0].prompt`（该档实际输入），仅 `output_len` 被覆盖；结果仍独立 `gather` 后丢弃，不计入统计。

### 2. `run_bench_multi.py`

CLI 新增：

- `--warmup-concurrency`（默认 `None`）
- `--warmup-output-len`（默认 `None`）

`build_base_config` 透传这两个参数到 serve cfg。第一次配置（`is_first_run`）时：

- 若 `warmup_concurrency` 已设：`cfg.num_warmups = warmup_concurrency`（凑齐一波满并发）；否则保持现有 `warmup_requests` 逻辑。
- 透传 `cfg.warmup_concurrency` / `cfg.warmup_output_len`。

默认全 `None` → 行为完全等同现状（向后兼容）。

### 3. `auto_bench.py` 配置 schema

`bench_profile` 新增可选字段：

- `warmup_concurrency: int | None`
- `warmup_output_len: int | None`

透传给 `run_bench_multi`。`configs/auto_bench.qwen2_5_1_5b.sglang_compare.json` 与 `smoke` 配置默认带 `warmup_concurrency=4`、`warmup_output_len=128`，让对比基准默认即修好。

## Configuration

auto_bench 配置示例：

```json
"bench_profiles": [{
  "name": "smoke",
  "warmup_concurrency": 4,
  "warmup_output_len": 128,
  ...
}]
```

CLI 直跑：

```bash
python run_bench_multi.py ... --warmup-concurrency 4 --warmup-output-len 128
```

## Trade-offs / Known Limitations

- **只预热首个输入长度**：后续输入长度（8192+）的 4 并发档不直接预热。但实测这些档本无尖峰（长 prefill 稳态占比大，首批冷启动被稀释）。若未来某长上下文小并发档冒尖峰，可 opt-in "每输入组预热"。
- **prefix-caching 启用时预热打折**：warmup 的 W 个请求同用 `input_requests[0]`，开启 prefix cache 时第 2 个起命中缓存。当前对比配置未启用，无影响。
- **不解决小样本长尾**：12 样本的 P90 仍有统计噪声，只是首批不再异常拉高。如需更稳，配合加大 `epochs`。

## Cost

全局一次 warmup = W 个并发 × 首个配置输入的 prefill + L=128 token 的 decode。

- 4096 输入：约 2-3 秒。
- 相比"每档用 `parallel_num` 完整预热"省一个数量级（后者长上下文大并发档单档即可 100s+）。

## Test Strategy

- **单元测试**
  - `serve.py`：`warmup_concurrency` / `warmup_output_len` 为 `None` 时回退到该档值（行为不变）；非 `None` 时正确覆盖信号量大小与 warmup `test_input.output_len`。
  - `run_bench_multi.py`：默认 `None` 时 `build_base_config` 不改变现有 cfg；CLI 传入时正确透传；首个配置 `num_warmups` 随 `warmup_concurrency` 设置。
  - `auto_bench.py`：新字段解析、缺省值、透传到 `run_bench_multi` 命令。
- **集成测试**：dry-run 验证 warmup 命令拼装（首个配置含 warmup，后续配置 `num_warmups=0`）。
- **回归**：现有 `tests/test_auto_bench.py`、`tests/test_shell_scripts.py` 等不破坏。

## First Implementation Scope

1. `serve.py` 加两参数 + 两处改动 + 单测。
2. `run_bench_multi.py` 加 CLI + 透传 + 单测。
3. `auto_bench.py` 配置 schema + 透传 + 单测。
4. 更新 `sglang_compare` / `smoke` 配置默认值。
5. README 补一段说明。
