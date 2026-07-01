# Qwen3-ASR vllm_standalone_bench 兼容设计

日期：2026-07-01

## 背景

`.magic/vllm` 已具备 Qwen3-ASR 模型与 OpenAI Audio transcription API 的支持，但本仓库当前主用的 `vllm_standalone_bench` 自动化压测工程仍主要面向文本生成：

- `auto_bench.py` 只接受 `openai` 和 `openai-chat`。
- `run_bench_multi.py` 固定使用 `random` 文本数据集。
- `run_bench_serve.py` 的轻量 dataset shim 只支持 `random` 和 `sharegpt`。
- `vllm_bench/lib/endpoint_request_func.py` 已有 `openai-audio` 请求函数，但能力没有贯通到配置、数据集和自动化 Docker 链路。

目标是在不影响现有文本 benchmark 的前提下，让 `vllm_standalone_bench` 支持 Qwen3-ASR-1.7B 的自动化 ASR 压测。

## 目标

1. 支持 `backend=openai-audio`，请求 `/v1/audio/transcriptions`。
2. 支持内置 ASR 测试数据集，默认可离线运行小型 Qwen3-ASR smoke/性能 sanity。
3. 支持外部 ASR 数据集通过只读 volume 挂载。
4. 保留现有文本 benchmark 行为、配置格式和结果列语义。
5. 保持 standalone 定位：bench-runner 不依赖完整 vLLM 包。
6. 输出 ASR 相关指标，至少包含音频总时长、平均音频时长和 RTFx。

## 非目标

1. 不实现 WER/CER 精度评测。
2. 不把完整 LibriSpeech/TED-LIUM/GigaSpeech 等大型数据集打进镜像。
3. 不改 vLLM 服务镜像构建方式。
4. 不把 ASR prefix caching 与文本 prefix caching 混为一套指标。
5. 不兼容旧 `benchmark_tools`，本次仅覆盖 `vllm_standalone_bench`。

## 兼容性原则

ASR 是可选增量能力。默认路径保持现状：

- 旧配置文件无需新增字段。
- `backend=openai` 和 `backend=openai-chat` 继续使用文本 random/sharegpt 逻辑。
- `input_lens`、`output_lens`、`prefix_ratio` 在文本路径保持原语义。
- CSV/XLSX 现有列不删除、不重命名。
- Docker run 命令仅在 ASR profile 或外部数据集配置出现时追加新参数或挂载。

新增列对文本路径使用空值或 0 值，避免旧分析脚本因列缺失失败。

## 内置数据集

内置数据集命名为 `librispeech_test_clean_256`。

数据来源为 LibriSpeech `test-clean`，只抽取一部分片段。OpenSLR 页面标注 LibriSpeech 使用 CC BY 4.0，内置数据需要保留 license 和 attribution 文件。

目录结构：

```text
vllm_standalone_bench/assets/librispeech_test_clean_256/
  LICENSE.LibriSpeech.txt
  ATTRIBUTION.md
  manifest.json
  asr_smoke.jsonl
  audio/
    000001.flac
    ...
    000256.flac
```

抽样规则：

- 从 LibriSpeech `test-clean` 固定 seed 抽取。
- 样本数量：256 条。
- 排除过短音频，最小时长为 5 秒。
- 最大时长建议限制为 30 秒，避免内置 smoke 过慢。
- 按时长分桶抽样，避免全是短句：
  - 5 到 10 秒：约 35%。
  - 10 到 20 秒：约 45%。
  - 20 到 30 秒：约 20%。
- 体积目标：优先控制在 100 MB 以内。如果实际超过该目标，允许在实现阶段下调到 192 条，但必须在 `manifest.json` 中记录实际样本数。

JSONL 格式：

```jsonl
{"prompt": "", "audio": "/opt/vllm_standalone_bench/assets/librispeech_test_clean_256/audio/000001.flac", "output_tokens": 128, "reference": "TRANSCRIPT TEXT"}
```

字段说明：

- `audio`：容器内音频路径。
- `prompt`：可选 transcription prompt，默认空字符串。
- `output_tokens`：可选样本级输出上限。profile 中的 `output_lens` 优先。
- `reference`：保留转写文本，第一版只用于人工排查和后续精度评测扩展。

内置数据只用于自动化链路验证和小型性能 sanity，不作为正式 ASR 精度 benchmark。

## 配置设计

文本配置保持现状。ASR profile 使用 `backend=openai-audio` 触发。

示例：

```json
{
  "bench_profiles": [
    {
      "name": "asr_smoke",
      "backend": "openai-audio",
      "dataset_name": "custom_audio",
      "output_lens": [128],
      "parallel_nums": [1, 4, 8, 16],
      "epochs": 8,
      "language": "en",
      "warmup_requests": 0
    }
  ]
}
```

新增字段：

- `dataset_name`：ASR 第一版只支持 `custom_audio`。未设置且 backend 为 `openai-audio` 时默认 `custom_audio`。
- `dataset_path`：可选。未设置时使用内置 `librispeech_test_clean_256/asr_smoke.jsonl`。
- `language`：可选，默认 `en`。作为 profile 级语言传给 transcription API。

外部数据集示例：

```json
{
  "mounts": {
    "models": "/Resource_Planning_Tool/model",
    "datasets": "/Resource_Planning_Tool/datasets"
  },
  "bench_profiles": [
    {
      "name": "asr_external",
      "backend": "openai-audio",
      "dataset_name": "custom_audio",
      "dataset_path": "/datasets/asr/custom.jsonl",
      "output_lens": [128],
      "parallel_nums": [1, 8],
      "epochs": 16,
      "language": "en"
    }
  ]
}
```

`mounts.datasets` 是可选字段。只有外部 `dataset_path` 位于 `/datasets` 下时才需要配置并挂载。

## 运行语义

ASR 不是一次性静态 batch 推理。请求仍按现有并发压测语义运行：

```text
num_prompts = parallel_num * epochs
```

例如 `parallel_nums=[16]`、`epochs=8`，会准备 128 个音频请求，客户端最多保持 16 个请求同时在途。vLLM server 端按其调度器动态批处理这些请求，Qwen3-ASR 的音频预处理和 audio encoder 会进入每个请求的前处理/编码阶段，后续文本生成进入 vLLM 连续调度和 decode 流程。

样本选择：

- 每个配置根据 seed 对数据集确定性 shuffle。
- 取前 `num_prompts` 条。
- 如果请求数超过数据集条数，则循环复用，保持顺序可复现。

ASR 路径中：

- `output_lens` 映射到 `max_completion_tokens`。
- `input_lens` 不再代表输入 token 长度，ASR 输入规模由音频时长决定。
- `prefix_ratio` 不参与 ASR prompt 构造。

## 请求链路

`run_bench_multi.py`：

- `backend=openai-audio` 时 endpoint 自动设为：
  - base-url 模式：`/audio/transcriptions`
  - host/port 模式：`/v1/audio/transcriptions`
- 设置 `dataset_name=custom_audio`。
- 透传 `dataset_path` 和 `language`。

`run_bench_serve.py`：

- dataset shim 增加 `custom_audio`。
- 支持读取 JSONL。
- 生成 `SampleRequest`，其中 `multi_modal_data={"audio_path": "/path/to/file.flac"}`。

`vllm_bench/lib/endpoint_request_func.py`：

- `async_request_openai_audio` 同时支持：
  - `multi_modal_content["audio"]`：内存数组和采样率。
  - `multi_modal_content["audio_path"]`：直接 multipart 上传文件。
- 记录 `input_audio_duration`。
- payload 包含 `language`，并允许 `extra_body` 覆盖。

## Docker 自动化

`Dockerfile.bench-runner`：

- COPY `assets/librispeech_test_clean_256` 到 `/opt/vllm_standalone_bench/assets/librispeech_test_clean_256`。
- 安装 `soundfile`。
- 如运行环境需要，安装系统库 `libsndfile1`。

`auto_bench.py`：

- `SUPPORTED_BACKENDS` 增加 `openai-audio`。
- `MountConfig` 增加可选 `datasets`。
- `BenchProfile` 增加 `dataset_name`、`dataset_path`、`language`。
- 构建 bench-runner 命令时：
  - 对 ASR profile 追加 `--dataset-name custom_audio`。
  - 未指定 `dataset_path` 时使用内置默认路径。
  - 指定外部 `/datasets/...` 路径时追加 `-v <mounts.datasets>:/datasets:ro`。
  - 追加 `--language <language>`。

旧文本 profile 不追加这些参数。

## 结果指标

保留所有现有 CSV/XLSX 列。新增列：

- `dataset_name`
- `audio_duration_s_avg`
- `audio_duration_s_total`
- `rtfx`
- `asr_language`

RTFx 定义：

```text
rtfx = total_audio_duration_s / benchmark_duration_s
```

含义：

- `rtfx > 1`：整体处理速度快于实时。
- `rtfx = 1`：接近实时。
- `rtfx < 1`：慢于实时。

文本 benchmark 的新增 ASR 列为 0 或空字符串。

## 错误处理

- `backend=openai-audio` 且 `dataset_name` 不是 `custom_audio`：配置错误。
- `dataset_path` 不存在：配置错误，提示内置默认路径和外部挂载方式。
- JSONL 缺少 `audio`：数据错误。
- 音频路径不存在：数据错误。
- `prefix_ratio` 出现在 ASR profile：忽略并记录 warning。
- `input_lens` 出现在 ASR profile：忽略并记录 warning。
- 数据集样本少于请求数：循环复用，不报错。
- 单条请求失败：沿用现有 failed 统计，不中断整组。

## 测试计划

单元测试：

- `auto_bench.py`
  - 接受 `openai-audio` backend。
  - ASR profile 默认 `dataset_name=custom_audio`。
  - 未指定 `dataset_path` 时使用内置默认路径。
  - 外部 `/datasets/...` 路径会要求 `mounts.datasets` 并生成只读 volume。
  - 旧文本配置构造出的 Docker 命令不新增 ASR 参数。

- `run_bench_serve.py`
  - `custom_audio` JSONL 能生成 `SampleRequest`。
  - 支持 `audio_path`、`prompt`、`output_tokens`。
  - 数据不够时循环复用。

- `vllm_bench/lib/endpoint_request_func.py`
  - `openai-audio` 支持 `audio_path` multipart 上传。
  - endpoint 后缀校验仍限制 `transcriptions` 或 `translations`。

- `run_bench_multi.py`
  - `openai-audio` 自动设置 `/audio/transcriptions`。
  - ASR 路径不调用 random prompt 生成。
  - CSV 新列存在。
  - 文本路径测试继续通过。

验证命令：

```bash
python3 -m pytest vllm_standalone_bench/tests -q
python3 vllm_standalone_bench/auto_bench.py run \
  --config vllm_standalone_bench/configs/auto_bench.qwen3_asr_1_7b.smoke.json \
  --run-id asr_dry_run \
  --dry-run
```

真实 Qwen3-ASR smoke 需要本地已有 Qwen3-ASR-1.7B 模型和支持音频的 vLLM 镜像。

## 文档与样例

新增：

- `configs/auto_bench.qwen3_asr_1_7b.smoke.json`
- README 中的 Qwen3-ASR 运行说明。
- 内置数据集 attribution 和 license 说明。

样例配置应默认使用内置数据集，不要求用户先准备外部音频。

## 风险与缓解

- 镜像体积增加：用 256 条短音频并设 100 MB 目标上限控制。
- 音频依赖缺失：在 runner 镜像显式安装 `soundfile` 和 `libsndfile1`。
- vLLM audio 支持不完整：真实 smoke 前通过 dry-run 和接口 ready check 分开验证。
- 旧结果消费方不识别新增列：只追加列，不删除或重命名现有列。
- ASR token 指标和文本 token 指标不可直接比较：文档中强调 ASR 主要看 RTFx、E2E、TTFT 和请求吞吐。

## 验收标准

1. 旧 `vllm_standalone_bench` 文本测试全部通过。
2. 旧 smoke 配置 dry-run 生成的核心命令不出现 ASR 参数。
3. 新 Qwen3-ASR smoke 配置 dry-run 包含：
   - `--backend openai-audio`
   - `--dataset-name custom_audio`
   - 默认内置 `dataset_path`
   - `/audio/transcriptions` endpoint 由 runner 自动选择
4. bench-runner 镜像内包含内置 ASR 数据集与 license/attribution。
5. ASR 结果 CSV/XLSX 包含新增 ASR 指标列。
6. 使用本地 Qwen3-ASR-1.7B 与支持音频的 vLLM 镜像时，smoke 能完成并产出 `result.csv`。
