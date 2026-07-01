# MTP 鉴权与真实数据集实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修复 vLLM standalone bench 在拉取 `/metrics` 时丢失鉴权头导致的 401；把 MTP/Spec Decode 接受率指标写入 CSV/XLSX；通过 `bench_profiles[].dataset` 增加内置真实风格 MTP 数据集，并保持现有 `input_lens`、`output_lens`、`parallel_nums`、`epochs` 矩阵配置兼容。

**架构：** 保持当前三层入口不变：`auto_bench.py` 解析 JSON 自动化配置并生成 `run_bench_multi.py` 命令；`run_bench_multi.py` 负责矩阵展开、CSV/XLSX 汇总和 vLLM benchmark 参数转换；`run_bench_serve.py` 继续 shim `vllm.benchmarks.datasets`，并新增 `builtin_mtp_chat` 数据集路由。MTP 数据集是离线内置的 chat prompt 生成器，按 `input_lens` 作为 token bucket 目标筛选样本，`output_lens` 继续映射为 `max_tokens`。

**技术栈：** Python 3、pytest、vLLM benchmark 兼容 shim、标准库 `dataclasses`/`json`/`random`/`pathlib`。

---

## 设计约束

- `dataset` 字段只新增到 `bench_profiles[]`，缺省时保持当前随机数据集行为。
- `dataset.name = "builtin_mtp_chat"` 时，`input_lens` 不失效，而是从“随机 prompt 精确长度”变为“真实 prompt token bucket 目标”。
- `dataset.length_policy = "bucket"` 时使用 `input_len_tolerance` 计算 `[target * (1 - tolerance), target * (1 + tolerance)]`。
- `output_lens` 继续控制每次请求的 `max_tokens`，用于近似真实生成长度压力。
- `parallel_nums`、`epochs`、`cross_product` 保持现有矩阵语义。
- `builtin_mtp_chat` 必须要求 tokenizer；如果没有传 `--tokenizer`，在启动 bench 前失败并给出清晰错误。
- `/v1/metrics` 请求必须复用业务请求同一组 `extra_headers`，包括 `Authorization=Bearer local-bench-key`。
- `serve_profiles[].args` 中出现 `speculative-config.` 但缺少前导 `--` 时，配置解析阶段报错，避免运行到 vLLM 后才失败。

## 数据集配置格式

```json
{
  "name": "MTP_real_bucket",
  "backend": "vllm",
  "dataset": {
    "name": "builtin_mtp_chat",
    "length_policy": "bucket",
    "input_len_tolerance": 0.2,
    "on_bucket_shortage": "error",
    "sampling": "shuffle"
  },
  "input_lens": [1024, 2048, 4096],
  "output_lens": [512],
  "parallel_nums": [1, 4, 8],
  "epochs": 2
}
```

## 任务 0：确认基线

**文件：**
- `vllm_standalone_bench/tests/`

**步骤：**
- [ ] 在 worktree 根目录确认当前分支是 `feat/mtp-auth-real-data`：

  ```bash
  git status --short --branch
  ```

- [ ] 运行基线测试：

  ```bash
  pytest -q vllm_standalone_bench/tests
  ```

**通过标准：**
- 测试通过。
- 如果基线失败，记录失败用例和堆栈，先修复阻塞实现的同域问题。

---

## 任务 1：修复 `/metrics` 鉴权 401

**文件：**
- `vllm_standalone_bench/vllm_bench/serve.py`
- `vllm_standalone_bench/tests/test_serve_metrics.py`

**步骤：**
- [ ] 先写失败测试，覆盖 `fetch_spec_decode_metrics()` 会把 header 传给 session：

  ```python
  import asyncio

  def test_fetch_spec_decode_metrics_passes_headers():
      class FakeResponse:
          status = 200

          async def text(self):
              return "\n".join(
                  [
                      "vllm:spec_decode_num_drafts_total 4",
                      "vllm:spec_decode_num_accepted_tokens_total 3",
                      "vllm:spec_decode_num_draft_tokens_total 4",
                  ]
              )

          async def __aenter__(self):
              return self

          async def __aexit__(self, exc_type, exc, tb):
              return False

      class FakeSession:
          def __init__(self):
              self.calls = []

          def get(self, url, headers=None):
              self.calls.append((url, headers))
              return FakeResponse()

      session = FakeSession()
      headers = {"Authorization": "Bearer local-bench-key"}

      metrics = asyncio.run(
          serve.fetch_spec_decode_metrics("http://127.0.0.1:8000/v1", session, headers)
      )

      assert session.calls == [
          ("http://127.0.0.1:8000/metrics", headers),
      ]
      assert metrics.num_drafts == 4
      assert metrics.num_accepted_tokens == 3
      assert metrics.num_draft_tokens == 4
  ```

- [ ] 修改 `fetch_spec_decode_metrics()` 签名，新增可选 `extra_headers` 参数：

  ```python
  async def fetch_spec_decode_metrics(
      base_url: str,
      session: aiohttp.ClientSession,
      extra_headers: Optional[dict[str, str]] = None,
  ) -> SpecDecodeMetrics:
      metrics_url = base_url.rstrip("/v1").rstrip("/") + "/metrics"
      try:
          async with session.get(metrics_url, headers=extra_headers) as response:
              if response.status != 200:
                  return None
              text = await response.text()
              # 继续使用函数内现有 Prometheus 文本解析逻辑解析 text。
  ```

- [ ] 在 `benchmark()` 里把现有 `extra_headers` 传给两次 metrics 拉取：

  ```python
  spec_decode_metrics_before = await fetch_spec_decode_metrics(
      base_url, session, extra_headers
  )
  benchmark_metrics = await run_requests_for_benchmark()
  spec_decode_metrics_after = await fetch_spec_decode_metrics(
      base_url, session, extra_headers
  )
  ```

- [ ] 运行定向测试：

  ```bash
  pytest -q vllm_standalone_bench/tests/test_serve_metrics.py
  ```

- [ ] 提交：

  ```bash
  git add vllm_standalone_bench/vllm_bench/serve.py vllm_standalone_bench/tests/test_serve_metrics.py
  git commit -m "fix: pass auth headers to vllm metrics"
  ```

**通过标准：**
- 新测试先失败后通过。
- `/v1/metrics` 的 URL 保持从 `/v1` base URL 正确归一到 `/metrics`。
- 无 API key 时 `headers=None` 仍可正常工作。

---

## 任务 2：把 MTP/Spec Decode 指标写入 CSV/XLSX

**文件：**
- `vllm_standalone_bench/run_bench_multi.py`
- `vllm_standalone_bench/tests/test_extract_row.py`

**步骤：**
- [ ] 先写失败测试，覆盖 `_extract_row()` 输出 MTP 指标：

  ```python
  def test_extract_row_includes_spec_decode_metrics():
      result = _result(
          spec_decode_acceptance_rate=75.0,
          spec_decode_system_efficiency=0.82,
          spec_decode_num_drafts=12,
          spec_decode_num_accepted_tokens=9,
          spec_decode_num_draft_tokens=12,
          spec_decode_per_position_acceptance_rates=[90.0, 60.0],
      )

      row = multi._extract_row(
          result,
          input_len=1024,
          output_len=512,
          parallel_num=4,
          epoch=1,
          prefix_len=0,
          random_range_ratio=0.0,
      )

      assert row["spec_decode_acceptance_rate"] == 75.0
      assert row["spec_decode_system_efficiency"] == 0.82
      assert row["spec_decode_num_drafts"] == 12
      assert row["spec_decode_num_accepted_tokens"] == 9
      assert row["spec_decode_num_draft_tokens"] == 12
      assert row["spec_decode_per_position_acceptance_rates"] == "[90.0,60.0]"
  ```

- [ ] 写表头测试，确保英文和中文表头都有新增列：

  ```python
  def test_csv_headers_include_spec_decode_metrics():
      for key in [
          "spec_decode_acceptance_rate",
          "spec_decode_system_efficiency",
          "spec_decode_num_drafts",
          "spec_decode_num_accepted_tokens",
          "spec_decode_num_draft_tokens",
          "spec_decode_per_position_acceptance_rates",
      ]:
          assert key in multi.CSV_HEADERS
          assert key in multi.CSV_HEADERS_ZH
  ```

- [ ] 修改 `CSV_HEADERS`、`CSV_HEADERS_ZH`，在缓存命中列之后追加：

  ```python
  "spec_decode_acceptance_rate",
  "spec_decode_system_efficiency",
  "spec_decode_num_drafts",
  "spec_decode_num_accepted_tokens",
  "spec_decode_num_draft_tokens",
  "spec_decode_per_position_acceptance_rates",
  ```

- [ ] 修改 `_extract_row()`，对缺省结果保持兼容：

  ```python
  import json

  def _json_list(key: str) -> str:
      value = result.get(key) or []
      return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

  row.update(
      {
          "spec_decode_acceptance_rate": _f("spec_decode_acceptance_rate"),
          "spec_decode_system_efficiency": _f("spec_decode_system_efficiency"),
          "spec_decode_num_drafts": _i("spec_decode_num_drafts"),
          "spec_decode_num_accepted_tokens": _i("spec_decode_num_accepted_tokens"),
          "spec_decode_num_draft_tokens": _i("spec_decode_num_draft_tokens"),
          "spec_decode_per_position_acceptance_rates": _json_list(
              "spec_decode_per_position_acceptance_rates"
          ),
      }
  )
  ```

- [ ] 运行定向测试：

  ```bash
  pytest -q vllm_standalone_bench/tests/test_extract_row.py
  ```

- [ ] 提交：

  ```bash
  git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/tests/test_extract_row.py
  git commit -m "feat: export spec decode metrics"
  ```

**通过标准：**
- 旧 result 没有 MTP 字段时仍能导出，数值列默认 `0`，列表列默认 `[]`。
- CSV 和 XLSX 共用表头逻辑，新增字段同时出现在两个输出格式。

---

## 任务 3：自动化配置支持 `dataset` 并增加 speculative 参数校验

**文件：**
- `vllm_standalone_bench/auto_bench.py`
- `vllm_standalone_bench/tests/test_auto_bench.py`

**步骤：**
- [ ] 先写失败测试，覆盖 `bench_profiles[].dataset` 解析和命令生成：

  ```python
  def test_build_bench_run_command_passes_builtin_dataset(tmp_path):
      config = minimal_config(tmp_path)
      config["bench_profiles"][0]["dataset"] = {
          "name": "builtin_mtp_chat",
          "length_policy": "bucket",
          "input_len_tolerance": 0.2,
          "on_bucket_shortage": "error",
          "sampling": "shuffle",
      }
      parsed = auto_bench.load_config(write_config(tmp_path, config))

      cmd = auto_bench.build_bench_run_command(
          parsed,
          parsed.models[0],
          parsed.serve_profiles[0],
          parsed.bench_profiles[0],
          Path("/tmp/run"),
      )

      assert value_after(cmd, "--dataset") == "builtin_mtp_chat"
      assert value_after(cmd, "--dataset-length-policy") == "bucket"
      assert value_after(cmd, "--dataset-input-len-tolerance") == "0.2"
      assert value_after(cmd, "--dataset-on-bucket-shortage") == "error"
      assert value_after(cmd, "--dataset-sampling") == "shuffle"
  ```

- [ ] 写失败测试，覆盖缺少 `--` 的 speculative 参数会在配置解析时报错：

  ```python
  def test_serve_profile_rejects_speculative_config_without_dashes(tmp_path):
      config = minimal_config(tmp_path)
      config["serve_profiles"][0]["args"] = [
          "speculative-config.num_speculative_tokens",
          "1",
      ]

      with pytest.raises(auto_bench.ConfigError, match="--speculative-config"):
          auto_bench.load_config(write_config(tmp_path, config))
  ```

- [ ] 在 `auto_bench.py` 增加数据结构：

  ```python
  @dataclass(frozen=True)
  class DatasetConfig:
      name: str
      length_policy: str = "exact"
      input_len_tolerance: float = 0.2
      on_bucket_shortage: str = "error"
      sampling: str = "shuffle"
  ```

- [ ] 给 `BenchProfile` 增加字段：

  ```python
  dataset: DatasetConfig | None = None
  ```

- [ ] 增加 `_parse_dataset_config(raw, path)`：

  ```python
  def _parse_dataset_config(raw: object, path: str) -> DatasetConfig | None:
      if raw is None:
          return None
      if not isinstance(raw, dict):
          raise ConfigError(f"{path} must be an object")
      name = _require_str(raw, "name", path)
      if name not in {"random", "builtin_mtp_chat"}:
          raise ConfigError(f"{path}.name unsupported dataset: {name}")
      length_policy = str(raw.get("length_policy", "exact"))
      if length_policy not in {"exact", "bucket"}:
          raise ConfigError(f"{path}.length_policy must be exact or bucket")
      tolerance = float(raw.get("input_len_tolerance", 0.2))
      if tolerance < 0 or tolerance >= 1:
          raise ConfigError(f"{path}.input_len_tolerance must be >= 0 and < 1")
      shortage = str(raw.get("on_bucket_shortage", "error"))
      if shortage != "error":
          raise ConfigError(f"{path}.on_bucket_shortage only supports error")
      sampling = str(raw.get("sampling", "shuffle"))
      if sampling not in {"shuffle", "round_robin"}:
          raise ConfigError(f"{path}.sampling must be shuffle or round_robin")
      return DatasetConfig(
          name=name,
          length_policy=length_policy,
          input_len_tolerance=tolerance,
          on_bucket_shortage=shortage,
          sampling=sampling,
      )
  ```

- [ ] 在 `_parse_bench_profiles()` 中传入 `dataset=_parse_dataset_config(profile.get("dataset"), "bench_profile.dataset")`。
- [ ] 在 `build_bench_run_command()` 中，当 `bench.dataset` 存在时追加：

  ```python
  cmd.extend(["--dataset", bench.dataset.name])
  cmd.extend(["--dataset-length-policy", bench.dataset.length_policy])
  cmd.extend(["--dataset-input-len-tolerance", str(bench.dataset.input_len_tolerance)])
  cmd.extend(["--dataset-on-bucket-shortage", bench.dataset.on_bucket_shortage])
  cmd.extend(["--dataset-sampling", bench.dataset.sampling])
  ```

- [ ] 增加 `_validate_serve_args(args, path)`，并在 `_parse_serve_profiles()` 调用：

  ```python
  def _validate_serve_args(args: Sequence[str], path: str) -> None:
      for value in args:
          if value.startswith("speculative-config."):
              raise ConfigError(
                  f"{path}.args contains {value!r}; use '--{value}' for vLLM dotted flags"
              )
  ```

- [ ] 运行定向测试：

  ```bash
  pytest -q vllm_standalone_bench/tests/test_auto_bench.py
  ```

- [ ] 提交：

  ```bash
  git add vllm_standalone_bench/auto_bench.py vllm_standalone_bench/tests/test_auto_bench.py
  git commit -m "feat: configure mtp benchmark datasets"
  ```

**通过标准：**
- 旧配置不带 `dataset` 时生成命令不新增 dataset 参数。
- 新配置生成 `run_bench_multi.py` 所需 dataset 参数。
- 缺少 `--` 的 `speculative-config.*` 参数在启动服务前失败。

---

## 任务 4：实现内置 `builtin_mtp_chat` 数据集

**文件：**
- `vllm_standalone_bench/vllm_bench/datasets/__init__.py`
- `vllm_standalone_bench/vllm_bench/datasets/builtin_mtp_chat.py`
- `vllm_standalone_bench/tests/test_builtin_mtp_chat_dataset.py`

**步骤：**
- [ ] 先写 tokenizer fake 和基础样本测试：

  ```python
  from types import SimpleNamespace

  from vllm_standalone_bench.vllm_bench.datasets import builtin_mtp_chat
  from vllm_standalone_bench import run_bench_serve

  class ChatTokenizer:
      def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
          text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
          if add_generation_prompt:
              text += "\nassistant:"
          if tokenize:
              return text.split()
          return text

      def encode(self, text):
          return text.split()

  def test_builtin_mtp_chat_builds_bucketed_requests():
      args = SimpleNamespace(
          input_len=80,
          random_input_len=80,
          output_len=16,
          random_output_len=16,
          num_prompts=3,
          dataset_length_policy="bucket",
          dataset_input_len_tolerance=0.5,
          dataset_on_bucket_shortage="error",
          dataset_sampling="round_robin",
          seed=7,
      )

      requests = builtin_mtp_chat.build_requests(
          args,
          ChatTokenizer(),
          run_bench_serve.SampleRequest,
      )

      assert len(requests) == 3
      assert all(40 <= item.prompt_len <= 120 for item in requests)
      assert all(item.expected_output_len == 16 for item in requests)
      assert all(isinstance(item.prompt, str) for item in requests)
  ```

- [ ] 写失败测试，覆盖没有 tokenizer 时拒绝：

  ```python
  def test_builtin_mtp_chat_requires_tokenizer():
      args = SimpleNamespace(
          input_len=80,
          random_input_len=80,
          output_len=16,
          random_output_len=16,
          num_prompts=1,
          dataset_length_policy="bucket",
          dataset_input_len_tolerance=0.2,
          dataset_on_bucket_shortage="error",
          dataset_sampling="shuffle",
          seed=1,
      )

      with pytest.raises(ValueError, match="requires --tokenizer"):
          builtin_mtp_chat.build_requests(args, None, run_bench_serve.SampleRequest)
  ```

- [ ] 创建 `datasets/__init__.py`：

  ```python
  """Offline benchmark datasets for vLLM standalone bench."""
  ```

- [ ] 创建 `builtin_mtp_chat.py`，公开函数名和签名固定为 `build_requests(args, tokenizer, sample_request_cls)`。

- [ ] 数据集内容固定为离线合成但真实风格的 chat 场景，覆盖这些样本族：

  ```python
  TOPICS = (
      "中文长文摘要与关键结论提取",
      "多轮技术问答和需求澄清",
      "Python 代码阅读与问题定位",
      "服务日志分析和根因判断",
      "JSON 配置审查和字段解释",
      "数学推理与步骤化演算",
      "中英混合 API 使用说明",
      "测试报告总结与风险分级",
  )
  ```

- [ ] 使用确定性扩展函数构造足够 prompt 长度：

  ```python
  def _repeat_to_budget(seed_text: str, target_words: int) -> str:
      words = seed_text.split()
      if not words:
          return seed_text
      chunks = []
      while len(" ".join(chunks).split()) < target_words:
          chunks.extend(words)
      return " ".join(chunks[:target_words])
  ```

- [ ] token 计数优先使用 chat template：

  ```python
  def _render_prompt(tokenizer, messages):
      if hasattr(tokenizer, "apply_chat_template"):
          return tokenizer.apply_chat_template(
              messages,
              tokenize=False,
              add_generation_prompt=True,
          )
      return "\n".join(f"{item['role']}: {item['content']}" for item in messages)

  def _count_tokens(tokenizer, prompt):
      encoded = tokenizer.encode(prompt)
      if hasattr(encoded, "ids"):
          return len(encoded.ids)
      return len(encoded)
  ```

- [ ] bucket 选择规则：

  ```python
  target_len = int(getattr(args, "input_len", None) or getattr(args, "random_input_len"))
  tolerance = float(getattr(args, "dataset_input_len_tolerance", 0.2))
  lower = max(1, int(target_len * (1 - tolerance)))
  upper = max(lower, int(target_len * (1 + tolerance)))
  ```

- [ ] 采样规则：

  ```python
  rng = random.Random(int(getattr(args, "seed", 0)))
  if sampling == "shuffle":
      rng.shuffle(candidates)
  selected = [candidates[index % len(candidates)] for index in range(num_prompts)]
  ```

- [ ] bucket 样本不足时抛出清晰错误：

  ```python
  if not candidates:
      raise ValueError(
          f"builtin_mtp_chat has no prompts in token bucket [{lower}, {upper}] "
          f"for target input_len={target_len}; increase input_len_tolerance"
      )
  ```

- [ ] 返回现有 `SampleRequest` 结构：

  ```python
  return [
      sample_request_cls(
          prompt=item.prompt,
          prompt_len=item.prompt_len,
          expected_output_len=output_len,
      )
      for item in selected
  ]
  ```

- [ ] 运行定向测试：

  ```bash
  pytest -q vllm_standalone_bench/tests/test_builtin_mtp_chat_dataset.py
  ```

- [ ] 提交：

  ```bash
  git add vllm_standalone_bench/vllm_bench/datasets vllm_standalone_bench/tests/test_builtin_mtp_chat_dataset.py
  git commit -m "feat: add builtin mtp chat dataset"
  ```

**通过标准：**
- 数据集全离线，不读取网络和外部文件。
- 相同 seed、相同参数下返回稳定顺序。
- 样本 prompt 是 chat 风格自然文本，不是随机 token 串。

---

## 任务 5：把内置数据集接入 benchmark CLI

**文件：**
- `vllm_standalone_bench/run_bench_multi.py`
- `vllm_standalone_bench/run_bench_serve.py`
- `vllm_standalone_bench/tests/test_random_dataset.py`
- `vllm_standalone_bench/tests/test_auto_bench.py`

**步骤：**
- [ ] 在 `test_random_dataset.py` 写失败测试，覆盖 `get_samples()` 路由到内置数据集：

  ```python
  def test_get_samples_supports_builtin_mtp_chat(monkeypatch):
      calls = []

      def fake_build_requests(args, tokenizer, sample_request_cls):
          calls.append((args.dataset_name, tokenizer, sample_request_cls))
          return [
              run_bench_serve.SampleRequest(
                  prompt="user: explain mtp\nassistant:",
                  prompt_len=4,
                  expected_output_len=8,
              )
          ]

      monkeypatch.setattr(
          run_bench_serve.builtin_mtp_chat,
          "build_requests",
          fake_build_requests,
      )
      args = argparse.Namespace(
          dataset_name="builtin_mtp_chat",
          input_len=32,
          output_len=8,
          num_prompts=1,
      )

      requests = run_bench_serve.get_samples(args, object())

      assert len(requests) == 1
      assert calls[0][0] == "builtin_mtp_chat"
      assert calls[0][2] is run_bench_serve.SampleRequest
  ```

- [ ] 在 `run_bench_serve.py` 引入内置数据集模块：

  ```python
  from vllm_standalone_bench.vllm_bench.datasets import builtin_mtp_chat
  ```

- [ ] 扩展 shim parser，新增参数：

  ```python
  parser.add_argument("--dataset-length-policy", default="exact")
  parser.add_argument("--dataset-input-len-tolerance", type=float, default=0.2)
  parser.add_argument("--dataset-on-bucket-shortage", default="error")
  parser.add_argument("--dataset-sampling", default="shuffle")
  ```

- [ ] 修改 `get_samples()` 路由：

  ```python
  if dataset_name == "builtin_mtp_chat":
      return builtin_mtp_chat.build_requests(args, tokenizer, SampleRequest)
  ```

- [ ] 在 `run_bench_multi.py` parser 增加 dataset 参数：

  ```python
  parser.add_argument("--dataset", default="random", choices=["random", "builtin_mtp_chat"])
  parser.add_argument("--dataset-length-policy", default="exact", choices=["exact", "bucket"])
  parser.add_argument("--dataset-input-len-tolerance", type=float, default=0.2)
  parser.add_argument("--dataset-on-bucket-shortage", default="error", choices=["error"])
  parser.add_argument("--dataset-sampling", default="shuffle", choices=["shuffle", "round_robin"])
  ```

- [ ] 修改 `_build_base_args()`：

  ```python
  base.dataset_name = our_args.dataset
  base.dataset_length_policy = our_args.dataset_length_policy
  base.dataset_input_len_tolerance = our_args.dataset_input_len_tolerance
  base.dataset_on_bucket_shortage = our_args.dataset_on_bucket_shortage
  base.dataset_sampling = our_args.dataset_sampling
  if our_args.dataset == "builtin_mtp_chat" and not our_args.tokenizer:
      raise ValueError("builtin_mtp_chat requires --tokenizer")
  ```

- [ ] 在 `test_auto_bench.py` 增加命令兼容测试，确认旧配置不会追加 dataset 参数：

  ```python
  def test_build_bench_run_command_omits_dataset_for_legacy_config(tmp_path):
      config = minimal_config(tmp_path)
      parsed = auto_bench.load_config(write_config(tmp_path, config))

      cmd = auto_bench.build_bench_run_command(
          parsed,
          parsed.models[0],
          parsed.serve_profiles[0],
          parsed.bench_profiles[0],
          Path("/tmp/run"),
      )

      assert "--dataset" not in cmd
  ```

- [ ] 运行定向测试：

  ```bash
  pytest -q vllm_standalone_bench/tests/test_random_dataset.py vllm_standalone_bench/tests/test_auto_bench.py
  ```

- [ ] 提交：

  ```bash
  git add vllm_standalone_bench/run_bench_multi.py vllm_standalone_bench/run_bench_serve.py vllm_standalone_bench/tests/test_random_dataset.py vllm_standalone_bench/tests/test_auto_bench.py
  git commit -m "feat: wire builtin mtp dataset into bench cli"
  ```

**通过标准：**
- 旧随机数据集路径不受影响。
- `--dataset builtin_mtp_chat` 会调用内置数据集。
- `auto_bench.py` 只在配置显式声明 `dataset` 时追加 dataset 参数。

---

## 任务 6：补充示例配置与使用说明

**文件：**
- `vllm_standalone_bench/configs/`
- `vllm_standalone_bench/README.md`

**步骤：**
- [ ] 查找现有配置目录：

  ```bash
  rg --files vllm_standalone_bench/configs
  ```

- [ ] 新增或更新一个 MTP 示例配置，文件名使用 `mtp_builtin_dataset*.json` 风格，包含：

  ```json
  {
    "bench_profiles": [
      {
        "name": "MTP_real_bucket",
        "backend": "vllm",
        "dataset": {
          "name": "builtin_mtp_chat",
          "length_policy": "bucket",
          "input_len_tolerance": 0.2,
          "on_bucket_shortage": "error",
          "sampling": "shuffle"
        },
        "input_lens": [1024, 2048, 4096],
        "output_lens": [512],
        "parallel_nums": [1, 4, 8],
        "epochs": 2
      }
    ]
  }
  ```

- [ ] README 增加简短说明：

  ```markdown
  ### MTP realistic dataset

  `bench_profiles[].dataset.name = "builtin_mtp_chat"` enables the built-in offline chat-style MTP dataset. With this dataset, `input_lens` are token bucket targets instead of exact random prompt lengths. `output_lens`, `parallel_nums`, and `epochs` keep their existing matrix behavior.
  ```

- [ ] 运行格式检查：

  ```bash
  git diff --check
  ```

- [ ] 提交：

  ```bash
  git add vllm_standalone_bench/configs vllm_standalone_bench/README.md
  git commit -m "docs: document builtin mtp dataset"
  ```

**通过标准：**
- 示例配置能表达完整 dataset 字段。
- 文档明确说明 `input_lens` 在内置数据集下是 bucket target，不是失效参数。

---

## 任务 7：全量验证与结果检查

**文件：**
- `vllm_standalone_bench/`

**步骤：**
- [ ] 运行全量测试：

  ```bash
  pytest -q vllm_standalone_bench/tests
  ```

- [ ] 运行 diff 空白检查：

  ```bash
  git diff --check
  ```

- [ ] 查看提交序列：

  ```bash
  git log --oneline --decorate -n 8
  ```

- [ ] 查看最终变更摘要：

  ```bash
  git diff --stat main..HEAD
  ```

**通过标准：**
- `pytest -q vllm_standalone_bench/tests` 通过。
- `git diff --check` 无输出。
- 分支提交包含设计文档、计划文档和每个功能提交。

## 验收清单

- [ ] `/v1/metrics` 使用 API key 时不再因为缺少 `Authorization` 报 401。
- [ ] MTP 接受率、系统效率、draft token 计数进入 CSV/XLSX。
- [ ] `bench_profiles[].dataset.name = "builtin_mtp_chat"` 可用。
- [ ] `input_lens` 在内置数据集下作为 token bucket 目标生效。
- [ ] 缺少 `--` 的 `speculative-config.*` 在配置解析阶段失败。
- [ ] 旧随机数据集配置和已有自动化测试继续通过。
