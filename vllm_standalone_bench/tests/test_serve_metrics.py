import asyncio

import run_bench_serve

serve = run_bench_serve._serve
RequestFuncOutput = serve.RequestFuncOutput
SampleRequest = serve.SampleRequest  # 由 datasets shim 注入，已挂在 serve 上


def _req(prompt_len, out_len):
    return SampleRequest(prompt="x", prompt_len=prompt_len,
                         expected_output_len=out_len)


def _out(success, *, output_tokens=0, finish_reason="", prompt_len=10,
         ttft=0.05, latency=1.0, itl=None, generated_text="abc",
         cached_tokens=0, cached_reported=False):
    return RequestFuncOutput(
        success=success, output_tokens=output_tokens,
        finish_reason=finish_reason, prompt_len=prompt_len,
        ttft=ttft, latency=latency, itl=itl or [0.05, 0.05],
        generated_text=generated_text,
        cached_tokens=cached_tokens, cached_reported=cached_reported,
    )


def test_metrics_aggregate_finish_reason_and_usage():
    outputs = [
        _out(True, output_tokens=8, finish_reason="length"),   # usage 上报 + length 停止
        _out(True, output_tokens=8, finish_reason="length"),   # usage 上报 + length 停止
        _out(True, output_tokens=0, finish_reason="stop"),     # 未上报 usage（将回退）
    ]
    inputs = [_req(10, 8) for _ in outputs]
    metrics, _lens = serve.calculate_metrics(
        input_requests=inputs, outputs=outputs, dur_s=2.0,
        tokenizer=None, selected_percentiles=[50, 90],
        goodput_config_dict={},
    )
    assert metrics.completed == 3
    assert metrics.finish_reason_length == 2   # 两个 "length"
    assert metrics.usage_reported_count == 2   # 两个 output_tokens>0（来自 usage）


def test_result_dict_carries_new_fields():
    outputs = [_out(True, output_tokens=8, finish_reason="length")]
    metrics, _ = serve.calculate_metrics(
        input_requests=[_req(10, 8)], outputs=outputs, dur_s=1.0,
        tokenizer=None, selected_percentiles=[50], goodput_config_dict={},
    )
    assert hasattr(metrics, "finish_reason_length")
    assert hasattr(metrics, "usage_reported_count")


def test_metrics_reported_without_tokenizer_when_usage_present():
    """无 tokenizer 但服务端上报了 output_tokens 时，TPOT 等指标仍应可算出（非 0）。"""
    outputs = [_out(True, output_tokens=8, finish_reason="length",
                    ttft=0.05, latency=0.5, itl=[0.05] * 7)]
    metrics, _ = serve.calculate_metrics(
        input_requests=[_req(10, 8)], outputs=outputs, dur_s=1.0,
        tokenizer=None, selected_percentiles=[50], goodput_config_dict={},
    )
    assert metrics.mean_tpot_ms > 0
    assert metrics.mean_ttft_ms > 0


def test_metrics_counts_tokenizer_fallback_outputs():
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            class Encoded:
                input_ids = text.split()

            return Encoded()

    outputs = [
        _out(True, output_tokens=0, finish_reason="stop",
             generated_text="a b c", ttft=0.05, latency=0.35, itl=[0.1, 0.1]),
    ]
    metrics, lens = serve.calculate_metrics(
        input_requests=[_req(10, 8)], outputs=outputs, dur_s=1.0,
        tokenizer=Tok(), selected_percentiles=[50], goodput_config_dict={},
    )

    assert lens == [3]
    assert metrics.usage_reported_count == 0
    assert metrics.tokenizer_fallback_count == 1
    assert metrics.total_output == 3


def test_metrics_aggregate_cached_tokens():
    """total_cached_tokens 累加成功请求的 cached_tokens；cached_reported_count 计上报数。"""
    outputs = [
        _out(True, output_tokens=8, finish_reason="length",
             cached_tokens=80, cached_reported=True),    # 命中 80
        _out(True, output_tokens=8, finish_reason="length",
             cached_tokens=0, cached_reported=True),     # 上报了但命中 0
        _out(True, output_tokens=0, finish_reason="stop"),  # 未上报 cached
    ]
    inputs = [_req(100, 8) for _ in outputs]
    metrics, _lens = serve.calculate_metrics(
        input_requests=inputs, outputs=outputs, dur_s=2.0,
        tokenizer=None, selected_percentiles=[50, 90], goodput_config_dict={},
    )
    assert metrics.total_cached_tokens == 80       # 80 + 0 + 0
    assert metrics.cached_reported_count == 2      # 前两个上报


def test_metrics_cached_fields_default_zero():
    """无 cached 数据时两个字段为 0。"""
    outputs = [_out(True, output_tokens=8, finish_reason="length")]
    metrics, _ = serve.calculate_metrics(
        input_requests=[_req(10, 8)], outputs=outputs, dur_s=1.0,
        tokenizer=None, selected_percentiles=[50], goodput_config_dict={},
    )
    assert metrics.total_cached_tokens == 0
    assert metrics.cached_reported_count == 0


def test_resolve_warmup_config_defaults_to_profile_values():
    cc, ol = serve.resolve_warmup_config(
        max_concurrency=8, warmup_concurrency=None,
        output_len=1024, warmup_output_len=None)
    assert cc == 8
    assert ol == 1024


def test_resolve_warmup_config_overrides_when_set():
    cc, ol = serve.resolve_warmup_config(
        max_concurrency=8, warmup_concurrency=4,
        output_len=1024, warmup_output_len=128)
    assert cc == 4
    assert ol == 128


def test_resolve_warmup_config_concurrency_none_falls_back_even_when_output_set():
    cc, ol = serve.resolve_warmup_config(
        max_concurrency=16, warmup_concurrency=None,
        output_len=1024, warmup_output_len=128)
    assert cc == 16
    assert ol == 128


def test_fetch_spec_decode_metrics_passes_headers_and_normalizes_url():
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
