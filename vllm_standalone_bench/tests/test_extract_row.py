import pytest

import run_bench_multi as m


# ---------- decide_token_usage_source ----------
def test_token_source_usage_when_all_reported():
    assert m.decide_token_usage_source(
        usage_reported_count=3, tokenizer_fallback_count=0,
        completed=3, has_tokenizer=True) == "usage"


def test_token_source_tokenizer_fallback_when_no_usage_but_has_tok():
    assert m.decide_token_usage_source(
        usage_reported_count=0, tokenizer_fallback_count=3,
        completed=3, has_tokenizer=True) == "tokenizer_fallback"


def test_token_source_partial_usage_when_some_usage_missing():
    assert m.decide_token_usage_source(
        usage_reported_count=2, tokenizer_fallback_count=1,
        completed=3, has_tokenizer=True) == "partial_usage"


def test_token_source_client_estimate_when_nothing_reported():
    assert m.decide_token_usage_source(
        usage_reported_count=0, tokenizer_fallback_count=0,
        completed=3, has_tokenizer=False) == "client_estimate"


def test_token_source_none_when_all_failed():
    assert m.decide_token_usage_source(
        usage_reported_count=0, tokenizer_fallback_count=0,
        completed=0, has_tokenizer=True) == "none"


def test_derive_prefix_suffix_tokens_from_total_input():
    assert m._derive_prefix_suffix_tokens(128, 0.8) == (102, 26)
    assert m._derive_prefix_suffix_tokens(128, 0.0) == (0, 128)
    assert m._derive_prefix_suffix_tokens(128, 1.0) == (128, 0)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan")])
def test_derive_prefix_suffix_tokens_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="--prefix-ratio"):
        m._derive_prefix_suffix_tokens(128, ratio)


# ---------- _extract_row: 真实 avg（不再回显 requested） ----------
def _result(total_in=30, total_out=24, completed=3, usage_reported=3,
            finish_reason_length=3, tokenizer_fallback=0, total_cached=0):
    """构造 serve.main_async 风格的最小 result dict（仅本测试关心的键）。"""
    return {
        "completed": completed, "failed": 0,
        "total_input_tokens": total_in, "total_output_tokens": total_out,
        "total_cached_tokens": total_cached,
        "usage_reported_count": usage_reported,
        "tokenizer_fallback_count": tokenizer_fallback,
        "finish_reason_length": finish_reason_length,
        "num_prompts": completed,
        "request_throughput": 1.0, "output_throughput": 12.0,
        "duration": 2.0,
    }


def test_extract_row_real_avg_from_totals():
    row = m._extract_row(
        _result(total_in=30, total_out=24, completed=3),  # 请求 out_len=8
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    # 真实平均 = 30/3=10（输入）, 24/3=8（输出）—— 而非回显 requested
    assert row["avg_input_tokens"] == 10.0
    assert row["avg_output_tokens"] == 8.0
    assert row["token_source"] == "usage"
    assert row["input_compliance"] == 100.0
    assert row["output_compliance"] == 100.0  # 8/8
    assert row["finish_reason_length_pct"] == 100.0


def test_extract_row_throughput_and_effective_token_rates():
    row = m._extract_row(
        {
            **_result(total_in=384, total_out=24, completed=3),
            "duration": 2.0,
            "output_throughput": 12.0,
            "mean_ttft_ms": 50.0,
            "mean_tpot_ms": 25.0,
        },
        in_len=128, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)

    assert row["throughput_tok_s"] == 12.0
    assert row["input_throughput_tok_s"] == 192.0
    assert row["prefill_effective_tok_s"] == 2560.0
    assert row["decode_effective_tok_s"] == 40.0
    assert "total_throughput_tok_s" not in row


def test_extract_row_compliance_when_undergenerated():
    # 服务端只生成了 12 token（请求 8×3=24，实测 12/3=4 < 8）
    row = m._extract_row(
        _result(total_out=12, completed=3, usage_reported=3),
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["avg_output_tokens"] == 4.0
    assert row["output_compliance"] == 50.0  # 4/8
    assert row["token_source"] == "usage"


def test_extract_row_prefix_total_input_len_uses_total_input_budget():
    row = m._extract_row(
        _result(completed=3, total_in=384, total_out=24),
        in_len=128, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai", prefix_tokens=102,
        prefix_ratio=0.8, has_tokenizer=True)
    assert row["total_input_len"] == 128
    assert row["input_len"] == 128
    assert row["prefix_tokens"] == 102
    assert row["avg_input_tokens"] == 128.0
    assert row["input_compliance"] == 100.0


def test_extract_row_cache_hit_rate_token_weighted():
    """cache_hit_rate = total_cached / total_input * 100（token 加权）；avg_cached per 请求。"""
    row = m._extract_row(
        _result(total_in=300, total_out=24, completed=3, total_cached=150),
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["cache_hit_rate"] == 50.0        # 150 / 300 * 100
    assert row["avg_cached_tokens"] == 50.0     # 150 / 3


def test_extract_row_cache_hit_rate_zero_when_no_cache():
    """无缓存数据（total_cached=0）→ 命中率 0、avg 0，不报错。"""
    row = m._extract_row(
        _result(total_in=300, total_out=24, completed=3, total_cached=0),
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["cache_hit_rate"] == 0.0
    assert row["avg_cached_tokens"] == 0.0


def test_extract_row_cache_hit_rate_safe_when_totals_zero():
    """全失败（completed=0、total_in=0）或键缺失 → 命中率回退 0，不抛除零。"""
    # completed=0 / total_in=0
    row_zero = m._extract_row(
        _result(total_in=0, total_out=0, completed=0, total_cached=0,
                usage_reported=0, finish_reason_length=0),
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row_zero["cache_hit_rate"] == 0.0
    assert row_zero["avg_cached_tokens"] == 0.0
    # 结果字典缺 total_cached_tokens 键（旧 serve 产物）
    row_missing = m._extract_row(
        {"completed": 3, "total_input_tokens": 300, "total_output_tokens": 24,
         "usage_reported_count": 3, "tokenizer_fallback_count": 0,
         "finish_reason_length": 3, "num_prompts": 3,
         "request_throughput": 1.0, "output_throughput": 12.0, "duration": 2.0},
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row_missing["cache_hit_rate"] == 0.0
    assert row_missing["avg_cached_tokens"] == 0.0


def test_extract_row_includes_spec_decode_metrics():
    result = {
        **_result(),
        "spec_decode_acceptance_rate": 75.0,
        "spec_decode_system_efficiency": 0.82,
        "spec_decode_num_drafts": 12,
        "spec_decode_num_accepted_tokens": 9,
        "spec_decode_num_draft_tokens": 12,
        "spec_decode_per_position_acceptance_rates": [90.0, 60.0],
    }

    row = m._extract_row(
        result,
        in_len=1024, out_len=512, parallel_num=4, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)

    assert row["spec_decode_acceptance_rate"] == 75.0
    assert row["spec_decode_system_efficiency"] == 0.82
    assert row["spec_decode_num_drafts"] == 12
    assert row["spec_decode_num_accepted_tokens"] == 9
    assert row["spec_decode_num_draft_tokens"] == 12
    assert row["spec_decode_per_position_acceptance_rates"] == "[90.0,60.0]"


def test_extract_row_spec_decode_defaults_when_metrics_missing():
    row = m._extract_row(
        _result(total_in=300, total_out=24, completed=3),
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)

    assert row["spec_decode_acceptance_rate"] == 0.0
    assert row["spec_decode_system_efficiency"] == 0.0
    assert row["spec_decode_num_drafts"] == 0
    assert row["spec_decode_num_accepted_tokens"] == 0
    assert row["spec_decode_num_draft_tokens"] == 0
    assert row["spec_decode_per_position_acceptance_rates"] == "[]"


def test_extract_row_includes_gpu_kv_cache_usage():
    result = {
        **_result(),
        "avg_gpu_kv_cache_usage": 9.12345,
        "peak_gpu_kv_cache_usage": 12.67891,
    }

    row = m._extract_row(
        result,
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True,
    )

    assert row["avg_gpu_kv_cache_usage"] == 9.1235
    assert row["peak_gpu_kv_cache_usage"] == 12.6789


def test_extract_row_gpu_kv_cache_usage_defaults_when_missing():
    row = m._extract_row(
        _result(),
        in_len=100, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True,
    )

    assert row["avg_gpu_kv_cache_usage"] == 0.0
    assert row["peak_gpu_kv_cache_usage"] == 0.0


def test_extract_row_spec_decode_token_aliases_from_serve_result():
    result = {
        **_result(),
        "spec_decode_accepted_tokens": 61,
        "spec_decode_draft_tokens": 280,
    }

    row = m._extract_row(
        result,
        in_len=1024, out_len=512, parallel_num=4, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True,
    )

    assert row["spec_decode_num_accepted_tokens"] == 61
    assert row["spec_decode_num_draft_tokens"] == 280


def test_token_source_tokenizer_fallback_when_no_usage():
    row = m._extract_row(
        _result(total_out=24, completed=3, usage_reported=0,
                tokenizer_fallback=3),
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["token_source"] == "tokenizer_fallback"


def test_extract_row_completed_zero_not_faking_compliance():
    """全部失败（completed==0）时：avg 应为 0、合规应为 0，不能回显 requested
    伪装成 100% 合规（Bug① 在零成功路径上的复发点）。"""
    row = m._extract_row(
        _result(total_in=0, total_out=0, completed=0, usage_reported=0,
                finish_reason_length=0),
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["avg_input_tokens"] == 0.0
    assert row["avg_output_tokens"] == 0.0
    assert row["output_compliance"] == 0.0
    assert row["token_source"] == "none"


def test_output_compliance_uses_unrounded_mean():
    """合规应基于未取整的真实均值，而非已 round 到 1 位的 avg_out。
    total_out=23, completed=3, out_len=8 → raw mean=7.667 → 95.8%；
    若误用 round 后的 avg_out=7.7 会算成 96.2%。二者不同，锁定用 raw。"""
    row = m._extract_row(
        _result(total_in=30, total_out=23, completed=3),
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["avg_output_tokens"] == 7.7          # 展示值取整
    assert row["output_compliance"] == 95.8          # 合规用未取整均值


def test_input_compliance_uses_unrounded_mean():
    """输入合规基于未取整均值和总输入目标长度。"""
    row = m._extract_row(
        _result(total_in=383, total_out=24, completed=3),
        in_len=128, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", prefix_tokens=102,
        prefix_ratio=0.8, has_tokenizer=True)
    assert row["avg_input_tokens"] == 127.7
    assert row["input_compliance"] == 99.7



def test_csv_headers_match_row_keys():
    """CSV_HEADERS 的每一列都必须能在 row 中取到；且新列必须已进表头（会被落盘）。"""
    row = m._extract_row(
        {"completed": 1, "total_input_tokens": 5, "total_output_tokens": 8,
         "usage_reported_count": 1, "tokenizer_fallback_count": 0,
         "finish_reason_length": 1, "num_prompts": 1,
         "request_throughput": 1.0, "output_throughput": 8.0, "duration": 1.0},
        in_len=5, out_len=8, parallel_num=1, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    missing = [h for h in m.CSV_HEADERS if h not in row]
    assert not missing, f"CSV_HEADERS 有列在 row 中缺失: {missing}"
    # 新增列必须被写入 CSV/XLSX
    for required in ("total_input_len", "input_compliance", "output_compliance",
                     "finish_reason_length_pct", "token_source", "seed",
                     "input_throughput_tok_s", "prefill_effective_tok_s",
                     "decode_effective_tok_s",
                     "avg_cached_tokens", "cache_hit_rate",
                     "avg_gpu_kv_cache_usage", "peak_gpu_kv_cache_usage",
                     "spec_decode_acceptance_rate",
                     "spec_decode_system_efficiency",
                     "spec_decode_num_drafts",
                     "spec_decode_num_accepted_tokens",
                     "spec_decode_num_draft_tokens",
                     "spec_decode_per_position_acceptance_rates"):
        assert required in m.CSV_HEADERS, f"新列 {required} 未进 CSV_HEADERS"
    assert "total_throughput_tok_s" not in m.CSV_HEADERS
    assert row["seed"] == 0
    assert len(m.CSV_HEADERS) == len(m.CSV_HEADERS_ZH), "中英文表头数量不一致"


def test_extract_row_records_effective_seed():
    row = m._extract_row(
        {"completed": 1, "total_input_tokens": 5, "total_output_tokens": 8,
         "usage_reported_count": 1, "tokenizer_fallback_count": 0,
         "finish_reason_length": 1, "num_prompts": 1,
         "request_throughput": 1.0, "output_throughput": 8.0, "duration": 1.0},
        in_len=5, out_len=8, parallel_num=1, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True, seed=98765)

    assert row["seed"] == 98765
