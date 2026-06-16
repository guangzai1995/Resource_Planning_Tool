import run_bench_multi as m


# ---------- decide_token_usage_source ----------
def test_token_source_usage_when_all_reported():
    assert m.decide_token_usage_source(
        usage_reported_count=3, completed=3, has_tokenizer=True) == "usage"


def test_token_source_tokenizer_when_none_reported_but_has_tok():
    assert m.decide_token_usage_source(
        usage_reported_count=0, completed=3, has_tokenizer=True) == "tokenizer"


def test_token_source_none_when_nothing():
    assert m.decide_token_usage_source(
        usage_reported_count=0, completed=3, has_tokenizer=False) == "none"


def test_token_source_none_when_all_failed():
    assert m.decide_token_usage_source(
        usage_reported_count=0, completed=0, has_tokenizer=True) == "none"


# ---------- _extract_row: 真实 avg（不再回显 requested） ----------
def _result(total_in=30, total_out=24, completed=3, usage_reported=3,
            finish_reason_length=3):
    """构造 serve.main_async 风格的最小 result dict（仅本测试关心的键）。"""
    return {
        "completed": completed, "failed": 0,
        "total_input_tokens": total_in, "total_output_tokens": total_out,
        "usage_reported_count": usage_reported,
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
    assert row["output_compliance"] == 100.0  # 8/8
    assert row["finish_reason_length_pct"] == 100.0


def test_extract_row_compliance_when_undergenerated():
    # 服务端只生成了 12 token（请求 8×3=24，实测 12/3=4 < 8）
    row = m._extract_row(
        _result(total_out=12, completed=3, usage_reported=3),
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["avg_output_tokens"] == 4.0
    assert row["output_compliance"] == 50.0  # 4/8
    assert row["token_source"] == "usage"


def test_extract_row_prefix_total_input_len():
    row = m._extract_row(
        _result(completed=3, total_in=690, total_out=24),  # prefix 场景
        in_len=128, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai", prefix_tokens=102,
        prefix_ratio=0.8, has_tokenizer=True)
    assert row["total_input_len"] == 128 + 102
    assert row["input_len"] == 128  # requested 后缀长度


def test_token_source_tokenizer_when_no_usage():
    row = m._extract_row(
        _result(total_out=24, completed=3, usage_reported=0),
        in_len=10, out_len=8, parallel_num=3, epochs=1,
        model="m", backend="openai-chat", has_tokenizer=True)
    assert row["token_source"] == "tokenizer"
