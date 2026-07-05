"""result.csv/xlsx 列顺序与中文字体配置的断言。

Part 1：CSV_HEADERS / CSV_HEADERS_ZH 的顺序——基本信息在前，吞吐+TTFT 紧随，
其余在后；列集合与中英表头一一对应。
Part 2：bench_compare 的 CJK 字体 fallback 配置（见文件末尾，任务 2 追加）。
"""
import pytest

import bench_compare as bc
import run_bench_multi as m


# 期望的新顺序（57 列）：基本信息(17) → 吞吐(5) → TTFT(4) → 其他(31)
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
    "cache_hit_rate_metrics",
    "cache_hit_tokens_device", "cache_hit_tokens_host",
    "cache_hit_tokens_storage", "cache_hit_tokens_storage_mooncake",
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
    "Metrics缓存命中率(%)",
    "Device缓存命中tokens", "Host缓存命中tokens",
    "Storage缓存命中tokens", "Mooncake缓存命中tokens",
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


def test_csv_headers_count_is_57():
    assert len(m.CSV_HEADERS) == 57
    assert len(m.CSV_HEADERS_ZH) == 57


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


# ── Part 2：绘图中文（CJK 字体配置）──────────────────────────────

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
