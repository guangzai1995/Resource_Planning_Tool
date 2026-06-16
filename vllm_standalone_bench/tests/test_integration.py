"""端到端冒烟：monkeypatch serve.main_async 返回固定结果，走 run_bench_multi
的提取与落盘路径，断言 CSV 的 avg/合规列真实。"""
import csv

import run_bench_multi as m

# 注意：run_bench_multi 内部用 importlib 把 run_bench_serve.py 加载为独立模块
# （'_run_bench_serve_shims'），所以必须 patch m._serve（即 run_bench_multi 实际
# 调用的那个 serve 模块对象），而不是 import run_bench_serve 得到的另一个实例。
serve = m._serve


def _fake_result(in_len, out_len, completed, *, undergen=False):
    """构造 serve.main_async 风格的 result dict。undergen=True 表示服务端少生成。"""
    real_out = (out_len // 2) if undergen else out_len
    return {
        "duration": 2.0,
        "completed": completed, "failed": 0,
        "total_input_tokens": in_len * completed,
        "total_output_tokens": real_out * completed,
        "request_throughput": completed / 2.0,
        "output_throughput": real_out * completed / 2.0,
        "total_token_throughput": (in_len + real_out) * completed / 2.0,
        "usage_reported_count": completed,
        "finish_reason_length": completed if not undergen else 0,
        "num_prompts": completed,
        "mean_ttft_ms": 50.0, "p50_ttft_ms": 50.0, "p90_ttft_ms": 60.0, "p99_ttft_ms": 70.0,
        "mean_tpot_ms": 30.0, "p50_tpot_ms": 30.0, "p90_tpot_ms": 31.0, "p99_tpot_ms": 32.0,
        "mean_e2el_ms": 1000.0, "p50_e2el_ms": 1000.0, "p90_e2el_ms": 1100.0, "p99_e2el_ms": 1200.0,
    }


def _run_to_rows(monkeypatch, results_seq):
    """把 _serve.main_async 替换为依次返回 results_seq 的假实现，跑 _run_all。"""
    import argparse
    it = iter(results_seq)

    async def _fake_main_async(cfg):
        return next(it)

    monkeypatch.setattr(serve, "main_async", _fake_main_async)

    our_args = argparse.Namespace(
        model="m", served_model_name=None, backend="openai-chat",
        base_url=None, host="127.0.0.1", port=8000, insecure=False, api_key=None,
        tokenizer="/some/tok",  # has_tokenizer=True
        input_lens=[128], output_lens=[8], cross_product=False,
        parallel_nums=[1], epochs=1, sleep_between=0, warmup_requests=0,
        prefix_ratio=0.0,
        max_ttft_ms=None, min_throughput_tok_s=None, min_output_compliance=0.95,
        output_csv=None, output_xlsx=None, result_dir=None,
    )
    return m._run_all(our_args)


def test_csv_records_real_avg_and_compliance(tmp_path, monkeypatch):
    rows = _run_to_rows(monkeypatch, [_fake_result(128, 8, 3)])
    assert rows[0]["avg_output_tokens"] == 8.0          # 真实，非 requested 回显
    assert rows[0]["output_compliance"] == 100.0
    assert rows[0]["token_source"] == "usage"

    csv_path = str(tmp_path / "bench.csv")
    m.save_csv(rows, csv_path)
    with open(csv_path, encoding="utf-8-sig") as f:
        data = list(csv.DictReader(f))
    assert len(data) == 1
    assert float(data[0]["avg_output_tokens"]) == 8.0
    assert float(data[0]["output_compliance"]) == 100.0
    assert data[0]["token_source"] == "usage"
    assert int(data[0]["total_input_len"]) == 128


def test_csv_flags_undergeneration(tmp_path, monkeypatch):
    rows = _run_to_rows(monkeypatch, [_fake_result(128, 8, 3, undergen=True)])
    assert rows[0]["avg_output_tokens"] == 4.0          # 服务端只生成一半
    assert rows[0]["output_compliance"] == 50.0         # 4/8
    assert rows[0]["finish_reason_length_pct"] == 0.0   # 都不是 length 停止
