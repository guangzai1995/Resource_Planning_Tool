"""warmup 固定并发预热的参数透传测试。"""
import argparse

import pytest

import run_bench_multi


def test_build_arg_parser_has_warmup_opts():
    args = run_bench_multi.build_arg_parser().parse_args([
        "--model", "m", "--backend", "openai",
        "--host", "127.0.0.1", "--port", "8000",
        "--warmup-concurrency", "4", "--warmup-output-len", "128",
    ])
    assert args.warmup_concurrency == 4
    assert args.warmup_output_len == 128


def test_build_arg_parser_warmup_opts_default_none():
    args = run_bench_multi.build_arg_parser().parse_args([
        "--model", "m", "--backend", "openai",
        "--host", "127.0.0.1", "--port", "8000",
    ])
    assert args.warmup_concurrency is None
    assert args.warmup_output_len is None


def test_build_base_args_passes_warmup_opts():
    args = run_bench_multi.build_arg_parser().parse_args([
        "--model", "m", "--backend", "openai",
        "--host", "127.0.0.1", "--port", "8000",
        "--warmup-concurrency", "4", "--warmup-output-len", "128",
    ])
    base = run_bench_multi._build_base_args(args)
    assert base.warmup_concurrency == 4
    assert base.warmup_output_len == 128


def test_build_arg_parser_and_base_args_pass_dataset_opts():
    args = run_bench_multi.build_arg_parser().parse_args([
        "--model", "m", "--backend", "openai-chat",
        "--host", "127.0.0.1", "--port", "8000",
        "--tokenizer", "/models/tok",
        "--dataset", "builtin_mtp_chat",
        "--dataset-length-policy", "bucket",
        "--dataset-input-len-tolerance", "0.2",
        "--dataset-on-bucket-shortage", "error",
        "--dataset-sampling", "shuffle",
    ])

    base = run_bench_multi._build_base_args(args)

    assert base.dataset_name == "builtin_mtp_chat"
    assert base.dataset_length_policy == "bucket"
    assert base.dataset_input_len_tolerance == 0.2
    assert base.dataset_on_bucket_shortage == "error"
    assert base.dataset_sampling == "shuffle"


def test_build_base_args_requires_tokenizer_for_builtin_mtp_chat():
    args = run_bench_multi.build_arg_parser().parse_args([
        "--model", "m", "--backend", "openai-chat",
        "--host", "127.0.0.1", "--port", "8000",
        "--dataset", "builtin_mtp_chat",
    ])

    with pytest.raises(ValueError, match="requires --tokenizer"):
        run_bench_multi._build_base_args(args)


def test_run_all_treats_warmup_requests_as_rounds(monkeypatch):
    seen_cfgs = []

    async def fake_main_async(cfg):
        seen_cfgs.append(cfg)
        return {
            "duration": 1.0,
            "completed": cfg.num_prompts,
            "failed": 0,
            "total_input_tokens": cfg.input_len * cfg.num_prompts,
            "total_output_tokens": cfg.output_len * cfg.num_prompts,
            "request_throughput": cfg.num_prompts,
            "output_throughput": cfg.output_len * cfg.num_prompts,
            "usage_reported_count": cfg.num_prompts,
            "finish_reason_length": cfg.num_prompts,
            "mean_ttft_ms": 1.0,
            "p50_ttft_ms": 1.0,
            "p90_ttft_ms": 1.0,
            "p99_ttft_ms": 1.0,
            "mean_tpot_ms": 1.0,
            "p50_tpot_ms": 1.0,
            "p90_tpot_ms": 1.0,
            "p99_tpot_ms": 1.0,
            "mean_e2el_ms": 1.0,
            "p50_e2el_ms": 1.0,
            "p90_e2el_ms": 1.0,
            "p99_e2el_ms": 1.0,
        }

    monkeypatch.setattr(run_bench_multi._serve, "main_async", fake_main_async)

    args = argparse.Namespace(
        model="m", served_model_name=None, backend="openai-chat",
        base_url="http://x/v1", host="127.0.0.1", port=8000,
        insecure=False, api_key=None, tokenizer="/some/tok",
        input_lens=[128], output_lens=[8], cross_product=False,
        parallel_nums=[1], epochs=1, sleep_between=0,
        warmup_requests=4, warmup_concurrency=8, warmup_output_len=None,
        prefix_ratio=0.0, seed=0, no_vary_seed_by_config=False,
        output_csv=None, output_xlsx=None, result_dir=None,
        max_ttft_ms=None, min_throughput_tok_s=None,
        min_output_compliance=0.0,
    )

    run_bench_multi._run_all(args)

    assert seen_cfgs[0].warmup_concurrency == 8
    assert seen_cfgs[0].num_warmups == 32
