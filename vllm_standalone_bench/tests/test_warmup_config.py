"""warmup 固定并发预热的参数透传测试。"""
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
