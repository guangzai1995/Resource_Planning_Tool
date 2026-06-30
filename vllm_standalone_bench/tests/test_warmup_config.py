"""warmup 固定并发预热的参数透传测试。"""
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
