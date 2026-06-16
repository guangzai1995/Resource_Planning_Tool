import asyncio

from conftest import FakeSession, sse


def _load():
    """先触发 shim，再取解析函数与数据类。"""
    import run_bench_serve
    serve = run_bench_serve._serve
    return (
        serve.ASYNC_REQUEST_FUNCS["openai"],
        serve.ASYNC_REQUEST_FUNCS["openai-chat"],
        serve.RequestFuncInput,
    )


def _run(fn, RequestFuncInput, backend_path, chunks):
    inp = RequestFuncInput(
        prompt="hi", api_url=f"http://x{backend_path}",
        prompt_len=1, output_len=8, model="m",
    )
    return asyncio.run(fn(inp, FakeSession(chunks)))


def test_chat_finish_reason_and_usage():
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 8}},
        "[DONE]",
    )
    out = _run(chat_fn, RequestFuncInput, "/v1/chat/completions", chunks)
    assert out.success
    assert out.finish_reason == "length"
    assert out.output_tokens == 8
    assert out.prompt_len == 5


def test_completions_finish_reason_and_usage():
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"text": "ab", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 8}},
        "[DONE]",
    )
    out = _run(completions_fn, RequestFuncInput, "/v1/completions", chunks)
    assert out.success
    assert out.finish_reason == "length"
    assert out.output_tokens == 8


def test_completions_usage_in_same_chunk_as_choices():
    """回归 Bug ②：choices 与 usage 出现在同一帧时，completions 不能漏读 completion_tokens。"""
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        # 服务端在最后一块同时带 choices（finish_reason）和 usage
        {"choices": [{"text": "ab", "finish_reason": "length"}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 8}},
        "[DONE]",
    )
    out = _run(completions_fn, RequestFuncInput, "/v1/completions", chunks)
    assert out.success
    assert out.output_tokens == 8, "completions 在 choices+usage 同帧时漏读了 usage"
    assert out.finish_reason == "length"

