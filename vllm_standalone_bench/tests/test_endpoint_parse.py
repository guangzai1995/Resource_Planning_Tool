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
