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


def test_chat_choice_without_delta_does_not_crash():
    """健壮性：chat 帧的 choice 缺 delta 键（如仅带 finish_reason 的终止帧变体）不应
    抛 KeyError 把整请求判失败；finish_reason 与 usage 仍应正确解析。"""
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"finish_reason": "length"}]},   # choice 无 delta 键
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}},
        "[DONE]",
    )
    out = _run(chat_fn, RequestFuncInput, "/v1/chat/completions", chunks)
    assert out.success
    assert out.finish_reason == "length"
    assert out.output_tokens == 5


def test_usage_without_completion_tokens_keeps_zero():
    """健壮性：usage 帧缺 completion_tokens 时 output_tokens 应保持 0（int），不能被赋成 None。"""
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
        {"choices": [], "usage": {"prompt_tokens": 3}},   # 无 completion_tokens
        "[DONE]",
    )
    out = _run(chat_fn, RequestFuncInput, "/v1/chat/completions", chunks)
    assert out.success
    assert out.output_tokens == 0          # 不是 None
    assert out.prompt_len == 3


def test_chat_parses_cached_tokens_nested():
    """OpenAI 标准：usage.prompt_tokens_details.cached_tokens（嵌套）。"""
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 8,
                   "prompt_tokens_details": {"cached_tokens": 80}}},
        "[DONE]",
    )
    out = _run(chat_fn, RequestFuncInput, "/v1/chat/completions", chunks)
    assert out.success
    assert out.cached_tokens == 80
    assert out.cached_reported is True


def test_completions_parses_cached_tokens_flat():
    """兼容：部分版本平铺为 usage.cached_tokens。"""
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"text": "ab", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": "length"}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 8,
                   "cached_tokens": 60}},
        "[DONE]",
    )
    out = _run(completions_fn, RequestFuncInput, "/v1/completions", chunks)
    assert out.success
    assert out.cached_tokens == 60
    assert out.cached_reported is True


def test_cached_tokens_absent_keeps_zero_and_unreported():
    """服务端未上报 cached_tokens（如未开 prefix caching）：保持 0、reported=False。"""
    completions_fn, chat_fn, RequestFuncInput = _load()
    chunks = sse(
        {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 8}},
        "[DONE]",
    )
    out = _run(chat_fn, RequestFuncInput, "/v1/chat/completions", chunks)
    assert out.success
    assert out.cached_tokens == 0
    assert out.cached_reported is False
