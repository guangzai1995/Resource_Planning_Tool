import asyncio

import run_bench_serve


serve = run_bench_serve._serve


def run_coroutine(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_benchmark_passes_language_to_openai_audio_request_inputs(monkeypatch):
    captured = []

    async def fake_request(request_func_input, session, pbar=None):
        captured.append(request_func_input)
        return serve.RequestFuncOutput(
            success=True,
            prompt_len=request_func_input.prompt_len,
            output_tokens=3,
            generated_text="你好",
            latency=0.3,
            ttft=0.05,
            itl=[0.1, 0.1],
            input_audio_duration=1.25,
        )

    async def fake_fetch_spec_decode_metrics(base_url, session, extra_headers=None):
        return None

    monkeypatch.setitem(serve.ASYNC_REQUEST_FUNCS, "openai-audio", fake_request)
    monkeypatch.setattr(serve, "fetch_spec_decode_metrics", fake_fetch_spec_decode_metrics)

    input_requests = [
        serve.SampleRequest(
            prompt="Transcribe the audio.",
            prompt_len=0,
            expected_output_len=32,
            multi_modal_data={"audio_path": "/tmp/sample.wav"},
            request_id="audio-1",
        )
    ]

    run_coroutine(
        serve.benchmark(
            task_type=serve.TaskType.GENERATION,
            endpoint_type="openai-audio",
            api_url="http://localhost:8000/v1/audio/transcriptions",
            base_url="http://localhost:8000",
            model_id="qwen3-asr",
            model_name="qwen3-asr",
            tokenizer=None,
            input_requests=input_requests,
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            num_warmups=0,
            profile=False,
            selected_percentile_metrics=["ttft", "tpot", "itl", "e2el"],
            selected_percentiles=[50],
            ignore_eos=True,
            goodput_config_dict={},
            max_concurrency=1,
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
            warmup_concurrency=None,
            warmup_output_len=None,
            ready_check_timeout_sec=0,
            language="zh",
        )
    )

    assert [request.language for request in captured] == ["zh"]
    assert captured[-1].request_id == "audio-1"
