import asyncio
import sys
from types import SimpleNamespace

import pytest

from conftest import FakeResponse, sse
from vllm_bench.lib.endpoint_request_func import (
    RequestFuncInput,
    async_request_openai_audio,
)


class CapturingSession:
    def __init__(self, chunks):
        self._chunks = chunks
        self.post_kwargs = None

    def post(self, **kwargs):
        self.post_kwargs = kwargs
        return FakeResponse(self._chunks)


def _fields_by_name(form):
    return {
        disposition["name"]: (disposition, headers, value)
        for disposition, headers, value in form._fields
    }


def _run_audio_request(request_func_input, session):
    return asyncio.run(async_request_openai_audio(request_func_input, session))


@pytest.fixture
def fake_soundfile(monkeypatch):
    def write(buffer, y, sr, format):
        buffer.write(b"fake-wav-bytes")

    module = SimpleNamespace(
        write=write,
        info=lambda fileobj: SimpleNamespace(duration=2.5),
    )
    monkeypatch.setitem(sys.modules, "soundfile", module)
    return module


def test_openai_audio_request_uploads_audio_path(fake_soundfile, tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake-audio")
    session = CapturingSession(
        sse(
            {"choices": [{"delta": {"content": "transcript"}}]},
            {"usage": {"completion_tokens": 17}},
            "[DONE]",
        )
    )
    request = RequestFuncInput(
        prompt="transcribe",
        api_url="http://localhost:8000/v1/audio/transcriptions",
        prompt_len=3,
        output_len=128,
        model="qwen3-asr",
        multi_modal_content={"audio_path": str(audio)},
        language="zh",
    )

    output = _run_audio_request(request, session)

    assert output.success is True
    assert output.generated_text == "transcript"
    assert output.output_tokens == 17
    assert output.input_audio_duration == 2.5
    assert output.prompt_len == 3
    assert session.post_kwargs["url"].endswith("/audio/transcriptions")

    fields = _fields_by_name(session.post_kwargs["data"])
    assert fields["language"][2] == "zh"
    assert fields["max_completion_tokens"][2] == "128"
    assert fields["model"][2] == "qwen3-asr"
    file_disposition, file_headers, audio_file = fields["file"]
    assert file_disposition["filename"] == "sample.wav"
    assert file_headers["Content-Type"] == "audio/wav"
    assert audio_file.closed is True


def test_openai_audio_request_preserves_audio_tuple_path(fake_soundfile):
    session = CapturingSession(
        sse(
            {"choices": [{"delta": {"content": "hello"}}]},
            {"usage": {"completion_tokens": 9}},
            "[DONE]",
        )
    )
    request = RequestFuncInput(
        prompt="transcribe",
        api_url="http://localhost:8000/v1/audio/transcriptions",
        prompt_len=4,
        output_len=32,
        model="qwen3-asr",
        multi_modal_content={"audio": ([0.0], 16000)},
    )

    output = _run_audio_request(request, session)

    assert output.success is True
    assert output.generated_text == "hello"
    assert output.output_tokens == 9
    assert output.input_audio_duration == 2.5


def test_openai_audio_request_reads_usage_in_same_chunk_as_choices(fake_soundfile):
    session = CapturingSession(
        sse(
            {
                "choices": [{"delta": {"content": "hello"}}],
                "usage": {"completion_tokens": 21},
            },
            "[DONE]",
        )
    )
    request = RequestFuncInput(
        prompt="transcribe",
        api_url="http://localhost:8000/v1/audio/transcriptions",
        prompt_len=4,
        output_len=32,
        model="qwen3-asr",
        multi_modal_content={"audio": ([0.0], 16000)},
    )

    output = _run_audio_request(request, session)

    assert output.success is True
    assert output.generated_text == "hello"
    assert output.output_tokens == 21


def test_openai_audio_request_requires_audio_path_or_audio(fake_soundfile):
    request = RequestFuncInput(
        prompt="transcribe",
        api_url="http://localhost:8000/v1/audio/transcriptions",
        prompt_len=1,
        output_len=32,
        model="qwen3-asr",
        multi_modal_content={},
    )

    with pytest.raises(TypeError, match="audio_path.*audio"):
        _run_audio_request(request, CapturingSession(b""))
