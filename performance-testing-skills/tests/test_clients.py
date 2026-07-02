from scripts.lib.clients import (
    build_request,
    classify_error,
    make_curl,
)
from scripts.lib.config import load_config
from scripts.lib.datasets import load_dataset


def test_build_openai_chat_request():
    request = build_request(
        {"protocol": "openai_chat", "base_url": "http://host/v1", "model": "m", "request": {"max_tokens": 8}},
        {"prompt": "hello"},
    )

    assert request["method"] == "POST"
    assert request["url"] == "http://host/v1/chat/completions"
    assert request["json"]["model"] == "m"
    assert request["json"]["messages"][0]["content"] == "hello"


def test_build_openai_completion_request():
    request = build_request(
        {"protocol": "openai_completion", "base_url": "http://host/v1", "model": "m", "request": {"max_tokens": 8}},
        {"prompt": "hello"},
    )

    assert request["url"] == "http://host/v1/completions"
    assert request["json"]["prompt"] == "hello"


def test_build_openai_asr_request():
    request = build_request(
        {"protocol": "openai_asr", "base_url": "http://host/v1", "model": "asr", "request": {"language": "en"}},
        {"audio": "sample.wav", "prompt": "transcribe"},
    )

    assert request["url"] == "http://host/v1/audio/transcriptions"
    assert request["multipart"]["model"] == "asr"
    assert request["multipart"]["language"] == "en"
    assert request["multipart_file"] == "sample.wav"


def test_build_generic_http_request_replaces_template_values():
    request = build_request(
        {
            "protocol": "generic_http",
            "method": "POST",
            "url": "http://host/infer",
            "body_template": {"input": "${prompt}", "max_tokens": "${max_tokens}"},
        },
        {"prompt": "hello", "expected_output_len": 9},
    )

    assert request["json"] == {"input": "hello", "max_tokens": 9}


def test_build_generic_http_request_preserves_loaded_body_templates():
    config = load_config("configs/generic_http.json")
    rows = load_dataset(config["dataset"])

    request = build_request(config, rows[0])

    assert request["json"]["prompt"] == rows[0]["prompt"]
    assert request["json"]["max_tokens"] == rows[0]["expected_output_len"]
    assert isinstance(request["json"]["max_tokens"], int)


def test_make_curl_contains_method_url_and_json():
    curl = make_curl({
        "method": "POST",
        "url": "http://host/v1/chat/completions",
        "headers": {"Content-Type": "application/json"},
        "json": {"model": "m"},
    })

    assert "curl" in curl
    assert "-X POST" in curl
    assert "http://host/v1/chat/completions" in curl
    assert "'Content-Type: application/json'" in curl


def test_classify_error_maps_common_status_codes():
    assert classify_error(status_code=401, exception=None) == "auth_error"
    assert classify_error(status_code=404, exception=None) == "not_found"
    assert classify_error(status_code=422, exception=None) == "bad_request"
    assert classify_error(status_code=500, exception=None) == "http_5xx"
