import importlib.util
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepares_openai_chat_request():
    clients = load_module("skill_clients_chat", "scripts/lib/clients.py")

    prepared = clients.prepare_request(
        target={
            "type": "openai-chat",
            "base_url": "http://localhost:8000",
            "endpoint": "/v1/chat/completions",
            "model": "demo-model",
            "headers": {"Authorization": "Bearer test"},
        },
        sample={"prompt": "Say hello"},
    )

    assert prepared["url"] == "http://localhost:8000/v1/chat/completions"
    assert prepared["method"] == "POST"
    assert prepared["headers"]["Content-Type"] == "application/json"
    assert prepared["headers"]["Authorization"] == "Bearer test"
    assert prepared["body"]["model"] == "demo-model"
    assert prepared["body"]["messages"][0]["content"] == "Say hello"


def test_prepares_completion_and_generic_requests():
    clients = load_module("skill_clients_generic", "scripts/lib/clients.py")

    completion = clients.prepare_request(
        target={
            "type": "openai-completion",
            "base_url": "http://localhost:8000",
            "endpoint": "/v1/completions",
            "model": "demo-model",
        },
        sample={"prompt": "Complete this"},
    )
    generic = clients.prepare_request(
        target={
            "type": "generic-http",
            "base_url": "http://localhost:8000",
            "endpoint": "/score",
            "body_template": {"text": "{prompt}", "tag": "smoke"},
        },
        sample={"prompt": "payload"},
    )

    assert completion["body"]["prompt"] == "Complete this"
    assert generic["body"] == {"text": "payload", "tag": "smoke"}


def test_request_json_posts_json_body(monkeypatch):
    clients = load_module("skill_clients_http", "scripts/lib/clients.py")
    seen = {}

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({"ok": True}).encode("utf-8")

    def fake_urlopen(request, timeout):
        seen["timeout"] = timeout
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(clients.urllib.request, "urlopen", fake_urlopen)
    response = clients.request_json(
        {
            "method": "POST",
            "url": "http://127.0.0.1:8000/echo",
            "headers": {"Content-Type": "application/json"},
            "body": {"prompt": "hello"},
        },
        timeout_sec=2,
    )

    assert response.status_code == 200
    assert response.json_body == {"ok": True}
    assert seen["timeout"] == 2
    assert seen["url"] == "http://127.0.0.1:8000/echo"
    assert seen["body"] == {"prompt": "hello"}
