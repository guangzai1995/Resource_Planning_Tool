import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from scripts.lib.clients import build_request, send_request


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        response = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "echo": payload,
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


class ErrorHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        encoded = b"bad request"
        self.send_response(422)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


def test_send_request_posts_json_to_local_server():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = {"protocol": "openai_chat", "base_url": f"http://127.0.0.1:{server.server_port}/v1", "model": "m"}
        request = build_request(config, {"prompt": "hello"})
        result = send_request("req-1", request, timeout_seconds=5)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert result["success"] is True
    assert result["status_code"] == 200
    assert result["output_tokens"] == 2
    assert result["latency_ms"] >= 0


def test_send_request_returns_failure_metadata_for_http_error():
    server = HTTPServer(("127.0.0.1", 0), ErrorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = {
            "method": "POST",
            "url": f"http://127.0.0.1:{server.server_port}/fail",
            "json": {"prompt": "hello"},
        }
        result = send_request("req-err", request, timeout_seconds=5)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert result["success"] is False
    assert result["status_code"] == 422
    assert result["error_type"] == "bad_request"
    assert result["error_message"]
    assert result["response_summary"] == "bad request"
