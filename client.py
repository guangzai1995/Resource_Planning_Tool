"""
运行脚本：调用 FastAPI 服务的各个接口

用途：命令行一站式调用健康检查、列出分词器、计算最大并发、计算 Token 数、成本优化。
默认通过 HTTP 调用运行中的服务，亦支持 --use-testclient 直接在进程内调用（开发调试用）。

示例：
  # 并发计算
  python client.py concurrency --gpu-model 4090 --model-name 32B --cards 8 --input-length 2048 --max-delay 5 --user-throughput 10 --output-length 1200

  # Token 计算
  python client.py tokens --system "你好" --user "今天天气如何" --tokenizer-name Qwen2___5-0___5B-Instruct

  # 成本优化
  python client.py cost --target 50 --model-name 32B --context-length 2048 --max-delay 5
"""

import argparse
import json
import sys
from typing import Optional, Tuple


def _pretty(obj) -> str:
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return str(obj)


def _build_base_url(host: str, port: int) -> str:
    return f"{host}:{port}"


def _http_get(base_url: str, path: str, params: dict | None, timeout: float) -> Tuple[int | None, str | None, dict | None]:
    import requests
    url = base_url.rstrip("/") + path
    try:
        r = requests.get(url, params=params, timeout=timeout)
        try:
            return r.status_code, r.text, r.json()
        except Exception:
            return r.status_code, r.text, None
    except Exception as e:
        return None, None, {"error": str(e)}


def _http_post(base_url: str, path: str, json_body: dict | None, timeout: float) -> Tuple[int | None, str | None, dict | None]:
    import requests
    url = base_url.rstrip("/") + path
    try:
        r = requests.post(url, json=json_body, timeout=timeout)
        try:
            return r.status_code, r.text, r.json()
        except Exception:
            return r.status_code, r.text, None
    except Exception as e:
        return None, None, {"error": str(e)}


class _TestClientWrapper:
    def __init__(self, app):
        from fastapi.testclient import TestClient
        self._client = TestClient(app)

    def get(self, path: str, params: dict | None):
        r = self._client.get(path, params=params)
        try:
            return r.status_code, r.text, r.json()
        except Exception:
            return r.status_code, r.text, None

    def post(self, path: str, json_body: dict | None):
        r = self._client.post(path, json=json_body)
        try:
            return r.status_code, r.text, r.json()
        except Exception:
            return r.status_code, r.text, None


def cmd_health(args) -> int:
    base_url = _build_base_url(args.host, args.port)
    if args.use_testclient:
        from server import app
        client = _TestClientWrapper(app)
        status, text, j = client.get("/healthz", None)
    else:
        status, text, j = _http_get(base_url, "/healthz", None, args.timeout)
    print(_pretty(j or {"status": status, "text": text}))
    return 0 if (status and status < 300) else 1


def cmd_tokenizers(args) -> int:
    base_url = _build_base_url(args.host, args.port)
    if args.use_testclient:
        from server import app
        client = _TestClientWrapper(app)
        status, text, j = client.get("/tokenizers", None)
    else:
        status, text, j = _http_get(base_url, "/tokenizers", None, args.timeout)
    print(_pretty(j or {"status": status, "text": text}))
    return 0 if (status and status < 300) else 1


def cmd_concurrency(args) -> int:
    base_url = _build_base_url(args.host, args.port)
    body = {
        "gpu_model": args.gpu_model,
        "model_name": args.model_name,
        "card_count": args.cards,
        "input_length": args.input_length,
        "max_delay": args.max_delay,
        "user_throughput": args.user_throughput,
        "output_length": args.output_length,
    }
    if args.use_testclient:
        from server import app
        client = _TestClientWrapper(app)
        status, text, j = client.post("/concurrency/calculate", body)
    else:
        status, text, j = _http_post(base_url, "/concurrency/calculate", body, args.timeout)
    print(_pretty(j or {"status": status, "text": text}))
    if j and isinstance(j, dict):
        # 根据 success 字段设置退出码
        return 0 if j.get("success") else 2
    return 0 if (status and status < 300) else 2


def cmd_tokens(args) -> int:
    base_url = _build_base_url(args.host, args.port)
    body = {
        "system_prompt": args.system or "",
        "user_prompt": args.user or "",
        "tokenizer_name": args.tokenizer_name,
    }
    if args.use_testclient:
        from server import app
        client = _TestClientWrapper(app)
        status, text, j = client.post("/tokens/calculate", body)
    else:
        status, text, j = _http_post(base_url, "/tokens/calculate", body, args.timeout)
    print(_pretty(j or {"status": status, "text": text}))
    if j and isinstance(j, dict):
        return 0 if j.get("success") else 2
    return 0 if (status and status < 300) else 2


def cmd_cost(args) -> int:
    base_url = _build_base_url(args.host, args.port)
    body = {
        "target_concurrency": args.target,
        "model_name": args.model_name,
        "context_length": args.context_length,
        "max_delay": args.max_delay,
    }
    if args.use_testclient:
        from server import app
        client = _TestClientWrapper(app)
        status, text, j = client.post("/cost/optimize", body)
    else:
        status, text, j = _http_post(base_url, "/cost/optimize", body, args.timeout)
    print(_pretty(j or {"status": status, "text": text}))
    if j and isinstance(j, dict):
        return 0 if j.get("success") else 2
    return 0 if (status and status < 300) else 2


def cmd_options(args) -> int:
    base_url = _build_base_url(args.host, args.port)
    params = {}
    if args.gpu_model:
        params["gpu_model"] = args.gpu_model
    if args.model_name:
        params["model_name"] = args.model_name
    if args.cards is not None:
        params["card_count"] = args.cards

    if args.use_testclient:
        from server import app
        client = _TestClientWrapper(app)
        status, text, j = client.get("/options", params)
    else:
        status, text, j = _http_get(base_url, "/options", params, args.timeout)
    print(_pretty(j or {"status": status, "text": text}))
    if j and isinstance(j, dict):
        return 0 if j.get("success") else 2
    return 0 if (status and status < 300) else 2

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="资源规划服务运行脚本")
    parser.add_argument("--host", default="http://localhost", help="服务Host（包含协议，如 http://localhost）")
    parser.add_argument("--port", default=15422, type=int, help="服务端口")
    parser.add_argument("--timeout", default=20.0, type=float, help="HTTP 超时时间(秒)")
    parser.add_argument("--use-testclient", action="store_true", help="在本进程内调用FastAPI（开发调试）")

    sub = parser.add_subparsers(dest="command", required=True)

    # health
    p_health = sub.add_parser("healthz", help="健康检查")
    p_health.set_defaults(func=cmd_health)

    # tokenizers
    p_tok = sub.add_parser("tokenizers", help="列出可用分词器")
    p_tok.set_defaults(func=cmd_tokenizers)

    # options
    p_opts = sub.add_parser("options", help="列举可选项 (GPU/模型/卡数/输入长度)")
    p_opts.add_argument("--gpu-model", dest="gpu_model", default=None)
    p_opts.add_argument("--model-name", dest="model_name", default=None)
    p_opts.add_argument("--cards", dest="cards", type=int, default=None)
    p_opts.set_defaults(func=cmd_options)
    
    # concurrency
    p_conc = sub.add_parser("concurrency", help="计算最大并发")
    p_conc.add_argument("--gpu-model", required=True, dest="gpu_model")
    p_conc.add_argument("--model-name", required=True, dest="model_name")
    p_conc.add_argument("--cards", required=True, type=int)
    p_conc.add_argument("--input-length", required=True, type=int)
    p_conc.add_argument("--max-delay", type=float, default=5.0)
    p_conc.add_argument("--user-throughput", type=float, default=10.0)
    p_conc.add_argument("--output-length", type=int, default=1200)
    p_conc.set_defaults(func=cmd_concurrency)

    # tokens
    p_tokens = sub.add_parser("tokens", help="计算输入Token统计")
    p_tokens.add_argument("--system", default="", help="系统提示词文本")
    p_tokens.add_argument("--user", default="", help="用户输入文本")
    p_tokens.add_argument("--tokenizer-name", default=None)
    p_tokens.set_defaults(func=cmd_tokens)

    # cost
    p_cost = sub.add_parser("cost", help="成本优化")
    p_cost.add_argument("--target", type=int, required=True, help="目标最大并发用户数")
    p_cost.add_argument("--model-name", required=True)
    p_cost.add_argument("--context-length", type=int, required=True)
    p_cost.add_argument("--max-delay", type=float, default=5.0)
    p_cost.set_defaults(func=cmd_cost)



    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
