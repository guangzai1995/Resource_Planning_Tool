"""测试基础设施：确保 bench 目录可导入 + 提供 SSE/HTTP 夹具。"""
import json
import os
import sys

# 把 vllm_standalone_bench/ 与 tests/ 都加入 sys.path：
#   - bench 目录：使 run_bench_serve / run_bench_multi 可直接 import
#   - tests 目录：使 `from conftest import FakeSession, sse` 可用
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_DIR = os.path.dirname(_HERE)
for _p in (_BENCH_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _FakeChunkStream:
    """模拟 aiohttp response.content，按 iter_any() 逐块吐出 SSE 字节。

    chunks 可传入单个 bytes（整段 SSE，常见用法）或 bytes 列表（模拟分块到达）。
    """

    def __init__(self, chunks):
        if isinstance(chunks, (bytes, bytearray)):
            self._chunks = [bytes(chunks)]
        else:
            self._chunks = [c if isinstance(c, (bytes, bytearray)) else str(c).encode()
                            for c in chunks]

    async def iter_any(self):
        for c in self._chunks:
            yield c


class FakeResponse:
    """模拟 aiohttp.ClientResponse 的最小子集（status/content/text/上下文）。"""

    def __init__(self, chunks, status=200, reason="OK", body=b""):
        self.status = status
        self.reason = reason
        self.content = _FakeChunkStream(chunks)
        self._body = body

    async def text(self):
        return self._body.decode("utf-8", "replace")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """模拟 aiohttp.ClientSession.post —— 总是返回同一份预置 SSE 响应。"""

    def __init__(self, chunks, status=200):
        self._chunks = chunks
        self._status = status

    def post(self, *args, **kwargs):
        return FakeResponse(self._chunks, status=self._status)

    async def close(self):
        pass


def sse(*events):
    """把若干 data 事件打包成原始 SSE 字节流。

    每个 event：dict → ``data: {json}\\n\\n``；str → ``data: {str}\\n\\n``（用于 "[DONE]"）。
    """
    out = b""
    for ev in events:
        payload = ev if isinstance(ev, str) else json.dumps(ev)
        out += f"data: {payload}\n\n".encode("utf-8")
    return out
