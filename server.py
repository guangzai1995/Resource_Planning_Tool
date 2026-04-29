import os
import re
import threading
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.data_loader import DataLoader
from src.performance import Performance
from src.token_processor import InitTokenizer
from src.utils import get_available_tokenizers


# ------------------------------
# Helpers
# ------------------------------

def strip_markup(text: str) -> str:
    """Convert markdown/HTML-like output to plain text for API responses.

    - Remove HTML tags like <span ...> ... </span>
    - Remove common markdown emphasis **, headers #, and emoji remains as text
    """
    if not isinstance(text, str):
        return str(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove markdown headers/emphasis
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)  # headers
    text = text.replace("**", "").replace("__", "")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ------------------------------
# Shared state (thread-safe where needed)
# ------------------------------

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 日志配置：按天滚动，保留7天，UTF-8，INFO级别
log_path = os.path.join(LOG_DIR, 'server.log')
handler = TimedRotatingFileHandler(log_path, when='midnight', interval=1, backupCount=7, encoding='utf-8', utc=False)
fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s - %(message)s')
handler.setFormatter(fmt)
root_logger = logging.getLogger('resource_planning')
root_logger.setLevel(logging.INFO)
if not any(isinstance(h, TimedRotatingFileHandler) and getattr(h, 'baseFilename', '') == handler.baseFilename for h in root_logger.handlers):
    root_logger.addHandler(handler)

app = FastAPI(title="Resource Planning HTTP Service", version="1.0.0")

@app.on_event("startup")
def _on_startup():
    root_logger.info("Service starting up")


class AppState:
    def __init__(self):
        # Load once at startup
        self.data_loader = DataLoader()
        self.performance = Performance(self.data_loader.performance_data)

        # Tokenizer cache by model directory name under ./model
        self._tok_lock = threading.RLock()
        self._tokenizers: dict[str, InitTokenizer] = {}

    def get_or_create_tokenizer(self, model_name: Optional[str]) -> InitTokenizer:
        """Return cached InitTokenizer instance for a given model folder name.
        If model_name is None, use a reasonable default.
        """
        if not model_name:
            model_name = "Qwen2___5-0___5B-Instruct"
        with self._tok_lock:
            tok = self._tokenizers.get(model_name)
            if tok is not None:
                return tok
            # InitTokenizer expects a full model path
            base_model_dir = os.path.join(os.path.dirname(__file__), "model")
            model_path = os.path.join(base_model_dir, model_name)
            tok = InitTokenizer(model_path=model_path)
            # keep even if tokenizer_available is False to reuse fallback
            self._tokenizers[model_name] = tok
            return tok


state = AppState()

# ------------------------------
# Request models (JSON bodies)
# ------------------------------

class ConcurrencyRequest(BaseModel):
    gpu_model: str
    model_name: str
    card_count: int
    input_length: int
    max_delay: float = 5.0
    user_throughput: float = 10.0
    output_length: int = 1200


class TokensRequest(BaseModel):
    system_prompt: Optional[str] = ""
    user_prompt: Optional[str] = ""
    tokenizer_name: Optional[str] = None


class CostOptimizeRequest(BaseModel):
    target_concurrency: int
    model_name: str
    context_length: int
    max_delay: float = 5.0
# ------------------------------
# Missing-data suggestions
# ------------------------------

def suggest_valid_options(
    gpu_model: Optional[str],
    model_name: Optional[str],
    card_count: Optional[int],
    input_length: Optional[int],
) -> str:
    """Given a possibly invalid selection, return available options at the first invalid level.

    Priority of checks:
    1) gpu_model not found -> list all GPU models
    2) model_name not found under gpu -> list models for that gpu
    3) card_count not found under gpu/model -> list card counts for that combo
    4) input_length not found under gpu/model/card -> list input lengths for that combo
    """
    data = state.data_loader.performance_data
    try:
        if not isinstance(data, dict) or not data:
            return "(无可用性能数据)"

        if not gpu_model or gpu_model not in data:
            gpus = ", ".join(sorted(data.keys()))
            return f"可用GPU型号: {gpus}"

        models = sorted(data[gpu_model].keys())
        if not model_name or model_name not in data[gpu_model]:
            return f"{gpu_model} 可用模型: {', '.join(models)}"

        cards = sorted(data[gpu_model][model_name].keys())
        if card_count is None or card_count not in data[gpu_model][model_name]:
            return f"{gpu_model}/{model_name} 可用卡数: {', '.join(str(c) for c in cards)}"

        inputs = sorted(data[gpu_model][model_name][card_count].keys())
        if input_length is None or input_length not in data[gpu_model][model_name][card_count]:
            return f"{gpu_model}/{model_name}/{card_count}卡 可用输入长度: {', '.join(str(i) for i in inputs)}"

        # All valid; no suggestion needed
        return ""
    except Exception:
        return "(建议生成失败)"



# ------------------------------
# Routes
# ------------------------------

@app.get("/healthz", response_class=JSONResponse)
@app.post("/healthz", response_class=JSONResponse)
def healthz():
    root_logger.info("Health check")
    return {"status_code": 200, "status": "ok", "service": "resource-planning", "version": "1.0.0"}


@app.get("/tokenizers", response_class=JSONResponse)
@app.post("/tokenizers", response_class=JSONResponse)
def list_tokenizers():
    names = get_available_tokenizers()
    root_logger.info("List tokenizers: count=%d", len(names))
    return {"status_code": 200, "tokenizers": names}

@app.get("/options", response_class=JSONResponse)
def list_options(gpu_model: Optional[str] = None, model_name: Optional[str] = None, card_count: Optional[int] = None):
    """枚举可选项（层级式，JSON）

    - 无参数：返回完整的 GPU -> 模型 -> 卡数 -> 输入长度 的树
    - 传参过滤：返回指定层级的子树，并附带 available_xxx 列表
    """
    data = state.data_loader.performance_data
    result_tree = []
    available = {}

    try:
        # 枚举 GPU 层
        gpu_keys = sorted(data.keys()) if isinstance(data, dict) else []
        if gpu_model and gpu_model in data:
            gpu_iter = [gpu_model]
        else:
            gpu_iter = gpu_keys
        available["gpus"] = gpu_keys

        for g in gpu_iter:
            models = data.get(g, {}) if isinstance(data, dict) else {}
            model_keys = sorted(models.keys())
            if g == gpu_model:
                available["models"] = model_keys
            if model_name and g == gpu_model and model_name in models:
                model_iter = [model_name]
            else:
                model_iter = model_keys

            model_nodes = []
            for m in model_iter:
                cards = models.get(m, {}) if isinstance(models, dict) else {}
                card_keys = sorted(cards.keys())
                if g == gpu_model and m == model_name:
                    available["card_counts"] = card_keys
                if card_count is not None and g == gpu_model and m == model_name and card_count in cards:
                    card_iter = [card_count]
                else:
                    card_iter = card_keys

                card_nodes = []
                for c in card_iter:
                    inputs = cards.get(c, {}) if isinstance(cards, dict) else {}
                    input_lengths = sorted(inputs.keys())
                    if g == gpu_model and m == model_name and c == card_count:
                        available["input_lengths"] = input_lengths
                    card_nodes.append({
                        "card_count": c,
                        "input_lengths": input_lengths,
                    })

                model_nodes.append({
                    "model_name": m,
                    "card_counts": card_nodes,
                })

            result_tree.append({
                "gpu_model": g,
                "models": model_nodes,
            })
        root_logger.info("Options query gpu=%s model=%s cards=%s", gpu_model, model_name, card_count)
        return {
                "status_code": 200,
                "success": True,
                "data": {
                    "filters": {"gpu_model": gpu_model, "model_name": model_name, "card_count": card_count},
                    "available": available,
                    "options": result_tree,
                },
            }
    except Exception as e:
        root_logger.exception("Options internal error")
        return JSONResponse(status_code=500, content={
            "status_code": 500,
            "success": False,
            "error": {"message": f"内部错误: {e}", "reason": "calculation_error"}
        })


@app.post("/concurrency/calculate", response_class=JSONResponse)
def calculate_concurrency(req: ConcurrencyRequest):
    """最大并发计算接口 (JSON ONLY)
    成功: { success: true, data: {...结构化字段...} }
    失败: { success: false, error: { message, reason, suggestions?, input } }
    reason 枚举:
      - data_missing: 组合不存在或无数据
      - constraints_unsatisfied: 无法同时满足延时/吞吐约束
      - calculation_error: 内部计算失败
      - invalid: 其他无效输入
        路径: /concurrency/calculate
    """
    try:
        result, chart, max_conc = state.performance.calculate_max_concurrency_optimized(
            req.gpu_model,
            req.model_name,
            req.card_count,
            req.input_length,
            req.max_delay,
            req.user_throughput,
            req.output_length,
        )
    except Exception as e:  # 真正的内部异常
        root_logger.exception("Concurrency calc internal error")
        return JSONResponse(
            status_code=500,
            content={
                "status_code": 500,
                "success": False,
                "error": {
                    "message": f"内部错误: {e}",
                    "reason": "calculation_error",
                },
            },
        )

    # 判定成功
    if isinstance(max_conc, int) and max_conc > 0:
        # 直接使用结构化结果
        optimal = result.get("optimal", {}) if isinstance(result, dict) else {}
        boundary = result.get("boundary", {}) if isinstance(result, dict) else {}
        root_logger.info("Concurrency success gpu=%s model=%s cards=%d input=%d max_conc=%s", req.gpu_model, req.model_name, req.card_count, req.input_length, optimal.get("concurrency"))
        return {
            "status_code": 200,
                "success": True,
                "data": {
                    "gpu_model": result.get("gpu_model", req.gpu_model) if isinstance(result, dict) else req.gpu_model,
                    "model_name": result.get("model_name", req.model_name) if isinstance(result, dict) else req.model_name,
                    "card_count": result.get("card_count", req.card_count) if isinstance(result, dict) else req.card_count,
                    "input_length": result.get("input_length", req.input_length) if isinstance(result, dict) else req.input_length,
                    "output_length": result.get("output_length", req.output_length) if isinstance(result, dict) else req.output_length,
                    "max_delay": result.get("max_first_token_delay", req.max_delay) if isinstance(result, dict) else req.max_delay,
                    "user_throughput_requirement": result.get("user_throughput_requirement", req.user_throughput) if isinstance(result, dict) else req.user_throughput,
                    "max_concurrency": optimal.get("concurrency"),
                    "per_user_throughput": optimal.get("per_user_throughput"),
                    "total_throughput": optimal.get("total_throughput"),
                    "interpolation_used": optimal.get("interpolation_used"),
                    "boundary": boundary,
                },
            }

    # 失败分支: 解析 reason
    base_msg = (
        result.get("message") if isinstance(result, dict) and "message" in result else "请求参数无效或数据缺失"
    )
    suggestions = suggest_valid_options(req.gpu_model, req.model_name, req.card_count, req.input_length)
    reason = result.get("reason", "invalid") if isinstance(result, dict) else "invalid"

    status = 400 if reason in {"data_missing", "constraints_unsatisfied", "invalid"} else 500
    root_logger.info("Concurrency failure gpu=%s model=%s cards=%d input=%d reason=%s", req.gpu_model, req.model_name, req.card_count, req.input_length, reason)
    return JSONResponse(
        status_code=status,
        content={
            "status_code": status,
            "success": False,
            "error": {
                "message": strip_markup(base_msg),
                "reason": reason,
                "suggestions": suggestions if suggestions else None,
                "input": {
                    "gpu_model": req.gpu_model,
                    "model_name": req.model_name,
                    "card_count": req.card_count,
                    "input_length": req.input_length,
                    "max_delay": req.max_delay,
                    "user_throughput": req.user_throughput,
                    "output_length": req.output_length,
                },
            },
        },
    )


@app.post("/tokens/calculate", response_class=JSONResponse)
def calculate_tokens(req: TokensRequest):
    """Token计算器（文本返回）
    - 可指定 tokenizer_name（位于 ./model 下的目录名）
    - 未指定时使用默认 Qwen2___5-0___5B-Instruct
    """
    try:
        tok = state.get_or_create_tokenizer(req.tokenizer_name)
        stats = tok.calculate_input_tokens(req.system_prompt or "", req.user_prompt or "")
        status = "已加载" if tok.tokenizer_available and tok.tokenizer else "降级字符计数"
        root_logger.info("Tokens calc tokenizer=%s status=%s total=%s", req.tokenizer_name or 'Qwen2___5-0___5B-Instruct', status, stats.get('total_tokens'))
        return {
            "status_code": 200,
            "success": True,
            "data": {
                "tokenizer": req.tokenizer_name or 'Qwen2___5-0___5B-Instruct',
                "status": status,
                **stats,
            }
        }
    except Exception as e:
        root_logger.exception("Tokens calc internal error")
        return JSONResponse(status_code=500, content={
            "status_code": 500,
            "success": False,
            "error": {"message": f"内部错误: {e}", "reason": "calculation_error"}
        })


@app.post("/cost/optimize", response_class=JSONResponse)
def optimize_cost(req: CostOptimizeRequest):
    """成本核算器（文本返回）
    - 返回TOP3低成本方案的纯文本报告
    """
    try:
        solutions = state.performance.calculate_cost_optimization(
            req.target_concurrency, req.model_name, req.context_length, req.max_delay
        )
        if not solutions:
            root_logger.info("Cost optimize no solutions model=%s target=%d context=%d", req.model_name, req.target_concurrency, req.context_length)
            return JSONResponse(status_code=404, content={
                "status_code": 404,
                "success": False,
                "error": {"message": "未找到符合约束的方案", "reason": "data_missing", "input": req.model_dump()},
            })
        root_logger.info("Cost optimize success model=%s target=%d context=%d solutions=%d", req.model_name, req.target_concurrency, req.context_length, len(solutions))
        return {
            "status_code": 200,
            "success": True,
            "data": {
                "solutions": solutions,
            }
        }
    except HTTPException as he:
        root_logger.exception("Cost optimize HTTPException")
        # Pass through existing HTTP errors but ensure JSON shape
        return JSONResponse(status_code=he.status_code, content={
            "status_code": he.status_code,
            "success": False,
            "error": {"message": str(he.detail), "reason": "calculation_error"}
        })
    except Exception as e:
        root_logger.exception("Cost optimize internal error")
        return JSONResponse(status_code=500, content={
            "status_code": 500,
            "success": False,
            "error": {"message": f"内部错误: {e}", "reason": "calculation_error"}
        })


# ------------------------------
# Entrypoint for local run (uvicorn)
# ------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "15422"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
