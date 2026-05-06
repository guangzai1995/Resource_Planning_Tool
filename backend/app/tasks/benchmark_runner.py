"""
异步后台任务：直接调用 OpenAI 兼容 API 进行并发压测，无需启动 vLLM 服务。

核心流程：
  对每个 (input_tokens, concurrency) 组合：
    1. 生成近似 input_tokens 长度的提示文本
    2. 并发发送 num_prompts 个流式请求
    3. 统计 TTFT / 增量延时 / 系统吞吐
  全部完成后自动写入 BenchmarkData 表
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import httpx
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── 全局任务状态（进程内） ────────────────────────────────────
_task_queues: dict[str, asyncio.Queue] = {}
_task_status: dict[str, str] = {}
_task_results: dict[str, list[dict]] = {}
_cancel_flags: dict[str, bool] = {}


# ── 工具函数 ──────────────────────────────────────────────────

def _update_status_db(task_id: str, status: str, error: Optional[str] = None):
    """同步更新 SQLite 中的任务状态"""
    from app.core.database import db_session
    from app.models.benchmark_run import BenchmarkRun

    try:
        with db_session() as db:
            run = db.query(BenchmarkRun).filter_by(run_uuid=task_id).first()
            if run:
                run.status = status
                if status == "running":
                    run.started_at = datetime.now().isoformat()
                elif status in ("done", "failed"):
                    run.finished_at = datetime.now().isoformat()
                if error:
                    run.error_message = error
    except Exception as e:
        logger.warning("db_update_error", task_id=task_id, error=str(e))


def _generate_prompt(input_tokens: int, tokenize_fn: Optional[Callable] = None) -> str:
    """生成约 input_tokens 个 token 的英文提示文本"""
    unit = (
        "The quick brown fox jumps over the lazy dog. "
        "In modern artificial intelligence research, large language models have demonstrated "
        "remarkable capabilities across diverse tasks including reasoning, code generation, "
        "and natural language understanding. "
    )
    if tokenize_fn is None:
        chars_needed = max(input_tokens * 4, 40)
        full = unit * (chars_needed // len(unit) + 2)
        return full[:chars_needed]

    # 使用 tokenizer 精确控制 token 数
    text = ""
    while len(tokenize_fn(text)) < input_tokens:
        text += unit
    actual = len(tokenize_fn(text))
    if actual > input_tokens:
        ratio = input_tokens / actual
        text = text[: int(len(text) * ratio)]
    return text


def _build_chat_payload(
    model_name: str,
    prompt: str,
    output_tokens: int,
    strategy: str = "prefill",
) -> dict:
    """
    为 chat/completions 路由构建 payload，支持两种策略：

    strategy="prefill"：
        在 messages 末尾加入空 assistant 消息，让模型从空内容续写。
        行为最接近 /completions，模型不会因"任务完成"提前停止。

    strategy="prompt_eng"：
        通过 system prompt + user 指令引导模型输出足够长度的内容。
        兼容性最好，适用于不支持 prefill 的 API。
    """
    if strategy == "prefill":
        return {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},  # 引导从空内容续写
            ],
            "max_tokens": output_tokens,
            "temperature": 0,
            "stream": True,
        }
    else:  # prompt_eng
        return {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Always generate complete, detailed responses. "
                        "Never stop early. Fill your entire response thoroughly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Please write a detailed technical analysis and thorough explanation of "
                        f"the following passage. Your response must be comprehensive and fill at "
                        f"least {output_tokens} tokens:\n\n{prompt}"
                    ),
                },
            ],
            "max_tokens": output_tokens,
            "temperature": 0,
            "stream": True,
        }


def _load_tokenizer(model_name_or_path: Optional[str]) -> Optional[Callable]:
    """按优先级加载 tokenizer：本地 model/ 目录 → modelscope → huggingface"""
    if not model_name_or_path:
        return None

    # 1. 尝试本地路径
    workspace_model_dir = Path(__file__).parents[4] / "model"
    candidates: list[Path] = [
        Path(model_name_or_path),
        workspace_model_dir / model_name_or_path,
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
                logger.info("tokenizer_loaded", source="local", path=str(p))
                return tok.encode
            except Exception as e:
                logger.warning("tokenizer_local_failed", path=str(p), error=str(e))

    # 2. modelscope 下载
    try:
        from modelscope import AutoTokenizer as MsAutoTokenizer  # type: ignore
        tok = MsAutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        logger.info("tokenizer_loaded", source="modelscope", model=model_name_or_path)
        return tok.encode
    except Exception as e:
        logger.warning("modelscope_tokenizer_failed", model=model_name_or_path, error=str(e))

    # 3. huggingface 下载
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        logger.info("tokenizer_loaded", source="huggingface", model=model_name_or_path)
        return tok.encode
    except Exception as e:
        logger.warning("hf_tokenizer_failed", model=model_name_or_path, error=str(e))

    logger.warning("tokenizer_unavailable", model=model_name_or_path)
    return None


def _pct(lst: list[float], p: int) -> Optional[float]:
    return float(np.percentile(lst, p)) if lst else None


# ── 单次请求 ──────────────────────────────────────────────────

async def _benchmark_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: dict,
    tokenize_fn: Optional[Callable] = None,
) -> dict:
    """
    执行一次流式请求，返回 ttft_ms、decode 间隔列表、token 数。

    Token 计数优先级：
      1. usage.completion_tokens（最准确，最后一个 SSE chunk 里）
      2. tokenizer 对收集到的文本进行编码计数
      3. 含文本的 SSE chunk 数量（兜底）

    同时支持：
      - /completions：choices[].text
      - /chat/completions：choices[].delta.content
      - 思考模型：choices[].delta.reasoning_content（与 text/content 一并计入 token）
    """
    ttft_ms: Optional[float] = None
    token_times: list[float] = []
    text_chunks: list[str] = []
    completion_tokens: Optional[int] = None
    start = time.perf_counter()

    try:
        async with client.stream("POST", url, headers=headers, json=payload, timeout=300.0) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                error_body = body.decode("utf-8", errors="replace")[:400]
                return {
                    "error": f"HTTP {resp.status_code}: {error_body}",
                    "ttft_ms": None,
                    "decode_intervals": [],
                    "n_tokens": 0,
                    "elapsed_s": time.perf_counter() - start,
                }

            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue

                now = time.perf_counter()
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # 从 usage 获取 completion_tokens（最准确）
                usage = chunk.get("usage") or {}
                ct = usage.get("completion_tokens")
                if ct is not None:
                    completion_tokens = int(ct)

                # 提取文本内容（同时支持 completions / chat / thinking）
                chunk_text = ""
                for choice in chunk.get("choices") or []:
                    chunk_text += choice.get("text") or ""
                    delta = choice.get("delta") or {}
                    chunk_text += delta.get("content") or ""
                    chunk_text += delta.get("reasoning_content") or ""

                if chunk_text:
                    if ttft_ms is None:
                        ttft_ms = (now - start) * 1000
                    token_times.append(now)
                    text_chunks.append(chunk_text)

    except Exception as e:
        return {
            "error": str(e),
            "ttft_ms": None,
            "decode_intervals": [],
            "n_tokens": 0,
            "elapsed_s": time.perf_counter() - start,
        }

    # Token 计数：usage > tokenizer > SSE chunk 数
    if completion_tokens is not None:
        n_tokens = completion_tokens
    elif tokenize_fn is not None and text_chunks:
        n_tokens = len(tokenize_fn("".join(text_chunks)))
    else:
        n_tokens = len(token_times)

    decode_intervals = [
        (token_times[i] - token_times[i - 1]) * 1000
        for i in range(1, len(token_times))
    ]
    return {
        "ttft_ms": ttft_ms,
        "decode_intervals": decode_intervals,
        "n_tokens": n_tokens,
        "elapsed_s": time.perf_counter() - start,
        "error": None,
    }


# ── 单个测试点（一组 input_tokens + concurrency） ────────────

async def _benchmark_point(
    api_url: str,
    headers: dict,
    payload: dict,
    concurrency: int,
    epochs: int,
    log_fn=None,
    tokenize_fn: Optional[Callable] = None,
) -> dict:
    """
    epoch 模式：与 benchmark_parallel.py 逻辑一致。
    每个 epoch 同时发 concurrency 个请求，等所有完成后进入下一 epoch。
    总请求数 = epochs × concurrency。
    """
    all_ttft: list[float] = []
    all_decode: list[float] = []
    total_tokens = 0
    error_count = 0
    total_requests = epochs * concurrency

    wall_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        for ep in range(epochs):
            ep_ttft: list[float] = []
            ep_tokens = 0
            ep_start = time.perf_counter()

            tasks = [
                _benchmark_request(client, api_url, headers, payload, tokenize_fn)
                for _ in range(concurrency)
            ]
            results = await asyncio.gather(*tasks)

            ep_elapsed = time.perf_counter() - ep_start
            for r in results:
                if r.get("error"):
                    error_count += 1
                    if log_fn:
                        await log_fn(f"  [WARN] {r['error'][:160]}")
                if r.get("ttft_ms") is not None:
                    all_ttft.append(r["ttft_ms"])
                    ep_ttft.append(r["ttft_ms"])
                all_decode.extend(r.get("decode_intervals", []))
                total_tokens += r.get("n_tokens", 0)
                ep_tokens += r.get("n_tokens", 0)

            if log_fn:
                ep_tpt = ep_tokens / ep_elapsed if ep_elapsed > 0 else 0.0
                ep_ttft_mean = float(np.mean(ep_ttft)) if ep_ttft else 0.0
                await log_fn(
                    f"  [Epoch {ep + 1}/{epochs}]  "
                    f"吞吐: {ep_tpt:.1f} tok/s  "
                    f"TTFT均值: {ep_ttft_mean:.1f}ms"
                )

    wall_elapsed = time.perf_counter() - wall_start
    throughput = total_tokens / wall_elapsed if wall_elapsed > 0 else 0.0
    throughput_per_user = throughput / concurrency if concurrency > 0 else 0.0

    return {
        "throughput_tokens_s": round(throughput, 2),
        "throughput_per_user_tokens_s": round(throughput_per_user, 2),
        "ttft_mean_ms": round(float(np.mean(all_ttft)), 2) if all_ttft else None,
        "ttft_p90_ms": round(_pct(all_ttft, 90), 2) if all_ttft else None,
        "ttft_p99_ms": round(_pct(all_ttft, 99), 2) if all_ttft else None,
        "ttft_max_ms": round(max(all_ttft), 2) if all_ttft else None,
        "decode_latency_mean_ms": round(float(np.mean(all_decode)), 2) if all_decode else None,
        "decode_latency_p90_ms": round(_pct(all_decode, 90), 2) if all_decode else None,
        "decode_latency_p99_ms": round(_pct(all_decode, 99), 2) if all_decode else None,
        "decode_latency_max_ms": round(max(all_decode), 2) if all_decode else None,
        "error_count": error_count,
        "total_requests": total_requests,
    }


# ── 写入数据库 ────────────────────────────────────────────────

def _import_results_to_db(
    task_id: str,
    results: list[dict],
    gpu_name: str,
    model_name: str,
    gpu_count: int,
) -> None:
    """将压测结果 UPSERT 到 BenchmarkData 表，并清除预测缓存"""
    from app.core.database import db_session
    from app.core.cache import clear_prediction_cache
    from app.models.benchmark_data import BenchmarkData
    from app.models.benchmark_run import BenchmarkRun

    try:
        with db_session() as db:
            run = db.query(BenchmarkRun).filter_by(run_uuid=task_id).first()
            run_id = run.id if run else None

            for row in results:
                key = dict(
                    gpu_name=gpu_name,
                    model_name=model_name,
                    gpu_count=gpu_count,
                    input_tokens=int(row["input_tokens"]),
                    output_tokens=int(row["output_tokens"]),
                    concurrency=int(row["concurrency"]),
                )
                existing = db.query(BenchmarkData).filter_by(**key).first()
                fields = {
                    **key,
                    "run_id": run_id,
                    "throughput_tokens_s": row.get("throughput_tokens_s"),
                    "throughput_per_user_tokens_s": row.get("throughput_per_user_tokens_s"),
                    "ttft_mean_ms": row.get("ttft_mean_ms"),
                    "ttft_p90_ms": row.get("ttft_p90_ms"),
                    "ttft_p99_ms": row.get("ttft_p99_ms"),
                    "ttft_max_ms": row.get("ttft_max_ms"),
                    "decode_latency_mean_ms": row.get("decode_latency_mean_ms"),
                    "decode_latency_p90_ms": row.get("decode_latency_p90_ms"),
                    "decode_latency_p99_ms": row.get("decode_latency_p99_ms"),
                    "decode_latency_max_ms": row.get("decode_latency_max_ms"),
                }
                if existing:
                    for k, v in fields.items():
                        setattr(existing, k, v)
                else:
                    db.add(BenchmarkData(**fields))

        clear_prediction_cache()
        logger.info("results_imported", task_id=task_id, count=len(results))
    except Exception as e:
        logger.error("import_error", task_id=task_id, error=str(e))


# ── 主压测任务 ────────────────────────────────────────────────

async def run_benchmark(task_id: str, config) -> None:
    """主压测任务：异步直接调用 OpenAI 兼容 API，无需子进程"""
    queue: asyncio.Queue = asyncio.Queue()
    _task_queues[task_id] = queue
    _task_status[task_id] = "running"
    _update_status_db(task_id, "running")

    async def log(msg: str):
        await queue.put(msg)

    _cancel_flags[task_id] = False
    all_results: list[dict] = []
    try:
        base = config.api_base_url.rstrip("/")
        api_url = (
            f"{base}/chat/completions"
            if config.backend_type == "openai-chat"
            else f"{base}/completions"
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }

        # 加载 tokenizer（用于精确 token 计数和 prompt 生成）
        tokenizer_src = getattr(config, "tokenizer_path", None) or config.model_name
        await log(f"[INFO] 正在加载 tokenizer: {tokenizer_src} …")
        tokenize_fn = _load_tokenizer(tokenizer_src)
        if tokenize_fn:
            await log(f"[INFO] Tokenizer 加载成功，将用于精确 token 计数")
        else:
            await log(f"[WARN] Tokenizer 未能加载，将使用 usage.completion_tokens 或 SSE chunk 计数")

        await log(f"[INFO] 模型: {config.model_name}  GPU: {config.gpu_name} × {config.gpu_count}")
        await log(f"[INFO] 接口: {api_url}")
        await log(f"[INFO] 输入tokens列表: {config.input_tokens_list}  输出tokens: {config.output_tokens}")
        await log(f"[INFO] 并发列表: {sorted(config.concurrency_list)}  测试轮数(epochs): {config.epochs}")

        # chat 路由策略：先尝试 prefill，全部失败后自动回退到 prompt_eng
        chat_strategy = "prefill" if config.backend_type == "openai-chat" else "completions"
        if config.backend_type == "openai-chat":
            await log("[INFO] Chat 路由策略：优先使用 prefill（空 assistant 消息续写），失败则自动回退到 prompt 工程模式")

        total = len(config.input_tokens_list) * len(config.concurrency_list)
        done = 0

        for input_tokens in config.input_tokens_list:
            if _cancel_flags.get(task_id):
                await log("[INFO] 用户已停止测试")
                break

            prompt = _generate_prompt(input_tokens, tokenize_fn)

            for concurrency in sorted(config.concurrency_list):
                if _cancel_flags.get(task_id):
                    await log("[INFO] 用户已停止测试")
                    break

                await log(
                    f"\n[RUN {done + 1}/{total}]  input={input_tokens}tok  "
                    f"conc={concurrency}  epochs={config.epochs}  "
                    f"总请求={config.epochs * concurrency}"
                    + (f"  [策略: {chat_strategy}]" if config.backend_type == "openai-chat" else "")
                )

                # 构建 payload
                if config.backend_type == "openai-chat":
                    payload: dict = _build_chat_payload(
                        config.model_name, prompt, config.output_tokens,
                        strategy=chat_strategy,
                    )
                else:
                    payload = {
                        "model": config.model_name,
                        "prompt": prompt,
                        "temperature": 0,
                        "max_tokens": config.output_tokens,
                        "stream": True,
                    }

                pt = await _benchmark_point(
                    api_url, headers, payload, concurrency, config.epochs, log,
                    tokenize_fn=tokenize_fn,
                )
                pt["input_tokens"] = input_tokens
                pt["output_tokens"] = config.output_tokens
                pt["concurrency"] = concurrency
                all_results.append(pt)

                tpt = pt.get("throughput_tokens_s") or 0.0
                tpu = pt.get("throughput_per_user_tokens_s") or 0.0
                ttft = pt.get("ttft_mean_ms") or 0.0
                dec = pt.get("decode_latency_mean_ms") or 0.0
                err_c = pt.get("error_count", 0)
                total_req = pt.get("total_requests", 1)
                err_rate = err_c / total_req if total_req > 0 else 0.0

                if err_rate >= 1.0:
                    # chat 路由：prefill 全败时自动回退到 prompt_eng 重试（仅一次）
                    if config.backend_type == "openai-chat" and chat_strategy == "prefill":
                        await log(
                            f"[WARN] prefill 策略全部失败，自动切换到 prompt 工程模式重试…"
                        )
                        chat_strategy = "prompt_eng"
                        payload = _build_chat_payload(
                            config.model_name, prompt, config.output_tokens,
                            strategy="prompt_eng",
                        )
                        pt = await _benchmark_point(
                            api_url, headers, payload, concurrency, config.epochs, log,
                            tokenize_fn=tokenize_fn,
                        )
                        pt["input_tokens"] = input_tokens
                        pt["output_tokens"] = config.output_tokens
                        pt["concurrency"] = concurrency
                        # 替换刚才追加的失败点
                        all_results[-1] = pt
                        tpt = pt.get("throughput_tokens_s") or 0.0
                        tpu = pt.get("throughput_per_user_tokens_s") or 0.0
                        ttft = pt.get("ttft_mean_ms") or 0.0
                        dec = pt.get("decode_latency_mean_ms") or 0.0
                        err_c = pt.get("error_count", 0)
                        total_req = pt.get("total_requests", 1)
                        err_rate = err_c / total_req if total_req > 0 else 0.0

                    if err_rate >= 1.0:
                        await log(
                            f"[ERROR] 全部 {total_req} 个请求失败，跳过本测试点。"
                            f"请检查 API URL / 模型名称是否正确"
                        )
                        done += 1
                    continue

                await log(
                    f"[RESULT] 吞吐: {tpt:.1f} tok/s | 单用户吞吐: {tpu:.2f} tok/s | "
                    f"TTFT均值: {ttft:.1f}ms | 增量延时均值: {dec:.1f}ms"
                    + (f" | 失败率: {err_rate*100:.0f}%" if err_c > 0 else "")
                )
                done += 1

                # ── 约束检查：仅在有有效数据时执行 ────
                if err_rate < 1.0:
                    violated: list[str] = []
                    if config.max_ttft_ms is not None and ttft > config.max_ttft_ms:
                        violated.append(
                            f"TTFT均值 {ttft:.1f}ms > 限制 {config.max_ttft_ms:.0f}ms"
                        )
                    if config.min_throughput_per_user is not None and tpu < config.min_throughput_per_user:
                        violated.append(
                            f"单用户吞吐 {tpu:.2f} tok/s < 限制 {config.min_throughput_per_user:.2f} tok/s"
                        )
                    if violated:
                        await log(
                            f"[STOP]  并发={concurrency} 已超限：{'; '.join(violated)}。"
                            f"跳过 input={input_tokens} 的更高并发测试"
                        )
                        break  # 不再测试该 input_tokens 下更高的并发数

        # 只写入有实际数据的测试点（排除全部失败的点）
        valid_results = [r for r in all_results if r.get("throughput_tokens_s") or r.get("ttft_mean_ms")]
        _task_results[task_id] = valid_results

        if valid_results:
            await log(f"\n[INFO] 正在写入数据库（{len(valid_results)} 个有效数据点）…")
            _import_results_to_db(
                task_id, valid_results, config.gpu_name, config.model_name, config.gpu_count
            )
            await log("[INFO] 写入完成，预测界面可立即使用新数据")
        else:
            await log("\n[WARN] 无有效测试数据，跳过写入数据库")

        if _cancel_flags.get(task_id):
            _task_status[task_id] = "cancelled"
            _update_status_db(task_id, "cancelled")
            await log("\n[INFO] 测试已停止")
        else:
            _task_status[task_id] = "done"
            _update_status_db(task_id, "done")
            await log(f"\n[DONE] 全部完成，有效数据点: {len(valid_results)}/{len(all_results)}")

    except Exception as e:
        logger.error("benchmark_error", task_id=task_id, error=str(e))
        _task_status[task_id] = "failed"
        _update_status_db(task_id, "failed", str(e))
        await log(f"[ERROR] {e}")
    finally:
        _cancel_flags.pop(task_id, None)
        await queue.put(None)  # 结束信号


# ── 状态查询接口（供路由层调用） ─────────────────────────────

def get_task_status(task_id: str) -> str:
    return _task_status.get(task_id, "unknown")


def cancel_task(task_id: str) -> bool:
    """请求取消正在运行的任务。返回 True 表示任务存在且已标记取消。"""
    if _task_status.get(task_id) == "running":
        _cancel_flags[task_id] = True
        return True
    return False


def get_task_queue(task_id: str) -> asyncio.Queue | None:
    return _task_queues.get(task_id)


def get_task_results(task_id: str) -> list[dict]:
    return _task_results.get(task_id, [])

