import asyncio
import csv
import json
import logging
import os
import stat
import time
from typing import Union
import random
import aiohttp
import numpy as np
import requests
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
IS_DEBUG = int(os.environ.get("BENCKMARK_DEBUG", 0))
EPOCH_NUM = 10
# The size of the data block returned in each iteration is not greater than 8192. Therefore, chunk_size is 8192.
CHUNK_SIZE = 8192
TIMEOUT = 3 * 3600

if IS_DEBUG:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt


def get_tokenizer(
        transformer_tokenizer_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    tokenizer = AutoTokenizer.from_pretrained(transformer_tokenizer_path, trust_remote_code=True)

    return tokenizer


def generate_hello_str(tokenizer, length, hello_token="Hello"):
    text = hello_token * (length - 1)
    completion_token_ids = tokenizer([text]).input_ids
    while len(completion_token_ids[0]) < length:
        text += hello_token
        completion_token_ids = tokenizer([text]).input_ids

    return text


def generate_str(tokenizer, length):
    vocab_size = tokenizer.vocab_size
    np.random.seed(int(time.time()) + random.randint(0, 1000))
    ids = np.random.randint(vocab_size / 4, vocab_size / 3, length)
    text = tokenizer.decode(ids)

    completion_token_ids = tokenizer([text]).input_ids
    if IS_DEBUG:
        logger.info("len(completion_token_ids[0]) %d, length %d ", len(completion_token_ids[0]), length)

    epoch = EPOCH_NUM
    while len(completion_token_ids[0]) != length and epoch > 0:
        while len(completion_token_ids[0]) > length:
            diff = len(completion_token_ids[0]) - length
            now_length = ids.shape[0] - diff
            ids = ids[:now_length]
            text = tokenizer.decode(ids)
            completion_token_ids = tokenizer([text]).input_ids
            if IS_DEBUG:
                logger.info("len(completion_token_ids[0]) %d, %d ", len(completion_token_ids[0]), length)

        while len(completion_token_ids[0]) < length:
            diff = length - len(completion_token_ids[0])
            diff_ids = np.random.randint(vocab_size / 4, vocab_size / 3, diff)
            ids = np.append(ids, diff_ids)
            text = tokenizer.decode(ids)
            completion_token_ids = tokenizer([text]).input_ids
            if IS_DEBUG:
                logger.info("len(completion_token_ids[0]) %d, %d ", len(completion_token_ids[0]), length)

        epoch -= 1

    if len(completion_token_ids[0]) != length:
        text = generate_hello_str(tokenizer, length)

    if IS_DEBUG:
        logger.info(text)
    return text


def get_api_url(backend, host, port, url):
    if url is not None and len(url) > 0:
        return url

    if backend == "mindspore":
        api_url = f"http://{host}:{port}/models/llama2/generate"
    elif backend == "base":
        api_url = f"http://{host}:{port}/v1/generate"
    elif backend == "tgi":
        api_url = f"https://{host}:{port}/generate_stream"
    elif backend == "openai":
        api_url = f"http://{host}:{port}/v1/completions"
    elif backend == "openai-chat":
        # OpenAI-compatible chat completions endpoint, used for multimodal (image+text)
        api_url = f"http://{host}:{port}/v1/chat/completions"
    elif backend == "trt":
        api_url = f"http://{host}:{port}/v2/models/ensemble/generate_stream"
    elif backend == "vllm-chat":
        # vLLM OpenAI-compatible chat completions endpoint
        api_url = f"http://{host}:{port}/v1/chat/completions"
    else:
        api_url = f"http://{host}:{port}/generate"
    return api_url


def get_request_data(
    backend: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    app_code: str = None,
    model: str = None,
    served_model_name: str = None,
    api_key: str | None = None,
    # Optional multimodal inputs. If provided and backend is chat-style,
    # an OpenAI-compatible chat payload with image_url will be constructed.
    image_url_or_b64: str | None = None,
    is_image_url: bool = False,
):
    confirm_error_output = False

    headers = {
        "User-Agent": "Benchmark Client",
        "Content-Type": "application/json",
    }
    if app_code is not None and len(app_code) > 0:
        headers["X-Apig-AppCode"] = app_code
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 0.8,
            "top_k": 5,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": True,
        }
        confirm_error_output = True
    elif backend == "vllm-chat":
        if served_model_name is None:
            served_name = model
        else:
            served_name = served_model_name
        # Build OpenAI-compatible chat messages with image_url support
        content_items = []
        if prompt is not None and len(prompt) > 0:
            content_items.append({"type": "text", "text": prompt})
        if image_url_or_b64:
            image_payload = {"url": image_url_or_b64}
            content_items.append({"type": "image_url", "image_url": image_payload})
        pload = {
            "model": served_name,
            "messages": [
                {"role": "user", "content": content_items}
            ],
            "temperature": 0,
            "top_p": 0.8,
            "max_tokens": output_len,
            "stream": True,
        }
        confirm_error_output = True
    elif backend == "openai":
        if served_model_name is None:
            served_name = model
        else:
            served_name = served_model_name
        pload = {
            "prompt": prompt,
            "temperature": 0,
            "top_p": 0.8,
            "top_k": 5,
            "max_tokens": output_len,
            "ignore_eos": True,
            "model": served_name,
            "stream": True,
        }
        confirm_error_output = True
    elif backend == "openai-chat":
        if served_model_name is None:
            served_name = model
        else:
            served_name = served_model_name
        content_items = []
        if prompt is not None and len(prompt) > 0:
            content_items.append({"type": "text", "text": prompt})
        if image_url_or_b64:
            # Use data URL or remote URL directly per OpenAI schema
            image_payload = {"url": image_url_or_b64}
            content_items.append({"type": "image_url", "image_url": image_payload})
        pload = {
            "model": served_name,
            "messages": [
                {"role": "user", "content": content_items}
            ],
            "temperature": 0,
            "top_p": 0.8,
            "max_tokens": output_len,
            "stream": True,
        }
        confirm_error_output = True
    elif backend == "mindspore":
        params = {
            "max_new_tokens": output_len,
            "do_sample": False,
            "ignore_eos": True,
            "return_full_text": False
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
            "stream": True
        }
    elif backend == "base":
        pload = {
            "prompt": prompt,
            "max_tokens": (prompt_len + output_len),
            "model_name": "llama2",
            "do_sample": False,
            "stream": True,
            "debug": 2
        }
    elif backend == "tgi":
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": False,
            "ignore_eos_token": True,
            "decoder_input_details": False
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
        confirm_error_output = True
    elif backend == "trt":
        headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        params = {
            "max_tokens": output_len,
            "min_length": output_len,
            "bad_words": "",
            "stop_words": "",
            "ignore_eos": True,
            "stream": True
        }
        pload = {
            "text_input": prompt,
            "parameters": params,
        }
        confirm_error_output = True
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return headers, pload, confirm_error_output


async def do_request(api_url, headers, pload, confirm_error_output, output_len, num_scheduler_steps,
                     use_spec_decode=False, log_outputs: bool = False):
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    first_token = True
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(ssl=False)) as session:
        while True:
            last_chunk = None
            prefill_start_time = time.perf_counter()
            time_record = [prefill_start_time]
            # 收集所有原始chunk（用于重建输出文本和调试）
            chunk_record = []
            async with session.post(api_url, headers=headers, json=pload) as response:
                async for chunk, _ in response.content.iter_chunks():
                    if len(chunk.strip()) > 0:
                        last_chunk = chunk
                        return_token_num = 1 if first_token else min(num_scheduler_steps, output_len)
                        time_record.extend([time.perf_counter()] * return_token_num)
                        first_token = False
                        output_len -= return_token_num
                    # 记录所有非空chunk，便于调试或后续解析
                    if len(chunk.strip()) > 0:
                        chunk_record.append(chunk)

            if confirm_error_output:
                # 检查是否接收到有效的chunk数据
                if last_chunk is None:
                    logger.error("未接收到有效的响应数据")
                    break
                    
                # if last_chunk.startswith(b'data:'):
                #     output = last_chunk[5:].strip().decode("utf-8")
                # elif last_chunk.endswith('data: [DONE]\n\n') and len(last_chunk) > len('data: [DONE]\n\n\n'):
                #     output = last_chunk[5:-len('data: [DONE]\n\n')].strip().decode("utf-8")
                # else:
                #     output = last_chunk.strip().strip().decode("utf-8").rstrip("\0")
                # 先将字节数据转换为字符串，便于处理
                try:
                    chunk_str = last_chunk.decode("utf-8").strip()
                except UnicodeDecodeError as e:
                    logger.error("解码响应数据失败: %s, 原始数据: %s", str(e), repr(last_chunk))
                    break
                
                # 分割可能存在的多个数据部分
                parts = [p.strip() for p in chunk_str.split('data:') if p.strip()]
                
                # 寻找有效的JSON部分（排除[DONE]标记）
                json_part = None
                for part in parts:
                    if part != '[DONE]':
                        json_part = part
                        break
                
                if json_part:
                    output = json_part
                else:
                    # 如果没有找到有效JSON部分，使用原始处理
                    output = chunk_str

                if IS_DEBUG:
                    logger.info(output)
                    
                # 检查是否为结束标记
                if output == '[DONE]' or output == 'data: [DONE]' or output.strip() == '[DONE]':
                    break
                    
                # 尝试解析JSON（只有当不是结束标记时）
                try:
                    output = json.loads(output)
                except Exception as e:
                    logger.error("JSON解析失败: %s, 原始数据: %s", str(e), repr(output))
                    break

                # Re-send the request if it failed.
                if "error" not in output:
                    break
                else:
                    logger.error("request failed, %s, retry", output)
                    await asyncio.sleep(0.1)
            else:
                break

        # 可选：解析输出文本并打印预览（受 log_outputs 控制）
        if log_outputs:
            try:
                merged = b"".join(chunk_record)
                text_preview = None
                if merged:
                    try:
                        data_str = merged.decode("utf-8", errors="ignore")
                        # SSE事件按空行分隔
                        events = [e.strip() for e in data_str.split("\n\n") if e.strip()]
                        acc = []
                        for ev in events:
                            # 仅处理以data:开头的行
                            if ev.startswith("data:"):
                                payload = ev[len("data:"):].strip()
                                if payload == "[DONE]":
                                    continue
                                try:
                                    obj = json.loads(payload)
                                    choice = obj.get("choices", [{}])[0]
                                    if "text" in choice:
                                        part = choice.get("text", "")
                                    else:
                                        part = choice.get("delta", {}).get("content", "")
                                    if part:
                                        acc.append(part)
                                except Exception:
                                    # 非JSON或部分内容，忽略
                                    pass
                        if acc:
                            full_text = "".join(acc)
                            # 只打印前500字符预览，避免日志过长
                            text_preview = (full_text[:500] + ("…" if len(full_text) > 500 else ""))
                    except Exception:
                        pass
                if text_preview is not None:
                    logger.info("模型输出预览: %s", text_preview)
            except Exception as e:
                logger.warning("解析模型输出预览失败: %s", str(e))

        return time_record, chunk_record


# def check_multi_step(args, api_url, tokenizer, prompt_len, output_len):
#     prompt = generate_str(tokenizer, prompt_len)
#     headers, pload, confirm_error_output = get_request_data(args.backend,
#                                                             prompt,
#                                                             prompt_len,
#                                                             output_len,
#                                                             args.best_of,
#                                                             args.use_beam_search,
#                                                             args.app_code,
#                                                             args.tokenizer,
#                                                             args.served_model_name,
#                                                             args.api_key)
#     return_num = 0
#     response = requests.post(api_url, headers=headers, json=pload, stream=True, timeout=TIMEOUT)
#     response.raise_for_status()
#     for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
#     #for chunk in response.iter_lines():
#         return_num += 1

#     # openai流式最后返回[done],其余接口没有
#     end_chunk_num = 1 if args.backend == 'openai' else 0
#     # 首token + 是否有返回[done] + chunk_num
#     theory_return_num = 1 + end_chunk_num + (output_len - 1) // args.num_scheduler_steps
#     # (output_len - 1) % num_scheduler_steps 非整除情况下最后返回包含剩余token的chunk
#     theory_return_num += 1 if (output_len - 1) % args.num_scheduler_steps else 0
#     print("theory_return_num:", theory_return_num, "return_num:", return_num)
#     if theory_return_num == return_num:
#         return True
#     return False
#     #return True

def check_multi_step(args, api_url, tokenizer, prompt_len, output_len):
    prompt = generate_str(tokenizer, prompt_len)
    headers, pload, confirm_error_output = get_request_data(args.backend,
                                                            prompt,
                                                            prompt_len,
                                                            output_len,
                                                            args.best_of,
                                                            args.use_beam_search,
                                                            args.app_code,
                                                            args.tokenizer,
                                                            args.served_model_name,
                                                            args.api_key)
    response = requests.post(api_url, headers=headers, json=pload, stream=True, timeout=TIMEOUT)
    response.raise_for_status()
    
    # 按 SSE 事件（data: 行）计数，避免 TCP 合包导致 iter_content 少计 chunk
    data_event_num = 0
    done_num = 0
    for line in response.iter_lines():
        if line and line.startswith(b'data:'):
            payload = line[5:].strip()
            if payload == b'[DONE]':
                done_num += 1
            else:
                data_event_num += 1

    # 期望的内容事件数（不含 [DONE]）：首 token 1 个 + 剩余 token 按 num_scheduler_steps 分组
    expected_data_events = 1
    remaining_tokens = output_len - 1
    if remaining_tokens > 0:
        expected_data_events += (remaining_tokens - 1) // args.num_scheduler_steps + 1

    # 服务端可能在单个 SSE 事件中合并多个 token（API 层缓冲），
    # 因此允许实际事件数略少于理论值，但比率应接近 num_scheduler_steps
    # 实际每个事件平均 token 数
    actual_tokens_per_event = output_len / data_event_num if data_event_num > 0 else float('inf')
    # 允许 2% 的浮动来适应 API 层的少量合包行为
    tolerance = 0.02 * args.num_scheduler_steps + 0.01
    is_match = abs(actual_tokens_per_event - args.num_scheduler_steps) <= tolerance

    print(f"expected_data_events: {expected_data_events}, actual_data_events: {data_event_num}, "
          f"done_events: {done_num}, tokens_per_event: {actual_tokens_per_event:.3f}, "
          f"expected_steps: {args.num_scheduler_steps}, match: {is_match}")
    
    return is_match

def statistics_and_print_performance_data(args, prompt_tokens, output_tokens, parallel_num,
                                          request_latency_record, all_latency_record):
    benchmark_start_time = np.min([time_record[0] for _, _, time_record, _ in request_latency_record])
    benchmark_end_time = np.max([time_record[-1] for _, _, time_record, _ in request_latency_record])
    benchmark_time = benchmark_end_time - benchmark_start_time
    logger.info("所有请求耗时: %.4f s", benchmark_time)

    benchmark_requests = args.epochs * parallel_num / benchmark_time
    logger.info("请求吞吐: %.4f requests/s", benchmark_requests)

    # 使用实际生成的 token 数（由时间戳长度推导）
    actual_output_tokens_list = [
        (len(time_record) - 1)
        for _, _, time_record, _ in request_latency_record
    ]
    total_output_tokens = int(np.sum(actual_output_tokens_list))
    total_output_token_throughput = total_output_tokens / benchmark_time
    logger.info("输出tokens总吞吐: %.4f tokens/s", total_output_token_throughput)

    prefill_latency_list = [
        time_record[1] - time_record[0]
        for _, _, time_record, _ in request_latency_record
    ]

    p90_prefill_latency = np.percentile(prefill_latency_list, 90) * 1000
    logger.info("首tokens时延TP90: %.4f ms", p90_prefill_latency)

    p99_prefill_latency = np.percentile(prefill_latency_list, 99) * 1000
    logger.info("首tokens时延TP99: %.4f ms", p99_prefill_latency)

    max_prefill_latency = np.max(prefill_latency_list) * 1000
    logger.info("最大首tokens时延: %.4f ms", max_prefill_latency)

    avg_prefill_latency = np.mean(prefill_latency_list) * 1000
    logger.info("平均首tokens时延: %.4f ms", avg_prefill_latency)

    prefill_ranges = [
        (time_record[0], time_record[1])
        for _, _, time_record, _ in request_latency_record
    ]

    def in_ranges(ranges, start, end):
        for time_range in ranges:
            if start <= time_range[1] and end >= time_range[0]:
                return True
        return False

    decode_latency_list = [
        end - start
        for _, _, time_record, _ in request_latency_record
        for start, end in zip(time_record[1:-1], time_record[2:])
        if not in_ranges(prefill_ranges, start, end)
    ]

    # 去掉开始的0（multi-step场景）
    start_index = 0
    for latency_index, latency in enumerate(decode_latency_list):
        if latency > 0:
            start_index = latency_index
            break
    decode_latency_list = decode_latency_list[start_index:]

    p90_decode_latency = np.percentile(decode_latency_list, 90) * 1000
    logger.info("增量时延TP90: %.4f ms", p90_decode_latency)

    p99_decode_latency = np.percentile(decode_latency_list, 99) * 1000
    logger.info("增量时延TP99: %.4f ms", p99_decode_latency)

    max_decode_latency = np.max(decode_latency_list) * 1000
    logger.info("最大增量时延: %.1f ms", max_decode_latency)

    avg_decode_latency = np.mean(decode_latency_list) * 1000
    logger.info("平均增量时延: %.1f ms", avg_decode_latency)

    if IS_DEBUG:
        plot_time_record(benchmark_start_time, benchmark_time, request_latency_record,
                         f"{parallel_num}_{prompt_tokens}_{output_tokens}.jpg")

    avg_prompt_token = np.mean([prompt_len for prompt_len, _, _, _ in request_latency_record])
    # 以实际生成的 token 数统计平均输出长度
    avg_output_token = np.mean(actual_output_tokens_list)

    latency_record = (avg_prompt_token, avg_output_token, parallel_num,
                      total_output_token_throughput,
                      p90_prefill_latency, p99_prefill_latency, max_prefill_latency, avg_prefill_latency,
                      p90_decode_latency, p99_decode_latency, max_decode_latency, avg_decode_latency)

    # If the benchmark backend supports speculative inference, request_latency_record is replaced with output_step,
    # which is an int value. Otherwise, the original chunk_list is retained as a list.
    is_spec_support_backend = isinstance(request_latency_record[0][-1], int)
    if getattr(args, "use_spec_decode", False) and getattr(args, "num_speculative_tokens", -1) >= 0 \
            and is_spec_support_backend:
        # 使用实际生成 token 数替换配置的 output_len，避免提前终止带来的偏差
        accept_rate_list = [
            ((len(time_record) - 1) - 1) / ((output_step - 1) * (args.num_speculative_tokens + 1))
            if (output_step - 1) * (args.num_speculative_tokens + 1) > 0 else 0.0
            for _, _, time_record, output_step in request_latency_record
        ]

        p90_accept_rate = np.percentile(accept_rate_list, 90)
        logger.info("投机接受率TP90: %.4f", p90_accept_rate)

        p99_accept_rate = np.percentile(accept_rate_list, 99)
        logger.info("投机接受率TP99: %.4f", p99_accept_rate)

        max_accept_rate = np.max(accept_rate_list)
        logger.info("投机最大接受率: %.2f", max_accept_rate)

        min_accept_rate = np.min(accept_rate_list)
        logger.info("投机最小接受率: %.2f", min_accept_rate)

        avg_accept_rate = np.mean(accept_rate_list)
        logger.info("投机平均接受率: %.2f", avg_accept_rate)

        accept_rate_record = (p90_accept_rate, p99_accept_rate, max_accept_rate, min_accept_rate, avg_accept_rate)

        latency_record = latency_record + accept_rate_record

    time.sleep(10)

    all_latency_record.append(latency_record)
    
    # 返回关键指标用于约束检查
    return {
        'avg_first_token_latency': avg_prefill_latency,  # 平均首token延时(ms)
        'total_throughput': total_output_token_throughput,  # 总吞吐量(tokens/s)
        'p90_first_token_latency': p90_prefill_latency,
        'p99_first_token_latency': p99_prefill_latency,
        'max_first_token_latency': max_prefill_latency,
        'avg_decode_latency': avg_decode_latency,
    }


def plot_time_record(benchmark_start_time, benchmark_time, request_latency_record, name="parallel.jpg"):
    def newline(ax, p1, p2, color='skyblue'):
        line = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=20, markersize=100, marker=".",
                             markerfacecolor=color)
        ax.add_line(line)

    fig_size_x = 256
    fig_size_y = 128
    fig, ax = plt.subplots(1, 1, figsize=(fig_size_x, fig_size_y), facecolor='#f7f7f7', dpi=80)
    time_records = [time_record for _, _, time_record in request_latency_record]
    time_records = (time_records - benchmark_start_time) * 1000
    for idx, time_record in enumerate(tqdm(time_records, desc="plot_time_record")):
        idx = idx * 1
        newline(ax, [time_record[0], idx], [time_record[1], idx], color='red')
        for start, end in zip(time_record[1:-1], time_record[2:]):
            newline(ax, [start, idx], [end, idx])

    ax.set_facecolor('#f7f7f7')
    ax.set(xlim=(0, (benchmark_time * 1000) + 10), ylim=(-1, len(time_records) * 1), ylabel='request')
    font_size = round(fig_size_x / 3)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('time', fontsize=fig_size_x)
    plt.ylabel('request', fontsize=fig_size_x)
    logger.info(f"save fig ...")
    plt.savefig(name)


def save_to_csv(benchmark_head, records, csv_path):
    logger.info(benchmark_head)
    sprtr_idx = csv_path.rfind('/')
    if sprtr_idx > 0:
        csv_dir = csv_path[0:sprtr_idx]
        if len(csv_dir) > 1:
            os.makedirs(csv_dir, exist_ok=True)

    # 覆盖写入，兼容重复跑基准
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(benchmark_head)
        for items in records:
            to_csv = []
            for item in items:
                if isinstance(item, float):
                    item = round(item, 4)
                to_csv.append(item)
            logger.info(to_csv)
            writer.writerow(to_csv)
