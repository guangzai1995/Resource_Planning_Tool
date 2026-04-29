import aiohttp
import argparse
import asyncio
import bisect
import csv
import json
import logging
import numpy as np
import os
import random
import time
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import PreTrainedTokenizerBase
from typing import AsyncGenerator, List, Tuple, Union

from benchmark_utils import get_tokenizer, get_api_url, get_request_data, do_request, save_to_csv, check_multi_step

logging.basicConfig(level=logging.DEBUG,
                    filename='serving_debug.log',
                    filemode='w',
                    format=
                    '%(asctime)s-%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def alpaca_prompt_format(prompt):
    """
    alpaca数据集格式
    """
    if 'input' in prompt:
        prompt_text = PROMPT_WITH_INPUT.format_map(prompt)
    else:
        prompt_text = PROMPT_WITHOUT_INPUT.format_map(prompt)
    return prompt_text


def get_alpaca_dataset(dataset):
    """
    获取alpaca数据集
    """
    dataset = [
        (data["input"], data["output"])
        for data in dataset
    ]

    return dataset


def get_sharegpt_dataset(dataset):
    """
    获取sharegpt数据集
    """
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]

    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    return dataset


def get_custom_dataset(dataset):
    """
    获取custom数据集
    """
    dataset = [
        (data["input"], data["output"])
        for data in dataset
    ]

    return dataset


def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        dataset_type: str,
        max_tokens: int,
        max_prompt_tokens: int,
) -> List[Tuple[str, int, int]]:
    """
    加载数据集采样请求
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    if dataset_type == "alpaca":
        dataset = get_alpaca_dataset(dataset)
    elif dataset_type == "sharegpt":
        dataset = get_sharegpt_dataset(dataset)
    else:
        dataset = get_custom_dataset(dataset)

    prompts = [prompt for prompt, _ in dataset]
    completions = [completion for _, completion in dataset]

    prompt_token_ids = tokenizer(prompts).input_ids
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > max_prompt_tokens:
            # Prune too long sequences.
            continue
        if prompt_len + output_len > max_tokens:
            output_len = max_tokens - prompt_len
            if output_len <= 0:
                continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def get_request(
        input_requests: List[Tuple[str, int, int]],
        request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    读取测试客户端请求
    """
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
        request_latency_record: List,
        backend: str,
        api_url: str,
        prompt: str,
        prompt_len: int,
        output_len: int,
        best_of: int,
        use_beam_search: bool,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
        api_key: str = "123456",
        num_scheduler_steps: int = 1
) -> None:
    """
    发送一次客户端请求
    """
    headers, pload, confirm_error_output = get_request_data(backend,
                                                            prompt,
                                                            prompt_len,
                                                            output_len,
                                                            best_of,
                                                            use_beam_search,
                                                            app_code,
                                                            model,
                                                            served_model_name,
                                                            api_key)

    time_record, _ = await do_request(api_url, headers, pload, confirm_error_output, output_len, num_scheduler_steps)

    output_tokens = len(time_record) - 1

    if output_tokens < output_len:
        logger.warning(f"output_tokens: {output_tokens} < output_len: {output_len} (模型可能提前停止)")

    request_latency_record.append((prompt_len, output_len, time_record))


async def benchmark(
        request_latency_record: List,
        backend: str,
        api_url: str,
        input_requests: List[Tuple[str, int, int]],
        best_of: int,
        use_beam_search: bool,
        request_rate: float,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
        num_scheduler_steps: int = 1,
        api_key: str = "123456"
) -> None:
    """
    benchmark test
    """
    tasks: List[asyncio.Task] = []
    pbar = tqdm(total=len(input_requests), desc="request")
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(send_request(request_latency_record,
                                                backend, api_url, prompt,
                                                prompt_len, output_len,
                                                best_of, use_beam_search,
                                                app_code,
                                                model,
                                                served_model_name,
                                                num_scheduler_steps,
                                                api_key
                                            ))
        tasks.append(task)
        pbar.update()
    pbar.close()
    await tqdm_asyncio.gather(*tasks, desc='finish')


def main(args: argparse.Namespace):
    """
    main entry
    """
    logger.info(args)
    if len(args.num_prompts) != len(args.request_rate):
        logger.error(f"The array length of num_prompts and request_rate is different!")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.benchmark_csv)), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = get_api_url(args.backend, args.host, args.port, args.url)
    tokenizer = get_tokenizer(args.tokenizer)

    is_multi_step = check_multi_step(args, api_url, tokenizer, args.max_prompt_tokens,
                                     args.max_tokens - args.max_prompt_tokens)
    if not is_multi_step:
        logger.error("The service does not use multi_step or num-scheduler-steps is different from the service.")
        return

    logger.info(f"Warmup ...")
    warmup_request_rate = 100
    warmup_input_samples = sample_requests(args.dataset, 256, tokenizer, args.dataset_type, args.max_tokens,
                                           args.max_prompt_tokens)
    asyncio.run(
        benchmark(
            [],
            args.backend, api_url,
            warmup_input_samples,
            args.best_of,
            args.use_beam_search,
            warmup_request_rate,
            args.app_code,
            args.tokenizer,
            args.served_model_name,
            args.num_scheduler_steps,
            args.api_key
        )
    )

    logger.info(f"Sample requests ...")
    request_time = args.request_time  # seconds
    num_samples = max(args.num_prompts)

    input_samples = sample_requests(args.dataset, num_samples, tokenizer, args.dataset_type, args.max_tokens,
                                    args.max_prompt_tokens)

    all_latency_record = []

    for i, request_rate in enumerate(args.request_rate):
        request_num = args.num_prompts[i]
        logger.info(f"Benchmark running with request_rate: {request_rate}, request_num: {request_num}")
        requests = input_samples[:request_num]
        latency_record = do_benchmark(api_url, args, request_rate, requests, args.num_scheduler_steps)
        all_latency_record.append(latency_record)

    all_latency_record.sort(key=lambda element: element[3])

    benchmark_head = ["数据集", "输入平均长度（tokens）", "输出平均长度（tokens）",
                      "请求频率（req/s）", "请求吞吐（req/s）", "请求平均时延（s）",
                      "平均输出tokens吞吐（tokens/s）", "单请求每tokens平均时延（ms）", "首tokens平均时延（ms）",
                      "输出tokens总吞吐（tokens/s）"]
    save_to_csv(benchmark_head, all_latency_record, args.benchmark_csv)

    logger.info(f"Benchmark serving with {args.dataset_type} dataset finished")


def do_benchmark(api_url, args, request_rate, requests, num_scheduler_steps):
    # (prompt len, output len, time_record)
    request_latency_record: List[Tuple[int, int, List]] = []
    asyncio.run(
        benchmark(
            request_latency_record,
            args.backend, api_url,
            requests,
            args.best_of,
            args.use_beam_search,
            request_rate,
            args.app_code,
            args.tokenizer,
            args.served_model_name,
            num_scheduler_steps
        )
    )

    benchmark_start_time = np.min([time_record[0] for _, _, time_record in request_latency_record])
    benchmark_end_time = np.max([time_record[-1] for _, _, time_record in request_latency_record])
    benchmark_time = benchmark_end_time - benchmark_start_time
    logger.info(f"所有请求耗时: {benchmark_time:.2f} s")

    request_num = len(requests)
    benchmark_requests = request_num / benchmark_time
    logger.info(f"请求吞吐: {benchmark_requests:.3f} requests/s")
    # Compute the latency statistics.
    avg_latency_list = [
        time_record[-1] - time_record[0]
        for _, _, time_record in request_latency_record]
    avg_latency = np.mean(avg_latency_list)
    logger.info(f"平均请求时延: {avg_latency:.3f} s")

    avg_latency_list = [round(item, 2) for item in avg_latency_list]
    logging.debug(
        "request_rate %s, avg: %s, max: %s, 请求时延 : %s",
        request_rate, avg_latency, np.max(avg_latency_list), avg_latency_list)

    avg_per_token_latency = np.mean([
        (time_record[-1] - time_record[0]) / (prompt_len + output_len)
        for prompt_len, output_len, time_record in request_latency_record
    ]) * 1000

    logger.info(f"平均每token(输入+输出)时延: {avg_per_token_latency:.2f} ms")
    avg_per_output_token_latency = np.mean([
        (time_record[-1] - time_record[1]) / output_len
        for _, output_len, time_record in request_latency_record
    ]) * 1000

    logger.info("平均每输出token时延: "
          f"{avg_per_output_token_latency:.2f} ms")

    avg_per_output_tokens = np.mean([
        output_len / (time_record[-1] - time_record[1])
        for _, output_len, time_record in request_latency_record
    ])
    logger.info("平均输出tokens吞吐: "
          f"{avg_per_output_tokens:.2f} tokens/s")
    avg_prefill_latency = np.mean([
        time_record[1] - time_record[0]
        for _, _, time_record in request_latency_record
    ]) * 1000
    logger.info("平均首tokens时延: "
          f"{avg_prefill_latency:.2f} ms")
    avg_prompt_len = np.mean([
        prompt_len
        for prompt_len, _, _ in request_latency_record
    ])
    logger.info(f"输入平均长度: {avg_prompt_len:.1f} tokens")
    avg_output_len = np.mean([
        output_len
        for _, output_len, _ in request_latency_record
    ])
    logger.info(f"输出平均长度: {avg_output_len:.1f} tokens")
    total_output_tokens = np.sum([
        output_len
        for _, output_len, _ in request_latency_record
    ])
    logger.info("输出总tokens: "
          f"{total_output_tokens} tokens")
    total_output_tokens_th = total_output_tokens / benchmark_time
    logger.info("输出tokens总吞吐: "
          f"{total_output_tokens_th:.3f} tokens/s")
    time.sleep(60)

    return (args.dataset_type, avg_prompt_len, avg_output_len,
            request_rate, benchmark_requests, avg_latency,
            avg_per_output_tokens, avg_per_output_token_latency, avg_prefill_latency,
            total_output_tokens_th)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="mindspore",
                        choices=["vllm", "mindspore", "base", "tgi", "openai", "trt"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9288)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--app-code", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--dataset-type", type=str, default="sharegpt",
                        choices=["alpaca", "sharegpt", "custom"])
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", nargs='+', type=int, default=[1000, 1000, 1000],
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate", nargs='+', type=float, default=[1, 4, 8],
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--request-time", type=int, default=300,
                        help="requests time in seconds.")
    parser.add_argument("--max-requests", type=int, default=3000,
                        help="max requests.")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens to process.")
    parser.add_argument("--max-prompt-tokens", type=int, default=900,
                        help="Max tokens to process.")
    parser.add_argument("--benchmark-csv", type=str, default="benchmark_serving.csv",
                        help="Path to the csv.")
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-scheduler-steps", type=int, default=1)
    args_global = parser.parse_args()
    main(args_global)
