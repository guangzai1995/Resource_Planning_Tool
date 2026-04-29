import argparse
import asyncio
import json
import os
import random
import logging
import base64
import mimetypes
from glob import glob
from typing import AsyncGenerator, List, Tuple, Optional, Union, Callable

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from benchmark_utils import (generate_str, get_tokenizer, get_api_url,
                             get_request_data, do_request,
                             save_to_csv, statistics_and_print_performance_data,
                             generate_hello_str, check_multi_step)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DATASET = None


def sample_random_requests(
        tokenizer: PreTrainedTokenizerBase,
        prompt_tokens: int,
        output_tokens: int,
        enable_prefix_caching: bool = False,
        prefix_caching_num: int = 0,
    image_payloads: Optional[List[str]] = None,
    random_image_gen: Optional[Callable[[], str]] = None,
) -> List[Tuple[Union[str, Tuple[str, Optional[str]]], int, int]]:
    prompt = generate_str(tokenizer, prompt_tokens)
    requests_list=[]
    #requests_list=[(prompt, prompt_tokens, output_tokens)]
    if enable_prefix_caching and prefix_caching_num:
        # construct prompt with the same prefix.
        caching_prompt = generate_hello_str(tokenizer, prefix_caching_num) + generate_str(tokenizer,
                                                                                   prompt_tokens - prefix_caching_num)
        if random_image_gen is not None:
            img = random_image_gen()
            requests_list.append(((caching_prompt, img), prompt_tokens, output_tokens))
        elif image_payloads:
            # round-robin choose an image
            img = random.choice(image_payloads)
            requests_list.append(((caching_prompt, img), prompt_tokens, output_tokens))
        else:
            requests_list.append((caching_prompt, prompt_tokens, output_tokens))
    else:
        if random_image_gen is not None:
            img = random_image_gen()
            requests_list=[((prompt, img), prompt_tokens, output_tokens)]
        elif image_payloads:
            img = random.choice(image_payloads)
            requests_list=[((prompt, img), prompt_tokens, output_tokens)]
        else:
            requests_list=[(prompt, prompt_tokens, output_tokens)] 
    return requests_list


# just copy from vllm.
def sample_sharegpt_requests(
        dataset_path,
        requests_num,
        tokenizer: PreTrainedTokenizerBase,
        modified_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if modified_output_len is not None and modified_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    global CACHE_DATASET
    if CACHE_DATASET is None:
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        # Only keep the first two turns of each conversation.
        dataset = [subset for subset in dataset if len(subset["conversations"]) >= 2]
        dataset = [(subset["conversations"][0]["value"],
                    subset["conversations"][1]["value"]) for subset in dataset]

        # Shuffle the dataset.
        random.shuffle(dataset)

        CACHE_DATASET = dataset
    else:
        dataset = CACHE_DATASET

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == requests_num:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        output = dataset[i][1]
        output_token_ids = tokenizer(output).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(output_token_ids
                         ) if modified_output_len is None else modified_output_len
        if prompt_len < 4 or output_len < 4:
            # Filter sequences that are too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Filter sequences that are too long.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def sample_human_eval_requests(
        dataset_path: str,
        requests_num: int,
        tokenizer: PreTrainedTokenizerBase,
        modified_output_len: Optional[int] = None,
):
    # Load the dataset.
    global CACHE_DATASET
    if CACHE_DATASET is None:
        with open(dataset_path, encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        # Filter out the conversations with less than 2 turns.
        dataset = [(data["prompt"], data["canonical_solution"]) for data in dataset]

        # Shuffle the dataset.

        random.shuffle(dataset)
        CACHE_DATASET = dataset
    else:
        dataset = CACHE_DATASET

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == requests_num:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        output = dataset[i][1]
        output_token_ids = tokenizer(output).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(output_token_ids
                         ) if modified_output_len is None else modified_output_len
        if prompt_len == 0 or output_len == 0:
            continue

        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


async def get_request(
    input_requests,
    req_rate
) -> AsyncGenerator[Tuple[Union[str, Tuple[str, Optional[str]]], int, int], None]:
    input_requests = iter(input_requests)
    for req in input_requests:
        yield req

        if req_rate == float("inf"):
            # If the request rate is infinite, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / req_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
        request_latency_record: List,
        backend: str,
        api_url: str,
        prompt_or_tuple: Union[str, Tuple[str, Optional[str]]],
        prompt_len: int,
        output_len: int,
        best_of: int,
        use_beam_search: bool,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
        num_scheduler_steps: int = 1,
        use_spec_decode: bool = False,
    api_key: str = "123456",
    log_messages: bool = False,
    log_outputs: bool = False,
) -> None:
    # Support text-only prompt (str) or (text, image_data_url) tuple
    if isinstance(prompt_or_tuple, tuple):
        prompt, image_data_url = prompt_or_tuple
    else:
        prompt, image_data_url = prompt_or_tuple, None
    headers, pload, confirm_error_output = get_request_data(backend,
                                                            prompt,
                                                            prompt_len,
                                                            output_len,
                                                            best_of,
                                                            use_beam_search,
                                                            app_code,
                                                            model,
                                                            served_model_name,
                                                            api_key,
                                                            image_url_or_b64=image_data_url)

    # 可选：打印本次请求的 messages/prompt 预览
    if log_messages:
        try:
            if backend in ("openai-chat", "vllm-chat"):
                msgs = pload.get("messages", [])
                def _shorten(s: str, n: int = 200) -> str:
                    return s[:n] + ("…" if len(s) > n else "")
                def _fmt_msg(m):
                    role = m.get("role", "")
                    content = m.get("content", "")
                    parts = []
                    if isinstance(content, list):
                        for it in content:
                            if isinstance(it, dict):
                                t = it.get("type")
                                if t == "text":
                                    parts.append(f"text:{_shorten(it.get('text',''))}")
                                elif t == "image_url":
                                    url_obj = it.get("image_url", {})
                                    url = url_obj.get("url", "")
                                    if isinstance(url, str) and url.startswith("data:"):
                                        parts.append(f"image:data-url(len={len(url)})")
                                    else:
                                        parts.append(f"image:{_shorten(str(url), 150)}")
                                else:
                                    parts.append(_shorten(json.dumps(it)))
                            else:
                                parts.append(_shorten(str(it)))
                    elif isinstance(content, str):
                        parts.append(f"text:{_shorten(content)}")
                    else:
                        parts.append(_shorten(str(content)))
                    return f"role={role} | " + " | ".join(parts)
                preview = "; ".join(_fmt_msg(m) for m in msgs)
                logger.info("请求messages: %s", preview)
            else:
                p = pload.get("prompt") or pload.get("inputs") or pload.get("text_input")
                if isinstance(p, str) and p:
                    short = p[:200] + ("…" if len(p) > 200 else "")
                    logger.info("请求prompt: %s", short)
        except Exception as e:
            logger.warning("打印messages失败: %s", str(e))

    time_record, chunk_record = await do_request(api_url, headers, pload, confirm_error_output,
                                                 output_len, num_scheduler_steps, use_spec_decode,
                                                 log_outputs=log_outputs)

    output_tokens = len(time_record) - 1

    # output_tokens will smaller than output_len when use spec decode, fix it in update_spec_output_tokens function.
    if not use_spec_decode and output_tokens < output_len:
        # 对于部分服务（如OpenAI兼容接口/多模态），模型可能因EOS/停止词提前结束，属正常行为，降级为警告
        logger.warning("output_tokens: %d < output_len: %d (模型可能提前停止，属正常现象)", output_tokens, output_len)

    request_latency_record.append((prompt_len, output_len, time_record, chunk_record))


async def benchmark(
        request_latency_record: List,
        backend: str,
        api_url: str,
    input_requests: List[Tuple[Union[str, Tuple[str, Optional[str]]], int, int]],
        best_of: int,
        use_beam_search: bool,
        request_rate: float,
        parallel_num: int,
        epochs: int,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
        num_scheduler_steps: int = 1,
        use_spec_decode: bool = False,
    api_key: str = None,
    log_messages: bool = False,
    log_outputs: bool = False,
) -> None:
    input_index = 0
    for ep in tqdm(range(epochs), desc="epoch"):
        input_parallel = []
        # print(f"请求prompt生成数量:{len(input_requests)}")
        # print(f"并行数量:{parallel_num}")
        # print(f"请求比例：{request_rate}")
        for id in range(parallel_num):
            #input_parallel.append(input_requests[input_index])
            input_parallel.append(input_requests[input_index])
            
            input_index += 1
            if input_index >= len(input_requests):
                input_index = 0
        # print(f"并行数量:{parallel_num}")
        # print(f"请求数量:{len(input_parallel)}")
        tasks: List[asyncio.Task] = []
        async for request in get_request(input_parallel, request_rate):
            prompt_or_tuple, prompt_len, output_len = request
            task = asyncio.create_task(send_request(request_latency_record,
                                                    backend, api_url, prompt_or_tuple,
                                                    prompt_len, output_len,
                                                    best_of, use_beam_search,
                                                    app_code,
                                                    model,
                                                    served_model_name,
                                                    num_scheduler_steps,
                                                    use_spec_decode,
                                                    api_key,
                                                    log_messages=log_messages,
                                                    log_outputs=log_outputs))
            tasks.append(task)
        await asyncio.gather(*tasks)


def group_get_output_tokens_per_step(output_list, tokenizer):
    token_list = tokenizer.batch_encode_plus(output_list, add_special_tokens=False, return_attention_mask=False)[
        "input_ids"]
    token_len_list = [0] + [len(token) for token in token_list]
    #  use max(end - start, 1) to ensure at lest 1 output token (when the model output eos token, the client will
    #  receive empty str).
    return [max(end - start, 1) for (end, start) in zip(token_len_list[1:], token_len_list[:-1])]


def update_spec_output_tokens(request_latency_record, tokenizer, backend):
    def decode_openai_output(chunk_list):
        output_list = []
        cur_output = ""
        for chunk in chunk_list:
            if chunk.startswith(b'data:'):
                output = chunk[5:].strip().decode("utf-8")
            else:
                output = chunk.strip().strip().decode("utf-8").rstrip("\0")

            if output != '[DONE]':
                output = json.loads(output)
                choice = output.get("choices", [{}])[0]
                # support both completions and chat-completions delta
                if "text" in choice:
                    part = choice.get("text", "")
                else:
                    part = choice.get("delta", {}).get("content", "")
                cur_output += part or ""
                output_list.append(cur_output)

        return output_list

    def decode_vllm_output(chunk_list):
        output_list = []
        for chunk in chunk_list:
            output = chunk.strip().strip().decode("utf-8").rstrip("\0")
            output = json.loads(output)["text"][0]
            output_list.append(output)

        return output_list

    if backend in ("openai", "openai-chat"):
        request_output_list = [decode_openai_output(chunk_list) for _, _, _, chunk_list in request_latency_record]
    elif backend == "vllm":
        request_output_list = [decode_vllm_output(chunk_list) for _, _, _, chunk_list in request_latency_record]
    elif backend == "vllm-chat":
        request_output_list = [decode_openai_output(chunk_list) for _, _, _, chunk_list in request_latency_record]
    else:
        logger.warning(
            f"Backend {backend} is not supported in spec decode benchmark, it might return the incorrect results.")
        return

    request_output_lens_list = [group_get_output_tokens_per_step(output_list, tokenizer) for output_list in
                                request_output_list]

    for req_index in range(len(request_latency_record)):
        (prompt_len, output_len, time_record, chunk_record) = request_latency_record[req_index]
        output_lens_per_step = request_output_lens_list[req_index]
        new_time_record = [time_record[0]]
        for o_len, timestamp in zip(output_lens_per_step, time_record[1:]):
            new_time_record.extend([timestamp] * o_len)

        output_tokens = len(new_time_record) - 1
        if len(new_time_record) - 1 < output_len:
            logger.warning("output_tokens: %d < output_len: %d (SpecDecode路径下由实际输出长度计算)", output_tokens, output_len)

        # replace chunk_record to output_step for acceptance rate of speculative decode.
        request_latency_record[req_index] = (prompt_len, output_len, new_time_record, len(output_lens_per_step))


def get_dataset_requests(args, tokenizer, prompt_tokens, output_tokens, parallel_num,
                         image_payloads: Optional[List[str]] = None,
                         random_image_gen: Optional[Callable[[], str]] = None):
    if args.dataset_type == "random":
        # logger.info(
        #     f"Benchmark running with parallel_num: %d, "
        #     f"prompt_tokens: %d, output_tokens: %d",
        #     parallel_num, prompt_tokens, output_tokens
        # )
        return sample_random_requests(tokenizer, prompt_tokens, output_tokens,
                          args.enable_prefix_caching, args.prefix_caching_num,
                          image_payloads, random_image_gen)
    else:
        if not os.path.exists(args.dataset_path):
            raise ValueError(f"Dataset path {args.dataset_path} is not existed.")

        logger.info(
            f"Benchmark running with parallel_num: %d, "
            f"dataset: %s, output_tokens: %s",
            parallel_num,
            args.dataset_type, "real" if args.use_real_dataset_output_tokens else str(output_tokens)
        )

        if args.dataset_type == "sharegpt":
            return sample_sharegpt_requests(args.dataset_path, parallel_num * args.epochs, tokenizer,
                                            None if args.use_real_dataset_output_tokens else output_tokens)
        elif args.dataset_type == "human-eval":
            return sample_human_eval_requests(args.dataset_path, parallel_num * args.epochs,
                                              tokenizer,
                                              None if args.use_real_dataset_output_tokens else output_tokens)
        else:
            raise ValueError("Unsupport dataset.")


def main(args: argparse.Namespace):
    logger.info(args)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    if args.enable_prefix_caching and not args.prefix_caching_num:
        logger.error("The prefix_caching_num parameter must be seted for prefix_caching.")
        return

    api_url = get_api_url(args.backend, args.host, args.port, args.url)
    tokenizer = get_tokenizer(args.tokenizer)

    if not args.use_spec_decode:
        chat_backends = {"openai-chat", "vllm-chat"}
        if args.backend in chat_backends or getattr(args, "multimodal", False):
            logger.info("Skip multi-step check for chat/multimodal backend to avoid early-stop false negatives.")
        else:
            is_multi_step = check_multi_step(args, api_url, tokenizer, args.prompt_tokens[0], args.output_tokens[0])
            if not is_multi_step:
                logger.error("The service does not use multi_step or num-scheduler-steps is different from the service.")
                return

    # Prepare image payloads if multimodal
    image_payloads: Optional[List[str]] = None
    random_image_gen: Optional[Callable[[], str]] = None
    if getattr(args, "multimodal", False):
        image_payloads = []
        # Single URL or data URL
        if args.image_url:
            image_payloads.append(args.image_url)
        # Directory images
        if args.image_dir and os.path.isdir(args.image_dir):
            patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif"]
            paths: List[str] = []
            for ptn in patterns:
                paths.extend(glob(os.path.join(args.image_dir, ptn)))

            def to_data_url(path: str) -> str:
                mime, _ = mimetypes.guess_type(path)
                mime = mime or "application/octet-stream"
                with open(path, "rb") as f:
                    b = f.read()
                return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

            for p in paths:
                try:
                    image_payloads.append(to_data_url(p))
                except Exception as e:
                    logger.warning("skip image %s due to error: %s", p, str(e))
        # Per-request random image
        if getattr(args, "random_images", False):
            def parse_size(sz: str) -> Tuple[int, int]:
                try:
                    if 'x' in sz:
                        w, h = sz.lower().split('x')
                    elif ',' in sz:
                        w, h = sz.split(',')
                    else:
                        parts = sz.split()
                        w, h = parts[0], parts[1]
                    return int(w), int(h)
                except Exception:
                    return 224, 224

            W, H = parse_size(getattr(args, "image_size", "224x224"))

            def rand_image_data_url(w: int, h: int) -> str:
                try:
                    from PIL import Image
                    import io
                    import numpy as np
                    arr = (np.random.rand(h, w, 3) * 255).astype('uint8')
                    img = Image.fromarray(arr, 'RGB')
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    b = buf.getvalue()
                    return f"data:image/png;base64,{base64.b64encode(b).decode('utf-8')}"
                except Exception:
                    one_px_png_b64 = (
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
                    )
                    return f"data:image/png;base64,{one_px_png_b64}"

            random_image_gen = lambda: rand_image_data_url(W, H)
        # Guard: need either provided images or on-the-fly generator
        if not image_payloads and random_image_gen is None:
            logger.error("multimodal enabled but no image provided via --image-url/--image-dir and --random-images not set")
            return

    logger.info(f"Warmup ...")
    input_requests = sample_random_requests(tokenizer, args.prompt_tokens[0], args.output_tokens[0],
                                            image_payloads=image_payloads,
                                            random_image_gen=random_image_gen)
    asyncio.run(
        benchmark([], args.backend, api_url, input_requests, args.best_of,
                  args.use_beam_search, args.request_rate, 4, 1, args.app_code, args.tokenizer,
                  args.served_model_name, args.num_scheduler_steps, args.use_spec_decode, args.api_key,
                  log_messages=getattr(args, 'log_messages', False),
                  log_outputs=getattr(args, 'log_outputs', False))
    )

    if args.dataset_type != "random":
        logging.info("When use sharegpt or human-eval dataset, the number of the --input-tokens will be ignored.")
    if args.use_real_dataset_output_tokens:
        logging.info("When use --use-real-dataset-output-tokens, the number of the --output-tokens will be ignored.")
    if args.use_spec_decode and args.num_speculative_tokens >= 0:
        logging.info("When enable --use-spec-decode and --num-speculative-tokens >= 0, the acceptance rate of "
                     "speculative inference will be collected. Ensure that --num-speculative-tokens in the benchmark is "
                     "equal to --num-speculative-tokens in the vllm service.")

    all_latency_record = []
    for parallel_num in args.parallel_num:
        for i, prompt_tokens in enumerate(args.prompt_tokens):
            output_tokens = args.output_tokens[i]
            input_requests=[]
            logger.info(
                f"Benchmark running with parallel_num: %d, "
                f"prompt_tokens: %d, output_tokens: %d",
                parallel_num, prompt_tokens, output_tokens
                )                                                   
            for i in range(parallel_num):
                input_requests.append(
                    get_dataset_requests(
                        args, tokenizer, prompt_tokens, output_tokens, parallel_num,
                        image_payloads, random_image_gen
                    )[0]
                )
            #input_requests = get_dataset_requests(args, tokenizer, prompt_tokens, output_tokens, parallel_num)

            request_latency_record: List[Tuple[int, int, List]] = []
            asyncio.run(benchmark(request_latency_record, args.backend, api_url, input_requests, args.best_of,
                                  args.use_beam_search, args.request_rate, parallel_num,
                                  args.epochs, args.app_code, args.tokenizer,
                                  args.served_model_name, args.num_scheduler_steps, args.use_spec_decode,
                                  args.api_key,
                                  log_messages=getattr(args, 'log_messages', False),
                                  log_outputs=getattr(args, 'log_outputs', False)))
            if args.use_spec_decode:
                update_spec_output_tokens(request_latency_record, tokenizer, args.backend)

            statistics_and_print_performance_data(args, prompt_tokens, output_tokens, parallel_num,
                                                  request_latency_record, all_latency_record)

    if args.use_spec_decode:
        all_latency_record.sort(key=lambda x: x[2])
    else:
        all_latency_record.sort()

    benchmark_head = ["输入长度", "输出长度", "并发数",
                      "输出tokens总吞吐",
                      "首tokens时延TP90（ms）", "首tokens时延TP99（ms）", "最大首tokens时延（ms）", "平均首tokens时延（ms）",
                      "增量时延TP90（ms）", "增量时延TP99（ms）", "最大增量时延（ms）", "平均增量时延（ms）"]

    if args.use_spec_decode and args.num_speculative_tokens >= 0 and len(all_latency_record[-1]) != len(benchmark_head):
        accept_head = ["投机接受率TP90", "投机接受率TP99", "投机最大接受率", "投机最小接受率", "投机平均接受率"]
        benchmark_head = benchmark_head + accept_head

    save_to_csv(benchmark_head, all_latency_record, args.benchmark_csv)

    logger.info("Benchmark parallel finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the serving prefill performance.")
    parser.add_argument("--backend", type=str, default="mindspore",
                        choices=["vllm", "mindspore", "base", "tgi", "openai", "trt", "openai-chat", "vllm-chat"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9288)
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--app-code", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs.")
    parser.add_argument("--parallel-num", nargs='+', type=int, default=[1, 4, 8],
                        help="Number of parallel request number.")
    parser.add_argument("--output-tokens", nargs='+', type=int, default=[256, 256, 256],
                        help="Max tokens to process.")
    parser.add_argument("--prompt-tokens", nargs='+', type=int, default=[512, 1024, 2048],
                        help="Max tokens to process.")
    parser.add_argument("--benchmark-csv", type=str, default="benchmark_parallel.csv",
                        help="Path to the csv.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--num-scheduler-steps", type=int, default=1)
    parser.add_argument("--enable-prefix-caching", type=bool, default=False)
    parser.add_argument("--prefix-caching-num", type=int, default=0)
    parser.add_argument("--use-spec-decode", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=-1, help="the step if spec decode, default -1 for disable accept rate statistic.")
    parser.add_argument("--dataset-type", default="random", choices=["random", "sharegpt", "human-eval"])
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--use-real-dataset-output-tokens", action="store_true")
    parser.add_argument("--api-key", type=str)
    # Logging controls
    parser.add_argument("--log-messages", action="store_true", help="打印每次请求的messages/prompt预览（默认关闭）")
    parser.add_argument("--log-outputs", action="store_true", help="打印模型输出预览（默认关闭）")
    # Multimodal options
    parser.add_argument("--multimodal", action="store_true", help="Enable image+text multimodal requests (chat endpoints)")
    parser.add_argument("--image-dir", type=str, default="", help="Directory containing images to use")
    parser.add_argument("--image-url", type=str, default="", help="Single image URL or data URL to reuse")
    parser.add_argument("--random-images", action="store_true", help="Enable per-request random image generation (data URL)")
    parser.add_argument("--image-size", type=str, default="224x224", help="Random image size WxH, e.g., 224x224 (used when --random-images)")
    args_global = parser.parse_args()
    print(f"Running benchmark with args: {args_global}")
    main(args_global)
