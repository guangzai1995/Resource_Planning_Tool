import argparse
import asyncio
import json
import os
import random
import stat
import time
from typing import AsyncGenerator, List, Tuple, Union, Optional

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

import logging
import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from benchmark_utils import generate_str, get_tokenizer


DATASET: List = []


def sample_requests(
    min_input: int,
    max_input: int,
    avg_input: int,
    std_input: int,
    min_output: int,
    max_output: int,
    avg_output: int,
    std_output: int,
    num_requests: int,
):
    input_samples = np.random.normal(loc=avg_input, scale=std_input, size=num_requests).astype(int)

    input_samples = np.clip(input_samples, min_input, max_input)

    output_samples = np.random.normal(loc=avg_output, scale=std_output, size=num_requests).astype(int)

    output_samples = np.clip(output_samples, min_output, max_output)

    return input_samples, output_samples


def get_outputs(
    tokenizer,
    input_len: int,
    output_len: int,
) -> None:
    input_str = generate_str(tokenizer, input_len)
    output_str = generate_str(tokenizer, output_len)

    request = {
        "input": input_str,
        "output": output_str,
        "instruction": "NA"
    }
    DATASET.append(request)


def do_generate(
    tokenizer,
    input_samples,
    output_samples
) -> None:
    for i in tqdm(range(input_samples.shape[0])):
        input_len = input_samples[i]
        output_len = output_samples[i]
        get_outputs(tokenizer, input_len, output_len)


def main(args: argparse.Namespace):
    """
    main entry
    """
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = get_tokenizer(args.tokenizer)
    input_samples, output_samples = sample_requests(
        args.min_input,
        args.max_input,
        args.avg_input,
        args.std_input,
        args.min_output,
        args.max_output,
        args.avg_output,
        args.std_output,
        args.num_requests,
    )

    do_generate(tokenizer, input_samples, output_samples)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(args.dataset, flags, mode), 'w') as f:
        json.dump(DATASET, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--min-input", type=int, default=100,
                        help="Number of min-input to process.")
    parser.add_argument("--max-input", type=int, default=3600,
                        help="Number of max-input to process.")
    parser.add_argument("--avg-input", type=int, default=1800,
                        help="Number of avg-input to process.")
    parser.add_argument("--std-input", type=int, default=500,
                        help="Number of std-input to process.")
    parser.add_argument("--min-output", type=int, default=40,
                        help="Number of min-output to process.")
    parser.add_argument("--max-output", type=int, default=256,
                        help="Number of max_output to process.")
    parser.add_argument("--avg-output", type=int, default=160,
                        help="Number of avg-output to process.")
    parser.add_argument("--std-output", type=int, default=30,
                        help="Number of std-output to process.")
    parser.add_argument("--num-requests", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    args_global = parser.parse_args()
    main(args_global)
