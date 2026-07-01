from types import SimpleNamespace

import pytest

import run_bench_serve as rbs
from vllm_bench.datasets import builtin_mtp_chat


class ChatTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        if tokenize:
            return text.split()
        return text

    def encode(self, text):
        return text.split()


def _args(**overrides):
    values = {
        "input_len": 80,
        "random_input_len": 80,
        "output_len": 16,
        "random_output_len": 16,
        "num_prompts": 3,
        "dataset_length_policy": "bucket",
        "dataset_input_len_tolerance": 0.5,
        "dataset_on_bucket_shortage": "error",
        "dataset_sampling": "round_robin",
        "seed": 7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_builtin_mtp_chat_builds_bucketed_requests():
    requests = builtin_mtp_chat.build_requests(
        _args(),
        ChatTokenizer(),
        rbs.SampleRequest,
    )

    assert len(requests) == 3
    assert all(40 <= item.prompt_len <= 120 for item in requests)
    assert all(item.expected_output_len == 16 for item in requests)
    assert all(isinstance(item.prompt, str) for item in requests)
    assert all("assistant:" in item.prompt for item in requests)


def test_builtin_mtp_chat_requires_tokenizer():
    with pytest.raises(ValueError, match="requires --tokenizer"):
        builtin_mtp_chat.build_requests(_args(num_prompts=1), None, rbs.SampleRequest)


def test_builtin_mtp_chat_is_stable_for_same_seed():
    first = builtin_mtp_chat.build_requests(_args(dataset_sampling="shuffle"),
                                            ChatTokenizer(), rbs.SampleRequest)
    second = builtin_mtp_chat.build_requests(_args(dataset_sampling="shuffle"),
                                             ChatTokenizer(), rbs.SampleRequest)

    assert [(item.prompt, item.prompt_len) for item in first] == [
        (item.prompt, item.prompt_len) for item in second
    ]


def test_builtin_mtp_chat_errors_when_bucket_has_no_candidates():
    with pytest.raises(ValueError, match="no prompts in token bucket"):
        builtin_mtp_chat.build_requests(
            _args(input_len=1, random_input_len=1, dataset_input_len_tolerance=0.0),
            ChatTokenizer(),
            rbs.SampleRequest,
        )
