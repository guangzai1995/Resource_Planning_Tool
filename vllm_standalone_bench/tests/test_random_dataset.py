import argparse

import pytest

import run_bench_serve as rbs


class FakeTokenizer:
    vocab_size = 128

    def decode(self, token_ids):
        return " ".join(f"tok{token_id}" for token_id in token_ids)

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            input_ids = text.split() if text else []

        return Encoded()


class TinyTokenizer(FakeTokenizer):
    vocab_size = 2


def test_random_prefix_prompt_len_uses_total_input_budget():
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=3,
        random_range_ratio=1.0,
    )

    requests = rbs._generate_random_requests(args, FakeTokenizer())

    assert [req.prompt_len for req in requests] == [8, 8]
    assert [req.expected_output_len for req in requests] == [4, 4]
    assert requests[0].prompt != requests[1].prompt


def test_random_full_prefix_keeps_total_input_budget():
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=8,
        random_range_ratio=1.0,
    )

    requests = rbs._generate_random_requests(args, FakeTokenizer())

    assert [req.prompt_len for req in requests] == [8, 8]
    assert [req.expected_output_len for req in requests] == [4, 4]
    assert requests[0].prompt == requests[1].prompt


def test_random_prefix_suffix_is_unique_even_when_random_collides(monkeypatch):
    def always_zero(*args):
        return 0

    monkeypatch.setattr(rbs.random, "randrange", always_zero)
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=3,
        random_range_ratio=1.0,
    )

    requests = rbs._generate_random_requests(args, FakeTokenizer())

    assert [req.prompt_len for req in requests] == [8, 8]
    assert requests[0].prompt != requests[1].prompt


def test_random_prefix_text_suffix_is_unique_even_when_random_collides(
    monkeypatch,
):
    def always_zero(*args):
        return 0

    monkeypatch.setattr(rbs.random, "randint", always_zero)
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=3,
        random_range_ratio=1.0,
    )

    requests = rbs._generate_random_requests(args, None)

    assert [req.prompt_len for req in requests] == [8, 8]
    assert requests[0].prompt != requests[1].prompt


def test_random_prefix_raises_when_unique_suffix_capacity_is_exhausted(
    monkeypatch,
):
    def always_zero(*args):
        return 0

    monkeypatch.setattr(rbs.random, "randrange", always_zero)
    args = argparse.Namespace(
        num_prompts=3,
        random_input_len=1,
        random_output_len=4,
        random_prefix_len=0,
        random_range_ratio=1.0,
    )

    with pytest.raises(ValueError, match="unique suffix"):
        rbs._generate_random_requests(args, TinyTokenizer())
