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


class CollapsingTokenizer(FakeTokenizer):
    vocab_size = 128

    def decode(self, token_ids):
        return " ".join("tok" for _ in token_ids)


class ShortEncodingTokenizer(FakeTokenizer):
    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            input_ids = text.split()[:-1] if text else []

        return Encoded()


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


def test_random_full_prefix_range_ratio_extends_shared_prefix(monkeypatch):
    def upper_bound(lo, hi):
        return hi

    monkeypatch.setattr(rbs.random, "randint", upper_bound)
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=8,
        random_range_ratio=2.0,
    )

    requests = rbs._generate_random_requests(args, FakeTokenizer())

    assert [req.prompt_len for req in requests] == [16, 16]
    assert requests[0].prompt == requests[1].prompt


def test_random_full_prefix_reuses_decoded_prompt_when_tokenizer_is_short(
    monkeypatch,
):
    next_token = iter(range(1000))

    def increasing_token(*args):
        return next(next_token)

    monkeypatch.setattr(rbs.random, "randrange", increasing_token)
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=8,
        random_range_ratio=1.0,
    )

    requests = rbs._generate_random_requests(args, ShortEncodingTokenizer())

    assert [req.prompt_len for req in requests] == [8, 8]
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


def test_random_prefix_raises_when_decoded_prompts_collide(monkeypatch):
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

    with pytest.raises(ValueError, match="unique prompt"):
        rbs._generate_random_requests(args, CollapsingTokenizer())


def test_random_prefix_range_ratio_keeps_suffix_for_short_inputs(monkeypatch):
    def lower_bound(lo, hi):
        return lo

    monkeypatch.setattr(rbs.random, "randint", lower_bound)
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=3,
        random_range_ratio=4.0,
    )

    requests = rbs._generate_random_requests(args, FakeTokenizer())

    assert [req.prompt_len for req in requests] == [2, 2]
    assert requests[0].prompt != requests[1].prompt


def test_get_samples_supports_builtin_mtp_chat(monkeypatch):
    calls = []

    def fake_build_requests(args, tokenizer, sample_request_cls):
        calls.append((args.dataset_name, tokenizer, sample_request_cls))
        return [
            rbs.SampleRequest(
                prompt="user: explain mtp\nassistant:",
                prompt_len=4,
                expected_output_len=8,
            )
        ]

    monkeypatch.setattr(
        rbs.builtin_mtp_chat,
        "build_requests",
        fake_build_requests,
    )
    args = argparse.Namespace(
        dataset_name="builtin_mtp_chat",
        input_len=32,
        output_len=8,
        num_prompts=1,
    )

    requests = rbs.get_samples(args, object())

    assert len(requests) == 1
    assert calls[0][0] == "builtin_mtp_chat"
    assert calls[0][2] is rbs.SampleRequest
