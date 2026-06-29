import argparse

import run_bench_serve as rbs


class FakeTokenizer:
    vocab_size = 128

    def decode(self, token_ids):
        return " ".join(f"tok{token_id}" for token_id in token_ids)

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            input_ids = text.split() if text else []

        return Encoded()


def test_random_prefix_prompt_len_includes_prefix_and_suffix():
    args = argparse.Namespace(
        num_prompts=2,
        random_input_len=8,
        random_output_len=4,
        random_prefix_len=3,
        random_range_ratio=1.0,
    )

    requests = rbs._generate_random_requests(args, FakeTokenizer())

    assert [req.prompt_len for req in requests] == [11, 11]
    assert [req.expected_output_len for req in requests] == [4, 4]
    assert requests[0].prompt != requests[1].prompt
