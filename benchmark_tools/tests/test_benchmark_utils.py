import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_tools.benchmark_utils import (
    add_response_end_timestamp_if_needed,
    get_request_data,
)


class RequestPayloadTest(unittest.TestCase):
    def test_openai_chat_payload_ignores_eos_for_fixed_length_benchmark(self):
        _, payload, _ = get_request_data(
            backend="openai-chat",
            prompt="hello",
            prompt_len=1,
            output_len=512,
            best_of=1,
            use_beam_search=False,
            served_model_name="qwen15b",
        )

        self.assertIs(payload["ignore_eos"], True)

    def test_response_end_timestamp_is_used_when_usage_reports_more_tokens(self):
        time_record = [10.0, 10.1]

        add_response_end_timestamp_if_needed(
            time_record,
            server_reported_output_tokens=512,
            response_end_time=15.2,
        )

        self.assertEqual(time_record, [10.0, 10.1, 15.2])


if __name__ == "__main__":
    unittest.main()
