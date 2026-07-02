---
name: manual-interface-performance-testing
description: Use when manually validating text, ASR, or generic HTTP model API requests, including smoke tests, small batches, curl generation, and request-shape debugging.
---

# Manual Interface Performance Testing

Use this skill when the user asks to try an endpoint, send one or a few requests, generate curl, validate ASR upload format, or debug request errors before a benchmark.

## Required Workflow

1. Identify the protocol and endpoint.
2. Generate or inspect the request shape.
3. Run a smoke request or print an equivalent curl command.
4. Report status code, latency, response summary, and error diagnosis.
5. Use small batches only for stability checks, not full bottleneck conclusions.

## Preferred Commands

```bash
python3 scripts/perf_manual.py --config configs/openai_chat.json --mode smoke --dry-run
python3 scripts/perf_manual.py --config configs/openai_asr.json --audio-file sample.wav --print-curl
```
