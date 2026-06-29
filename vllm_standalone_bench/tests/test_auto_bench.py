import json

import pytest

import auto_bench as ab


def write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def minimal_config(tmp_path):
    model_root = tmp_path / "model"
    model_dir = model_root / "Qwen2.5-1.5B-Instruct"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return {
        "run": {
            "name": "smoke",
            "results_dir": str(tmp_path / "results"),
            "vllm_image": "009e4cb46541",
            "bench_image": "vllm-bench-runner:offline",
            "network": "vllm-bench-net",
            "create_network": True,
            "cleanup_network": True,
            "container_port": 8000,
            "publish_host_port": False,
            "api_key": "local-bench-key",
            "ready_timeout_sec": 30,
            "cooldown_sec": 0
        },
        "mounts": {"models": str(model_root)},
        "models": [{
            "name": "qwen2_5_1_5b",
            "model_path": "/models/Qwen2.5-1.5B-Instruct",
            "tokenizer_path": "/models/Qwen2.5-1.5B-Instruct",
            "served_model_name": "qwen2_5_1_5b"
        }],
        "serve_profiles": [{
            "name": "bf16_default",
            "gpus": "all",
            "args": ["--dtype", "bfloat16"]
        }],
        "bench_profiles": [{
            "name": "smoke",
            "backend": "openai-chat",
            "input_lens": [64],
            "output_lens": [32],
            "parallel_nums": [1],
            "epochs": 1,
            "prefix_ratio": 0.0,
            "warmup_requests": 0
        }]
    }


def test_load_config_applies_defaults_and_expands_cases(tmp_path):
    path = write_config(tmp_path, minimal_config(tmp_path))

    config = ab.load_config(path)
    cases = ab.expand_cases(config)

    assert config.run.name == "smoke"
    assert config.run.container_port == 8000
    assert config.run.publish_host_port is False
    assert config.models[0].host_model_path == tmp_path / "model" / "Qwen2.5-1.5B-Instruct"
    assert len(cases) == 1
    assert cases[0].api_model_name == "qwen2_5_1_5b"
    assert cases[0].container_name.startswith("bench-vllm-qwen2_5_1_5b-bf16_default-")


def test_invalid_name_is_rejected(tmp_path):
    data = minimal_config(tmp_path)
    data["models"][0]["name"] = "bad/name"

    with pytest.raises(ab.ConfigError, match="safe filename"):
        ab.load_config(write_config(tmp_path, data))


def test_output_lens_must_broadcast_or_match_input_lens(tmp_path):
    data = minimal_config(tmp_path)
    data["bench_profiles"][0]["input_lens"] = [64, 128, 256]
    data["bench_profiles"][0]["output_lens"] = [32, 64]

    with pytest.raises(ab.ConfigError, match="output_lens"):
        ab.load_config(write_config(tmp_path, data))
