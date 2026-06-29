import json
import math

import pytest

import auto_bench as ab


def write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_config_at(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def value_after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


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
    data = minimal_config(tmp_path)
    del data["run"]["container_port"]
    del data["run"]["publish_host_port"]
    path = write_config(tmp_path, data)

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


@pytest.mark.parametrize(("section", "value"), [
    ("models", ".."),
    ("serve_profiles", "."),
])
def test_dot_names_are_rejected(tmp_path, section, value):
    data = minimal_config(tmp_path)
    data[section][0]["name"] = value

    with pytest.raises(ab.ConfigError, match="safe filename"):
        ab.load_config(write_config(tmp_path, data))


def test_relative_model_mount_is_resolved_from_config_dir(tmp_path):
    config_dir = tmp_path / "configs"
    model_root = config_dir / "model"
    model_dir = model_root / "Qwen2.5-1.5B-Instruct"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    data = minimal_config(tmp_path)
    data["mounts"]["models"] = "model"

    config = ab.load_config(write_config_at(config_dir / "config.json", data))

    assert config.mounts.models == (config_dir / "model").resolve()
    assert config.models[0].host_model_path == (
        config_dir / "model" / "Qwen2.5-1.5B-Instruct"
    ).resolve()


def test_model_container_path_must_not_escape_models_root(tmp_path):
    data = minimal_config(tmp_path)
    data["models"][0]["model_path"] = "/models/../outside"

    with pytest.raises(ab.ConfigError, match="/models|model path"):
        ab.load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("prefix_ratio", [1.5, math.nan])
def test_prefix_ratio_must_be_finite_ratio(tmp_path, prefix_ratio):
    data = minimal_config(tmp_path)
    data["bench_profiles"][0]["prefix_ratio"] = prefix_ratio

    with pytest.raises(ab.ConfigError, match="prefix_ratio"):
        ab.load_config(write_config(tmp_path, data))


def test_backend_must_be_supported(tmp_path):
    data = minimal_config(tmp_path)
    data["bench_profiles"][0]["backend"] = "bad"

    with pytest.raises(ab.ConfigError, match="backend"):
        ab.load_config(write_config(tmp_path, data))


def test_output_lens_must_broadcast_or_match_input_lens(tmp_path):
    data = minimal_config(tmp_path)
    data["bench_profiles"][0]["input_lens"] = [64, 128, 256]
    data["bench_profiles"][0]["output_lens"] = [32, 64]

    with pytest.raises(ab.ConfigError, match="output_lens"):
        ab.load_config(write_config(tmp_path, data))


def test_build_vllm_command_uses_bridge_network_without_host_port(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    run_dir = tmp_path / "results" / "run123"

    cmd = ab.build_vllm_run_command(config, case, run_dir)

    assert "--network" in cmd
    assert "vllm-bench-net" in cmd
    assert value_after(cmd, "--name") == case.container_name
    assert "--network=host" not in cmd
    assert "-p" not in cmd
    assert "vllm serve" not in " ".join(cmd)
    assert "serve" in cmd
    serve_index = cmd.index("serve")
    assert cmd[serve_index + 1] == "/models/Qwen2.5-1.5B-Instruct"
    assert cmd[-10:] == [
        "vllm", "serve", "/models/Qwen2.5-1.5B-Instruct",
        "--served-model-name", "qwen2_5_1_5b",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", "local-bench-key",
        "--dtype", "bfloat16",
    ][-10:]


def test_build_bench_command_targets_container_dns(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    bench_dir = tmp_path / "results" / "run123" / "qwen2_5_1_5b" / "bf16_default" / "smoke"

    cmd = ab.build_bench_run_command(config, case, bench_dir)

    assert "--network" in cmd
    assert "vllm-bench-net" in cmd
    assert "--base-url" in cmd
    assert f"http://{case.container_name}:8000/v1" in cmd
    assert "--model" in cmd
    assert "qwen2_5_1_5b" in cmd
    assert "--output-csv" in cmd
    assert "/results/result.csv" in cmd
    assert value_after(cmd, "--served-model-name") == "qwen2_5_1_5b"
    assert value_after(cmd, "--output-xlsx") == "/results/result.xlsx"


def test_network_cleanup_only_removes_owned_empty_network():
    assert ab.should_cleanup_network(owned=True, cleanup_enabled=True, connected_containers=[]) is True
    assert ab.should_cleanup_network(owned=False, cleanup_enabled=True, connected_containers=[]) is False
    assert ab.should_cleanup_network(owned=True, cleanup_enabled=False, connected_containers=[]) is False
    assert ab.should_cleanup_network(owned=True, cleanup_enabled=True, connected_containers=["external"]) is False


def test_validate_local_paths_rejects_missing_model_dir(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    config.models[0].host_model_path.rename(tmp_path / "missing-model")

    with pytest.raises(ab.ConfigError, match="model path"):
        ab.validate_local_paths(config)
