import csv
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path

import pytest

import auto_bench as ab

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


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


def values_after(cmd, flag):
    return [cmd[index + 1] for index, value in enumerate(cmd) if value == flag]


def labels_from(cmd):
    labels = {}
    for value in values_after(cmd, "--label"):
        key, separator, label_value = value.partition("=")
        if separator:
            labels[key] = label_value
    return labels


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


def asr_config(tmp_path):
    data = minimal_config(tmp_path)
    data["models"][0]["name"] = "qwen3_asr_1_7b"
    data["models"][0]["served_model_name"] = "qwen3-asr"
    data["bench_profiles"][0] = {
        "name": "asr_smoke",
        "backend": "openai-audio",
        "output_lens": [128],
        "parallel_nums": [1, 4],
        "epochs": 1,
        "warmup_requests": 0,
        "dataset_name": "custom_audio",
        "language": "en",
    }
    return data


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


def test_expand_cases_uses_topology_profiles(tmp_path):
    from test_remote_topology import pd_topology_config

    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    cases = ab.expand_cases(config, run_id="run123")

    assert len(cases) == 1
    case = cases[0]
    assert case.serve_profile is None
    assert case.topology_profile.name == "sglang_pd_2p2d"
    assert case.serving_name == "sglang_pd_2p2d"


def test_start_detached_accepts_topology_profiles_before_lock(tmp_path, monkeypatch):
    from test_remote_topology import pd_topology_config

    config_path = write_config(tmp_path, pd_topology_config(tmp_path))
    config = ab.load_config(config_path)
    run_dir = tmp_path / "results" / "run123"
    calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)

    exit_code = ab.start_detached(config_path, config, "run123")

    assert exit_code == 0
    assert calls
    assert (run_dir / "state.json").exists()
    assert (run_dir / ".run.lock").exists()


def test_legacy_command_helpers_reject_topology_case(tmp_path):
    from test_remote_topology import pd_topology_config

    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    with pytest.raises(ab.ConfigError, match="legacy"):
        ab.build_serve_run_command(config, case, tmp_path / "run")


def test_topology_layout_uses_topology_profile(tmp_path):
    from test_remote_topology import (
        pd_topology_config,
        write_config as write_topology_config,
    )

    config = ab.load_config(write_topology_config(tmp_path, pd_topology_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    layout = ab.build_layout(config, "run123", case)

    model = "qwen2_5_1_5b"
    assert layout.serve_dir == tmp_path / "results" / "run123" / model / "sglang_pd_2p2d"
    assert layout.bench_dir == layout.serve_dir / "smoke"


def test_topology_bench_command_targets_frontend_endpoint(tmp_path):
    from test_remote_topology import pd_topology_config

    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_bench_run_command(config, case, tmp_path / "bench")

    labels = labels_from(cmd)
    assert value_after(cmd, "--base-url") == "http://10.0.0.31:8000/v1"
    assert labels["vllm_auto_bench.topology_profile"] == "sglang_pd_2p2d"
    assert "vllm_auto_bench.serve_profile" not in labels


def test_topology_dry_run_masks_passwords_and_prints_remote_commands(tmp_path, capsys):
    from test_remote_topology import pd_topology_config

    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["image"] = "sglang:pd"
    topology["hosts"]["p1"]["auth"] = {
        "type": "password",
        "password": "secret-p1-password",
    }
    topology["hosts"]["p2"]["auth"]["key_path"] = "/home/bench/.ssh/id_ed25519"
    topology["frontend"]["args"] = [
        "--api-key",
        "router-secret",
        "--openai-api-key",
        "router-openai-secret",
    ]
    topology["env"] = {
        "OPENAI_API_KEY": "profile-openai-secret",
        "DB_PASSWORD": "profile-db-password",
        "VISIBLE_ENV": "profile-visible",
    }
    topology["prefill"][0]["env"] = {
        "SERVICE_TOKEN": "node-service-token",
        "VISIBLE_NODE_ENV": "node-visible",
    }
    topology["prefill"][0]["args"] = [
        "--db-password",
        "node-db-secret",
        "--tokenizer-path",
        "/models/Qwen2.5-1.5B-Instruct",
    ]
    config = ab.load_config(write_config(tmp_path, data))

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=True)

    out = capsys.readouterr().out
    resolved = json.loads(
        (tmp_path / "results" / "run123" / "config.resolved.json").read_text(
            encoding="utf-8"
        )
    )
    rendered_resolved = json.dumps(resolved)
    assert result == 0
    assert "sglang.launch_server" in out
    assert "sglang_router.launch_router" in out
    assert "run_bench_multi.py" in out
    assert "--pd-disaggregation" in out
    assert "docker network create" not in out
    assert "docker network rm" not in out
    assert "secret-p1-password" not in out
    assert "router-secret" not in out
    assert "router-openai-secret" not in out
    assert "node-db-secret" not in out
    assert "local-bench-key" not in out
    assert "secret-p1-password" not in rendered_resolved
    assert "router-secret" not in rendered_resolved
    assert "router-openai-secret" not in rendered_resolved
    assert "node-db-secret" not in rendered_resolved
    assert "local-bench-key" not in rendered_resolved
    assert "profile-openai-secret" not in rendered_resolved
    assert "profile-db-password" not in rendered_resolved
    assert "node-service-token" not in rendered_resolved
    assert resolved["models"][0]["tokenizer_path"] == (
        "/models/Qwen2.5-1.5B-Instruct"
    )
    assert resolved["models"][0]["host_tokenizer_path"] != "***"
    assert resolved["run"]["api_key"] == "***"
    assert resolved["topology_profiles"][0]["frontend"]["args"] == [
        "--api-key",
        "***",
        "--openai-api-key",
        "***",
    ]
    assert resolved["topology_profiles"][0]["env"]["OPENAI_API_KEY"] == "***"
    assert resolved["topology_profiles"][0]["env"]["DB_PASSWORD"] == "***"
    assert resolved["topology_profiles"][0]["env"]["VISIBLE_ENV"] == "profile-visible"
    assert (
        resolved["topology_profiles"][0]["hosts"]["p2"]["auth"]["key_path"]
        == "/home/bench/.ssh/id_ed25519"
    )
    assert resolved["topology_profiles"][0]["prefill"][0]["env"] == {
        "SERVICE_TOKEN": "***",
        "VISIBLE_NODE_ENV": "node-visible",
    }
    assert resolved["topology_profiles"][0]["prefill"][0]["args"] == [
        "--db-password",
        "***",
        "--tokenizer-path",
        "/models/Qwen2.5-1.5B-Instruct",
    ]
    assert (
        resolved["topology_profiles"][0]["hosts"]["p1"]["auth"].get("password")
        == "***"
    )


def secret_topology_config(tmp_path):
    from test_remote_topology import pd_topology_config

    data = pd_topology_config(tmp_path)
    data["run"]["api_key"] = "run-secret"
    data["run"]["resource_monitor"] = {"enabled": False}
    topology = data["topology_profiles"][0]
    topology["image"] = "sglang:pd"
    topology["hosts"]["p1"]["auth"] = {
        "type": "password",
        "password": "ssh-secret",
    }
    topology["env"] = {
        "OPENAI_API_KEY": "env-secret",
        "VISIBLE_ENV": "visible",
    }
    topology["frontend"]["args"] = ["--api-key", "router-secret"]
    topology["prefill"][0]["env"] = {"SERVICE_TOKEN": "node-token-secret"}
    topology["prefill"][0]["args"] = ["--db-password", "node-password-secret"]
    return data


def test_real_run_writes_redacted_resolved_config_and_private_resume_config(
    tmp_path,
    monkeypatch,
):
    data = secret_topology_config(tmp_path)
    config = ab.load_config(write_config(tmp_path, data))
    remote = FakeRemoteDockerRunner()
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner())

    assert result == 0
    run_dir = tmp_path / "results" / "run123"
    public_path = run_dir / "config.resolved.json"
    resume_path = run_dir / ab.RESUME_CONFIG_FILE
    public_text = public_path.read_text(encoding="utf-8")
    for secret_value in (
        "run-secret",
        "ssh-secret",
        "env-secret",
        "router-secret",
        "node-token-secret",
        "node-password-secret",
    ):
        assert secret_value not in public_text
    public_config = json.loads(public_text)
    topology = public_config["topology_profiles"][0]
    assert public_config["run"]["api_key"] == "***"
    assert topology["hosts"]["p1"]["auth"]["password"] == "***"
    assert topology["env"]["OPENAI_API_KEY"] == "***"
    assert topology["env"]["VISIBLE_ENV"] == "visible"
    assert topology["frontend"]["args"] == ["--api-key", "***"]
    assert topology["prefill"][0]["env"]["SERVICE_TOKEN"] == "***"
    assert topology["prefill"][0]["args"] == ["--db-password", "***"]

    assert resume_path.exists()
    assert stat.S_IMODE(resume_path.stat().st_mode) == 0o600
    resume_text = resume_path.read_text(encoding="utf-8")
    for secret_value in (
        "run-secret",
        "ssh-secret",
        "env-secret",
        "router-secret",
        "node-token-secret",
        "node-password-secret",
    ):
        assert secret_value in resume_text
    context = ab.load_resume_context(tmp_path / "results", "run123")
    assert context.config.run.api_key == "run-secret"
    assert (
        context.config.topology_profiles[0].hosts["p1"].auth.password
        == "ssh-secret"
    )


def test_resource_monitor_defaults_enabled(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))

    assert config.run.resource_monitor.enabled is True
    assert config.run.resource_monitor.backend == "nvidia-smi"
    assert config.run.resource_monitor.interval_sec == 1.0


def test_resource_monitor_can_be_disabled(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["resource_monitor"] = {"enabled": False}

    config = ab.load_config(write_config(tmp_path, data))

    assert config.run.resource_monitor.enabled is False
    assert config.run.resource_monitor.backend == "nvidia-smi"


def test_resource_monitor_rejects_unsupported_backend(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["resource_monitor"] = {
        "enabled": True,
        "backend": "dcmi",
        "interval_sec": 1.0,
    }

    with pytest.raises(ab.ConfigError, match="resource_monitor.backend"):
        ab.load_config(write_config(tmp_path, data))


def test_resource_monitor_interval_must_be_positive(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["resource_monitor"] = {
        "enabled": True,
        "backend": "nvidia-smi",
        "interval_sec": 0,
    }

    with pytest.raises(ab.ConfigError, match="resource_monitor.interval_sec"):
        ab.load_config(write_config(tmp_path, data))


def test_config_to_dict_includes_resource_monitor(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["resource_monitor"] = {
        "enabled": True,
        "backend": "nvidia-smi",
        "interval_sec": 2.5,
    }
    config = ab.load_config(write_config(tmp_path, data))

    payload = ab.config_to_dict(config)

    assert payload["run"]["resource_monitor"] == {
        "enabled": True,
        "backend": "nvidia-smi",
        "interval_sec": 2.5,
    }


def test_make_run_id_is_unique_within_same_second():
    first = ab.make_run_id("smoke", now=1782849600.0)
    second = ab.make_run_id("smoke", now=1782849600.0)

    assert first != second
    assert ab.SAFE_NAME_RE.fullmatch(first)
    assert ab.SAFE_NAME_RE.fullmatch(second)


def test_invalid_name_is_rejected(tmp_path):
    data = minimal_config(tmp_path)
    data["models"][0]["name"] = "bad/name"

    with pytest.raises(ab.ConfigError, match="safe filename"):
        ab.load_config(write_config(tmp_path, data))


def test_host_network_is_rejected(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["network"] = "host"

    with pytest.raises(ab.ConfigError, match="bridge|host|network"):
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
    assert value_after(cmd, "--entrypoint") == "vllm"
    image_index = cmd.index(config.run.vllm_image)
    assert "serve" in cmd[image_index + 1:]
    serve_index = cmd.index("serve")
    assert cmd[serve_index + 1] == "/models/Qwen2.5-1.5B-Instruct"
    assert value_after(cmd, "--served-model-name") == case.api_model_name
    assert value_after(cmd, "--host") == "0.0.0.0"
    assert value_after(cmd, "--port") == "8000"
    assert value_after(cmd, "--api-key") == "local-bench-key"
    assert value_after(cmd, "--dtype") == "bfloat16"


def test_save_vllm_artifacts_masks_api_key_in_serve_command(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["api_key"] = "legacy-secret"
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)

    ab.save_vllm_artifacts(config, FakeRunner(), case, layout)

    serve_command = (layout.serve_dir / "serve_command.txt").read_text(
        encoding="utf-8"
    )
    assert "legacy-secret" not in serve_command
    assert "***" in serve_command


def test_save_vllm_artifacts_redacts_secrets_in_docker_inspect(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["api_key"] = "legacy-api-secret"
    data["serve_profiles"][0]["args"] = [
        "--dtype",
        "bfloat16",
        "--db-password",
        "db-secret",
        "--service-token=token-secret",
    ]
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)

    class InspectSecretRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True,
                stdout=None, stderr=None):
            if args[:2] == ["docker", "inspect"]:
                payload = [{
                    "Config": {
                        "Cmd": [
                            "vllm",
                            "serve",
                            "--api-key",
                            "legacy-api-secret",
                            "--db-password",
                            "db-secret",
                            "--service-token=token-secret",
                            "--visible-flag",
                            "ordinary-value",
                        ],
                        "Args": ["--service-token=token-secret"],
                    },
                    "Plain": "ordinary-value",
                    "Raw": "legacy-api-secret db-secret token-secret",
                }]
                return ab.Completed(list(args), 0, json.dumps(payload), "")
            return super().run(
                args,
                check=check,
                capture=capture,
                text=text,
                stdout=stdout,
                stderr=stderr,
            )

    ab.save_vllm_artifacts(config, InspectSecretRunner(), case, layout)

    inspect_text = (layout.serve_dir / "docker.inspect.json").read_text(
        encoding="utf-8"
    )
    serve_command = (layout.serve_dir / "serve_command.txt").read_text(
        encoding="utf-8"
    )
    assert "legacy-api-secret" not in inspect_text
    assert "db-secret" not in inspect_text
    assert "token-secret" not in inspect_text
    assert "***" in inspect_text
    assert "ordinary-value" in inspect_text
    assert "legacy-api-secret" not in serve_command
    assert "db-secret" not in serve_command
    assert "token-secret" not in serve_command
    assert "***" in serve_command


def test_build_vllm_command_includes_ownership_labels(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    run_dir = tmp_path / "results" / "run123"

    cmd = ab.build_vllm_run_command(config, case, run_dir)

    labels = values_after(cmd, "--label")
    assert "vllm_auto_bench.managed=true" in labels
    assert "vllm_auto_bench.run_id=run123" in labels
    assert f"vllm_auto_bench.run_dir={run_dir.resolve()}" in labels
    assert "vllm_auto_bench.model=qwen2_5_1_5b" in labels
    assert "vllm_auto_bench.serve_profile=bf16_default" in labels


def test_build_bench_command_targets_container_dns(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    bench_dir = Path("relative-results") / "run123" / "qwen2_5_1_5b" / "bf16_default" / "smoke"

    cmd = ab.build_bench_run_command(config, case, bench_dir)

    mounts = [cmd[index + 1] for index, value in enumerate(cmd) if value == "-v"]
    assert "--network" in cmd
    assert "vllm-bench-net" in cmd
    assert "--user" in cmd
    assert value_after(cmd, "--user") == f"{os.getuid()}:{os.getgid()}"
    assert f"{bench_dir.resolve()}:/results" in mounts
    assert "--base-url" in cmd
    assert f"http://{case.container_name}:8000/v1" in cmd
    assert "--model" in cmd
    assert "qwen2_5_1_5b" in cmd
    assert value_after(cmd, "--model") == "qwen2_5_1_5b"
    assert value_after(cmd, "--tokenizer") == "/models/Qwen2.5-1.5B-Instruct"
    assert value_after(cmd, "--input-lens") == "64"
    assert "--prefix-ratio" not in cmd
    assert "--output-csv" in cmd
    assert "/results/result.csv" in cmd
    assert value_after(cmd, "--served-model-name") == "qwen2_5_1_5b"
    assert value_after(cmd, "--output-xlsx") == "/results/result.xlsx"


def test_build_bench_command_passes_builtin_dataset(tmp_path):
    data = minimal_config(tmp_path)
    data["bench_profiles"][0]["dataset"] = {
        "name": "builtin_mtp_chat",
        "length_policy": "bucket",
        "input_len_tolerance": 0.2,
        "on_bucket_shortage": "error",
        "sampling": "shuffle",
    }
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    bench_dir = Path("relative-results") / "run123" / "qwen2_5_1_5b" / "bf16_default" / "smoke"

    cmd = ab.build_bench_run_command(config, case, bench_dir)

    assert value_after(cmd, "--dataset") == "builtin_mtp_chat"
    assert value_after(cmd, "--dataset-length-policy") == "bucket"
    assert value_after(cmd, "--dataset-input-len-tolerance") == "0.2"
    assert value_after(cmd, "--dataset-on-bucket-shortage") == "error"
    assert value_after(cmd, "--dataset-sampling") == "shuffle"


def test_build_bench_command_omits_dataset_for_legacy_config(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    bench_dir = Path("relative-results") / "run123" / "qwen2_5_1_5b" / "bf16_default" / "smoke"

    cmd = ab.build_bench_run_command(config, case, bench_dir)

    assert "--dataset" not in cmd


def test_serve_profile_rejects_speculative_config_without_dashes(tmp_path):
    data = minimal_config(tmp_path)
    data["serve_profiles"][0]["args"] = [
        "speculative-config.num_speculative_tokens",
        "1",
    ]

    with pytest.raises(ab.ConfigError, match="--speculative-config"):
        ab.load_config(write_config(tmp_path, data))


def test_asr_profile_defaults_to_builtin_dataset_path(tmp_path):
    config = ab.load_config(write_config(tmp_path, asr_config(tmp_path)))
    bench = config.bench_profiles[0]
    assert bench.backend == "openai-audio"
    assert bench.input_lens == (0,)
    assert bench.prefix_ratio == 0.0
    assert bench.dataset_name == "custom_audio"
    assert bench.dataset_path == ab.BUILTIN_ASR_DATASET_PATH
    assert bench.language == "en"


def test_build_bench_command_passes_asr_dataset_args(tmp_path):
    config = ab.load_config(write_config(tmp_path, asr_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    bench_dir = (
        tmp_path
        / "results"
        / "run123"
        / "qwen3_asr_1_7b"
        / "bf16_default"
        / "asr_smoke"
    )
    cmd = ab.build_bench_run_command(config, case, bench_dir)
    assert value_after(cmd, "--backend") == "openai-audio"
    assert value_after(cmd, "--dataset-name") == "custom_audio"
    assert value_after(cmd, "--dataset-path") == ab.BUILTIN_ASR_DATASET_PATH
    assert value_after(cmd, "--language") == "en"
    assert "--input-lens" not in cmd
    assert "--prefix-ratio" not in cmd
    assert "--tokenizer" not in cmd
    assert value_after(cmd, "--output-lens") == "128"


def test_external_asr_dataset_requires_datasets_mount(tmp_path):
    data = asr_config(tmp_path)
    data["bench_profiles"][0]["dataset_path"] = "/datasets/asr/custom.jsonl"
    with pytest.raises(ab.ConfigError, match="mounts.datasets"):
        ab.load_config(write_config(tmp_path, data))


def test_external_asr_dataset_must_be_under_datasets_mount(tmp_path):
    data = asr_config(tmp_path)
    data["bench_profiles"][0]["dataset_path"] = "/tmp/asr.jsonl"

    with pytest.raises(ab.ConfigError, match="/datasets|dataset_path"):
        ab.load_config(write_config(tmp_path, data))


def test_external_asr_dataset_mount_is_added(tmp_path):
    data = asr_config(tmp_path)
    host_datasets = tmp_path / "datasets"
    host_dataset_file = host_datasets / "asr" / "custom.jsonl"
    host_dataset_file.parent.mkdir(parents=True)
    host_dataset_file.write_text("{}", encoding="utf-8")
    data["mounts"]["datasets"] = str(host_datasets)
    data["bench_profiles"][0]["dataset_path"] = "/datasets/asr/custom.jsonl"
    config = ab.load_config(write_config(tmp_path, data))
    ab.validate_local_paths(config)
    case = ab.expand_cases(config, run_id="run123")[0]
    cmd = ab.build_bench_run_command(config, case, tmp_path / "bench")
    mounts = [cmd[index + 1] for index, value in enumerate(cmd) if value == "-v"]
    assert f"{host_datasets.resolve()}:/datasets:ro" in mounts


def test_validate_local_paths_rejects_missing_external_asr_dataset_file(tmp_path):
    data = asr_config(tmp_path)
    host_datasets = tmp_path / "datasets"
    host_datasets.mkdir()
    data["mounts"]["datasets"] = str(host_datasets)
    data["bench_profiles"][0]["dataset_path"] = "/datasets/asr/missing.jsonl"
    config = ab.load_config(write_config(tmp_path, data))

    with pytest.raises(ab.ConfigError, match="dataset"):
        ab.validate_local_paths(config)


@pytest.mark.parametrize(
    "dataset_path",
    [
        "datasets/asr/custom.jsonl",
        "/datasets/../custom.jsonl",
        "//datasets/asr/custom.jsonl",
    ],
)
def test_asr_dataset_path_must_be_absolute_and_not_escape(tmp_path, dataset_path):
    data = asr_config(tmp_path)
    data["bench_profiles"][0]["dataset_path"] = dataset_path

    with pytest.raises(ab.ConfigError, match="dataset_path"):
        ab.load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dataset_name", 123, "dataset_name"),
        ("language", "", "language"),
        ("dataset_path", 123, "dataset_path"),
        ("dataset_path", "", "dataset_path"),
    ],
)
def test_asr_dataset_fields_must_be_non_empty_strings(tmp_path, field, value, match):
    data = asr_config(tmp_path)
    data["bench_profiles"][0][field] = value

    with pytest.raises(ab.ConfigError, match=match):
        ab.load_config(write_config(tmp_path, data))


def test_asr_output_lens_can_have_multiple_values_without_cross_product(tmp_path):
    data = asr_config(tmp_path)
    data["bench_profiles"][0]["output_lens"] = [64, 128]
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_bench_run_command(config, case, tmp_path / "bench")

    output_index = cmd.index("--output-lens")
    assert cmd[output_index + 1:output_index + 3] == ["64", "128"]
    assert "--cross-product" not in cmd


def test_build_bench_command_includes_name_and_ownership_labels(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    bench_dir = tmp_path / "results" / "run123" / "qwen2_5_1_5b" / "bf16_default" / "smoke"

    cmd = ab.build_bench_run_command(config, case, bench_dir)

    assert value_after(cmd, "--name") == "bench-runner-qwen2_5_1_5b-bf16_default-smoke-run123"
    labels = values_after(cmd, "--label")
    assert "vllm_auto_bench.managed=true" in labels
    assert "vllm_auto_bench.run_id=run123" in labels
    assert f"vllm_auto_bench.run_dir={(tmp_path / 'results' / 'run123').resolve()}" in labels
    assert "vllm_auto_bench.model=qwen2_5_1_5b" in labels
    assert "vllm_auto_bench.serve_profile=bf16_default" in labels
    assert "vllm_auto_bench.bench_profile=smoke" in labels


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


def test_case_paths_and_state_files_are_written(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)

    assert layout.run_dir == tmp_path / "results" / "run123"
    assert layout.serve_dir == layout.run_dir / "qwen2_5_1_5b" / "bf16_default"
    assert layout.bench_dir == layout.serve_dir / "smoke"

    ab.write_state(layout.run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": {"model": "qwen2_5_1_5b", "serve_profile": "bf16_default", "bench_profile": "smoke"},
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1}
    })
    state = json.loads((layout.run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "running"


def test_write_json_atomic_uses_unique_tmp_for_interleaved_writes(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    original_replace = Path.replace
    triggered = False

    def interleaved_replace(self, target):
        nonlocal triggered
        if Path(target) == path and not triggered:
            triggered = True
            ab.write_json_atomic(path, {"writer": "nested"})
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", interleaved_replace)

    ab.write_json_atomic(path, {"writer": "outer"})

    assert triggered is True
    assert json.loads(path.read_text(encoding="utf-8"))["writer"] == "outer"


def test_manifest_records_relative_artifact_paths(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    manifest = ab.Manifest(run_id="run123", total=1)

    manifest.record(case, layout, "passed")
    ab.write_manifest(layout.run_dir, manifest)

    data = json.loads((layout.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["run_id"] == "run123"
    assert data["status"] == "completed"
    assert data["cases"][0]["csv"] == "qwen2_5_1_5b/bf16_default/smoke/result.csv"
    assert data["cases"][0]["serve_profile"] == "bf16_default"
    assert data["cases"][0]["topology_profile"] is None


def test_topology_manifest_records_null_serve_profile(tmp_path, monkeypatch):
    from test_remote_topology import pd_topology_config, write_config as write_topology_config

    config = ab.load_config(write_topology_config(tmp_path, pd_topology_config(tmp_path)))
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: FakeRemoteDockerRunner())
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *args, **kwargs: True)

    ab.run_controller(config, run_id="run123", runner=FakeRunner())

    manifest = json.loads(
        (tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8")
    )
    row = manifest["cases"][0]
    assert row["serve_profile"] is None
    assert row["topology_profile"] == "sglang_pd_2p2d"


def test_manifest_interrupted_wins_over_incomplete_cases(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    manifest = ab.Manifest(run_id="run123", total=2)

    manifest.record(case, layout, "interrupted", error="stopped")

    assert manifest.status() == "interrupted"
    data = manifest.to_dict()
    assert data["status"] == "interrupted"
    assert data["cases"][0]["error"] == "stopped"
    assert data["cases"][0]["xlsx"] == "qwen2_5_1_5b/bf16_default/smoke/result.xlsx"


def test_manifest_status_matrix(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)

    running = ab.Manifest(run_id="run123", total=2)
    running.record(case, layout, "passed")
    assert running.status() == "running"

    completed = ab.Manifest(run_id="run123", total=1)
    completed.record(case, layout, "passed")
    assert completed.status() == "completed"

    failed = ab.Manifest(run_id="run123", total=1)
    failed.record(case, layout, "failed", error="boom")
    assert failed.status() == "completed_with_failures"


def two_bench_config(tmp_path):
    data = minimal_config(tmp_path)
    second_profile = dict(data["bench_profiles"][0])
    second_profile["name"] = "smoke2"
    data["bench_profiles"].append(second_profile)
    return data


def write_resolved_config_for_resume(tmp_path, data=None):
    data = data or two_bench_config(tmp_path)
    config = ab.load_config(write_config(tmp_path, data))
    run_dir = tmp_path / "results" / "run123"
    ab.write_json_atomic(run_dir / "config.resolved.json", ab.config_to_dict(config))
    return config, run_dir


def test_plan_resume_cases_keeps_passed_and_selects_pending(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    first_layout = ab.build_layout(config, "run123", cases[0])
    old_manifest = {
        "run_id": "run123",
        "status": "interrupted",
        "cases": [
            {
                "model": cases[0].model.name,
                "serve_profile": cases[0].serve_profile.name,
                "bench_profile": cases[0].bench_profile.name,
                "status": "passed",
                "csv": str((first_layout.bench_dir / "result.csv").relative_to(first_layout.run_dir)),
                "xlsx": str((first_layout.bench_dir / "result.xlsx").relative_to(first_layout.run_dir)),
                "extra": {"source": "old"},
            },
            {
                "model": cases[1].model.name,
                "serve_profile": cases[1].serve_profile.name,
                "bench_profile": cases[1].bench_profile.name,
                "status": "interrupted",
                "csv": "old.csv",
                "xlsx": "old.xlsx",
            },
        ],
    }

    initial_manifest, pending, unknown = ab.plan_resume_cases(
        run_id="run123",
        cases=cases,
        manifest_data=old_manifest,
    )

    assert unknown == []
    assert [row["bench_profile"] for row in initial_manifest.cases] == ["smoke"]
    assert initial_manifest.cases[0]["topology_profile"] is None
    assert initial_manifest.cases[0]["status"] == "passed"
    assert initial_manifest.cases[0]["extra"] == {"source": "old"}
    assert [case.bench_profile.name for case in pending] == ["smoke2"]
    assert initial_manifest.total == 2


def test_plan_resume_cases_supports_topology_key(tmp_path):
    from test_remote_topology import pd_topology_config, write_config as write_topology_config

    config = ab.load_config(write_topology_config(tmp_path, pd_topology_config(tmp_path)))
    cases = ab.expand_cases(config, run_id="run123")
    manifest_data = {
        "run_id": "run123",
        "cases": [{
            "model": "qwen2_5_1_5b",
            "serve_profile": None,
            "topology_profile": "sglang_pd_2p2d",
            "bench_profile": "smoke",
            "status": "passed",
        }],
    }

    initial, pending, unknown = ab.plan_resume_cases(
        run_id="run123",
        cases=cases,
        manifest_data=manifest_data,
    )

    assert len(initial.cases) == 1
    assert pending == ()
    assert unknown == []


def test_plan_resume_cases_migrates_legacy_topology_row(tmp_path):
    from test_remote_topology import pd_topology_config, write_config as write_topology_config

    config = ab.load_config(write_topology_config(tmp_path, pd_topology_config(tmp_path)))
    cases = ab.expand_cases(config, run_id="run123")
    manifest_data = {
        "run_id": "run123",
        "cases": [{
            "model": "qwen2_5_1_5b",
            "serve_profile": "sglang_pd_2p2d",
            "bench_profile": "smoke",
            "status": "passed",
            "extra": {"source": "legacy-topology"},
        }],
    }

    initial, pending, unknown = ab.plan_resume_cases(
        run_id="run123",
        cases=cases,
        manifest_data=manifest_data,
    )

    assert pending == ()
    assert unknown == []
    assert len(initial.cases) == 1
    row = initial.cases[0]
    assert row["serve_profile"] is None
    assert row["topology_profile"] == "sglang_pd_2p2d"
    assert row["extra"] == {"source": "legacy-topology"}


def test_plan_resume_cases_reruns_failed_skipped_and_missing(tmp_path):
    data = two_bench_config(tmp_path)
    third_profile = dict(data["bench_profiles"][0])
    third_profile["name"] = "smoke3"
    data["bench_profiles"].append(third_profile)
    config, run_dir = write_resolved_config_for_resume(tmp_path, data)
    cases = ab.expand_cases(config, run_id="run123")
    manifest_data = {
        "run_id": "run123",
        "status": "completed_with_failures",
        "cases": [
            {
                "model": cases[0].model.name,
                "serve_profile": cases[0].serve_profile.name,
                "bench_profile": cases[0].bench_profile.name,
                "status": "failed",
                "csv": "old.csv",
                "xlsx": "old.xlsx",
            },
            {
                "model": cases[1].model.name,
                "serve_profile": cases[1].serve_profile.name,
                "bench_profile": cases[1].bench_profile.name,
                "status": "skipped",
                "csv": "old.csv",
                "xlsx": "old.xlsx",
            }
        ],
    }

    initial_manifest, pending, unknown = ab.plan_resume_cases(
        run_id="run123",
        cases=cases,
        manifest_data=manifest_data,
    )

    assert unknown == []
    assert initial_manifest.cases == []
    assert [case.bench_profile.name for case in pending] == ["smoke", "smoke2", "smoke3"]


@pytest.mark.parametrize(("manifest_data", "match"), [
    ({"run_id": "run123", "cases": "bad"}, "manifest cases must be a list"),
    ({"run_id": "run123", "cases": ["bad"]}, "manifest cases must contain objects"),
    (
        {
            "run_id": "run123",
            "cases": [{
                "model": "qwen2_5_1_5b",
                "serve_profile": "bf16_default",
                "bench_profile": None,
                "status": "passed",
            }],
        },
        "manifest case row must include model, bench_profile, and exactly one of serve_profile/topology_profile",
    ),
    (
        {
            "run_id": "run123",
            "cases": [{
                "model": "qwen2_5_1_5b",
                "serve_profile": "bf16_default",
                "topology_profile": "sglang_pd_2p2d",
                "bench_profile": "smoke",
                "status": "passed",
            }],
        },
        "manifest case row must include model, bench_profile, and exactly one of serve_profile/topology_profile",
    ),
    ({"run_id": "other", "cases": []}, "manifest run_id mismatch"),
])
def test_plan_resume_cases_rejects_invalid_manifest(tmp_path, manifest_data, match):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")

    with pytest.raises(ab.ConfigError, match=match):
        ab.plan_resume_cases(
            run_id="run123",
            cases=cases,
            manifest_data=manifest_data,
        )


def test_plan_resume_cases_reports_manifest_rows_not_in_config(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    manifest_data = {
        "run_id": "run123",
        "status": "interrupted",
        "cases": [
            {
                "model": "old_model",
                "serve_profile": "old_serve",
                "bench_profile": "old_bench",
                "status": "passed",
                "csv": "old.csv",
                "xlsx": "old.xlsx",
            }
        ],
    }

    initial_manifest, pending, unknown = ab.plan_resume_cases(
        run_id="run123",
        cases=cases,
        manifest_data=manifest_data,
    )

    assert initial_manifest.cases == []
    assert [case.bench_profile.name for case in pending] == ["smoke", "smoke2"]
    assert unknown == [("old_model", "old_serve", None, "old_bench")]


def test_plan_resume_cases_reports_unknown_topology_rows(tmp_path):
    from test_remote_topology import pd_topology_config, write_config as write_topology_config

    config = ab.load_config(write_topology_config(tmp_path, pd_topology_config(tmp_path)))
    cases = ab.expand_cases(config, run_id="run123")
    manifest_data = {
        "run_id": "run123",
        "status": "interrupted",
        "cases": [
            {
                "model": "qwen2_5_1_5b",
                "serve_profile": None,
                "topology_profile": "old_topology",
                "bench_profile": "smoke",
                "status": "passed",
            }
        ],
    }

    initial_manifest, pending, unknown = ab.plan_resume_cases(
        run_id="run123",
        cases=cases,
        manifest_data=manifest_data,
    )

    assert initial_manifest.cases == []
    assert pending == cases
    assert unknown == [("qwen2_5_1_5b", None, "old_topology", "smoke")]


def write_resume_state(run_dir, status="interrupted"):
    state = {
        "run_id": run_dir.name,
        "status": status,
        "current": None,
        "counts": {"passed": 1, "failed": 0, "skipped": 0, "running": 0, "total": 2},
    }
    if status is None:
        del state["status"]
    ab.write_state(run_dir, state)


def write_resume_manifest(run_dir, cases, config, statuses):
    rows = []
    for case, status in zip(cases, statuses):
        layout = ab.build_layout(config, run_dir.name, case)
        rows.append({
            "model": case.model.name,
            "serve_profile": case.serve_profile.name if case.serve_profile else None,
            "topology_profile": (
                case.topology_profile.name if case.topology_profile else None
            ),
            "bench_profile": case.bench_profile.name,
            "status": status,
            "csv": str((layout.bench_dir / "result.csv").relative_to(layout.run_dir)),
            "xlsx": str((layout.bench_dir / "result.xlsx").relative_to(layout.run_dir)),
        })
    ab.write_json_atomic(run_dir / "manifest.json", {
        "run_id": run_dir.name,
        "status": "interrupted",
        "cases": rows,
    })


def test_load_resume_context_rejects_redacted_inline_password_without_private_config(
    tmp_path,
):
    from test_remote_topology import pd_topology_config

    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["hosts"]["p1"]["auth"] = {
        "type": "password",
        "password": "***",
    }
    config = ab.load_config(write_config(tmp_path, data))
    run_dir = tmp_path / "results" / "run123"
    ab.write_json_atomic(run_dir / "config.resolved.json", ab.config_to_dict(config))
    write_resume_state(run_dir)
    write_resume_manifest(run_dir, [], config, [])

    with pytest.raises(
        ab.ConfigError,
        match="private resume config|inline password.*redacted|password_env|rerun",
    ):
        ab.load_resume_context(tmp_path / "results", "run123")


def test_load_resume_context_uses_cli_results_dir_over_resolved_config(tmp_path):
    original_data = two_bench_config(tmp_path)
    config, original_run_dir = write_resolved_config_for_resume(tmp_path, original_data)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(original_run_dir)
    write_resume_manifest(original_run_dir, cases[:1], config, ["passed"])

    other_results = tmp_path / "other-results"
    moved_run_dir = other_results / "run123"
    moved_run_dir.parent.mkdir()
    original_run_dir.rename(moved_run_dir)

    context = ab.load_resume_context(other_results, "run123")

    assert context.config.run.results_dir == other_results
    assert [case.bench_profile.name for case in context.pending_cases] == ["smoke2"]
    assert [row["bench_profile"] for row in context.initial_manifest.cases] == ["smoke"]


def test_load_resume_context_rejects_active_state(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir, status="running")
    write_resume_manifest(run_dir, cases[:1], config, ["passed"])

    with pytest.raises(ab.ConfigError, match="active|running"):
        ab.load_resume_context(tmp_path / "results", "run123")


def test_load_resume_context_completed_with_no_pending_is_empty(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir, status="completed")
    write_resume_manifest(run_dir, cases, config, ["passed", "passed"])

    context = ab.load_resume_context(tmp_path / "results", "run123")

    assert context.pending_cases == ()
    assert [row["status"] for row in context.initial_manifest.cases] == ["passed", "passed"]


@pytest.mark.parametrize("status", [None, "complete", 123])
def test_load_resume_context_rejects_unknown_state_even_without_pending(tmp_path, status):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir, status=status)
    write_resume_manifest(run_dir, cases, config, ["passed", "passed"])

    with pytest.raises(ab.ConfigError, match="run status cannot be resumed"):
        ab.load_resume_context(tmp_path / "results", "run123")


def test_load_resume_context_rejects_completed_with_pending(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir, status="completed")
    write_resume_manifest(run_dir, cases[:1], config, ["passed"])

    with pytest.raises(ab.ConfigError, match="run status cannot be resumed: completed"):
        ab.load_resume_context(tmp_path / "results", "run123")


def test_load_resume_context_rejects_state_run_id_mismatch(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir)
    write_resume_manifest(run_dir, cases, config, ["passed", "passed"])
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    state["run_id"] = "other"
    ab.write_json_atomic(run_dir / "state.json", state)

    with pytest.raises(ab.ConfigError, match="state run_id mismatch"):
        ab.load_resume_context(tmp_path / "results", "run123")


@pytest.mark.parametrize(("filename", "label"), [
    ("state.json", "state"),
    ("manifest.json", "manifest"),
])
def test_load_resume_context_rejects_non_object_state_or_manifest(tmp_path, filename, label):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir)
    write_resume_manifest(run_dir, cases, config, ["passed", "passed"])
    (run_dir / filename).write_text("[]", encoding="utf-8")

    with pytest.raises(ab.ConfigError, match=f"{label} must be a JSON object"):
        ab.load_resume_context(tmp_path / "results", "run123")


def test_load_resume_context_returns_unknown_manifest_cases(tmp_path):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir)
    write_resume_manifest(run_dir, cases[:1], config, ["passed"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["cases"].append({
        "model": "old_model",
        "serve_profile": "old_serve",
        "bench_profile": "old_bench",
        "status": "passed",
        "csv": "old.csv",
        "xlsx": "old.xlsx",
    })
    ab.write_json_atomic(run_dir / "manifest.json", manifest)

    context = ab.load_resume_context(tmp_path / "results", "run123")

    assert context.unknown_manifest_cases == (("old_model", "old_serve", None, "old_bench"),)


def test_controller_command_matches_resume_child(tmp_path):
    cmd = [
        sys.executable,
        str(Path(ab.__file__).resolve()),
        "resume",
        "--run-id",
        "run123",
        "--child",
        "--results-dir",
        str(tmp_path / "results"),
    ]

    assert ab.controller_command_matches(cmd, "run123", tmp_path / "results") is True


def test_main_resume_foreground_runs_pending_cases(tmp_path, monkeypatch):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir)
    write_resume_manifest(run_dir, cases[:1], config, ["passed"])
    calls = []

    def fake_run_controller(config_arg, run_id, runner=None, dry_run=False, lock_token=None,
                            initial_manifest=None, cases_to_run=None):
        calls.append((run_id, tuple(case.bench_profile.name for case in cases_to_run), len(initial_manifest.cases)))
        return 0

    monkeypatch.setattr(ab, "install_signal_handlers", lambda: calls.append("signals"), raising=False)
    monkeypatch.setattr(ab, "run_controller", fake_run_controller)

    exit_code = ab.main([
        "resume",
        "--results-dir", str(tmp_path / "results"),
        "--run-id", "run123",
    ])

    assert exit_code == 0
    assert calls == ["signals", ("run123", ("smoke2",), 1)]


def test_main_resume_detach_starts_resume_child(tmp_path, monkeypatch):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir)
    write_resume_manifest(run_dir, cases[:1], config, ["passed"])
    popen_calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        popen_calls.append(command)
        return FakeProcess()

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)

    exit_code = ab.main([
        "resume",
        "--results-dir", str(tmp_path / "results"),
        "--run-id", "run123",
        "--detach",
    ])

    controller = json.loads((run_dir / "controller.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert popen_calls[0][2] == "resume"
    assert "--child" in popen_calls[0]
    assert controller["command"][2] == "resume"


def test_main_resume_detach_child_uses_saved_resume_state_after_parent_starting(
    tmp_path,
    monkeypatch,
):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir)
    write_resume_manifest(run_dir, cases[:1], config, ["passed"])
    child_exits = []
    controller_calls = []

    class FakeProcess:
        pid = 12345

    def fake_run_controller(config_arg, run_id, runner=None, dry_run=False, lock_token=None,
                            initial_manifest=None, cases_to_run=None):
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        controller_calls.append((
            run_id,
            tuple(case.bench_profile.name for case in cases_to_run),
            tuple(row["bench_profile"] for row in initial_manifest.cases),
            state["status"],
        ))
        return 0

    def fake_popen(command, **kwargs):
        child_exits.append(ab.main(command[2:]))
        return FakeProcess()

    monkeypatch.setattr(ab, "install_signal_handlers", lambda: None, raising=False)
    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ab, "run_controller", fake_run_controller)

    exit_code = ab.main([
        "resume",
        "--results-dir", str(tmp_path / "results"),
        "--run-id", "run123",
        "--detach",
    ])

    assert exit_code == 0
    assert child_exits == [0]
    assert controller_calls == [("run123", ("smoke2",), ("smoke",), "starting")]


@pytest.mark.parametrize("lock_token", [None, "wrong"])
def test_resume_child_startup_context_requires_matching_lock_token(tmp_path, lock_token):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir)
    write_resume_manifest(run_dir, cases[:1], config, ["passed"])
    original_state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "starting",
        "current": None,
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 0, "total": 2},
    })
    ab.write_json_atomic(run_dir / ".resume-startup-state.json", original_state)
    (run_dir / ".run.lock").write_text(
        json.dumps({"pid": os.getpid(), "token": "tok", "created_at": 1.0}),
        encoding="utf-8",
    )

    with pytest.raises(ab.ConfigError, match="active.*starting"):
        ab.load_resume_child_startup_context(tmp_path / "results", "run123", lock_token)


def test_main_resume_empty_pending_does_not_start_controller(tmp_path, monkeypatch, capsys):
    config, run_dir = write_resolved_config_for_resume(tmp_path)
    cases = ab.expand_cases(config, run_id="run123")
    write_resume_state(run_dir, status="completed")
    write_resume_manifest(run_dir, cases, config, ["passed", "passed"])

    def fail_run_controller(*args, **kwargs):
        raise AssertionError("empty resume should not start controller")

    monkeypatch.setattr(ab, "run_controller", fail_run_controller)

    exit_code = ab.main([
        "resume",
        "--results-dir", str(tmp_path / "results"),
        "--run-id", "run123",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "no pending" in captured.out.lower() or "nothing to resume" in captured.out.lower()


def test_main_resume_child_load_context_failure_releases_lock(tmp_path, monkeypatch, capsys):
    results_dir = tmp_path / "results"
    run_dir = results_dir / "run123"
    run_dir.mkdir(parents=True)
    (run_dir / ".run.lock").write_text(
        json.dumps({"pid": os.getpid(), "token": "tok", "created_at": 1.0}),
        encoding="utf-8",
    )

    def fail_context(*args, **kwargs):
        raise ab.ConfigError("bad resume context")

    monkeypatch.setattr(ab, "install_signal_handlers", lambda: None, raising=False)
    monkeypatch.setattr(ab, "load_resume_context", fail_context)

    exit_code = ab.main([
        "resume",
        "--results-dir", str(results_dir),
        "--run-id", "run123",
        "--child",
        "--lock-token", "tok",
    ])

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert state["status"] == "failed"
    assert "bad resume context" in state["error"]
    assert "bad resume context" in captured.err
    assert not (run_dir / ".run.lock").exists()


class FakeRunner:
    def __init__(self, failures=None):
        self.commands = []
        self.failures = failures or {}
        self.container_labels = {}

    def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
        self.commands.append(list(args))
        key = " ".join(args[:3])
        if key in self.failures:
            return ab.Completed(list(args), self.failures[key], "", "forced failure")
        if args[:3] == ["docker", "run", "-d"]:
            container_name = value_after(args, "--name")
            self.container_labels[container_name] = {
                label.split("=", 1)[0]: label.split("=", 1)[1]
                for label in values_after(args, "--label")
                if "=" in label
            }
            return ab.Completed(list(args), 0, "container-id\n", "")
        if args[:4] == ["docker", "inspect", "--format", "{{json .Config.Labels}}"]:
            labels = self.container_labels.get(args[4])
            if labels is None:
                return ab.Completed(list(args), 1, "", "not found")
            return ab.Completed(list(args), 0, json.dumps(labels) + "\n", "")
        if args[:5] == ["docker", "network", "inspect", "--format", "{{.Driver}}"]:
            return ab.Completed(list(args), 0, "bridge\n", "")
        if args[:4] == ["docker", "network", "inspect", "--format"]:
            return ab.Completed(
                list(args),
                0,
                json.dumps({
                    "vllm_auto_bench.managed": "true",
                    "vllm_auto_bench.run_id": "run123",
                }) + "\n",
                "",
            )
        if args[:3] == ["docker", "network", "inspect"]:
            return ab.Completed(list(args), 1, "", "not found")
        if args[:3] == ["docker", "network", "create"]:
            return ab.Completed(list(args), 0, "network-id\n", "")
        if args[:3] == ["docker", "logs", "--timestamps"]:
            return ab.Completed(list(args), 0, "vllm log\n", "")
        if args[:3] == ["docker", "inspect", "--format"]:
            return ab.Completed(list(args), 0, "[]\n", "")
        if args[:3] == ["docker", "network", "rm"]:
            return ab.Completed(list(args), 0, "", "")
        if args[:2] == ["docker", "stop"]:
            return ab.Completed(list(args), 0, "", "")
        if args[:3] == ["docker", "inspect", "--type=image"]:
            return ab.Completed(list(args), 0, "image\n", "")
        if args[:3] == ["docker", "run", "--rm"] and any(
            "run_bench_multi.py" in str(arg) for arg in args
        ):
            for mount in values_after(args, "-v"):
                host_dir, separator, container_dir = mount.partition(":")
                if separator and container_dir == "/results":
                    result_csv = Path(host_dir) / "result.csv"
                    result_csv.parent.mkdir(parents=True, exist_ok=True)
                    result_csv.write_text(
                        "model,throughput_tok_s\nm,12.5\n",
                        encoding="utf-8-sig",
                    )
                    break
            return ab.Completed(list(args), 0, "ok\n", "")
        return ab.Completed(list(args), 0, "ok\n", "")


class FakeRemoteDockerRunner:
    def __init__(self, failures=None):
        self.commands = []
        self.failures = failures or {}
        self.labels = {}

    def run(self, host, command, *, check=False, capture=True, text=True,
            stdout=None, stderr=None):
        self.commands.append((host.name, list(command)))
        key = (host.name, " ".join(command[:3]))
        if key in self.failures:
            return ab.Completed(list(command), self.failures[key], "", "forced failure")
        if command[:3] == ["docker", "run", "-d"]:
            self.labels[value_after(command, "--name")] = {
                label.split("=", 1)[0]: label.split("=", 1)[1]
                for label in values_after(command, "--label")
                if "=" in label
            }
            return ab.Completed(list(command), 0, "container-id\n", "")
        if command[:4] == ["docker", "inspect", "--format", "{{json .Config.Labels}}"]:
            labels = self.labels.get(command[4])
            if labels is None:
                return ab.Completed(list(command), 1, "", "not found")
            return ab.Completed(list(command), 0, json.dumps(labels) + "\n", "")
        if command[:2] == ["docker", "logs"]:
            return ab.Completed(list(command), 0, f"{host.name} log\n", "")
        if command[:2] == ["docker", "inspect"]:
            return ab.Completed(list(command), 0, "[]\n", "")
        if command[:2] == ["docker", "stop"]:
            return ab.Completed(list(command), 0, "", "")
        if command[:3] == ["docker", "rm", "-f"]:
            return ab.Completed(list(command), 0, "", "")
        return ab.Completed(list(command), 0, "ok\n", "")

    def inspect_labels(self, host, container_name):
        result = self.run(
            host,
            [
                "docker",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                container_name,
            ],
            check=False,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout.strip() or "{}")


class InspectSecretRemoteRunner(FakeRemoteDockerRunner):
    def run(self, host, command, *, check=False, capture=True, text=True,
            stdout=None, stderr=None):
        if command[:2] == ["docker", "inspect"] and command[:3] != [
            "docker",
            "inspect",
            "--format",
        ]:
            self.commands.append((host.name, list(command)))
            payload = [{
                "Config": {
                    "Env": [
                        "OPENAI_API_KEY=remote-env-secret",
                        "DB_PASSWORD=remote-db-secret",
                        "VISIBLE=value",
                    ],
                    "Cmd": [
                        "python",
                        "-m",
                        "server",
                        "--api-key",
                        "router-secret",
                    ],
                    "Args": [
                        "--password",
                        "arg-password",
                    ],
                },
                "Args": ["--token", "arg-token"],
                "Raw": "remote-env-secret router-secret arg-password arg-token",
            }]
            return ab.Completed(list(command), 0, json.dumps(payload), "")
        return super().run(
            host,
            command,
            check=check,
            capture=capture,
            text=text,
            stdout=stdout,
            stderr=stderr,
        )


class LogSecretRemoteRunner(FakeRemoteDockerRunner):
    def run(self, host, command, *, check=False, capture=True, text=True,
            stdout=None, stderr=None):
        if command[:2] == ["docker", "logs"]:
            self.commands.append((host.name, list(command)))
            return ab.Completed(
                list(command),
                0,
                f"{host.name} run-secret router-secret env-secret\n",
                "ssh-secret node-token-secret node-password-secret\n",
            )
        return super().run(
            host,
            command,
            check=check,
            capture=capture,
            text=text,
            stdout=stdout,
            stderr=stderr,
        )


class ArtifactFailingRemoteRunner(FakeRemoteDockerRunner):
    def run(self, host, command, *, check=False, capture=True, text=True,
            stdout=None, stderr=None):
        if command[:2] == ["docker", "logs"]:
            self.commands.append((host.name, list(command)))
            raise RuntimeError("logs unavailable")
        return super().run(
            host,
            command,
            check=check,
            capture=capture,
            text=text,
            stdout=stdout,
            stderr=stderr,
        )


class ForeignLabelRemoteRunner(FakeRemoteDockerRunner):
    def inspect_labels(self, host, container_name):
        self.commands.append((
            host.name,
            [
                "docker",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                container_name,
            ],
        ))
        if container_name not in self.labels:
            return None
        return {
            "vllm_auto_bench.managed": "true",
            "vllm_auto_bench.run_id": "foreign-run",
        }


class ForeignSameNameRemoteRunner(FakeRemoteDockerRunner):
    def __init__(self, container_name):
        super().__init__()
        self.container_name = container_name

    def inspect_labels(self, host, container_name):
        self.commands.append((
            host.name,
            [
                "docker",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                container_name,
            ],
        ))
        if container_name != self.container_name:
            return None
        return {
            "vllm_auto_bench.managed": "true",
            "vllm_auto_bench.run_id": "foreign-run",
            "vllm_auto_bench.role_name": "p1",
        }


class FakeResourceMonitor:
    instances = []

    def __init__(
        self,
        *,
        output_dir,
        interval_sec,
        enabled,
        backend,
        readers=None,
        passthrough_exceptions=(),
    ):
        self.output_dir = Path(output_dir)
        self.interval_sec = interval_sec
        self.enabled = enabled
        self.backend = backend
        self.readers = readers
        self.passthrough_exceptions = tuple(passthrough_exceptions)
        self.started = False
        self.stopped = False
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        summary = {
            "available": True,
            "sample_count": 1,
            "aggregate": {
                "cpu_util_avg_pct": 12.5,
                "gpu_mem_used_max_mb": 1234.0,
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "resource_summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        return summary


class StartFailingResourceMonitor(FakeResourceMonitor):
    def start(self):
        raise RuntimeError("start failed")


class StopFailingResourceMonitor(FakeResourceMonitor):
    def stop(self):
        self.stopped = True
        raise RuntimeError("stop failed")


class StopRequestedResourceMonitor(FakeResourceMonitor):
    def stop(self):
        self.stopped = True
        raise ab.StopRequested("monitor background interrupted")


class MiddleStopRequestedResourceMonitor(FakeResourceMonitor):
    instances = []

    def stop(self):
        self.stopped = True
        if self.output_dir.name == "p2":
            raise ab.StopRequested("p2 monitor background interrupted")
        return super().stop()


class SecondStartInterruptingResourceMonitor(FakeResourceMonitor):
    instances = []

    def start(self):
        self.started = True
        if len(type(self).instances) == 2:
            raise ab.StopRequested("monitor interrupted")


class SecondStartFailsCleanupStopRequestedResourceMonitor(FakeResourceMonitor):
    instances = []

    def start(self):
        self.started = True
        if self.output_dir.name == "p2":
            raise RuntimeError("start failed")

    def stop(self):
        self.stopped = True
        if self.output_dir.name == "p2":
            raise ab.StopRequested("cleanup stop requested")
        return super().stop()


class ReaderStopRequestedRemoteRunner(FakeRemoteDockerRunner):
    def __init__(self, interrupt_host):
        super().__init__()
        self.interrupt_host = interrupt_host
        self.capture_calls = []

    def capture(self, host, command):
        self.capture_calls.append((host.name, list(command)))
        if host.name == self.interrupt_host:
            raise ab.StopRequested("reader interrupted")
        if command == ["cat", "/proc/stat"]:
            return "cpu  100 0 50 850 0 0 0 0 0 0\n"
        if command == ["cat", "/proc/meminfo"]:
            return "MemTotal: 1024000 kB\nMemAvailable: 512000 kB\n"
        if command in (
            ["cat", "/proc/net/dev"],
            ["cat", "/proc/diskstats"],
        ):
            return ""
        if command[:1] == ["nvidia-smi"]:
            return ""
        raise AssertionError(f"unexpected capture command: {command!r}")


def command_index(commands, prefix):
    for index, command in enumerate(commands):
        if command[:len(prefix)] == prefix:
            return index
    raise AssertionError(f"command not found: {prefix!r}")


def assert_removed_after_stop(commands, container_name):
    stop_index = command_index(commands, ["docker", "stop", container_name])
    assert any(
        command[:4] == ["docker", "rm", "-f", container_name]
        for command in commands[stop_index + 1:]
    )


def ready_probe_commands(commands):
    return [
        command for command in commands
        if command[:3] == ["docker", "run", "--rm"]
        and "-c" in command
        and "run_bench_multi.py" not in command
    ]


def bench_run_commands(commands):
    return [
        command for command in commands
        if command[:3] == ["docker", "run", "--rm"]
        and any("run_bench_multi.py" in str(arg) for arg in command)
    ]


def network_create_command(commands):
    for command in commands:
        if command[:3] == ["docker", "network", "create"]:
            return command
    raise AssertionError("docker network create command not found")


def topology_config_with_image(tmp_path):
    from test_remote_topology import pd_topology_config, write_config

    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["image"] = "sglang:pd"
    return ab.load_config(write_config(tmp_path, data))


def topology_role_command(config, role_name, run_id="run123"):
    case = ab.expand_cases(config, run_id=run_id)[0]
    layout = ab.build_layout(config, run_id, case)
    return case.topology_profile.build_commands(config, case, layout.run_dir)[role_name]


def test_topology_run_starts_roles_then_bench_and_cleans_up(tmp_path, monkeypatch):
    from test_remote_topology import pd_topology_config, write_config

    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["image"] = "sglang:pd"
    data["topology_profiles"][0]["hosts"]["p1"]["auth"] = {
        "type": "password",
        "password": "inline-secret",
    }
    config = ab.load_config(write_config(tmp_path, data))
    remote = FakeRemoteDockerRunner()
    local = FakeRunner()
    ready_roles = []
    FakeResourceMonitor.instances = []
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)
    monkeypatch.setattr(
        ab,
        "wait_for_remote_ready",
        lambda _config, _case, role_name: ready_roles.append(role_name) or True,
        raising=False,
    )

    result = ab.run_controller(config, run_id="run123", runner=local)

    assert result == 0
    names = [
        cmd[1][cmd[1].index("--name") + 1]
        for cmd in remote.commands
        if cmd[1][:3] == ["docker", "run", "-d"]
    ]
    assert names[:5] == [
        "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-p1",
        "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-p2",
        "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-d1",
        "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-d2",
        "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-router",
    ]
    assert ready_roles == ["p1", "p2", "d1", "d2", "router"]
    assert len(bench_run_commands(local.commands)) == 1
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    role_commands = case.topology_profile.build_commands(config, case, layout.run_dir)
    monitors_by_role = {
        monitor.output_dir.name: monitor
        for monitor in FakeResourceMonitor.instances
    }
    assert set(monitors_by_role) == {"p1", "p2", "d1", "d2", "router"}
    for role_name, monitor in monitors_by_role.items():
        assert monitor.output_dir == layout.bench_dir / "resources" / role_name
        assert monitor.started is True
        assert monitor.stopped is True
        assert monitor.readers.runner is remote
        assert monitor.readers.host.name == role_commands[role_name].host_name
    with (layout.bench_dir / "result.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["p1_resource_monitor_available"] == "true"
    assert rows[0]["p1_cpu_util_avg_pct"] == "12.5"
    assert rows[0]["router_gpu_mem_used_max_mb"] == "1234.0"
    assert any(
        cmd[1][:2] == ["docker", "stop"] and "router" in cmd[1][2]
        for cmd in remote.commands
    )
    assert (layout.bench_dir / "commands" / "p1.txt").read_text(
        encoding="utf-8"
    ).startswith("docker run -d")
    assert (layout.bench_dir / "logs" / "router.log").read_text(
        encoding="utf-8"
    ) == "router log\n"
    assert (layout.bench_dir / "inspect" / "router.json").read_text(
        encoding="utf-8"
    ) == "[]\n"
    topology_text = (layout.bench_dir / "topology.resolved.json").read_text(
        encoding="utf-8"
    )
    assert "inline-secret" not in topology_text
    resolved = json.loads(topology_text)
    assert resolved["topology_profiles"][0]["hosts"]["p1"]["auth"]["password"] == "***"


def test_topology_remote_resource_monitor_start_failed_does_not_fail_bench(
    tmp_path,
    monkeypatch,
):
    config = topology_config_with_image(tmp_path)
    remote = FakeRemoteDockerRunner()
    local = FakeRunner()
    StartFailingResourceMonitor.instances = []
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)
    monkeypatch.setattr(ab, "ResourceMonitor", StartFailingResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=local)

    assert result == 0
    assert len(bench_run_commands(local.commands)) == 1
    assert len(StartFailingResourceMonitor.instances) == 5
    assert all(monitor.stopped for monitor in StartFailingResourceMonitor.instances)


def test_topology_resource_monitor_start_interrupt_stops_started_monitors(
    tmp_path,
    monkeypatch,
):
    config = topology_config_with_image(tmp_path)
    remote = FakeRemoteDockerRunner()
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    role_commands = case.topology_profile.build_commands(config, case, layout.run_dir)
    started_roles = [role_commands["p1"], role_commands["p2"]]
    SecondStartInterruptingResourceMonitor.instances = []
    monkeypatch.setattr(ab, "ResourceMonitor", SecondStartInterruptingResourceMonitor)

    with pytest.raises(ab.StopRequested, match="monitor interrupted"):
        ab._start_topology_resource_monitors(
            config,
            remote,
            case,
            layout,
            started_roles,
        )

    assert len(SecondStartInterruptingResourceMonitor.instances) == 2
    assert [monitor.started for monitor in SecondStartInterruptingResourceMonitor.instances] == [
        True,
        True,
    ]
    assert [monitor.stopped for monitor in SecondStartInterruptingResourceMonitor.instances] == [
        True,
        True,
    ]


def test_topology_resource_monitor_start_failure_cleanup_stop_requested_stops_started(
    tmp_path,
    monkeypatch,
):
    config = topology_config_with_image(tmp_path)
    remote = FakeRemoteDockerRunner()
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    role_commands = case.topology_profile.build_commands(config, case, layout.run_dir)
    started_roles = [role_commands["p1"], role_commands["p2"]]
    SecondStartFailsCleanupStopRequestedResourceMonitor.instances = []
    monkeypatch.setattr(
        ab,
        "ResourceMonitor",
        SecondStartFailsCleanupStopRequestedResourceMonitor,
    )

    with pytest.raises(ab.StopRequested, match="cleanup stop requested"):
        ab._start_topology_resource_monitors(
            config,
            remote,
            case,
            layout,
            started_roles,
        )

    monitors_by_role = {
        monitor.output_dir.name: monitor
        for monitor in SecondStartFailsCleanupStopRequestedResourceMonitor.instances
    }
    assert set(monitors_by_role) == {"p1", "p2"}
    assert monitors_by_role["p1"].stopped is True
    assert monitors_by_role["p2"].stopped is True


def test_topology_remote_reader_stop_requested_interrupts_start_and_stops_started_monitor(
    tmp_path,
):
    config = topology_config_with_image(tmp_path)
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    role_commands = case.topology_profile.build_commands(config, case, layout.run_dir)
    started_roles = [role_commands["p1"], role_commands["p2"]]
    remote = ReaderStopRequestedRemoteRunner(
        interrupt_host=role_commands["p2"].host_name,
    )

    with pytest.raises(ab.StopRequested, match="reader interrupted"):
        ab._start_topology_resource_monitors(
            config,
            remote,
            case,
            layout,
            started_roles,
        )

    assert (
        role_commands["p2"].host_name,
        ["cat", "/proc/stat"],
    ) in remote.capture_calls
    assert (layout.bench_dir / "resources" / "p1" / "resource_summary.json").is_file()


def test_topology_remote_resource_monitor_stop_failed_does_not_fail_bench(
    tmp_path,
    monkeypatch,
):
    config = topology_config_with_image(tmp_path)
    remote = FakeRemoteDockerRunner()
    local = FakeRunner()
    StopFailingResourceMonitor.instances = []
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)
    monkeypatch.setattr(ab, "ResourceMonitor", StopFailingResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=local)

    assert result == 0
    assert len(bench_run_commands(local.commands)) == 1
    assert len(StopFailingResourceMonitor.instances) == 5
    assert all(monitor.stopped for monitor in StopFailingResourceMonitor.instances)


def test_topology_remote_resource_monitor_stop_requested_interrupts_run(
    tmp_path,
    monkeypatch,
):
    config = topology_config_with_image(tmp_path)
    remote = FakeRemoteDockerRunner()
    local = FakeRunner()
    StopRequestedResourceMonitor.instances = []
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)
    monkeypatch.setattr(ab, "ResourceMonitor", StopRequestedResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=local)

    assert result == 130
    assert len(bench_run_commands(local.commands)) == 1
    assert any(
        cmd[1][:2] == ["docker", "stop"] and "router" in cmd[1][2]
        for cmd in remote.commands
    )
    manifest = json.loads(
        (tmp_path / "results" / "run123" / "manifest.json").read_text(
            encoding="utf-8",
        )
    )
    assert manifest["status"] == "interrupted"
    assert manifest["cases"][0]["status"] == "interrupted"
    assert "monitor background interrupted" in manifest["cases"][0]["error"]


def test_topology_remote_resource_monitor_stop_requested_stops_remaining_monitors(
    tmp_path,
    monkeypatch,
):
    config = topology_config_with_image(tmp_path)
    remote = FakeRemoteDockerRunner()
    local = FakeRunner()
    MiddleStopRequestedResourceMonitor.instances = []
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)
    monkeypatch.setattr(ab, "ResourceMonitor", MiddleStopRequestedResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=local)

    assert result == 130
    assert len(MiddleStopRequestedResourceMonitor.instances) == 5
    monitors_by_role = {
        monitor.output_dir.name: monitor
        for monitor in MiddleStopRequestedResourceMonitor.instances
    }
    assert [role for role, monitor in monitors_by_role.items() if monitor.stopped] == [
        "p1",
        "p2",
        "d1",
        "d2",
        "router",
    ]
    manifest = json.loads(
        (tmp_path / "results" / "run123" / "manifest.json").read_text(
            encoding="utf-8",
        )
    )
    assert manifest["status"] == "interrupted"
    assert manifest["cases"][0]["status"] == "interrupted"
    assert "p2 monitor background interrupted" in manifest["cases"][0]["error"]


def test_topology_prefixed_resource_merge_failed_does_not_fail_bench(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = topology_config_with_image(tmp_path)
    remote = FakeRemoteDockerRunner()
    local = FakeRunner()
    FakeResourceMonitor.instances = []
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)

    def fail_merge(*args, **kwargs):
        raise RuntimeError("prefixed merge failed")

    monkeypatch.setattr(ab, "append_prefixed_summaries_to_result_files", fail_merge)

    result = ab.run_controller(config, run_id="run123", runner=local)

    assert result == 0
    assert len(bench_run_commands(local.commands)) == 1
    assert "remote resource monitor result merge failed" in caplog.text
    assert any(
        cmd[1][:2] == ["docker", "stop"] and "router" in cmd[1][2]
        for cmd in remote.commands
    )
    manifest = json.loads(
        (tmp_path / "results" / "run123" / "manifest.json").read_text(
            encoding="utf-8",
        )
    )
    assert manifest["cases"][0]["status"] == "passed"
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    status = json.loads((layout.bench_dir / "status.json").read_text(encoding="utf-8"))
    assert status == {"status": "passed", "error": None}


def test_topology_owned_stale_role_removed_before_start(tmp_path, monkeypatch):
    config = topology_config_with_image(tmp_path)
    role_command = topology_role_command(config, "p1")
    remote = FakeRemoteDockerRunner()
    remote.labels[role_command.container_name] = labels_from(list(role_command.argv))
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner())

    assert result == 0
    inspect_index = command_index(
        [command for _host, command in remote.commands],
        [
            "docker",
            "inspect",
            "--format",
            "{{json .Config.Labels}}",
            role_command.container_name,
        ],
    )
    rm_index = command_index(
        [command for _host, command in remote.commands],
        ["docker", "rm", "-f", role_command.container_name],
    )
    run_index = command_index(
        [command for _host, command in remote.commands],
        ["docker", "run", "-d"],
    )
    assert inspect_index < rm_index < run_index


def test_topology_foreign_same_name_role_fails_without_removal(tmp_path, monkeypatch):
    config = topology_config_with_image(tmp_path)
    role_command = topology_role_command(config, "p1")
    remote = ForeignSameNameRemoteRunner(role_command.container_name)
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner())

    assert result == 1
    assert not any(
        cmd[1][:2] == ["docker", "stop"] and cmd[1][2] == role_command.container_name
        for cmd in remote.commands
    )
    assert not any(
        cmd[1][:3] == ["docker", "rm", "-f"] and cmd[1][3] == role_command.container_name
        for cmd in remote.commands
    )
    manifest = json.loads(
        (tmp_path / "results" / "run123" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["cases"][0]["status"] == "failed"


def test_topology_run_does_not_append_legacy_local_resource_columns(tmp_path, monkeypatch):
    config = topology_config_with_image(tmp_path)
    remote = FakeRemoteDockerRunner()
    FakeResourceMonitor.instances = []
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner())

    assert result == 0
    assert len(FakeResourceMonitor.instances) == 5
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    with (layout.bench_dir / "result.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert "resource_monitor_available" not in rows[0]
    assert rows[0]["p1_resource_monitor_available"] == "true"


def test_wait_for_remote_ready_uses_api_key_only_for_router(tmp_path, monkeypatch):
    config = topology_config_with_image(tmp_path)
    case = ab.expand_cases(config, run_id="run123")[0]
    probes = []

    def fake_wait_for_ready(url, api_key, timeout_sec):
        probes.append((url, api_key, timeout_sec))
        return True

    monkeypatch.setattr(ab, "wait_for_ready", fake_wait_for_ready)

    assert ab.wait_for_remote_ready(config, case, "p1") is True
    assert ab.wait_for_remote_ready(config, case, "router") is True

    assert probes == [
        ("http://10.0.0.11:30000/v1", None, 30),
        ("http://10.0.0.31:8000/v1", "local-bench-key", 30),
    ]


def test_topology_inspect_artifact_redacts_secrets(tmp_path, monkeypatch):
    from test_remote_topology import pd_topology_config, write_config

    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["image"] = "sglang:pd"
    topology["env"] = {
        "OPENAI_API_KEY": "remote-env-secret",
        "DB_PASSWORD": "remote-db-secret",
    }
    topology["frontend"]["args"] = ["--api-key", "router-secret"]
    topology["prefill"][0]["args"] = ["--password", "arg-password"]
    topology["decode"][0]["args"] = ["--token", "arg-token"]
    config = ab.load_config(write_config(tmp_path, data))
    remote = InspectSecretRemoteRunner()
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner())

    assert result == 0
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    inspect_text = (layout.bench_dir / "inspect" / "router.json").read_text(
        encoding="utf-8"
    )
    assert "remote-env-secret" not in inspect_text
    assert "remote-db-secret" not in inspect_text
    assert "router-secret" not in inspect_text
    assert "arg-password" not in inspect_text
    assert "arg-token" not in inspect_text
    assert "***" in inspect_text


def test_topology_artifacts_role_logs_redact_known_secrets(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, secret_topology_config(tmp_path)))
    remote = LogSecretRemoteRunner()
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner())

    assert result == 0
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    log_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((layout.bench_dir / "logs").glob("*.log"))
    )
    for secret_value in (
        "run-secret",
        "ssh-secret",
        "env-secret",
        "router-secret",
        "node-token-secret",
        "node-password-secret",
    ):
        assert secret_value not in log_text
    assert "***" in log_text


def test_topology_foreign_labels_skip_cleanup(tmp_path, monkeypatch):
    config = topology_config_with_image(tmp_path)
    remote = ForeignLabelRemoteRunner()
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner())

    assert result == 0
    assert not any(cmd[1][:2] == ["docker", "stop"] for cmd in remote.commands)
    assert not any(cmd[1][:3] == ["docker", "rm", "-f"] for cmd in remote.commands)


def test_topology_artifact_failure_still_cleans_up_and_passes(tmp_path, monkeypatch):
    config = topology_config_with_image(tmp_path)
    remote = ArtifactFailingRemoteRunner()
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)
    monkeypatch.setattr(ab, "wait_for_remote_ready", lambda *a, **k: True, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner())

    assert result == 0
    assert any(cmd[1][:2] == ["docker", "stop"] for cmd in remote.commands)
    manifest = json.loads(
        (tmp_path / "results" / "run123" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["cases"][0]["status"] == "passed"
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    assert "failed to save topology artifacts" in (
        layout.bench_dir / "artifact.warning.txt"
    ).read_text(encoding="utf-8")


def test_topology_prefill_start_failure_cleans_started_roles(tmp_path, monkeypatch):
    from test_remote_topology import pd_topology_config, write_config

    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["image"] = "sglang:pd"
    config = ab.load_config(write_config(tmp_path, data))
    remote = FakeRemoteDockerRunner(failures={("p2", "docker run -d"): 1})
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner())

    assert result == 1
    assert any(
        cmd[0] == "p1" and cmd[1][:2] == ["docker", "stop"]
        for cmd in remote.commands
    )
    manifest = json.loads(
        (tmp_path / "results" / "run123" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["cases"][0]["status"] == "failed"


def test_topology_start_failure_writes_start_log_and_status_error(
    tmp_path,
    monkeypatch,
):
    from test_remote_topology import pd_topology_config, write_config

    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["image"] = "sglang:pd"
    config_path = write_config(tmp_path, data)
    config = ab.load_config(config_path)
    remote = FakeRemoteDockerRunner(failures={("p1", "docker run -d"): 125})
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)

    result = ab.run_controller(
        config,
        run_id="run123",
        runner=FakeRunner(),
        config_path=config_path,
    )

    assert result == 1
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    start_log = (layout.bench_dir / "logs" / "p1.start.log").read_text(
        encoding="utf-8",
    )
    assert "returncode: 125" in start_log
    assert "forced failure" in start_log
    status = json.loads((layout.bench_dir / "status.json").read_text(
        encoding="utf-8",
    ))
    assert "forced failure" in status["error"]


def test_controller_runs_case_and_cleans_owned_network(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    assert result == 0
    assert "vllm-bench-net" in network_create_command(runner.commands)
    assert any("docker run -d" in cmd for cmd in joined)
    assert any("run_bench_multi.py" in cmd for cmd in joined)
    assert any("docker stop bench-vllm-qwen2_5_1_5b-bf16_default-run123" in cmd for cmd in joined)
    assert_removed_after_stop(runner.commands, "bench-vllm-qwen2_5_1_5b-bf16_default-run123")
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_run_controller_starts_and_stops_resource_monitor(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    FakeResourceMonitor.instances = []
    summaries = []
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)
    monkeypatch.setattr(
        ab,
        "append_summary_to_result_files",
        lambda output_dir, summary: summaries.append((Path(output_dir), summary)),
    )

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 0
    assert len(FakeResourceMonitor.instances) == 1
    monitor = FakeResourceMonitor.instances[0]
    assert monitor.started is True
    assert monitor.stopped is True
    assert monitor.interval_sec == 1.0
    assert monitor.backend == "nvidia-smi"
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    assert monitor.output_dir == layout.bench_dir
    assert monitor.readers is None
    assert summaries[0][1]["aggregate"]["cpu_util_avg_pct"] == 12.5


def test_run_controller_stops_resource_monitor_when_bench_fails(tmp_path, monkeypatch):
    data = minimal_config(tmp_path)
    config = ab.load_config(write_config(tmp_path, data))
    runner = FakeRunner(failures={"docker run --rm": 7})
    FakeResourceMonitor.instances = []
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)
    monkeypatch.setattr(ab, "append_summary_to_result_files", lambda *args, **kwargs: None)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 1
    assert len(FakeResourceMonitor.instances) == 1
    assert FakeResourceMonitor.instances[0].stopped is True


def test_run_controller_does_not_start_resource_monitor_for_dry_run(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    FakeResourceMonitor.instances = []
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=True)

    assert result == 0
    assert FakeResourceMonitor.instances == []


def test_run_controller_does_not_start_resource_monitor_when_disabled(tmp_path, monkeypatch):
    data = minimal_config(tmp_path)
    data["run"]["resource_monitor"] = {"enabled": False}
    config = ab.load_config(write_config(tmp_path, data))
    FakeResourceMonitor.instances = []
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    assert result == 0
    assert FakeResourceMonitor.instances == []


def test_run_controller_resource_monitor_start_failed_does_not_fail_bench(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    StartFailingResourceMonitor.instances = []
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "ResourceMonitor", StartFailingResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 0
    assert len(bench_run_commands(runner.commands)) == 1
    assert "resource monitor start failed" in caplog.text


def test_run_controller_resource_monitor_stop_failed_does_not_fail_bench(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    StopFailingResourceMonitor.instances = []
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "ResourceMonitor", StopFailingResourceMonitor)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    assert result == 0
    assert "resource monitor stop failed" in caplog.text


def test_run_controller_resource_monitor_result_merge_failed_does_not_fail_bench(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    FakeResourceMonitor.instances = []
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "ResourceMonitor", FakeResourceMonitor)

    def fail_merge(*args, **kwargs):
        raise RuntimeError("merge failed")

    monkeypatch.setattr(ab, "append_summary_to_result_files", fail_merge)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    assert result == 0
    assert "resource monitor result merge failed" in caplog.text


def test_network_create_command_has_ownership_labels(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    create = network_create_command(runner.commands)
    labels = [create[index + 1] for index, value in enumerate(create) if value == "--label"]
    assert result == 0
    assert "vllm_auto_bench.managed=true" in labels
    assert "vllm_auto_bench.run_id=run123" in labels


def test_controller_rejects_existing_non_bridge_network(tmp_path):
    class HostNetworkRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            if args[:3] == ["docker", "network", "inspect"] and len(args) == 4:
                self.commands.append(list(args))
                return ab.Completed(list(args), 0, "network\n", "")
            if args[:5] == ["docker", "network", "inspect", "--format", "{{.Driver}}"]:
                self.commands.append(list(args))
                return ab.Completed(list(args), 0, "host\n", "")
            return super().run(
                args,
                check=check,
                capture=capture,
                text=text,
                stdout=stdout,
                stderr=stderr,
            )

    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = HostNetworkRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 1
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_controller_does_not_remove_network_with_other_run_label(tmp_path):
    class OtherRunLabelRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            if args[:4] == ["docker", "network", "inspect", "--format"]:
                self.commands.append(list(args))
                return ab.Completed(
                    list(args),
                    0,
                    json.dumps({
                        "vllm_auto_bench.managed": "true",
                        "vllm_auto_bench.run_id": "other",
                    }) + "\n",
                    "",
                )
            return super().run(args, check=check, capture=capture, text=text, stdout=stdout, stderr=stderr)

    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = OtherRunLabelRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    assert result == 0
    assert any("docker network create" in cmd for cmd in joined)
    assert not any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_controller_rejects_existing_vllm_container_with_foreign_labels(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    runner = FakeRunner()
    runner.container_labels[case.container_name] = {
        "vllm_auto_bench.managed": "true",
        "vllm_auto_bench.run_id": "run123",
        "vllm_auto_bench.run_dir": str((tmp_path / "other-results" / "run123").resolve()),
    }

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert not any(command[:4] == ["docker", "rm", "-f", case.container_name] for command in runner.commands)
    assert not any(command[:3] == ["docker", "stop", case.container_name] for command in runner.commands)
    assert manifest["cases"][0]["status"] == "skipped"
    assert "container" in manifest["cases"][0]["error"]
    assert state["status"] == "completed_with_failures"


def test_controller_skips_bench_when_vllm_not_ready(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: False)
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: False, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 1
    joined = [" ".join(cmd) for cmd in runner.commands]
    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cases"][0]["status"] == "skipped"
    assert any("docker stop bench-vllm-qwen2_5_1_5b-bf16_default-run123" in cmd for cmd in joined)
    assert_removed_after_stop(runner.commands, "bench-vllm-qwen2_5_1_5b-bf16_default-run123")
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_controller_ready_exception_stops_and_removes_container(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()

    def raise_ready(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ab, "wait_for_ready", raise_ready)
    monkeypatch.setattr(ab, "wait_for_container_ready", raise_ready, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    assert result == 1
    assert any("docker stop bench-vllm-qwen2_5_1_5b-bf16_default-run123" in cmd for cmd in joined)
    assert_removed_after_stop(runner.commands, "bench-vllm-qwen2_5_1_5b-bf16_default-run123")
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_controller_stop_requested_writes_interrupted_and_cleans(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()

    def stop_ready(*args, **kwargs):
        raise ab.StopRequested("stopped")

    monkeypatch.setattr(ab, "wait_for_container_ready", stop_ready, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert result != 0
    assert manifest["status"] == "interrupted"
    assert manifest["cases"][0]["status"] == "interrupted"
    assert state["status"] == "interrupted"
    assert any("docker stop bench-vllm-qwen2_5_1_5b-bf16_default-run123" in cmd for cmd in joined)
    assert_removed_after_stop(runner.commands, "bench-vllm-qwen2_5_1_5b-bf16_default-run123")
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_run_controller_with_initial_manifest_skips_passed_case(tmp_path, monkeypatch):
    data = two_bench_config(tmp_path)
    config = ab.load_config(write_config(tmp_path, data))
    all_cases = ab.expand_cases(config, run_id="run123")
    first_layout = ab.build_layout(config, "run123", all_cases[0])
    initial = ab.Manifest(run_id="run123", total=2)
    initial.record(all_cases[0], first_layout, "passed")
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True, raising=False)

    result = ab.run_controller(
        config,
        run_id="run123",
        runner=runner,
        initial_manifest=initial,
        cases_to_run=(all_cases[1],),
    )

    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    bench_commands = bench_run_commands(runner.commands)
    assert result == 0
    assert [row["bench_profile"] for row in manifest["cases"]] == ["smoke", "smoke2"]
    assert [row["status"] for row in manifest["cases"]] == ["passed", "passed"]
    assert len(bench_commands) == 1
    assert "smoke2" in " ".join(bench_commands[0])
    assert "smoke-run123" not in " ".join(bench_commands[0])


def test_run_controller_pending_empty_writes_finished_state_without_docker(tmp_path):
    config = ab.load_config(write_config(tmp_path, two_bench_config(tmp_path)))
    all_cases = ab.expand_cases(config, run_id="run123")
    initial = ab.Manifest(run_id="run123", total=2)
    for case in all_cases:
        initial.record(case, ab.build_layout(config, "run123", case), "passed")
    runner = FakeRunner()

    result = ab.run_controller(
        config,
        run_id="run123",
        runner=runner,
        initial_manifest=initial,
        cases_to_run=(),
    )

    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert result == 0
    assert state["status"] == "completed"
    assert state["counts"]["passed"] == 2
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_run_controller_resume_stop_preserves_old_passed_row(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, two_bench_config(tmp_path)))
    all_cases = ab.expand_cases(config, run_id="run123")
    initial = ab.Manifest(run_id="run123", total=2)
    initial.record(all_cases[0], ab.build_layout(config, "run123", all_cases[0]), "passed")

    def stop_ready(*args, **kwargs):
        raise ab.StopRequested("resume stopped")

    monkeypatch.setattr(ab, "wait_for_container_ready", stop_ready, raising=False)
    result = ab.run_controller(
        config,
        run_id="run123",
        runner=FakeRunner(),
        initial_manifest=initial,
        cases_to_run=(all_cases[1],),
    )

    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    assert result == 130
    assert manifest["status"] == "interrupted"
    assert [row["bench_profile"] for row in manifest["cases"]] == ["smoke", "smoke2"]
    assert [row["status"] for row in manifest["cases"]] == ["passed", "interrupted"]


def test_install_signal_handlers_raises_stop_requested(monkeypatch):
    handlers = {}

    def fake_signal(signum, handler):
        handlers[signum] = handler

    monkeypatch.setattr(ab.signal, "signal", fake_signal)

    ab.install_signal_handlers()

    assert ab.signal.SIGTERM in handlers
    with pytest.raises(ab.StopRequested):
        handlers[ab.signal.SIGTERM](ab.signal.SIGTERM, None)


def test_main_child_run_installs_signal_handlers(tmp_path, monkeypatch):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    calls = []

    monkeypatch.setattr(ab, "install_signal_handlers", lambda: calls.append("signals"), raising=False)
    monkeypatch.setattr(
        ab,
        "run_controller",
        lambda config, run_id, runner=None, dry_run=False: calls.append(("controller", dry_run)) or 0,
    )

    exit_code = ab.main([
        "run",
        "--config", str(config_path),
        "--run-id", "run123",
        "--child",
    ])

    assert exit_code == 0
    assert calls == ["signals", ("controller", False)]


def test_main_foreground_run_installs_signal_handlers(tmp_path, monkeypatch):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    calls = []

    monkeypatch.setattr(ab, "install_signal_handlers", lambda: calls.append("signals"), raising=False)
    monkeypatch.setattr(
        ab,
        "run_controller",
        lambda config, run_id, runner=None, dry_run=False: calls.append(("controller", dry_run)) or 0,
    )

    exit_code = ab.main([
        "run",
        "--config", str(config_path),
        "--run-id", "run123",
    ])

    assert exit_code == 0
    assert calls == ["signals", ("controller", False)]


def test_main_dry_run_does_not_install_signal_handlers(tmp_path, monkeypatch):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    calls = []

    monkeypatch.setattr(ab, "install_signal_handlers", lambda: calls.append("signals"), raising=False)
    monkeypatch.setattr(
        ab,
        "run_controller",
        lambda config, run_id, runner=None, dry_run=False: calls.append(("controller", dry_run)) or 0,
    )

    exit_code = ab.main([
        "run",
        "--config", str(config_path),
        "--run-id", "run123",
        "--dry-run",
    ])

    assert exit_code == 0
    assert calls == [("controller", True)]


def test_main_child_load_config_failure_writes_failed_state(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "bad-config.json"
    results_dir = tmp_path / "results"

    def fail_load_config(path):
        raise ab.ConfigError("bad child config")

    monkeypatch.setattr(ab, "load_config", fail_load_config)
    monkeypatch.setattr(ab, "install_signal_handlers", lambda: None, raising=False)

    exit_code = ab.main([
        "run",
        "--config", str(config_path),
        "--run-id", "run123",
        "--child",
        "--results-dir", str(results_dir),
    ])

    captured = capsys.readouterr()
    state = json.loads((results_dir / "run123" / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert state["status"] == "failed"
    assert "bad child config" in state["error"]
    assert "bad child config" in captured.err


def test_controller_default_ready_probe_runs_inside_docker_network(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()

    def reject_host_ready(*args, **kwargs):
        raise AssertionError("host ready check should not be used by default")

    monkeypatch.setattr(ab, "wait_for_ready", reject_host_ready)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    probes = ready_probe_commands(runner.commands)
    assert result == 0
    assert len(probes) == 1
    assert value_after(probes[0], "--name") == "bench-ready-qwen2_5_1_5b-bf16_default-run123"
    assert value_after(probes[0], "--network") == "vllm-bench-net"
    labels = values_after(probes[0], "--label")
    assert "vllm_auto_bench.managed=true" in labels
    assert "vllm_auto_bench.run_id=run123" in labels
    assert f"vllm_auto_bench.run_dir={(tmp_path / 'results' / 'run123').resolve()}" in labels
    assert "vllm-bench-runner:offline" in probes[0]
    assert "http://bench-vllm-qwen2_5_1_5b-bf16_default-run123:8000/v1/models" in probes[0]


def test_ready_probe_uses_container_ip_when_available(tmp_path):
    """Long container names exceed urllib's IDNA label length limit; use IP instead."""
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    class IPInspectRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True,
                stdout=None, stderr=None):
            if (
                args[:3] == ["docker", "inspect", "--format"]
                and "NetworkSettings.Networks" in args[3]
            ):
                return ab.Completed(list(args), 0, "172.18.0.5\n", "")
            return super().run(
                args,
                check=check,
                capture=capture,
                text=text,
                stdout=stdout,
                stderr=stderr,
            )

    runner = IPInspectRunner()
    assert ab.wait_for_container_ready(config, case, runner) is True
    probes = ready_probe_commands(runner.commands)
    assert len(probes) == 1
    assert "http://172.18.0.5:8000/v1/models" in probes[0]


@pytest.mark.parametrize("exception_type", [ab.StopRequested, KeyboardInterrupt])
def test_ready_probe_interruption_removes_owned_probe_container(tmp_path, exception_type):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    probe_name = "bench-ready-qwen2_5_1_5b-bf16_default-run123"

    class InterruptedProbeRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            if ready_probe_commands([list(args)]):
                self.commands.append(list(args))
                if "--name" in args:
                    self.container_labels[value_after(args, "--name")] = {
                        label.split("=", 1)[0]: label.split("=", 1)[1]
                        for label in values_after(args, "--label")
                        if "=" in label
                    }
                raise exception_type("probe interrupted")
            return super().run(
                args,
                check=check,
                capture=capture,
                text=text,
                stdout=stdout,
                stderr=stderr,
            )

    runner = InterruptedProbeRunner()

    with pytest.raises(exception_type):
        ab.wait_for_container_ready(config, case, runner)

    assert any(command[:3] == ["docker", "stop", probe_name] for command in runner.commands)
    assert any(command[:4] == ["docker", "rm", "-f", probe_name] for command in runner.commands)


def test_ready_probe_cleans_owned_leftover_before_starting_new_probe(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    run_dir = tmp_path / "results" / "run123"
    probe_name = "bench-ready-qwen2_5_1_5b-bf16_default-run123"

    class ConflictingProbeRunner(FakeRunner):
        def __init__(self):
            super().__init__()
            self.container_labels[probe_name] = {
                "vllm_auto_bench.managed": "true",
                "vllm_auto_bench.run_id": "run123",
                "vllm_auto_bench.run_dir": str(run_dir.resolve()),
            }

        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            if args[:4] == ["docker", "rm", "-f", probe_name]:
                self.commands.append(list(args))
                self.container_labels.pop(probe_name, None)
                return ab.Completed(list(args), 0, "", "")
            if ready_probe_commands([list(args)]):
                self.commands.append(list(args))
                if probe_name in self.container_labels:
                    return ab.Completed(list(args), 125, "", "name conflict")
                return ab.Completed(list(args), 0, "ready\n", "")
            return super().run(
                args,
                check=check,
                capture=capture,
                text=text,
                stdout=stdout,
                stderr=stderr,
            )

    runner = ConflictingProbeRunner()

    assert ab.wait_for_container_ready(config, case, runner) is True

    rm_index = command_index(runner.commands, ["docker", "rm", "-f", probe_name])
    run_index = command_index(runner.commands, ["docker", "run", "--rm"])
    assert rm_index < run_index


def test_ready_probe_label_mismatch_fails_closed_without_removing_container(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    probe_name = "bench-ready-qwen2_5_1_5b-bf16_default-run123"
    runner = FakeRunner()
    runner.container_labels[probe_name] = {
        "vllm_auto_bench.managed": "true",
        "vllm_auto_bench.run_id": "other-run",
        "vllm_auto_bench.run_dir": str((tmp_path / "other-results" / "run123").resolve()),
    }

    assert ab.wait_for_container_ready(config, case, runner) is False

    assert ready_probe_commands(runner.commands) == []
    assert not any(command[:3] == ["docker", "stop", probe_name] for command in runner.commands)
    assert not any(command[:4] == ["docker", "rm", "-f", probe_name] for command in runner.commands)
    assert runner.container_labels[probe_name]["vllm_auto_bench.run_id"] == "other-run"


def test_controller_stop_during_bench_removes_owned_bench_container(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    bench_name = "bench-runner-qwen2_5_1_5b-bf16_default-smoke-run123"

    class StopBenchRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            if bench_run_commands([list(args)]):
                self.commands.append(list(args))
                self.container_labels[value_after(args, "--name")] = {
                    label.split("=", 1)[0]: label.split("=", 1)[1]
                    for label in values_after(args, "--label")
                    if "=" in label
                }
                raise ab.StopRequested("bench interrupted")
            return super().run(
                args,
                check=check,
                capture=capture,
                text=text,
                stdout=stdout,
                stderr=stderr,
            )

    runner = StopBenchRunner()
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 130
    assert any(command[:3] == ["docker", "stop", bench_name] for command in runner.commands)
    assert any(command[:4] == ["docker", "rm", "-f", bench_name] for command in runner.commands)


def test_controller_rejects_existing_bench_container_with_foreign_labels(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    bench_name = "bench-runner-qwen2_5_1_5b-bf16_default-smoke-run123"
    runner = FakeRunner()
    runner.container_labels[bench_name] = {
        "vllm_auto_bench.managed": "true",
        "vllm_auto_bench.run_id": "other-run",
        "vllm_auto_bench.run_dir": str((tmp_path / "other-results" / "run123").resolve()),
    }
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 1
    assert bench_run_commands(runner.commands) == []
    assert not any(command[:3] == ["docker", "stop", bench_name] for command in runner.commands)
    assert not any(command[:4] == ["docker", "rm", "-f", bench_name] for command in runner.commands)
    assert runner.container_labels[bench_name]["vllm_auto_bench.run_id"] == "other-run"


def test_controller_published_port_ready_uses_localhost(tmp_path, monkeypatch):
    data = minimal_config(tmp_path)
    data["run"]["publish_host_port"] = True
    data["run"]["host_port"] = 18080
    config = ab.load_config(write_config(tmp_path, data))
    runner = FakeRunner()
    captured = {}

    def capture_host_ready(base_url, api_key, timeout_sec):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["timeout_sec"] = timeout_sec
        return True

    def reject_container_ready(*args, **kwargs):
        raise AssertionError("container ready check should not be used with published port")

    monkeypatch.setattr(ab, "wait_for_ready", capture_host_ready)
    monkeypatch.setattr(ab, "wait_for_container_ready", reject_container_ready, raising=False)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 0
    assert captured == {
        "base_url": "http://127.0.0.1:18080/v1",
        "api_key": "local-bench-key",
        "timeout_sec": 30,
    }
    assert ready_probe_commands(runner.commands) == []


def test_controller_artifact_failure_still_stops_and_removes_container(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)

    def raise_artifact(*args, **kwargs):
        raise RuntimeError("artifact boom")

    monkeypatch.setattr(ab, "save_vllm_artifacts", raise_artifact)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    assert result == 0
    assert any("docker stop bench-vllm-qwen2_5_1_5b-bf16_default-run123" in cmd for cmd in joined)
    assert_removed_after_stop(runner.commands, "bench-vllm-qwen2_5_1_5b-bf16_default-run123")
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_controller_artifact_stop_requested_interrupts_and_cleans(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)

    def stop_artifact(*args, **kwargs):
        raise ab.StopRequested("stopped")

    monkeypatch.setattr(ab, "save_vllm_artifacts", stop_artifact)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert result == 130
    assert manifest["status"] == "interrupted"
    assert state["status"] == "interrupted"
    assert any("docker stop bench-vllm-qwen2_5_1_5b-bf16_default-run123" in cmd for cmd in joined)
    assert_removed_after_stop(runner.commands, "bench-vllm-qwen2_5_1_5b-bf16_default-run123")
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_controller_stop_during_vllm_start_cleans_container_and_network(tmp_path):
    class StopDuringVllmStartRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            self.commands.append(list(args))
            if args[:3] == ["docker", "run", "-d"]:
                container_name = value_after(args, "--name")
                self.container_labels[container_name] = {
                    label.split("=", 1)[0]: label.split("=", 1)[1]
                    for label in values_after(args, "--label")
                    if "=" in label
                }
                raise ab.StopRequested("stopped during vllm start")
            return super().run(args, check=check, capture=capture, text=text, stdout=stdout, stderr=stderr)

    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = StopDuringVllmStartRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    assert result == 130
    assert any("docker stop bench-vllm-qwen2_5_1_5b-bf16_default-run123" in cmd for cmd in joined)
    assert_removed_after_stop(runner.commands, "bench-vllm-qwen2_5_1_5b-bf16_default-run123")
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_controller_stop_during_network_create_does_not_remove_unowned_network(tmp_path):
    class StopDuringNetworkCreateRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            self.commands.append(list(args))
            if args[:3] == ["docker", "network", "create"]:
                raise ab.StopRequested("stopped during network create")
            return super().run(args, check=check, capture=capture, text=text, stdout=stdout, stderr=stderr)

    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = StopDuringNetworkCreateRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    assert result == 130
    assert any("docker network create" in cmd and "vllm-bench-net" in cmd for cmd in joined)
    assert not any("docker network rm vllm-bench-net" in cmd for cmd in joined)


@pytest.mark.parametrize("failure_prefix", [
    ["docker", "inspect", "--format"],
    ["docker", "network", "rm"],
])
def test_controller_cleanup_stop_requested_marks_interrupted(tmp_path, failure_prefix):
    class StopDuringCleanupRunner(FakeRunner):
        def __init__(self):
            super().__init__()
            self.bench_finished = False

        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            if any("run_bench_multi.py" in str(arg) for arg in args):
                self.bench_finished = True
            if self.bench_finished and args[:len(failure_prefix)] == failure_prefix:
                self.commands.append(list(args))
                raise ab.StopRequested("stopped during cleanup")
            return super().run(args, check=check, capture=capture, text=text, stdout=stdout, stderr=stderr)

    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = StopDuringCleanupRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert result == 130
    assert manifest["status"] == "interrupted"
    assert state["status"] == "interrupted"


def test_controller_validate_failure_writes_failed_state(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))

    def fail_validate(config_arg):
        raise ab.ConfigError("bad local paths")

    monkeypatch.setattr(ab, "validate_local_paths", fail_validate)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert state["status"] == "failed"
    assert "bad local paths" in state["error"]


def test_controller_config_resolved_write_failure_writes_failed_state(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    original_write_json = ab.write_json_atomic

    def fail_config_resolved(path, payload):
        if path.name == "config.resolved.json":
            raise OSError("cannot write resolved config")
        original_write_json(path, payload)

    monkeypatch.setattr(ab, "write_json_atomic", fail_config_resolved)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert state["status"] == "failed"
    assert "cannot write resolved config" in state["error"]


def test_controller_init_stop_requested_writes_interrupted_state(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))

    def stop_validate(config_arg):
        raise ab.StopRequested("stopped during init")

    monkeypatch.setattr(ab, "validate_local_paths", stop_validate)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    run_dir = tmp_path / "results" / "run123"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result == 130
    assert state["status"] == "interrupted"
    assert manifest["status"] == "interrupted"


def test_controller_dry_run_prints_commands_without_result_files(tmp_path, capsys):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=True)

    out = capsys.readouterr().out
    run_dir = tmp_path / "results" / "run123"
    assert result == 0
    assert "docker network create" in out
    assert "vllm_auto_bench.run_id=run123" in out
    assert "docker run -d" in out
    assert "run_bench_multi.py" in out
    assert (run_dir / "config.resolved.json").exists()
    assert not (run_dir / "manifest.json").exists()
    assert not (run_dir / "state.json").exists()
    assert not (
        run_dir / "qwen2_5_1_5b" / "bf16_default" / "smoke" / "status.json"
    ).exists()
    assert runner.commands == []


def test_cleanup_network_warns_when_external_containers_connected(tmp_path, caplog):
    class ConnectedNetworkRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            if args[:4] == ["docker", "inspect", "--format", "{{json .Containers}}"]:
                self.commands.append(list(args))
                return ab.Completed(list(args), 0, json.dumps({"external": {}}), "")
            return super().run(args, check=check, capture=capture, text=text, stdout=stdout, stderr=stderr)

    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = ConnectedNetworkRunner()

    stop_requested = ab.cleanup_network(config, runner, owned=True, dry_run=False, run_id="run123")

    assert stop_requested is False
    assert not any(command[:3] == ["docker", "network", "rm"] for command in runner.commands)
    assert "network cleanup skipped" in caplog.text.lower()
    assert "external" in caplog.text.lower()


def test_cleanup_network_warns_when_network_rm_fails(tmp_path, caplog):
    class FailingNetworkRmRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            if args[:3] == ["docker", "network", "rm"]:
                self.commands.append(list(args))
                return ab.Completed(list(args), 1, "", "network busy")
            return super().run(args, check=check, capture=capture, text=text, stdout=stdout, stderr=stderr)

    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FailingNetworkRmRunner()

    stop_requested = ab.cleanup_network(config, runner, owned=True, dry_run=False, run_id="run123")

    assert stop_requested is False
    assert any(command[:3] == ["docker", "network", "rm"] for command in runner.commands)
    assert "network cleanup failed" in caplog.text.lower()
    assert "network busy" in caplog.text


def test_controller_group_exception_skips_only_unrecorded_cases(tmp_path, monkeypatch):
    data = minimal_config(tmp_path)
    second_profile = dict(data["bench_profiles"][0])
    second_profile["name"] = "smoke2"
    data["bench_profiles"].append(second_profile)
    config = ab.load_config(write_config(tmp_path, data))
    runner = FakeRunner()
    original_build_bench = ab.build_bench_run_command
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(ab, "wait_for_container_ready", lambda *args, **kwargs: True, raising=False)

    def fail_second_bench(config_arg, case, bench_dir):
        if case.bench_profile.name == "smoke2":
            raise RuntimeError("bench boom")
        return original_build_bench(config_arg, case, bench_dir)

    monkeypatch.setattr(ab, "build_bench_run_command", fail_second_bench)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    manifest = json.loads((tmp_path / "results" / "run123" / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert len(manifest["cases"]) == 2
    assert [case["bench_profile"] for case in manifest["cases"]] == ["smoke", "smoke2"]
    assert [case["status"] for case in manifest["cases"]] == ["passed", "skipped"]
    assert state["counts"]["passed"] == 1
    assert state["counts"]["skipped"] == 1
    assert state["counts"]["total"] == 2


def test_current_state_counts_manifest_cases(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    layout = ab.build_layout(config, "run123", case)
    manifest = ab.Manifest(run_id="run123", total=3)
    manifest.record(case, layout, "failed", error="failed")
    manifest.record(case, layout, "skipped", error="skipped")

    state = ab.current_state(
        "run123",
        (case, case, case),
        2,
        case,
        "running",
        manifest=manifest,
    )

    assert state["counts"]["passed"] == 0
    assert state["counts"]["failed"] == 1
    assert state["counts"]["skipped"] == 1
    assert state["counts"]["running"] == 1
    assert state["counts"]["completed"] == 2
    assert state["counts"]["total"] == 3


def test_runner_protocol_type_is_exposed():
    assert hasattr(ab, "RunnerProtocol")


def test_detach_command_reinvokes_child_with_run_id(tmp_path):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    run_id = "smoke_20260629_120000"
    results_dir = tmp_path / "results"
    cmd = ab.build_detach_command(config_path, run_id, results_dir)

    assert cmd[:2] == [sys.executable, str(Path(ab.__file__).resolve())]
    assert cmd[2:] == [
        "run", "--config", str(config_path),
        "--run-id", run_id,
        "--child",
        "--results-dir", str(results_dir),
    ]


def test_status_reads_state_file(tmp_path, capsys):
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": {"model": "m", "serve_profile": "s", "bench_profile": "b"},
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "run123",
            "status": "completed_with_failures",
            "cases": [{}, {}, {}],
        }),
        encoding="utf-8",
    )

    exit_code = ab.print_status(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "running" in captured.out
    assert "m/s/b" in captured.out
    assert "pid: 12345" in captured.out
    assert "manifest: completed_with_failures cases=3" in captured.out


def test_status_reads_topology_current(tmp_path, capsys):
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": {
            "model": "m",
            "serve_profile": None,
            "topology_profile": "topo",
            "bench_profile": "b",
        },
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })

    exit_code = ab.print_status(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "m/topo/b" in captured.out


def test_current_bench_log_path_uses_topology_current(tmp_path):
    run_dir = tmp_path / "results" / "run123"
    log_path = run_dir / "m" / "topo" / "b" / "bench.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("bench log\n", encoding="utf-8")
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": {
            "model": "m",
            "serve_profile": None,
            "topology_profile": "topo",
            "bench_profile": "b",
        },
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })

    assert ab.current_bench_log_path(run_dir) == log_path


def test_status_corrupt_state_returns_error(tmp_path, capsys):
    run_dir = tmp_path / "results" / "run123"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{bad json", encoding="utf-8")

    exit_code = ab.print_status(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "state" in captured.err.lower()


def test_status_invalid_utf8_state_returns_error(tmp_path, capsys):
    run_dir = tmp_path / "results" / "run123"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_bytes(b"\xff")

    exit_code = ab.print_status(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "state" in captured.err.lower()
    assert "invalid" in captured.err.lower()


def test_status_rejects_non_object_state(tmp_path, capsys):
    run_dir = tmp_path / "results" / "run123"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("[]", encoding="utf-8")

    exit_code = ab.print_status(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "state" in captured.err.lower()
    assert "invalid" in captured.err.lower()


def test_parse_args_run_dry_run():
    args = ab.parse_args(["run", "--config", "c.json", "--dry-run"])

    assert args.command == "run"
    assert args.config == Path("c.json")
    assert args.dry_run is True


def test_child_startup_exception_releases_lock_when_state_write_fails(tmp_path, monkeypatch):
    calls = []

    def fail_load_config(path):
        raise ab.ConfigError("bad config")

    def fail_startup_state(*args, **kwargs):
        raise OSError("state write failed")

    def fake_release(run_dir, token):
        calls.append((run_dir, token))

    monkeypatch.setattr(ab, "load_config", fail_load_config)
    monkeypatch.setattr(ab, "write_child_startup_state", fail_startup_state)
    monkeypatch.setattr(ab, "release_run_lock_for_token", fake_release)

    exit_code = ab.main([
        "run",
        "--config",
        str(tmp_path / "config.json"),
        "--run-id",
        "run123",
        "--child",
        "--results-dir",
        str(tmp_path / "results"),
        "--lock-token",
        "token123",
    ])

    assert exit_code == 1
    assert calls == [(tmp_path / "results" / "run123", "token123")]


def test_child_startup_stop_releases_lock_when_state_write_fails(tmp_path, monkeypatch):
    calls = []

    def stop_during_signal_install():
        raise ab.StopRequested("stopped")

    def fail_startup_state(*args, **kwargs):
        raise OSError("state write failed")

    def fake_release(run_dir, token):
        calls.append((run_dir, token))

    monkeypatch.setattr(ab, "install_signal_handlers", stop_during_signal_install)
    monkeypatch.setattr(ab, "write_child_startup_state", fail_startup_state)
    monkeypatch.setattr(ab, "release_run_lock_for_token", fake_release)

    exit_code = ab.main([
        "run",
        "--config",
        str(tmp_path / "config.json"),
        "--run-id",
        "run123",
        "--child",
        "--results-dir",
        str(tmp_path / "results"),
        "--lock-token",
        "token123",
    ])

    assert exit_code == 130
    assert calls == [(tmp_path / "results" / "run123", "token123")]


def test_main_status_reads_state_file(tmp_path, capsys):
    ab.write_state(tmp_path / "run123", {
        "run_id": "run123",
        "status": "completed",
        "current": None,
        "counts": {"passed": 1, "failed": 0, "skipped": 0, "running": 0, "total": 1},
    })

    exit_code = ab.main(["status", "--results-dir", str(tmp_path), "--run-id", "run123"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "completed" in captured.out


def test_logs_prints_controller_log(tmp_path, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.log").write_text("line 1\nline 2\n", encoding="utf-8")

    exit_code = ab.main(["logs", "--results-dir", str(tmp_path), "--run-id", "run123"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "line 1" in captured.out
    assert "line 2" in captured.out


def test_logs_prefers_current_bench_log(tmp_path, capsys):
    run_dir = tmp_path / "run123"
    bench_dir = run_dir / "model_a" / "serve_a" / "bench_a"
    bench_dir.mkdir(parents=True)
    (run_dir / "controller.log").write_text("controller only\n", encoding="utf-8")
    (bench_dir / "bench.log").write_text("active bench line\n", encoding="utf-8")
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": {
            "model": "model_a",
            "serve_profile": "serve_a",
            "bench_profile": "bench_a",
        },
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })

    exit_code = ab.main(["logs", "--results-dir", str(tmp_path), "--run-id", "run123"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "active bench line" in captured.out
    assert "controller only" not in captured.out


def test_logs_controller_flag_prints_controller_log(tmp_path, capsys):
    run_dir = tmp_path / "run123"
    bench_dir = run_dir / "model_a" / "serve_a" / "bench_a"
    bench_dir.mkdir(parents=True)
    (run_dir / "controller.log").write_text("controller line\n", encoding="utf-8")
    (bench_dir / "bench.log").write_text("bench line\n", encoding="utf-8")
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": {
            "model": "model_a",
            "serve_profile": "serve_a",
            "bench_profile": "bench_a",
        },
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })

    exit_code = ab.main([
        "logs",
        "--results-dir",
        str(tmp_path),
        "--run-id",
        "run123",
        "--controller",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "controller line" in captured.out
    assert "bench line" not in captured.out


def test_logs_replace_invalid_utf8(tmp_path, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.log").write_bytes(b"ok\n\xff\n")

    exit_code = ab.main(["logs", "--results-dir", str(tmp_path), "--run-id", "run123"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ok" in captured.out
    assert "\ufffd" in captured.out


def test_follow_file_tails_from_end(tmp_path, monkeypatch, capsys):
    path = tmp_path / "controller.log"
    path.write_text("old\n", encoding="utf-8")
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            with path.open("a", encoding="utf-8") as handle:
                handle.write("new\n")
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(ab.time, "sleep", fake_sleep)

    exit_code = ab.follow_file(path)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "old" not in captured.out
    assert "new" in captured.out


def test_readme_documents_vllm_cache_persistence():
    readme = (Path(ab.__file__).resolve().parent / "README.md").read_text(encoding="utf-8")
    readme_lower = readme.lower()

    assert "vllm_cache" in readme
    assert "VLLM_CACHE_ROOT" in readme
    assert "DG_JIT_CACHE_DIR" in readme
    assert "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" in readme
    assert "可省略" in readme
    assert ".cache/vllm_auto_bench" in readme
    assert "model_path" in readme
    assert "gpus" in readme
    assert "serve args" in readme
    assert "image id" in readme_lower
    assert "digest" in readme_lower
    assert "cache_key" in readme
    assert "更换" in readme or "清理" in readme


def test_gitignore_ignores_local_cache_dir():
    gitignore = (Path(ab.__file__).resolve().parents[1] / ".gitignore").read_text(
        encoding="utf-8"
    )

    assert ".cache/" in gitignore.splitlines()


def test_follow_file_handles_open_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "controller.log"
    path.write_text("data", encoding="utf-8")

    def fail_open(*args, **kwargs):
        raise OSError("cannot read")

    monkeypatch.setattr(Path, "open", fail_open)

    exit_code = ab.follow_file(path)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "cannot read" in captured.err


def test_print_status_renders_counts_summary(tmp_path, capsys, monkeypatch):
    run_dir = tmp_path / "rd"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(json.dumps({
        "run_id": "rd", "status": "running",
        "current": {"model": "m", "serve_profile": "sp", "bench_profile": "bp"},
        "counts": {"passed": 2, "failed": 1, "running": 1, "total": 4},
    }), encoding="utf-8")
    # _stdout_supports_color() checks stdout (not stderr); force no-color path.
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    rc = ab.print_status(run_dir)
    out = capsys.readouterr().out
    assert rc == 0
    assert "passed=2" in out
    assert "failed=1" in out
    assert "\x1b[" not in out  # no-color path emits plain text


def test_print_status_renders_colored_counts_when_tty(tmp_path, capsys, monkeypatch):
    run_dir = tmp_path / "rd"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(json.dumps({
        "run_id": "rd", "status": "running",
        "counts": {"passed": 2, "failed": 1, "total": 4},
    }), encoding="utf-8")
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    ab.print_status(run_dir)
    out = capsys.readouterr().out
    assert "\033[32m" in out and "passed=2" in out  # 绿色 ANSI + 文本


def test_format_counts_default_is_plain_text_even_when_tty(monkeypatch):
    """logger.info("run finished: %s", _format_counts(...)) 走默认调用，
    即便前台 stdout 是 tty 也必须返回纯文本，避免 ANSI 经 FileFormatter 污染
    controller.log。"""
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    text = ab._format_counts({"passed": 2, "failed": 1, "running": 1, "total": 4})
    assert "\x1b[" not in text
    assert "passed=2" in text
    assert "failed=1" in text


def test_format_counts_color_true_emits_ansi():
    text = ab._format_counts({"passed": 2, "failed": 1}, color=True)
    assert "\x1b[32m" in text  # GREEN for passed
    assert "\x1b[31m" in text  # RED for failed
    assert "passed=2" in text and "failed=1" in text


def test_follow_file_level_filter(tmp_path, monkeypatch, capsys):
    log = tmp_path / "controller.log"
    log.write_text("", encoding="utf-8")
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(
                    "2026-07-04 12:00:00 INFO  [case 1/2][bench] ok\n"
                    "2026-07-04 12:00:01 ERROR [case 1/2][bench] boom\n"
                )
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(ab.time, "sleep", fake_sleep)

    rc = ab.follow_file(log, level="ERROR")
    out = capsys.readouterr().out
    assert rc == 0
    assert "boom" in out
    assert "ok" not in out  # INFO 行被 level 过滤


def write_stop_state(run_dir, status="running"):
    ab.write_state(run_dir, {
        "run_id": run_dir.name,
        "status": status,
        "current": None,
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })


def matching_controller_cmdline(run_id="run123", results_dir=None):
    command = [
        sys.executable,
        str(Path(ab.__file__).resolve()),
        "run",
        "--config",
        "config.json",
        "--run-id",
        run_id,
        "--child",
    ]
    if results_dir is not None:
        command.extend(["--results-dir", str(results_dir)])
    return command


def write_controller_metadata(run_dir, pid=12345, run_id=None, command=None):
    run_id = run_id or run_dir.name
    command = command or matching_controller_cmdline(run_id, run_dir.parent)
    (run_dir / "controller.json").write_text(
        json.dumps({
            "pid": pid,
            "run_id": run_id,
            "command": command,
            "config_path": "config.json",
            "started_at": 1,
        }),
        encoding="utf-8",
    )


def test_stop_run_sends_sigterm(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    write_controller_metadata(run_dir)
    signals = []

    def fake_kill(pid, sig):
        signals.append((pid, sig))

    monkeypatch.setattr(ab.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(
        ab,
        "read_process_cmdline",
        lambda pid: matching_controller_cmdline(run_dir.name, run_dir.parent),
        raising=False,
    )

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert signals == [(12345, ab.signal.SIGTERM)]
    assert "12345" in captured.out


def test_cleanup_run_removes_remote_managed_containers_by_run_id(
    tmp_path,
    monkeypatch,
):
    config = topology_config_with_image(tmp_path)
    run_dir = config.run.results_dir / "run123"
    run_dir.mkdir(parents=True)
    ab.write_private_json_atomic(
        run_dir / ab.RESUME_CONFIG_FILE,
        ab.resume_config_to_dict(config),
    )

    class CleanupRemoteRunner(FakeRemoteDockerRunner):
        def run(self, host, command, *, check=False, capture=True, text=True,
                stdout=None, stderr=None):
            if command[:3] == ["docker", "ps", "-aq"]:
                self.commands.append((host.name, list(command)))
                return ab.Completed(list(command), 0, "old-a\nold-b\n", "")
            return super().run(
                host,
                command,
                check=check,
                capture=capture,
                text=text,
                stdout=stdout,
                stderr=stderr,
            )

    remote = CleanupRemoteRunner()
    monkeypatch.setattr(ab, "RemoteDockerRunner", lambda: remote, raising=False)

    exit_code = ab.cleanup_run(run_dir)

    assert exit_code == 0
    ps_commands = [
        command for _host, command in remote.commands
        if command[:3] == ["docker", "ps", "-aq"]
    ]
    assert ps_commands
    assert all(
        f"label={ab.NETWORK_MANAGED_LABEL}=true" in command
        and f"label={ab.NETWORK_RUN_ID_LABEL}=run123" in command
        for command in ps_commands
    )
    assert any(
        command == ["docker", "rm", "-f", "old-a", "old-b"]
        for _host, command in remote.commands
    )


def test_stop_run_rejects_unsafe_pid(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("0\n", encoding="utf-8")
    write_stop_state(run_dir)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for unsafe pid")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "unsafe pid" in captured.err


def test_stop_run_handles_missing_process(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    write_controller_metadata(run_dir)

    def missing_process(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(ab.os, "kill", missing_process, raising=False)
    monkeypatch.setattr(
        ab,
        "read_process_cmdline",
        lambda pid: matching_controller_cmdline(run_dir.name, run_dir.parent),
        raising=False,
    )

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not running" in captured.err or "process not found" in captured.err


def test_stop_run_rejects_invalid_pid(tmp_path, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("not-a-pid\n", encoding="utf-8")

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "invalid pid" in captured.err


def test_stop_run_handles_os_error(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    write_controller_metadata(run_dir)

    def fail_stop(pid, sig):
        raise OSError("denied")

    monkeypatch.setattr(ab.os, "kill", fail_stop, raising=False)
    monkeypatch.setattr(
        ab,
        "read_process_cmdline",
        lambda pid: matching_controller_cmdline(run_dir.name, run_dir.parent),
        raising=False,
    )

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "failed to stop" in captured.err


def test_stop_run_rejects_inactive_state(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir, status="completed")

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for inactive state")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "run is not active" in captured.err or "state invalid" in captured.err


def test_stop_run_invalid_utf8_state_returns_error(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    (run_dir / "state.json").write_bytes(b"\xff")

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for invalid state")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "state invalid" in captured.err


def test_stop_run_rejects_non_object_state(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    (run_dir / "state.json").write_text("[]", encoding="utf-8")

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for invalid state")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "state invalid" in captured.err


def test_stop_run_rejects_mismatched_process(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    write_controller_metadata(run_dir)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for mismatched process")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)
    monkeypatch.setattr(
        ab,
        "read_process_cmdline",
        lambda pid: [sys.executable, str(Path(ab.__file__).resolve()), "run", "--run-id", "other"],
        raising=False,
    )

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "process does not match controller" in captured.err or "stale pid" in captured.err


def test_stop_run_rejects_run_id_as_other_argument(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    write_controller_metadata(run_dir)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for mismatched run id")

    malicious_cmdline = matching_controller_cmdline("other", run_dir.parent) + ["--config", "run123"]
    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)
    monkeypatch.setattr(ab, "read_process_cmdline", lambda pid: malicious_cmdline, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "process does not match controller" in captured.err or "stale pid" in captured.err


def test_stop_run_requires_controller_metadata(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called without controller metadata")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)
    monkeypatch.setattr(
        ab,
        "read_process_cmdline",
        lambda pid: matching_controller_cmdline(run_dir.name, run_dir.parent),
        raising=False,
    )

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "metadata" in captured.err or "stale pid" in captured.err


def test_stop_run_rejects_bad_controller_metadata(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    (run_dir / "controller.json").write_text("{bad json", encoding="utf-8")
    write_stop_state(run_dir)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called with bad controller metadata")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)
    monkeypatch.setattr(
        ab,
        "read_process_cmdline",
        lambda pid: matching_controller_cmdline(run_dir.name, run_dir.parent),
        raising=False,
    )

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "metadata" in captured.err or "stale pid" in captured.err


@pytest.mark.parametrize("metadata", [
    {"pid": 999, "run_id": "run123", "command": matching_controller_cmdline("run123")},
    {"pid": 12345, "run_id": "other", "command": matching_controller_cmdline("run123")},
    {"pid": 12345, "run_id": "run123", "command": matching_controller_cmdline("other")},
    {"pid": 12345, "run_id": "run123", "command": ["ok", 1]},
])
def test_stop_run_rejects_controller_metadata_mismatch(tmp_path, monkeypatch, capsys, metadata):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    (run_dir / "controller.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_stop_state(run_dir)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called with mismatched controller metadata")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)
    monkeypatch.setattr(
        ab,
        "read_process_cmdline",
        lambda pid: matching_controller_cmdline(run_dir.name),
        raising=False,
    )

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "metadata" in captured.err or "stale pid" in captured.err


def test_stop_run_rejects_metadata_command_that_is_not_controller(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    command = ["sleep", "30"]
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    write_controller_metadata(run_dir, command=command)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for non-controller command")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)
    monkeypatch.setattr(ab, "read_process_cmdline", lambda pid: command, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "controller" in captured.err or "metadata" in captured.err


def test_stop_run_rejects_auto_bench_path_outside_controller_position(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    command = [
        "tail",
        "-f",
        str(Path(ab.__file__).resolve()),
        "run",
        "--child",
        "--run-id", "run123",
        "--results-dir", str(run_dir.parent),
    ]
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    write_controller_metadata(run_dir, command=command)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for tail command")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)
    monkeypatch.setattr(ab, "read_process_cmdline", lambda pid: command, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "controller" in captured.err or "metadata" in captured.err


def test_stop_run_rejects_duplicate_run_id_arguments(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    command = [
        sys.executable,
        str(Path(ab.__file__).resolve()),
        "run",
        "--config", "config.json",
        "--run-id", "run123",
        "--run-id", "other",
        "--child",
        "--results-dir", str(run_dir.parent),
    ]
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    write_controller_metadata(run_dir, command=command)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for duplicate run-id")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)
    monkeypatch.setattr(ab, "read_process_cmdline", lambda pid: command, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "controller" in captured.err or "metadata" in captured.err


def test_stop_run_rejects_controller_from_other_results_dir(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "results" / "run123"
    other_results = tmp_path / "other-results"
    run_dir.mkdir(parents=True)
    other_results.mkdir()
    command = matching_controller_cmdline("run123", other_results)
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    write_controller_metadata(run_dir, command=command)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for other results-dir")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)
    monkeypatch.setattr(ab, "read_process_cmdline", lambda pid: command, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "controller" in captured.err or "metadata" in captured.err


def test_stop_run_rejects_invalid_utf8_pid_file(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_bytes(b"\xff")
    write_stop_state(run_dir)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for invalid pid")

    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "invalid pid" in captured.err


def test_stop_run_handles_pid_file_read_error(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    pid_path = run_dir / "controller.pid"
    pid_path.write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    original_read_text = Path.read_text

    def fail_pid_read(self, *args, **kwargs):
        if self == pid_path:
            raise OSError("cannot read pid")
        return original_read_text(self, *args, **kwargs)

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called when pid cannot be read")

    monkeypatch.setattr(Path, "read_text", fail_pid_read)
    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "invalid pid" in captured.err


def test_prepare_model_args_parse_expected_options():
    args = ab.parse_args([
        "prepare-model",
        "--modelscope-id", "Qwen/Qwen2.5-1.5B-Instruct",
        "--target", "/tmp/model",
        "--bench-image", "bench:offline",
        "--force",
    ])

    assert args.modelscope_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.target == "/tmp/model"
    assert args.bench_image == "bench:offline"
    assert args.force is True


def test_main_prepare_model_calls_stub_with_expected_keywords(monkeypatch):
    captured = {}

    def fake_prepare_model(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(ab, "prepare_model", fake_prepare_model)

    exit_code = ab.main([
        "prepare-model",
        "--modelscope-id", "Qwen/Qwen2.5-1.5B-Instruct",
        "--target", "/tmp/model",
        "--bench-image", "bench:offline",
        "--force",
    ])

    assert exit_code == 0
    assert captured["modelscope_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert captured["target"] == Path("/tmp/model")
    assert captured["bench_image"] == "bench:offline"
    assert captured["force"] is True
    assert isinstance(captured["runner"], ab.DockerRunner)


def write_model_files(path, *, safetensors=True, marker="complete"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    if safetensors:
        (path / "model.safetensors").write_text(marker, encoding="utf-8")


class ModelDownloadRunner(FakeRunner):
    def __init__(self, target_parent, marker="fresh", require_clean_tmp=False):
        super().__init__()
        self.target_parent = Path(target_parent)
        self.marker = marker
        self.require_clean_tmp = require_clean_tmp

    def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
        self.commands.append(list(args))
        target_arg = args[-1]
        prefix = "/model-target/"
        if not target_arg.startswith(prefix):
            return ab.Completed(list(args), 1, "", "unexpected target")
        host_tmp = self.target_parent / target_arg.removeprefix(prefix)
        if host_tmp.exists() and self.require_clean_tmp:
            return ab.Completed(list(args), 0, "download skipped existing tmp\n", "")
        write_model_files(host_tmp, marker=self.marker)
        return ab.Completed(list(args), 0, "ok\n", "")


def test_model_dir_requires_complete_safetensors(tmp_path):
    target = tmp_path / "model"
    target.mkdir()
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "tokenizer.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors.parts").mkdir()

    with pytest.raises(ab.ConfigError, match="safetensors"):
        ab.validate_prepared_model_dir(target)


def test_model_dir_requires_all_indexed_safetensors_shards(tmp_path):
    target = tmp_path / "model"
    write_model_files(target, safetensors=False)
    (target / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors",
            }
        }),
        encoding="utf-8",
    )
    (target / "model-00001-of-00002.safetensors").write_text("shard-1", encoding="utf-8")

    with pytest.raises(ab.ConfigError, match="safetensors|model-00002"):
        ab.validate_prepared_model_dir(target)


def test_model_dir_accepts_complete_indexed_safetensors_shards(tmp_path):
    target = tmp_path / "model"
    write_model_files(target, safetensors=False)
    (target / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors",
            }
        }),
        encoding="utf-8",
    )
    (target / "model-00001-of-00002.safetensors").write_text("shard-1", encoding="utf-8")
    (target / "model-00002-of-00002.safetensors").write_text("shard-2", encoding="utf-8")

    ab.validate_prepared_model_dir(target)


def test_model_dir_rejects_index_shard_without_safetensors_suffix(tmp_path):
    target = tmp_path / "model"
    write_model_files(target, safetensors=False)
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": "config.json"}}),
        encoding="utf-8",
    )

    with pytest.raises(ab.ConfigError, match="safetensors|shard"):
        ab.validate_prepared_model_dir(target)


def test_prepare_model_uses_bench_image_and_temp_dir(tmp_path):
    target = tmp_path / "Qwen2.5-1.5B-Instruct"
    runner = ModelDownloadRunner(tmp_path)

    result = ab.prepare_model(
        modelscope_id="Qwen/Qwen2.5-1.5B-Instruct",
        target=target,
        bench_image="bench:offline",
        runner=runner,
    )

    assert result == 0
    assert target.is_dir()
    assert (target / "model.safetensors").exists()
    cmd = runner.commands[0]
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert f"{tmp_path.resolve()}:/model-target" in cmd
    assert "bench:offline" in cmd
    assert "Qwen/Qwen2.5-1.5B-Instruct" in cmd
    assert "/model-target/Qwen2.5-1.5B-Instruct.download-tmp" in cmd


def test_prepare_model_removes_stale_download_tmp_before_download(tmp_path):
    target = tmp_path / "model"
    tmp_download = tmp_path / "model.download-tmp"
    write_model_files(tmp_download, marker="stale-other-model")
    runner = ModelDownloadRunner(tmp_path, marker="fresh-model", require_clean_tmp=True)

    result = ab.prepare_model(
        modelscope_id="Qwen/Qwen2.5-1.5B-Instruct",
        target=target,
        bench_image="bench:offline",
        runner=runner,
    )

    assert result == 0
    assert (target / "model.safetensors").read_text(encoding="utf-8") == "fresh-model"


def test_prepare_model_removes_broken_download_tmp_symlink(tmp_path):
    target = tmp_path / "model"
    tmp_download = tmp_path / "model.download-tmp"
    try:
        tmp_download.symlink_to(tmp_path / "missing-download-target", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink is not supported: {exc}")
    runner = ModelDownloadRunner(tmp_path, marker="fresh-model")

    result = ab.prepare_model(
        modelscope_id="Qwen/Qwen2.5-1.5B-Instruct",
        target=target,
        bench_image="bench:offline",
        runner=runner,
    )

    assert result == 0
    assert target.is_dir()
    assert not tmp_download.is_symlink()
    assert (target / "model.safetensors").read_text(encoding="utf-8") == "fresh-model"


def test_prepare_model_existing_complete_skips_download(tmp_path, capsys):
    target = tmp_path / "model"
    write_model_files(target)
    runner = FakeRunner()

    result = ab.prepare_model(
        modelscope_id="Qwen/Qwen2.5-1.5B-Instruct",
        target=target,
        bench_image="bench:offline",
        runner=runner,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert runner.commands == []
    assert "already exists" in captured.out


def test_prepare_model_existing_incomplete_without_force_fails(tmp_path):
    target = tmp_path / "model"
    write_model_files(target, safetensors=False)
    runner = FakeRunner()

    with pytest.raises(ab.ConfigError, match="safetensors"):
        ab.prepare_model(
            modelscope_id="Qwen/Qwen2.5-1.5B-Instruct",
            target=target,
            bench_image="bench:offline",
            runner=runner,
        )

    assert runner.commands == []


def test_prepare_model_force_backs_up_existing_target(tmp_path):
    target = tmp_path / "model"
    write_model_files(target, marker="old")
    runner = ModelDownloadRunner(tmp_path, marker="new")

    result = ab.prepare_model(
        modelscope_id="Qwen/Qwen2.5-1.5B-Instruct",
        target=target,
        bench_image="bench:offline",
        force=True,
        runner=runner,
    )

    backups = list(tmp_path.glob("model.backup-*"))
    assert result == 0
    assert (target / "model.safetensors").read_text(encoding="utf-8") == "new"
    assert len(backups) == 1
    assert (backups[0] / "model.safetensors").read_text(encoding="utf-8") == "old"


def test_prepare_model_download_failure_keeps_tmp_and_no_target(tmp_path):
    class FailingDownloadRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            self.commands.append(list(args))
            return ab.Completed(list(args), 1, "", "download failed")

    target = tmp_path / "model"
    tmp_download = tmp_path / "model.download-tmp"
    tmp_download.mkdir()
    runner = FailingDownloadRunner()

    with pytest.raises(RuntimeError, match="download failed"):
        ab.prepare_model(
            modelscope_id="Qwen/Qwen2.5-1.5B-Instruct",
            target=target,
            bench_image="bench:offline",
            runner=runner,
        )

    assert not target.exists()
    assert runner.commands


def test_main_dry_run_takes_precedence_over_detach(tmp_path, monkeypatch):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    calls = []

    def fake_start_detached(*args, **kwargs):
        calls.append("detach")
        return 0

    def fake_run_controller(config, run_id, runner=None, dry_run=False):
        calls.append(("controller", dry_run))
        return 0

    monkeypatch.setattr(ab, "start_detached", fake_start_detached)
    monkeypatch.setattr(ab, "run_controller", fake_run_controller)

    exit_code = ab.main([
        "run",
        "--config", str(config_path),
        "--dry-run",
        "--detach",
        "--run-id", "run123",
    ])

    assert exit_code == 0
    assert calls == [("controller", True)]


def test_start_detached_uses_devnull_stdin(tmp_path, monkeypatch):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)

    exit_code = ab.start_detached(config_path, config, "run123")

    assert exit_code == 0
    assert calls
    assert calls[0][1]["stdin"] is ab.subprocess.DEVNULL


def test_start_detached_rejects_existing_active_run_dir(tmp_path, monkeypatch, caplog):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": None,
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    monkeypatch.setattr(ab, "is_process_running", lambda pid: pid == 12345, raising=False)

    def fail_popen(*args, **kwargs):
        raise AssertionError("active run should not start another controller")

    monkeypatch.setattr(ab.subprocess, "Popen", fail_popen)

    exit_code = ab.start_detached(config_path, config, "run123")

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert state["status"] == "running"
    assert "active" in caplog.text


def test_run_controller_rejects_existing_active_run_dir(tmp_path, monkeypatch, caplog):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": None,
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    monkeypatch.setattr(ab, "is_process_running", lambda pid: pid == 12345, raising=False)
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert state["status"] == "running"
    assert "active" in caplog.text
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_run_controller_rejects_active_state_without_pid_file(tmp_path, caplog):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": None,
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert state["status"] == "running"
    assert "active" in caplog.text
    assert not (run_dir / "config.resolved.json").exists()
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_run_controller_rejects_active_state_with_invalid_pid_file(tmp_path, caplog):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "starting",
        "current": None,
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })
    (run_dir / "controller.pid").write_text("not-a-pid\n", encoding="utf-8")
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert state["status"] == "starting"
    assert "active" in caplog.text
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_start_detached_rejects_existing_run_lock(tmp_path, monkeypatch, caplog):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_dir = tmp_path / "results" / "run123"
    run_dir.mkdir(parents=True)
    (run_dir / ".run.lock").write_text("locked\n", encoding="utf-8")

    def fail_popen(*args, **kwargs):
        raise AssertionError("locked run should not start another controller")

    monkeypatch.setattr(ab.subprocess, "Popen", fail_popen)

    exit_code = ab.start_detached(config_path, config, "run123")

    assert exit_code == 1
    assert "active" in caplog.text
    assert (run_dir / ".run.lock").exists()


def test_start_detached_invalid_run_id_does_not_create_lock_path(tmp_path):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)

    with pytest.raises(ab.ConfigError, match="run_id"):
        ab.start_detached(config_path, config, "bad/name")

    assert not (tmp_path / "results" / "bad").exists()


def test_run_controller_releases_lock_after_success(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_dir = tmp_path / "results" / "run123"
    first_runner = FakeRunner()

    first = ab.run_controller(config, run_id="run123", runner=first_runner, dry_run=False)
    second = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    assert first == 0
    assert second == 0
    assert not (run_dir / ".run.lock").exists()


def test_run_controller_reclaims_terminal_stale_lock(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "completed",
        "current": None,
        "counts": {"passed": 1, "failed": 0, "skipped": 0, "running": 0, "total": 1},
    })
    (run_dir / ".run.lock").write_text(
        json.dumps({"pid": 12345, "token": "old-token", "created_at": 1.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ab, "is_process_running", lambda pid: False)

    result = ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    assert result == 0
    assert not (run_dir / ".run.lock").exists()


def test_run_controller_does_not_delete_new_lock_after_stale_recheck(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "completed",
        "current": None,
        "counts": {"passed": 1, "failed": 0, "skipped": 0, "running": 0, "total": 1},
    })
    stale_payload = {"pid": 12345, "token": "old-token", "created_at": 1.0}
    new_payload = {"pid": 67890, "token": "new-token", "created_at": 2.0}
    (run_dir / ".run.lock").write_text(json.dumps(stale_payload), encoding="utf-8")
    original_read_run_lock = ab._read_run_lock
    reads = 0

    def racing_read_run_lock(lock_run_dir):
        nonlocal reads
        reads += 1
        payload = original_read_run_lock(lock_run_dir)
        if reads == 1:
            (run_dir / ".run.lock").write_text(json.dumps(new_payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(ab, "_read_run_lock", racing_read_run_lock)
    monkeypatch.setattr(ab, "is_process_running", lambda pid: False)
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 1
    assert "active" in caplog.text
    assert json.loads((run_dir / ".run.lock").read_text(encoding="utf-8")) == new_payload
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_release_run_lock_does_not_delete_replaced_lock(tmp_path, monkeypatch):
    run_dir = tmp_path / "results" / "run123"
    run_dir.mkdir(parents=True)
    lock_path = run_dir / ".run.lock"
    old_payload = {"pid": 12345, "token": "old-token", "created_at": 1.0}
    new_payload = {"pid": 67890, "token": "new-token", "created_at": 2.0}
    lock_path.write_text(json.dumps(old_payload), encoding="utf-8")
    original_read_run_lock = ab._read_run_lock
    reads = 0

    def racing_read_run_lock(lock_run_dir):
        nonlocal reads
        reads += 1
        payload = original_read_run_lock(lock_run_dir)
        if reads == 1:
            lock_path.write_text(json.dumps(new_payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(ab, "_read_run_lock", racing_read_run_lock)

    ab.release_run_lock(ab.RunLock(run_dir=run_dir, token="old-token"))

    assert json.loads(lock_path.read_text(encoding="utf-8")) == new_payload


def test_run_controller_keeps_active_dead_pid_lock_fail_closed(tmp_path, monkeypatch, caplog):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": None,
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    (run_dir / ".run.lock").write_text(
        json.dumps({"pid": 12345, "token": "old-token", "created_at": 1.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ab, "is_process_running", lambda pid: False)
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 1
    assert "active" in caplog.text
    assert (run_dir / ".run.lock").exists()
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_run_controller_corrupt_lock_fail_closed(tmp_path, caplog):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "completed",
        "current": None,
        "counts": {"passed": 1, "failed": 0, "skipped": 0, "running": 0, "total": 1},
    })
    (run_dir / ".run.lock").write_text("not-json", encoding="utf-8")
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    assert result == 1
    assert "active" in caplog.text
    assert (run_dir / ".run.lock").exists()
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_run_controller_releases_lock_when_cleanup_terminal_state_write_fails(
    tmp_path,
    monkeypatch,
):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_dir = tmp_path / "results" / "run123"

    monkeypatch.setattr(ab, "cleanup_network", lambda *args, **kwargs: True)

    def fail_terminal_state(*args, **kwargs):
        raise OSError("terminal state write failed")

    monkeypatch.setattr(ab, "write_terminal_state", fail_terminal_state)

    with pytest.raises(OSError, match="terminal state write failed"):
        ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=False)

    assert not (run_dir / ".run.lock").exists()


def test_start_detached_popen_failure_writes_failed_state(tmp_path, monkeypatch, capsys):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)

    def fail_popen(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(ab.subprocess, "Popen", fail_popen)

    exit_code = ab.start_detached(config_path, config, "run123")

    captured = capsys.readouterr()
    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert state["status"] == "failed"
    assert "spawn failed" in captured.err


def test_start_detached_validate_local_paths_failure_releases_lock(tmp_path, capsys):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_dir = tmp_path / "results" / "run123"
    config.models[0].host_model_path.rename(tmp_path / "missing-model")

    exit_code = ab.start_detached(config_path, config, "run123")

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert state["status"] == "failed"
    assert "model path does not exist" in captured.err
    assert not (run_dir / ".run.lock").exists()


def test_start_detached_starting_state_write_failure_releases_lock(tmp_path, monkeypatch, capsys):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_dir = tmp_path / "results" / "run123"
    original_write_state = ab.write_state

    def fail_starting_state(path, state):
        if state.get("status") == "starting":
            raise OSError("starting write failed")
        return original_write_state(path, state)

    monkeypatch.setattr(ab, "write_state", fail_starting_state)

    exit_code = ab.start_detached(config_path, config, "run123")

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert state["status"] == "failed"
    assert "starting write failed" in captured.err
    assert not (run_dir / ".run.lock").exists()


def test_start_detached_writes_failed_state_before_releasing_lock(tmp_path, monkeypatch, capsys):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_dir = tmp_path / "results" / "run123"
    real_release = ab.release_run_lock

    def fail_popen(*args, **kwargs):
        raise OSError("spawn failed")

    def racing_release(lock):
        real_release(lock)
        ab.write_state(run_dir, {
            "run_id": "run123",
            "status": "running",
            "current": None,
            "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
        })

    monkeypatch.setattr(ab.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(ab, "release_run_lock", racing_release)

    exit_code = ab.start_detached(config_path, config, "run123")

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert state["status"] == "running"
    assert "spawn failed" in captured.err


def test_start_detached_pid_write_failure_terminates_child(tmp_path, monkeypatch, capsys):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    original_write_text = Path.write_text

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

    process = FakeProcess()

    def fake_popen(*args, **kwargs):
        return process

    def fail_pid_write(self, *args, **kwargs):
        if self.name == "controller.pid":
            raise OSError("pid write failed")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(Path, "write_text", fail_pid_write)

    exit_code = ab.start_detached(config_path, config, "run123")

    captured = capsys.readouterr()
    state = json.loads((tmp_path / "results" / "run123" / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert process.terminated is True
    assert state["status"] == "failed"
    assert "pid write failed" in captured.err


def test_start_detached_metadata_failure_keeps_child_lock_when_exit_unconfirmed(
    tmp_path,
    monkeypatch,
    capsys,
):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_dir = tmp_path / "results" / "run123"
    lock_path = run_dir / ".run.lock"
    original_write_text = Path.write_text

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.terminated = False
            self.wait_calls = 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.wait_calls += 1
            raise ab.subprocess.TimeoutExpired(cmd="controller", timeout=timeout)

        def kill(self):
            self.terminated = True

    process = FakeProcess()

    def fake_popen(*args, **kwargs):
        return process

    def fail_pid_write_after_child_takes_lock(self, *args, **kwargs):
        if self.name == "controller.pid":
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            payload["pid"] = process.pid
            lock_path.write_text(json.dumps(payload), encoding="utf-8")
            raise OSError("pid write failed")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(Path, "write_text", fail_pid_write_after_child_takes_lock)

    exit_code = ab.start_detached(config_path, config, "run123")

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert process.terminated is True
    assert process.wait_calls >= 1
    assert state["status"] == "failed"
    assert lock_payload["pid"] == process.pid
    assert "pid write failed" in captured.err


def test_start_detached_metadata_failure_releases_lock_after_child_exits(
    tmp_path,
    monkeypatch,
    capsys,
):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_dir = tmp_path / "results" / "run123"
    lock_path = run_dir / ".run.lock"
    original_write_text = Path.write_text

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.terminated = False
            self.wait_calls = 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 0

    process = FakeProcess()

    def fake_popen(*args, **kwargs):
        return process

    def fail_pid_write_after_child_takes_lock(self, *args, **kwargs):
        if self.name == "controller.pid":
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            payload["pid"] = process.pid
            lock_path.write_text(json.dumps(payload), encoding="utf-8")
            raise OSError("pid write failed")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(Path, "write_text", fail_pid_write_after_child_takes_lock)

    exit_code = ab.start_detached(config_path, config, "run123")

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert process.terminated is True
    assert process.wait_calls == 1
    assert state["status"] == "failed"
    assert "pid write failed" in captured.err
    assert not lock_path.exists()


def test_example_configs_are_parseable():
    root = Path(ab.__file__).resolve().parent
    config_paths = [
        root / "configs" / "auto_bench.example.json",
        root / "configs" / "auto_bench.qwen2_5_1_5b.smoke.json",
    ]

    for path in config_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        config = ab.load_config(path)

        assert config.run.network == "vllm-bench-net"
        assert config.run.publish_host_port is False
        assert config.run.vllm_image == "009e4cb46541"
        assert config.run.bench_image == "vllm-bench-runner:offline"
        assert config.models
        assert config.serve_profiles
        assert config.bench_profiles
        assert data["mounts"]["models"] == "/Resource_Planning_Tool/model"
        assert data["run"]["results_dir"] == "vllm_standalone_bench/results"
        assert data["run"]["host_port"] == 18000
        assert any(
            model["model_path"] == "/models/Qwen2.5-1.5B-Instruct"
            for model in data["models"]
        )
        if path.name == "auto_bench.example.json":
            assert len(data["serve_profiles"]) == 2
            for profile in data["serve_profiles"]:
                assert "--gpu-memory-utilization" in profile["args"]
                gpu_util = float(value_after(profile["args"], "--gpu-memory-utilization"))
                assert 0 < gpu_util <= 1
            latency_matrix = data["bench_profiles"][0]
            assert latency_matrix["name"] == "latency_matrix"
            assert latency_matrix["output_lens"] == [1024]
            assert latency_matrix["parallel_nums"] == [1, 4, 8]
            assert latency_matrix["prefix_ratio"] == 0.8
            assert latency_matrix["max_ttft_ms"] == 15000
            assert latency_matrix["min_throughput_tok_s"] == 5
        else:
            smoke_args = data["serve_profiles"][0]["args"]
            assert "--gpu-memory-utilization" in smoke_args
            gpu_util = float(value_after(smoke_args, "--gpu-memory-utilization"))
            assert 0 < gpu_util <= 1


def test_bench_runner_dockerfile_contains_offline_dependencies():
    root = Path(ab.__file__).resolve().parent
    dockerfile = (root / "Dockerfile.bench-runner").read_text(encoding="utf-8")

    for expected in [
        "openpyxl",
        "modelscope",
        "run_bench_multi.py",
        "run_bench_serve.py",
        "vllm_bench",
        "requirements.txt",
    ]:
        assert expected in dockerfile

    for line in dockerfile.splitlines():
        if not line.startswith("COPY "):
            continue
        parts = line.split()
        sources = parts[1:-1]
        for source in sources:
            assert (root / source).exists(), f"COPY source does not exist: {source}"


def test_serve_profile_engine_defaults_to_vllm(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    assert config.serve_profiles[0].engine == "vllm"


def test_invalid_engine_rejected(tmp_path):
    data = minimal_config(tmp_path)
    data["serve_profiles"][0]["engine"] = "trtllm"
    with pytest.raises(ab.ConfigError, match="engine"):
        ab.load_config(write_config(tmp_path, data))


def test_images_falls_back_to_vllm_image(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    assert config.run.images == {"vllm": "009e4cb46541"}


def test_images_missing_engine_rejected(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["images"] = {"vllm": data["run"]["vllm_image"]}
    data["serve_profiles"][0]["engine"] = "sglang"
    with pytest.raises(ab.ConfigError, match="missing image"):
        ab.load_config(write_config(tmp_path, data))


def test_images_without_vllm_image_supported(tmp_path):
    data = minimal_config(tmp_path)
    del data["run"]["vllm_image"]
    data["run"]["images"] = {"sglang": "sglang:latest"}
    data["serve_profiles"][0]["engine"] = "sglang"
    config = ab.load_config(write_config(tmp_path, data))
    assert config.run.images == {"sglang": "sglang:latest"}
    assert config.run.vllm_image is None


def test_explicit_images_vllm_not_overridden_by_vllm_image(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["images"] = {"vllm": "explicit-vllm:tag"}
    config = ab.load_config(write_config(tmp_path, data))
    assert config.run.images["vllm"] == "explicit-vllm:tag"


def sglang_config(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["images"] = {"vllm": data["run"]["vllm_image"], "sglang": "sglang:latest"}
    profile = {
        "name": "sglang_bf16",
        "engine": "sglang",
        "gpus": "all",
        "args": ["--dtype", "bfloat16", "--mem-fraction-static", "0.70"],
    }
    data["serve_profiles"] = [profile]
    return data


def test_build_sglang_command_uses_launch_server(tmp_path):
    config = ab.load_config(write_config(tmp_path, sglang_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_serve_run_command(config, case, tmp_path / "results" / "run123")

    assert value_after(cmd, "--entrypoint") == "python3"
    assert "sglang:latest" in cmd
    assert value_after(cmd, "-m") == "sglang.launch_server"
    assert value_after(cmd, "--model-path") == "/models/Qwen2.5-1.5B-Instruct"
    assert value_after(cmd, "--host") == "0.0.0.0"
    assert value_after(cmd, "--port") == "8000"
    assert value_after(cmd, "--served-model-name") == "qwen2_5_1_5b"
    assert value_after(cmd, "--api-key") == "local-bench-key"
    assert value_after(cmd, "--mem-fraction-static") == "0.70"
    assert value_after(cmd, "--gpus") == "all"


def test_build_serve_command_dispatches_vllm(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_serve_run_command(config, case, tmp_path / "results" / "run123")

    assert value_after(cmd, "--entrypoint") == "vllm"
    assert "serve" in cmd


def test_build_vllm_serve_command_wraps_specific_gpus(tmp_path):
    # serve 路径也要用 _docker_gpus 包装：指定卡时 docker 需要 '"device=..."' 形式，
    # 否则 docker 解析异常、容器看到的卡数不对（实测 TP=4 报 available GPUs=3）。
    data = minimal_config(tmp_path)
    data["serve_profiles"][0]["gpus"] = "0,1,2,3"
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_serve_run_command(config, case, tmp_path / "results" / "run123")

    assert value_after(cmd, "--gpus") == '"device=0,1,2,3"'


def test_build_vllm_serve_command_keeps_gpus_all(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_serve_run_command(config, case, tmp_path / "results" / "run123")

    assert value_after(cmd, "--gpus") == "all"


def test_build_postprocess_container_command_uses_bench_image_and_host_user(
    tmp_path,
    monkeypatch,
):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    monkeypatch.setattr(ab.os, "getuid", lambda: 1001)
    monkeypatch.setattr(ab.os, "getgid", lambda: 1002)

    cmd = ab.build_postprocess_container_command(
        config,
        config_path=config_path,
        run_id="run123",
    )

    mounts = [cmd[index + 1] for index, value in enumerate(cmd) if value == "-v"]
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert value_after(cmd, "--user") == "1001:1002"
    assert value_after(cmd, "-w") == "/workspace"
    assert f"{ab.project_root()}:/workspace" in mounts
    assert f"{config_path.parent.resolve()}:/auto-bench-config:ro" in mounts
    assert f"{config.run.results_dir.resolve()}:/auto-bench-results" in mounts
    assert "vllm-bench-runner:offline" in cmd
    assert "postprocess" in cmd
    assert value_after(cmd, "--config") == "/auto-bench-config/config.json"
    assert value_after(cmd, "--results-dir") == "/auto-bench-results"
    assert values_after(cmd, "--run-id") == ["run123"]


def test_postprocess_run_merges_resource_summaries_and_aggregates(
    tmp_path,
    monkeypatch,
):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    run_id = "run123"
    run_dir = config.run.results_dir / run_id
    case = ab.expand_cases(config, run_id=run_id)[0]
    layout = ab.build_layout(config, run_id, case)
    layout.bench_dir.mkdir(parents=True)
    (layout.bench_dir / "result.csv").write_text(
        "model,throughput_tok_s\nm,1\n",
        encoding="utf-8-sig",
    )
    (layout.bench_dir / "resource_summary.json").write_text(
        json.dumps({
            "available": True,
            "sample_count": 1,
            "aggregate": {"cpu_util_avg_pct": 12.5},
        }) + "\n",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        ab,
        "aggregate_compare",
        lambda config_arg, run_dir_arg: calls.append(Path(run_dir_arg)) or None,
    )

    assert ab.run_postprocess(config, run_id) == 0

    assert calls == [run_dir]
    with (layout.bench_dir / "result.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["resource_monitor_available"] == "true"
    assert rows[0]["resource_sample_count"] == "1"
    assert rows[0]["cpu_util_avg_pct"] == "12.5"


def test_controller_invokes_postprocess_container_after_groups(tmp_path, monkeypatch):
    data = minimal_config(tmp_path)
    data["run"]["images"] = {"vllm": data["run"]["vllm_image"], "sglang": "sglang:latest"}
    sglang_profile = {
        "name": "sglang_bf16",
        "engine": "sglang",
        "gpus": "all",
        "args": ["--dtype", "bfloat16"],
    }
    data["serve_profiles"].append(sglang_profile)
    config_path = write_config(tmp_path, data)
    config = ab.load_config(config_path)
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_ready", lambda *a, **k: True)

    result = ab.run_controller(
        config,
        run_id="run123",
        runner=runner,
        dry_run=False,
        config_path=config_path,
    )

    postprocess_commands = [
        command for command in runner.commands
        if command[:3] == ["docker", "run", "--rm"] and "postprocess" in command
    ]
    assert result == 0
    assert len(postprocess_commands) == 1
    assert "vllm-bench-runner:offline" in postprocess_commands[0]


def test_controller_dry_run_skips_aggregate(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()

    ab.run_controller(config, run_id="run123", runner=runner, dry_run=True)

    assert not any(
        command[:3] == ["docker", "run", "--rm"] and "postprocess" in command
        for command in runner.commands
    )


def test_shipped_sglang_compare_config_parses():
    path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "auto_bench.qwen2_5_1_5b.sglang_compare.json"
    )
    config = ab.load_config(path)
    engines = {profile.engine for profile in config.serve_profiles}
    assert engines == {"vllm", "sglang"}
    assert "vllm" in config.run.images
    assert "sglang" in config.run.images


def test_shipped_sglang_pd_remote_config_parses(tmp_path):
    path = CONFIG_DIR / "auto_bench.sglang_pd_remote.example.json"
    config = ab.load_config(path)
    assert config.serve_profiles == ()
    assert config.topology_profiles[0].engine == "sglang"
    assert config.topology_profiles[0].frontend.kind == "sglang_router"
    case = ab.expand_cases(config, run_id="dryrun")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "dryrun")
    assert {"p1", "p2", "d1", "d2", "router"} <= set(commands)
    router = commands["router"].argv
    assert "sglang_router.launch_router" in router
    assert "http://192.0.2.11:30000" in values_after(router, "--prefill")
    assert "http://192.0.2.21:30001" in values_after(router, "--decode")


def test_shipped_sglang_pd_hicache_remote_config_parses(tmp_path):
    path = CONFIG_DIR / "auto_bench.sglang_pd_hicache_remote.example.json"
    config = ab.load_config(path)
    assert config.serve_profiles == ()
    profile = config.topology_profiles[0]
    assert profile.engine == "sglang"
    assert profile.sglang_hicache is not None
    assert profile.sglang_hicache.mode == "full_async_offload"

    case = ab.expand_cases(config, run_id="dryrun")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "dryrun")
    p1 = commands["p1"].argv
    d1 = commands["d1"].argv
    router = commands["router"].argv
    assert "--enable-hierarchical-cache" in p1
    assert "--disaggregation-decode-enable-offload-kvcache" in d1
    assert value_after(p1, "--hicache-storage-backend") == "mooncake"
    assert value_after(d1, "--hicache-storage-backend") == "mooncake"
    assert value_after(p1, "--device") == "/dev/infiniband"
    assert "--device" not in router


def test_shipped_sglang_pd_hicache_minimax_config_parses(tmp_path):
    path = CONFIG_DIR / "auto_bench.sglang_pd_hicache_remote_minimax.json"
    config = ab.load_config(path)
    assert [profile.name for profile in config.serve_profiles] == [
        "sglang_tp4_minimax_m27_baseline_14_4gpu",
    ]
    assert [profile.name for profile in config.topology_profiles] == [
        "sglang_pd_hicache_minimax_m27_2p2d",
        "sglang_pd_hicache_minimax_m27_prefill_only_2p2d",
    ]
    assert config.serve_profiles[0].engine == "sglang"
    assert config.serve_profiles[0].gpus == "0,1,2,3"
    assert all(profile.engine == "sglang" for profile in config.topology_profiles)
    assert all(
        profile.sglang_hicache is not None
        for profile in config.topology_profiles
    )
    bench = config.bench_profiles[0]
    assert bench.name == "latency_matrix"
    assert bench.input_lens == (4096, 8192, 16384, 32768, 65536, 131072, 172032)
    assert bench.parallel_nums == (1, 4, 8, 16, 24, 32, 48, 64, 96, 128)
    assert bench.prefix_ratio == 0.1

    baseline_case = next(
        case for case in ab.expand_cases(config, run_id="dryrun")
        if case.serve_profile is not None
    )
    baseline_cmd = ab.build_serve_run_command(config, baseline_case, tmp_path / "dryrun")
    assert value_after(baseline_cmd, "--gpus") == '"device=0,1,2,3"'
    assert value_after(baseline_cmd, "--quantization") == "compressed-tensors"
    assert value_after(baseline_cmd, "--kv-cache-dtype") == "fp8_e4m3"
    assert value_after(baseline_cmd, "--mem-fraction-static") == "0.92"
    assert value_after(baseline_cmd, "--max-running-requests") == "128"
    assert value_after(baseline_cmd, "--context-length") == "172032"
    assert "--enable-metrics" in baseline_cmd
    assert "--enable-cache-report" in baseline_cmd
    assert "--stream-response-default-include-usage" in baseline_cmd
    assert "--log-requests" in baseline_cmd
    assert value_after(baseline_cmd, "--log-requests-level") == "2"
    assert value_after(baseline_cmd, "--log-requests-format") == "json"
    assert value_after(baseline_cmd, "--log-requests-target") == "stdout"
    assert value_after(baseline_cmd, "--tool-call-parser") == "minimax-m2"
    assert value_after(baseline_cmd, "--reasoning-parser") == "minimax"

    pd_case = next(
        case for case in ab.expand_cases(config, run_id="dryrun")
        if (
            case.topology_profile is not None
            and case.topology_profile.name == "sglang_pd_hicache_minimax_m27_2p2d"
        )
    )
    commands = pd_case.topology_profile.build_commands(config, pd_case, tmp_path / "dryrun")
    p1 = commands["p1"].argv
    d1 = commands["d1"].argv
    assert value_after(p1, "--quantization") == "compressed-tensors"
    assert value_after(d1, "--kv-cache-dtype") == "fp8_e4m3"
    assert "--log-requests" in p1
    assert "--log-requests" in d1
    assert "--enable-metrics" in p1
    assert "--enable-cache-report" in d1


def test_controller_dry_run_prints_sglang_command(tmp_path, capsys):
    data = minimal_config(tmp_path)
    data["run"]["images"] = {"vllm": data["run"]["vllm_image"], "sglang": "sglang:latest"}
    data["serve_profiles"][0]["engine"] = "sglang"
    config = ab.load_config(write_config(tmp_path, data))

    ab.run_controller(config, run_id="run123", runner=FakeRunner(), dry_run=True)

    out = capsys.readouterr().out
    assert "sglang.launch_server" in out
    assert "sglang:latest" in out
    assert "--model-path" in out


def test_load_config_parses_warmup_opts(tmp_path):
    data = minimal_config(tmp_path)
    data["bench_profiles"][0]["warmup_concurrency"] = 4
    data["bench_profiles"][0]["warmup_output_len"] = 128
    config = ab.load_config(write_config(tmp_path, data))
    bp = config.bench_profiles[0]
    assert bp.warmup_concurrency == 4
    assert bp.warmup_output_len == 128


def test_load_config_warmup_opts_default_none(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    bp = config.bench_profiles[0]
    assert bp.warmup_concurrency is None
    assert bp.warmup_output_len is None


def enable_vllm_cache(data, root):
    data["run"]["vllm_cache"] = {
        "enabled": True,
        "root": str(root),
        "container_path": "/vllm-cache",
        "set_default_env": True,
    }
    return data


def expected_vllm_cache_key_inputs(config, case):
    return {
        "vllm_image_ref": config.run.images["vllm"],
        "model": {
            "name": case.model.name,
            "model_path": case.model.model_path,
            "tokenizer_path": case.model.tokenizer_path,
            "served_model_name": case.model.served_model_name,
        },
        "serve_profile": {
            "name": case.serve_profile.name,
            "gpus": case.serve_profile.gpus,
            "args": list(case.serve_profile.args),
        },
    }


def expected_vllm_cache_fingerprint(inputs):
    canonical = json.dumps(
        inputs,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def test_load_config_vllm_cache_defaults_disabled(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))

    assert config.run.vllm_cache.enabled is False
    assert config.run.vllm_cache.root is None
    assert config.run.vllm_cache.container_path == "/vllm-cache"
    assert config.run.vllm_cache.set_default_env is True
    assert config.serve_profiles[0].cache_key is None


def test_load_config_parses_enabled_vllm_cache(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"

    config = ab.load_config(write_config(tmp_path, data))

    assert config.run.vllm_cache.enabled is True
    assert config.run.vllm_cache.root == (tmp_path / "cache").resolve()
    assert config.run.vllm_cache.container_path == "/vllm-cache"
    assert config.run.vllm_cache.set_default_env is True
    assert config.serve_profiles[0].cache_key == "glm52-fp8-tp8-h20-o2"


def test_load_config_defaults_enabled_vllm_cache_root_from_config_dir(tmp_path):
    config_dir = tmp_path / "configs"
    data = minimal_config(tmp_path)
    data["run"]["vllm_cache"] = {"enabled": True}

    config = ab.load_config(write_config_at(config_dir / "config.json", data))

    assert config.run.vllm_cache.enabled is True
    assert config.run.vllm_cache.root == (
        config_dir / ".cache" / "vllm_auto_bench"
    ).resolve()
    assert config.run.vllm_cache.container_path == "/vllm-cache"
    assert config.run.vllm_cache.set_default_env is True


def test_load_config_resolves_relative_vllm_cache_root_from_config_dir(tmp_path):
    config_dir = tmp_path / "configs"
    data = enable_vllm_cache(minimal_config(tmp_path), "relative-cache")

    config = ab.load_config(write_config_at(config_dir / "config.json", data))

    assert config.run.vllm_cache.root == (config_dir / "relative-cache").resolve()


def test_vllm_cache_null_is_rejected(tmp_path):
    data = minimal_config(tmp_path)
    data["run"]["vllm_cache"] = None

    with pytest.raises(ab.ConfigError, match="vllm_cache.*object"):
        ab.load_config(write_config(tmp_path, data))


def test_vllm_cache_explicit_null_root_is_rejected(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["run"]["vllm_cache"]["root"] = None

    with pytest.raises(ab.ConfigError, match="vllm_cache.root"):
        ab.load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("container_path", [
    "relative/cache",
    "/cache/../bad",
    "/",
    "/models",
    "/models/cache",
])
def test_vllm_cache_container_path_must_be_absolute_and_safe(tmp_path, container_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["run"]["vllm_cache"]["container_path"] = container_path

    with pytest.raises(ab.ConfigError, match="container_path|absolute|contain"):
        ab.load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("cache_key", ["bad/name", ".", ".."])
def test_serve_profile_cache_key_must_be_safe(tmp_path, cache_key):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = cache_key

    with pytest.raises(ab.ConfigError, match="cache_key|safe filename"):
        ab.load_config(write_config(tmp_path, data))


def test_resolve_vllm_cache_dir_uses_explicit_cache_key(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    assert ab.resolve_vllm_cache_dir(config, case) == (
        tmp_path / "cache" / "glm52-fp8-tp8-h20-o2"
    ).resolve()


def test_resolve_vllm_cache_dir_uses_stable_default_key(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    cache_dir = ab.resolve_vllm_cache_dir(config, case)
    key_inputs = expected_vllm_cache_key_inputs(config, case)
    expected_fingerprint = expected_vllm_cache_fingerprint(key_inputs)
    expected_name = (
        f"{case.model.name}__{case.serve_profile.name}__"
        f"{expected_fingerprint}"
    )

    assert cache_dir is not None
    assert cache_dir.parent == (tmp_path / "cache").resolve()
    assert cache_dir.name == expected_name
    assert ab.resolve_vllm_cache_dir(config, case) == cache_dir


def test_default_vllm_cache_dir_changes_when_image_changes(tmp_path):
    first_data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    first_config = ab.load_config(write_config(tmp_path, first_data))
    first_case = ab.expand_cases(first_config, run_id="run123")[0]

    second_data = json.loads(json.dumps(first_data))
    second_data["run"]["vllm_image"] = "vllm-openai:changed"
    second_config = ab.load_config(write_config(tmp_path, second_data))
    second_case = ab.expand_cases(second_config, run_id="run123")[0]

    assert (
        ab.resolve_vllm_cache_dir(first_config, first_case).name
        != ab.resolve_vllm_cache_dir(second_config, second_case).name
    )


def test_default_vllm_cache_dir_changes_when_serve_args_change(tmp_path):
    first_data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    first_config = ab.load_config(write_config(tmp_path, first_data))
    first_case = ab.expand_cases(first_config, run_id="run123")[0]

    second_data = json.loads(json.dumps(first_data))
    second_data["serve_profiles"][0]["args"] = ["--dtype", "float16"]
    second_config = ab.load_config(write_config(tmp_path, second_data))
    second_case = ab.expand_cases(second_config, run_id="run123")[0]

    assert (
        ab.resolve_vllm_cache_dir(first_config, first_case).name
        != ab.resolve_vllm_cache_dir(second_config, second_case).name
    )


def test_default_vllm_cache_dir_changes_when_serve_gpus_change(tmp_path):
    first_data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    first_config = ab.load_config(write_config(tmp_path, first_data))
    first_case = ab.expand_cases(first_config, run_id="run123")[0]

    second_data = json.loads(json.dumps(first_data))
    second_data["serve_profiles"][0]["gpus"] = "device=0,1"
    second_config = ab.load_config(write_config(tmp_path, second_data))
    second_case = ab.expand_cases(second_config, run_id="run123")[0]

    assert (
        ab.resolve_vllm_cache_dir(first_config, first_case).name
        != ab.resolve_vllm_cache_dir(second_config, second_case).name
    )


def test_default_vllm_cache_dir_changes_when_model_path_changes(tmp_path):
    first_data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    first_config = ab.load_config(write_config(tmp_path, first_data))
    first_case = ab.expand_cases(first_config, run_id="run123")[0]

    second_data = json.loads(json.dumps(first_data))
    second_data["models"][0]["model_path"] = "/models/Qwen2.5-1.5B-Instruct-Alt"
    second_config = ab.load_config(write_config(tmp_path, second_data))
    second_case = ab.expand_cases(second_config, run_id="run123")[0]

    assert (
        ab.resolve_vllm_cache_dir(first_config, first_case).name
        != ab.resolve_vllm_cache_dir(second_config, second_case).name
    )


def test_validate_local_paths_creates_vllm_cache_dirs(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    cache_dir = ab.resolve_vllm_cache_dir(config, case)

    assert cache_dir is not None
    assert not cache_dir.exists()

    ab.validate_local_paths(config)

    assert cache_dir.is_dir()


def test_build_vllm_cache_env_defaults(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    config = ab.load_config(write_config(tmp_path, data))

    assert ab.build_vllm_cache_env(config) == {
        "VLLM_CACHE_ROOT": "/vllm-cache",
        "DG_JIT_CACHE_DIR": "/vllm-cache/deep_gemm",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "/vllm-cache/flashinfer_autotune",
    }


def test_vllm_cache_container_path_rejects_container_root(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["run"]["vllm_cache"]["container_path"] = "/"

    with pytest.raises(ab.ConfigError, match="container_path|root|/models"):
        ab.load_config(write_config(tmp_path, data))


def test_build_vllm_cache_env_can_be_disabled(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["run"]["vllm_cache"]["set_default_env"] = False
    config = ab.load_config(write_config(tmp_path, data))

    assert ab.build_vllm_cache_env(config) == {}


def test_build_vllm_command_omits_cache_when_disabled(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_vllm_run_command(config, case, tmp_path / "results" / "run123")

    assert "/vllm-cache" not in " ".join(cmd)
    assert "VLLM_CACHE_ROOT=/vllm-cache" not in cmd


def test_build_vllm_command_includes_cache_mount_and_env(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_vllm_run_command(config, case, tmp_path / "results" / "run123")

    mounts = values_after(cmd, "-v")
    envs = values_after(cmd, "-e")
    assert f"{(tmp_path / 'cache' / 'glm52-fp8-tp8-h20-o2').resolve()}:/vllm-cache:rw" in mounts
    assert "VLLM_CACHE_ROOT=/vllm-cache" in envs
    assert "DG_JIT_CACHE_DIR=/vllm-cache/deep_gemm" in envs
    assert "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/vllm-cache/flashinfer_autotune" in envs


def test_vllm_cache_metadata_payload(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    payload = ab.vllm_cache_metadata(config, case)

    assert payload == {
        "enabled": True,
        "cache_key": "glm52-fp8-tp8-h20-o2",
        "cache_key_source": "explicit",
        "cache_key_inputs": expected_vllm_cache_key_inputs(config, case),
        "host_dir": str((tmp_path / "cache" / "glm52-fp8-tp8-h20-o2").resolve()),
        "container_path": "/vllm-cache",
        "env": {
            "VLLM_CACHE_ROOT": "/vllm-cache",
            "DG_JIT_CACHE_DIR": "/vllm-cache/deep_gemm",
            "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "/vllm-cache/flashinfer_autotune",
        },
    }


def test_vllm_cache_metadata_payload_for_default_key_includes_audit_inputs(tmp_path):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    payload = ab.vllm_cache_metadata(config, case)

    assert payload is not None
    assert payload["cache_key_source"] == "default"
    assert payload["cache_key_inputs"] == expected_vllm_cache_key_inputs(config, case)


def test_run_controller_writes_vllm_cache_metadata(tmp_path, monkeypatch):
    data = enable_vllm_cache(minimal_config(tmp_path), tmp_path / "cache")
    data["serve_profiles"][0]["cache_key"] = "glm52-fp8-tp8-h20-o2"
    config = ab.load_config(write_config(tmp_path, data))
    monkeypatch.setattr(ab, "wait_for_ready", lambda *a, **k: True)
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner)

    metadata_path = (
        tmp_path / "results" / "run123" / "qwen2_5_1_5b" / "bf16_default" / "vllm_cache.json"
    )
    assert result == 0
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["cache_key"] == (
        "glm52-fp8-tp8-h20-o2"
    )


def test_build_sglang_command_omits_vllm_cache_mount_and_env(tmp_path):
    data = enable_vllm_cache(sglang_config(tmp_path), tmp_path / "cache")
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]

    cmd = ab.build_serve_run_command(config, case, tmp_path / "results" / "run123")

    assert "/vllm-cache" not in " ".join(cmd)
    assert "VLLM_CACHE_ROOT=/vllm-cache" not in cmd


def test_build_bench_command_includes_warmup_opts(tmp_path):
    data = minimal_config(tmp_path)
    data["bench_profiles"][0]["warmup_concurrency"] = 4
    data["bench_profiles"][0]["warmup_output_len"] = 128
    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config)[0]
    cmd = ab.build_bench_run_command(config, case, tmp_path / "bench")
    assert value_after(cmd, "--warmup-concurrency") == "4"
    assert value_after(cmd, "--warmup-output-len") == "128"


def test_build_bench_command_omits_warmup_opts_when_none(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config)[0]
    cmd = ab.build_bench_run_command(config, case, tmp_path / "bench")
    assert "--warmup-concurrency" not in cmd
    assert "--warmup-output-len" not in cmd


def test_sglang_compare_config_enables_fixed_warmup():
    cfg = json.loads((CONFIG_DIR / "auto_bench.qwen2_5_1_5b.sglang_compare.json").read_text())
    bp = cfg["bench_profiles"][0]
    assert bp["warmup_concurrency"] == 4
    assert bp["warmup_output_len"] == 128


def test_smoke_config_enables_fixed_warmup():
    cfg = json.loads((CONFIG_DIR / "auto_bench.qwen2_5_1_5b.smoke.json").read_text())
    bp = cfg["bench_profiles"][0]
    assert bp["warmup_concurrency"] == 4
    assert bp["warmup_output_len"] == 128


def test_qwen3_asr_sample_config_loads():
    path = CONFIG_DIR / "auto_bench.qwen3_asr_1_7b.smoke.json"

    config = ab.load_config(path)

    assert config.bench_profiles[0].backend == "openai-audio"
    assert config.bench_profiles[0].dataset_path == ab.BUILTIN_ASR_DATASET_PATH
    assert config.bench_profiles[0].parallel_nums == (1, 4, 8)


def test_builtin_asr_dataset_manifest_is_valid():
    root = Path(__file__).resolve().parents[1] / "assets" / "librispeech_test_clean_256"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    jsonl = root / "asr_smoke.jsonl"
    audio_dir = root / "audio"

    import soundfile

    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    audio_refs = [row["audio"] for row in rows]
    audio_paths = [root / audio_ref for audio_ref in audio_refs]
    root_resolved = root.resolve()
    referenced_files = {path.resolve() for path in audio_paths}
    actual_flacs = {path.resolve() for path in audio_dir.glob("*.flac")}
    missing_flacs = sorted(
        path.relative_to(root_resolved).as_posix() for path in referenced_files - actual_flacs
    )
    extra_flacs = sorted(
        path.relative_to(root_resolved).as_posix() for path in actual_flacs - referenced_files
    )

    def duration_bucket(duration_s):
        if 5.0 <= duration_s < 10.0:
            return "medium"
        if 10.0 <= duration_s < 20.0:
            return "long"
        if 20.0 <= duration_s <= 30.0:
            return "xlong"
        return None

    assert len(rows) == 256
    assert manifest["sample_count"] == 256
    assert manifest["requested_sample_count"] == 256
    assert 5.0 <= manifest["min_duration_s"] <= manifest["max_duration_s"] <= 30.0
    assert manifest["total_audio_bytes"] <= 104857600
    assert len(audio_refs) == len(set(audio_refs))
    assert all(audio_ref.startswith("audio/") for audio_ref in audio_refs)
    assert all(Path(audio_ref).name == audio_ref.removeprefix("audio/") for audio_ref in audio_refs)
    assert all(Path(audio_ref).suffix == ".flac" for audio_ref in audio_refs)
    assert missing_flacs == []
    assert extra_flacs == []

    durations = [soundfile.info(path).duration for path in audio_paths]
    duration_buckets = {"medium": 0, "long": 0, "xlong": 0}
    for duration in durations:
        bucket = duration_bucket(duration)
        assert bucket is not None
        duration_buckets[bucket] += 1

    assert duration_buckets == manifest["duration_buckets"]
    assert min(durations) == pytest.approx(manifest["min_duration_s"])
    assert max(durations) == pytest.approx(manifest["max_duration_s"])
    assert round(sum(durations), 3) == manifest["total_duration_s"]
    assert sum(path.stat().st_size for path in audio_paths) == manifest["total_audio_bytes"]
    assert (root / "ATTRIBUTION.md").is_file()
    assert (root / "LICENSE.LibriSpeech.txt").is_file()


def test_example_config_includes_resource_monitor():
    path = CONFIG_DIR / "auto_bench.example.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["run"]["resource_monitor"] == {
        "enabled": True,
        "backend": "nvidia-smi",
        "interval_sec": 1.0,
    }


def test_run_controller_dry_run_logs_to_controller_log(tmp_path):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_id = "log_dry_run_001"
    rc = ab.run_controller(config, run_id, runner=FakeRunner(), dry_run=True)
    assert rc == 0
    log_path = config.run.results_dir / run_id / "controller.log"
    assert log_path.exists(), "controller.log should be created"
    text = log_path.read_text(encoding="utf-8")
    assert "controller started" in text


def test_controller_log_contains_case_prefix(tmp_path):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_id = "log_prefix_001"
    rc = ab.run_controller(config, run_id, runner=FakeRunner(), dry_run=True)
    assert rc == 0
    text = (config.run.results_dir / run_id / "controller.log").read_text(encoding="utf-8")
    assert ("[case " in text) or ("[serve]" in text), text


def test_controller_log_has_run_finished_node(tmp_path):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_id = "log_progress_001"
    rc = ab.run_controller(config, run_id, runner=FakeRunner(), dry_run=True)
    assert rc == 0
    text = (config.run.results_dir / run_id / "controller.log").read_text(encoding="utf-8")
    assert "run finished" in text


def test_controller_log_has_run_finished_node_non_dry_run(tmp_path):
    """非 dry-run 路径同样在 controller.log 落 run finished 节点，
    并携带 passed=N 等计数摘要。"""
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_id = "log_progress_real_001"
    rc = ab.run_controller(config, run_id, runner=FakeRunner(), dry_run=False)
    assert rc == 0
    text = (config.run.results_dir / run_id / "controller.log").read_text(encoding="utf-8")
    assert "run finished" in text
    assert "passed=" in text
