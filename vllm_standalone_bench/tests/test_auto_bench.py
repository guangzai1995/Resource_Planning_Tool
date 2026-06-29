import json
import math
from pathlib import Path

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


def test_build_bench_command_targets_container_dns(tmp_path):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    case = ab.expand_cases(config, run_id="run123")[0]
    bench_dir = Path("relative-results") / "run123" / "qwen2_5_1_5b" / "bf16_default" / "smoke"

    cmd = ab.build_bench_run_command(config, case, bench_dir)

    mounts = [cmd[index + 1] for index, value in enumerate(cmd) if value == "-v"]
    assert "--network" in cmd
    assert "vllm-bench-net" in cmd
    assert f"{bench_dir.resolve()}:/results" in mounts
    assert "--base-url" in cmd
    assert f"http://{case.container_name}:8000/v1" in cmd
    assert "--model" in cmd
    assert "qwen2_5_1_5b" in cmd
    assert value_after(cmd, "--model") == "qwen2_5_1_5b"
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


class FakeRunner:
    def __init__(self, failures=None):
        self.commands = []
        self.failures = failures or {}

    def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
        self.commands.append(list(args))
        key = " ".join(args[:3])
        if key in self.failures:
            return ab.Completed(list(args), self.failures[key], "", "forced failure")
        if args[:3] == ["docker", "network", "inspect"]:
            return ab.Completed(list(args), 1, "", "not found")
        if args[:3] == ["docker", "network", "create"]:
            return ab.Completed(list(args), 0, "network-id\n", "")
        if args[:3] == ["docker", "run", "-d"]:
            return ab.Completed(list(args), 0, "container-id\n", "")
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
        return ab.Completed(list(args), 0, "ok\n", "")


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


def test_controller_runs_case_and_cleans_owned_network(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: True)

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    assert result == 0
    assert any("docker network create vllm-bench-net" in cmd for cmd in joined)
    assert any("docker run -d" in cmd for cmd in joined)
    assert any("run_bench_multi.py" in cmd for cmd in joined)
    assert any("docker stop bench-vllm-qwen2_5_1_5b-bf16_default-run123" in cmd for cmd in joined)
    assert_removed_after_stop(runner.commands, "bench-vllm-qwen2_5_1_5b-bf16_default-run123")
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_controller_skips_bench_when_vllm_not_ready(tmp_path, monkeypatch):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()
    monkeypatch.setattr(ab, "wait_for_ready", lambda *args, **kwargs: False)

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

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=False)

    joined = [" ".join(cmd) for cmd in runner.commands]
    assert result == 1
    assert any("docker stop bench-vllm-qwen2_5_1_5b-bf16_default-run123" in cmd for cmd in joined)
    assert_removed_after_stop(runner.commands, "bench-vllm-qwen2_5_1_5b-bf16_default-run123")
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


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
