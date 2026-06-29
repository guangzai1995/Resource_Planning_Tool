import json
import math
import sys
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


def ready_probe_commands(commands):
    return [
        command for command in commands
        if command[:3] == ["docker", "run", "--rm"]
        and "-c" in command
        and "run_bench_multi.py" not in command
    ]


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
    assert value_after(probes[0], "--network") == "vllm-bench-net"
    assert "vllm-bench-runner:offline" in probes[0]
    assert "http://bench-vllm-qwen2_5_1_5b-bf16_default-run123:8000/v1/models" in probes[0]


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


def test_controller_stop_during_network_create_attempts_network_cleanup(tmp_path):
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
    assert any("docker network create vllm-bench-net" in cmd for cmd in joined)
    assert any("docker network rm vllm-bench-net" in cmd for cmd in joined)


def test_controller_dry_run_prints_commands_without_result_files(tmp_path, capsys):
    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FakeRunner()

    result = ab.run_controller(config, run_id="run123", runner=runner, dry_run=True)

    out = capsys.readouterr().out
    run_dir = tmp_path / "results" / "run123"
    assert result == 0
    assert "docker network create vllm-bench-net" in out
    assert "docker run -d" in out
    assert "run_bench_multi.py" in out
    assert not (run_dir / "config.resolved.json").exists()
    assert not (run_dir / "manifest.json").exists()
    assert not (run_dir / "state.json").exists()
    assert not (
        run_dir / "qwen2_5_1_5b" / "bf16_default" / "smoke" / "status.json"
    ).exists()
    assert runner.commands == []


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
    cmd = ab.build_detach_command(config_path, run_id)

    assert cmd[:2] == [sys.executable, str(Path(ab.__file__).resolve())]
    assert cmd[2:] == [
        "run", "--config", str(config_path),
        "--run-id", run_id,
        "--child",
    ]


def test_status_reads_state_file(tmp_path, capsys):
    run_dir = tmp_path / "results" / "run123"
    ab.write_state(run_dir, {
        "run_id": "run123",
        "status": "running",
        "current": {"model": "m", "serve_profile": "s", "bench_profile": "b"},
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })

    exit_code = ab.print_status(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "running" in captured.out
    assert "m/s/b" in captured.out


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


def test_logs_replace_invalid_utf8(tmp_path, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.log").write_bytes(b"ok\n\xff\n")

    exit_code = ab.main(["logs", "--results-dir", str(tmp_path), "--run-id", "run123"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ok" in captured.out
    assert "\ufffd" in captured.out


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


def write_stop_state(run_dir, status="running"):
    ab.write_state(run_dir, {
        "run_id": run_dir.name,
        "status": status,
        "current": None,
        "counts": {"passed": 0, "failed": 0, "skipped": 0, "running": 1, "total": 1},
    })


def matching_controller_cmdline(run_id="run123"):
    return [
        sys.executable,
        str(Path(ab.__file__).resolve()),
        "run",
        "--config",
        "config.json",
        "--run-id",
        run_id,
        "--child",
    ]


def test_stop_run_sends_sigterm(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    (run_dir / "controller.pid").write_text("12345\n", encoding="utf-8")
    write_stop_state(run_dir)
    signals = []

    def fake_kill(pid, sig):
        signals.append((pid, sig))

    monkeypatch.setattr(ab.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(
        ab,
        "read_process_cmdline",
        lambda pid: matching_controller_cmdline(run_dir.name),
        raising=False,
    )

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert signals == [(12345, ab.signal.SIGTERM)]
    assert "12345" in captured.out


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

    def missing_process(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(ab.os, "kill", missing_process, raising=False)
    monkeypatch.setattr(
        ab,
        "read_process_cmdline",
        lambda pid: matching_controller_cmdline(run_dir.name),
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

    def fail_stop(pid, sig):
        raise OSError("denied")

    monkeypatch.setattr(ab.os, "kill", fail_stop, raising=False)
    monkeypatch.setattr(
        ab,
        "read_process_cmdline",
        lambda pid: matching_controller_cmdline(run_dir.name),
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

    def fail_if_called(pid, sig):
        raise AssertionError("os.kill should not be called for mismatched run id")

    malicious_cmdline = matching_controller_cmdline("other") + ["--config", "run123"]
    monkeypatch.setattr(ab.os, "kill", fail_if_called, raising=False)
    monkeypatch.setattr(ab, "read_process_cmdline", lambda pid: malicious_cmdline, raising=False)

    exit_code = ab.stop_run(run_dir)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "process does not match controller" in captured.err or "stale pid" in captured.err


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
