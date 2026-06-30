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


def values_after(cmd, flag):
    return [cmd[index + 1] for index, value in enumerate(cmd) if value == flag]


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


def test_cleanup_network_warns_when_external_containers_connected(tmp_path, capsys):
    class ConnectedNetworkRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            if args[:4] == ["docker", "inspect", "--format", "{{json .Containers}}"]:
                self.commands.append(list(args))
                return ab.Completed(list(args), 0, json.dumps({"external": {}}), "")
            return super().run(args, check=check, capture=capture, text=text, stdout=stdout, stderr=stderr)

    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = ConnectedNetworkRunner()

    stop_requested = ab.cleanup_network(config, runner, owned=True, dry_run=False, run_id="run123")

    captured = capsys.readouterr()
    assert stop_requested is False
    assert not any(command[:3] == ["docker", "network", "rm"] for command in runner.commands)
    assert "warning" in captured.err.lower()
    assert "external" in captured.err.lower()


def test_cleanup_network_warns_when_network_rm_fails(tmp_path, capsys):
    class FailingNetworkRmRunner(FakeRunner):
        def run(self, args, *, check=False, capture=True, text=True, stdout=None, stderr=None):
            if args[:3] == ["docker", "network", "rm"]:
                self.commands.append(list(args))
                return ab.Completed(list(args), 1, "", "network busy")
            return super().run(args, check=check, capture=capture, text=text, stdout=stdout, stderr=stderr)

    config = ab.load_config(write_config(tmp_path, minimal_config(tmp_path)))
    runner = FailingNetworkRmRunner()

    stop_requested = ab.cleanup_network(config, runner, owned=True, dry_run=False, run_id="run123")

    captured = capsys.readouterr()
    assert stop_requested is False
    assert any(command[:3] == ["docker", "network", "rm"] for command in runner.commands)
    assert "warning" in captured.err.lower()
    assert "network busy" in captured.err


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


def test_start_detached_rejects_existing_active_run_dir(tmp_path, monkeypatch, capsys):
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

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert state["status"] == "running"
    assert "active" in captured.err


def test_run_controller_rejects_existing_active_run_dir(tmp_path, monkeypatch, capsys):
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

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert state["status"] == "running"
    assert "active" in captured.err
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_run_controller_rejects_active_state_without_pid_file(tmp_path, capsys):
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

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert state["status"] == "running"
    assert "active" in captured.err
    assert not (run_dir / "config.resolved.json").exists()
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_run_controller_rejects_active_state_with_invalid_pid_file(tmp_path, capsys):
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

    captured = capsys.readouterr()
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert state["status"] == "starting"
    assert "active" in captured.err
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_start_detached_rejects_existing_run_lock(tmp_path, monkeypatch, capsys):
    config_path = write_config(tmp_path, minimal_config(tmp_path))
    config = ab.load_config(config_path)
    run_dir = tmp_path / "results" / "run123"
    run_dir.mkdir(parents=True)
    (run_dir / ".run.lock").write_text("locked\n", encoding="utf-8")

    def fail_popen(*args, **kwargs):
        raise AssertionError("locked run should not start another controller")

    monkeypatch.setattr(ab.subprocess, "Popen", fail_popen)

    exit_code = ab.start_detached(config_path, config, "run123")

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "active" in captured.err
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
    capsys,
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

    captured = capsys.readouterr()
    assert result == 1
    assert "active" in captured.err
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


def test_run_controller_keeps_active_dead_pid_lock_fail_closed(tmp_path, monkeypatch, capsys):
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

    captured = capsys.readouterr()
    assert result == 1
    assert "active" in captured.err
    assert (run_dir / ".run.lock").exists()
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_run_controller_corrupt_lock_fail_closed(tmp_path, capsys):
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

    captured = capsys.readouterr()
    assert result == 1
    assert "active" in captured.err
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
                assert "0.90" in profile["args"]
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
            assert "0.90" in smoke_args


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
