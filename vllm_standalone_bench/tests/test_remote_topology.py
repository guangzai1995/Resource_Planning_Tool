import json

import pytest

import auto_bench as ab
from test_auto_bench import minimal_config


def write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def value_after(argv, flag):
    return argv[argv.index(flag) + 1]


def values_after(argv, flag):
    return [argv[index + 1] for index, value in enumerate(argv) if value == flag]


def labels_from(argv):
    labels = {}
    for value in values_after(argv, "--label"):
        key, separator, label_value = value.partition("=")
        if separator:
            labels[key] = label_value
    return labels


def pd_topology_config(tmp_path):
    data = minimal_config(tmp_path)
    del data["serve_profiles"]
    data["topology_profiles"] = [{
        "name": "sglang_pd_2p2d",
        "engine": "sglang",
        "mode": "pd",
        "provider": "ssh_docker",
        "hosts": {
            "p1": {
                "address": "10.0.0.11",
                "ssh_user": "bench",
                "auth": {"type": "key"},
            },
            "p2": {
                "address": "10.0.0.12",
                "ssh_user": "bench",
                "auth": {"type": "key"},
            },
            "d1": {
                "address": "10.0.0.21",
                "ssh_user": "bench",
                "auth": {"type": "key"},
            },
            "d2": {
                "address": "10.0.0.22",
                "ssh_user": "bench",
                "auth": {"type": "key"},
            },
            "router": {
                "address": "10.0.0.31",
                "ssh_user": "bench",
                "auth": {"type": "key"},
            },
        },
        "prefill": [
            {"name": "p1", "host": "p1", "port": 30000},
            {"name": "p2", "host": "p2", "port": 30000},
        ],
        "decode": [
            {"name": "d1", "host": "d1", "port": 31000},
            {"name": "d2", "host": "d2", "port": 31000},
        ],
        "frontend": {
            "kind": "router",
            "host": "router",
            "port": 8000,
        },
    }]
    return data


def test_sglang_pd_commands_render_worker_and_router_flags(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["image"] = "sglang:pd"
    topology["router_image"] = "sglang-router:offline"
    topology["network"] = "pd-net"
    topology["transfer_backend"] = "nixl"
    topology["prefill"][0]["bootstrap_port"] = 12335

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    p1 = commands["p1"].argv
    router = commands["router"].argv
    assert "sglang.launch_server" in p1
    assert value_after(p1, "--disaggregation-mode") == "prefill"
    assert value_after(p1, "--disaggregation-bootstrap-port") == "12335"
    pd_flag_index = router.index("--pd-disaggregation")
    assert router[pd_flag_index + 1] == "--prefill"
    assert "repeated" not in router
    assert values_after(router, "--prefill") == [
        "http://10.0.0.11:30000",
        "http://10.0.0.12:30000",
    ]
    assert values_after(router, "--decode") == [
        "http://10.0.0.21:31000",
        "http://10.0.0.22:31000",
    ]
    assert "12335" not in values_after(router, "--prefill")
    assert (
        commands["p1"].container_name
        == "bench-pd-run123-qwen2_5_1_5b-sglang_pd_2p2d-p1"
    )


def test_sglang_pd_commands_include_expected_labels_and_masked_argv(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["image"] = "sglang:pd"
    topology["frontend"]["args"] = ["--api-key", "router-secret"]

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    router = commands["router"]
    assert "router-secret" in router.argv
    assert "router-secret" not in router.masked_argv

    labels = labels_from(commands["p1"].argv)
    assert labels["vllm_auto_bench.managed"] == "true"
    assert labels["vllm_auto_bench.run_id"] == "run123"
    assert labels["vllm_auto_bench.model"] == "qwen2_5_1_5b"
    assert labels["vllm_auto_bench.topology_profile"] == "sglang_pd_2p2d"
    assert labels["vllm_auto_bench.role"] == "prefill"
    assert labels["vllm_auto_bench.role_name"] == "p1"


def test_sglang_pd_commands_render_disaggregation_ib_device(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["image"] = "sglang:pd"
    topology["disaggregation_ib_device"] = "mlx5_0"

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    p1 = commands["p1"].argv
    d1 = commands["d1"].argv
    router = commands["router"].argv
    assert value_after(p1, "--disaggregation-ib-device") == "mlx5_0"
    assert value_after(d1, "--disaggregation-ib-device") == "mlx5_0"
    assert "--disaggregation-ib-device" not in router


def test_vllm_pd_worker_command_renders_kv_template(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["kv_transfer_config_template"] = {
        "kv_connector": "NixlConnector",
        "kv_role": "{kv_role}",
        "kv_rank": "{kv_rank}",
        "kv_parallel_size": "{kv_parallel_size}",
        "node_name": "{node_name}",
        "node_address": "{node_address}",
        "node_port": "{node_port}",
        "run_id": "{run_id}",
    }
    topology["frontend"] = {
        "kind": "external",
        "host": "router",
        "port": 8000,
        "image": "pd-proxy:offline",
        "command": ["python", "/opt/proxy.py", "--port", "{frontend_port}", "--run", "{run_id}"],
    }

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    p1 = commands["p1"].argv
    d1 = commands["d1"].argv
    assert value_after(p1, "--entrypoint") == "vllm"

    p1_kv_config = json.loads(value_after(p1, "--kv-transfer-config"))
    assert p1_kv_config["kv_connector"] == "NixlConnector"
    assert p1_kv_config["kv_role"] == "kv_producer"
    assert str(p1_kv_config["kv_rank"]) == "0"
    assert str(p1_kv_config["kv_parallel_size"]) == "4"
    assert p1_kv_config["node_name"] == "p1"
    assert p1_kv_config["node_address"] == "10.0.0.11"
    assert str(p1_kv_config["node_port"]) == "30000"

    d1_kv_config = json.loads(value_after(d1, "--kv-transfer-config"))
    assert d1_kv_config["kv_role"] == "kv_consumer"
    assert str(d1_kv_config["kv_rank"]) == "2"

    external = commands["router"].argv
    assert "pd-proxy:offline" in external
    assert value_after(external, "--port") == "8000"
    assert value_after(external, "--run") == "run123"


def test_load_config_accepts_topology_profiles_without_serve_profiles(tmp_path):
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))

    assert config.serve_profiles == ()
    assert len(config.topology_profiles) == 1
    topology = config.topology_profiles[0]
    assert topology.name == "sglang_pd_2p2d"
    assert [node.name for node in topology.prefill] == ["p1", "p2"]
    assert topology.frontend.host == "router"


def test_topology_only_config_does_not_require_run_engine_images(tmp_path):
    data = pd_topology_config(tmp_path)
    del data["run"]["vllm_image"]
    data["run"].pop("images", None)
    data["topology_profiles"][0]["image"] = "lmsysorg/sglang:latest"
    data["topology_profiles"][0]["router_image"] = "sglang-router:offline"

    config = ab.load_config(write_config(tmp_path, data))
    cases = ab.expand_cases(config, run_id="run123")

    assert config.run.images == {}
    assert cases[0].topology_profile.name == "sglang_pd_2p2d"


def test_topology_profile_rejects_missing_host_reference(tmp_path):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["prefill"][0]["host"] = "missing"

    with pytest.raises(ab.ConfigError, match="missing"):
        ab.load_config(write_config(tmp_path, data))


def test_config_to_dict_masks_inline_password(tmp_path):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["hosts"]["p1"]["auth"] = {
        "type": "password",
        "password": "secret-for-review",
    }

    config = ab.load_config(write_config(tmp_path, data))
    resolved = ab.config_to_dict(config)

    assert "secret-for-review" not in json.dumps(resolved)
    assert (
        resolved["topology_profiles"][0]["hosts"]["p1"]["auth"].get("password")
        == "***"
    )


def test_topology_profile_name_cannot_duplicate_serve_profile(tmp_path):
    data = minimal_config(tmp_path)
    topology = pd_topology_config(tmp_path / "topology")["topology_profiles"][0]
    topology["name"] = data["serve_profiles"][0]["name"]
    data["topology_profiles"] = [topology]

    with pytest.raises(ab.ConfigError, match="duplicate|profile name"):
        ab.load_config(write_config(tmp_path, data))


def test_topology_profile_names_must_be_unique(tmp_path):
    data = pd_topology_config(tmp_path)
    duplicate = dict(data["topology_profiles"][0])
    data["topology_profiles"].append(duplicate)

    with pytest.raises(ab.ConfigError, match="duplicate"):
        ab.load_config(write_config(tmp_path, data))


def test_serve_and_topology_profiles_with_distinct_names_parse(tmp_path):
    data = minimal_config(tmp_path)
    data["topology_profiles"] = pd_topology_config(tmp_path / "topology")[
        "topology_profiles"
    ]

    config = ab.load_config(write_config(tmp_path, data))

    assert [profile.name for profile in config.serve_profiles] == ["bf16_default"]
    assert [profile.name for profile in config.topology_profiles] == [
        "sglang_pd_2p2d"
    ]


def test_password_env_requires_existing_env(tmp_path, monkeypatch):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["hosts"]["p1"]["auth"] = {
        "type": "password_env",
        "env": "MISSING_PD_PASSWORD",
    }
    monkeypatch.delenv("MISSING_PD_PASSWORD", raising=False)

    with pytest.raises(ab.ConfigError, match="MISSING_PD_PASSWORD"):
        ab.load_config(write_config(tmp_path, data))


def test_password_env_resolved_config_keeps_env_name_not_value(tmp_path, monkeypatch):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["hosts"]["p1"]["auth"] = {
        "type": "password_env",
        "env": "P1_PASSWORD",
    }
    monkeypatch.setenv("P1_PASSWORD", "secret-env-value")

    config = ab.load_config(write_config(tmp_path, data))
    resolved = ab.config_to_dict(config)
    rendered = json.dumps(resolved)

    assert "secret-env-value" not in rendered
    assert "P1_PASSWORD" in rendered
