import json
from pathlib import Path

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


def kv_config_after(argv):
    return json.loads(value_after(argv, "--kv-transfer-config"))


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
            {"name": "p1", "host": "p1", "port": 30000, "bootstrap_port": 12335},
            {"name": "p2", "host": "p2", "port": 30000, "bootstrap_port": 12336},
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
    prefill_positions = [
        index for index, value in enumerate(router) if value == "--prefill"
    ]
    assert [
        router[position + 1:position + 3]
        for position in prefill_positions
    ] == [
        ("http://10.0.0.11:30000", "12335"),
        ("http://10.0.0.12:30000", "12336"),
    ]
    assert values_after(router, "--decode") == [
        "http://10.0.0.21:31000",
        "http://10.0.0.22:31000",
    ]
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


def test_sglang_pd_rejects_prefill_without_bootstrap_port(tmp_path):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["prefill"][0].pop("bootstrap_port")

    with pytest.raises(ab.ConfigError, match="prefill.*bootstrap_port"):
        ab.load_config(write_config(tmp_path, data))


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


def test_vllm_pd_p2p_rejects_missing_kv_port(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {"connector": "p2p_nccl", "proxy": {"kind": "builtin"}}
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}

    with pytest.raises(ab.ConfigError, match="kv_port"):
        ab.load_config(write_config(tmp_path, data))


def test_vllm_pd_nixl_rejects_missing_side_channel_port(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {"connector": "nixl", "proxy": {"kind": "builtin"}}
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}

    with pytest.raises(ab.ConfigError, match="side_channel_port"):
        ab.load_config(write_config(tmp_path, data))


def test_vllm_pd_rejects_unknown_structured_key(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {
        "connector": "p2p_nccl",
        "proxy": {"kind": "builtin"},
        "unknown": True,
    }
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}

    with pytest.raises(ab.ConfigError, match="unknown"):
        ab.load_config(write_config(tmp_path, data))


def test_vllm_pd_accepts_normalized_proxy_kind_from_resume_config(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {
        "connector": "p2p_nccl",
        "proxy_kind": "builtin",
        "p2p_send_type": "PUT_ASYNC",
        "nccl_num_channels": 16,
    }
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}
    topology["prefill"][0]["kv_port"] = 21001
    topology["prefill"][1]["kv_port"] = 21002
    topology["decode"][0]["kv_port"] = 22001
    topology["decode"][1]["kv_port"] = 22002

    config = ab.load_config(write_config(tmp_path, data))

    assert config.topology_profiles[0].vllm_pd.proxy_kind == "builtin"


def test_vllm_pd_p2p_commands_render_structured_builtin_proxy(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {
        "connector": "p2p_nccl",
        "proxy": {"kind": "builtin"},
        "p2p_send_type": "PUT_ASYNC",
        "nccl_num_channels": 16,
    }
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}
    topology["prefill"][0]["gpus"] = "0,1,2,3"
    topology["prefill"][0]["kv_port"] = 21001
    topology["prefill"][1]["kv_port"] = 21002
    topology["decode"][0]["kv_port"] = 22001
    topology["decode"][1]["kv_port"] = 22002

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    p1_config = kv_config_after(commands["p1"].argv)
    assert value_after(commands["p1"].argv, "--gpus") == '"device=0,1,2,3"'
    assert p1_config["kv_connector"] == "P2pNcclConnector"
    assert p1_config["kv_role"] == "kv_producer"
    assert p1_config["kv_port"] == 21001
    assert "kv_rank" not in p1_config
    assert "kv_parallel_size" not in p1_config
    assert p1_config["kv_connector_extra_config"] == {
        "http_port": 30000,
        "send_type": "PUT_ASYNC",
        "nccl_num_channels": 16,
    }

    d1_config = kv_config_after(commands["d1"].argv)
    assert d1_config["kv_connector"] == "P2pNcclConnector"
    assert d1_config["kv_role"] == "kv_consumer"
    assert d1_config["kv_port"] == 22001

    frontend = commands["router"].argv
    assert "vllm_bench.pd_proxy" in frontend
    assert value_after(frontend, "--connector") == "p2p_nccl"
    assert value_after(frontend, "--port") == "8000"
    assert len(values_after(frontend, "--prefill")) == 2
    assert len(values_after(frontend, "--decode")) == 2
    first_prefill = json.loads(values_after(frontend, "--prefill")[0])
    assert first_prefill == {
        "name": "p1",
        "url": "http://10.0.0.11:30000",
        "kv_address": "10.0.0.11:21001",
    }


def test_vllm_pd_nixl_commands_render_side_channel_env(tmp_path):
    data = pd_topology_config(tmp_path)
    topology = data["topology_profiles"][0]
    topology["engine"] = "vllm"
    topology["image"] = "vllm:pd"
    topology["vllm_pd"] = {"connector": "nixl", "proxy": {"kind": "builtin"}}
    topology["frontend"] = {"kind": "builtin", "host": "router", "port": 8000}
    topology["prefill"][0]["side_channel_port"] = 5601
    topology["prefill"][1]["side_channel_port"] = 5602
    topology["decode"][0]["side_channel_port"] = 5701
    topology["decode"][1]["side_channel_port"] = 5702

    config = ab.load_config(write_config(tmp_path, data))
    case = ab.expand_cases(config, run_id="run123")[0]
    commands = case.topology_profile.build_commands(config, case, tmp_path / "run123")

    p1_config = kv_config_after(commands["p1"].argv)
    assert p1_config == {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer",
    }
    p1_env = values_after(commands["p1"].argv, "-e")
    assert "VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.11" in p1_env
    assert "VLLM_NIXL_SIDE_CHANNEL_PORT=5601" in p1_env

    d1_config = kv_config_after(commands["d1"].argv)
    assert d1_config == {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer",
    }
    frontend = commands["router"].argv
    first_prefill = json.loads(values_after(frontend, "--prefill")[0])
    assert first_prefill == {
        "name": "p1",
        "url": "http://10.0.0.11:30000",
    }


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


def test_topology_profile_rejects_duplicate_prefill_decode_role_names(tmp_path):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["decode"][0]["name"] = "p1"

    with pytest.raises(ab.ConfigError) as exc_info:
        ab.load_config(write_config(tmp_path, data))

    message = str(exc_info.value)
    assert "topology_profiles[0]" in message
    assert "sglang_pd_2p2d" in message
    assert "p1" in message
    assert "duplicate" in message or "unique" in message


def test_topology_profile_rejects_frontend_role_name_collision(tmp_path):
    data = pd_topology_config(tmp_path)
    data["topology_profiles"][0]["prefill"][0]["name"] = "router"

    with pytest.raises(ab.ConfigError) as exc_info:
        ab.load_config(write_config(tmp_path, data))

    message = str(exc_info.value)
    assert "topology_profiles[0]" in message
    assert "sglang_pd_2p2d" in message
    assert "router" in message
    assert "duplicate" in message or "unique" in message


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


@pytest.mark.parametrize("filename", [
    "auto_bench.vllm_pd_p2p_remote.example.json",
    "auto_bench.vllm_pd_nixl_remote.example.json",
])
def test_vllm_pd_example_configs_load(filename):
    config = ab.load_config(
        Path("vllm_standalone_bench/configs") / filename
    )

    assert config.topology_profiles
    assert all(profile.engine == "vllm" for profile in config.topology_profiles)
    assert all(profile.vllm_pd is not None for profile in config.topology_profiles)


def test_minimax_p2p_compare_config_expands_all_profiles():
    config = ab.load_config(
        Path("vllm_standalone_bench/configs")
        / "auto_bench.vllm_pd_p2p_remote.example.json"
    )
    cases = ab.expand_cases(config, run_id="minimax_compare")

    assert [profile.name for profile in config.serve_profiles] == [
        "minimax_tp8_single",
    ]
    assert [profile.name for profile in config.topology_profiles] == [
        "vllm_pd_p2p_minimax_m27_2p2d",
        "vllm_pd_p2p_minimax_m27_3p1d",
        "vllm_pd_p2p_minimax_m27_1p3d",
    ]
    assert [case.serve_profile.name for case in cases if case.serve_profile] == [
        "minimax_tp8_single",
    ]
    assert [
        case.topology_profile.name for case in cases if case.topology_profile
    ] == [
        "vllm_pd_p2p_minimax_m27_2p2d",
        "vllm_pd_p2p_minimax_m27_3p1d",
        "vllm_pd_p2p_minimax_m27_1p3d",
    ]


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
