import json

import pytest

import auto_bench as ab
from test_auto_bench import minimal_config


def write_config(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


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


def test_load_config_accepts_topology_profiles_without_serve_profiles(tmp_path):
    config = ab.load_config(write_config(tmp_path, pd_topology_config(tmp_path)))

    assert config.serve_profiles == ()
    assert len(config.topology_profiles) == 1
    topology = config.topology_profiles[0]
    assert topology.name == "sglang_pd_2p2d"
    assert [node.name for node in topology.prefill] == ["p1", "p2"]
    assert topology.frontend.host == "router"


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
