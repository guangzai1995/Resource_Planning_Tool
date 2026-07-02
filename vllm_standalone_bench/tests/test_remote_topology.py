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
            "prefill-a": {
                "address": "10.0.0.11",
                "ssh_user": "bench",
                "auth": {"type": "key"},
            },
            "prefill-b": {
                "address": "10.0.0.12",
                "ssh_user": "bench",
                "auth": {"type": "key"},
            },
            "decode-a": {
                "address": "10.0.0.21",
                "ssh_user": "bench",
                "auth": {"type": "key"},
            },
            "decode-b": {
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
            {"name": "p1", "host": "prefill-a", "port": 30000},
            {"name": "p2", "host": "prefill-b", "port": 30000},
        ],
        "decode": [
            {"name": "d1", "host": "decode-a", "port": 31000},
            {"name": "d2", "host": "decode-b", "port": 31000},
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
