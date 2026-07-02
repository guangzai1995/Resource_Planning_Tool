import json
import shlex
import shutil
import subprocess

import pytest

from auto_bench import ConfigError
from remote_docker import RemoteDockerRunner, build_ssh_base_command
from remote_topology import RemoteAuth, RemoteHost


def value_after(argv, flag):
    index = argv.index(flag)
    return argv[index + 1]


def test_key_auth_uses_plain_ssh_command():
    host = RemoteHost(
        "p1",
        "10.0.0.11",
        "root",
        RemoteAuth("key", key_path="/keys/id_rsa"),
    )

    cmd, env = build_ssh_base_command(host)

    assert cmd[0] == "ssh"
    assert value_after(cmd, "-i") == "/keys/id_rsa"
    assert "root@10.0.0.11" in cmd
    assert env == {}


def test_password_env_uses_sshpass_env_without_password_in_args(monkeypatch):
    monkeypatch.setenv("P1_PASSWORD", "secret-pass")
    host = RemoteHost(
        "p1",
        "10.0.0.11",
        "root",
        RemoteAuth("password_env", env="P1_PASSWORD"),
    )

    cmd, env = build_ssh_base_command(host, sshpass_path="/usr/bin/sshpass")

    assert cmd[:2] == ["/usr/bin/sshpass", "-e"]
    assert "secret-pass" not in " ".join(cmd)
    assert env["SSHPASS"] == "secret-pass"


def test_password_auth_uses_sshpass_env_without_password_in_args():
    host = RemoteHost(
        "p1",
        "10.0.0.11",
        "root",
        RemoteAuth("password", password="inline-secret"),
    )

    cmd, env = build_ssh_base_command(host, sshpass_path="/usr/bin/sshpass")

    assert cmd[:2] == ["/usr/bin/sshpass", "-e"]
    assert "inline-secret" not in " ".join(cmd)
    assert env["SSHPASS"] == "inline-secret"


def test_password_auth_requires_sshpass(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    host = RemoteHost(
        "p1",
        "10.0.0.11",
        "root",
        RemoteAuth("password", password="inline-secret"),
    )

    with pytest.raises(ConfigError, match="sshpass"):
        build_ssh_base_command(host)


def test_remote_runner_runs_ssh_command_without_password_in_args(monkeypatch):
    monkeypatch.setenv("P1_PASSWORD", "secret-pass")
    captured = {}

    def fake_run(args, *, capture_output, text, env):
        captured["args"] = args
        captured["env"] = env
        captured["capture_output"] = capture_output
        captured["text"] = text
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    host = RemoteHost(
        "p1",
        "10.0.0.11",
        "root",
        RemoteAuth("password_env", env="P1_PASSWORD"),
    )

    result = RemoteDockerRunner(sshpass_path="/usr/bin/sshpass").run(
        host,
        ["docker", "ps"],
    )

    assert result.returncode == 0
    assert captured["args"][-1] == shlex.join(["docker", "ps"])
    assert "secret-pass" not in " ".join(captured["args"])
    assert captured["env"]["SSHPASS"] == "secret-pass"
    assert captured["capture_output"] is True
    assert captured["text"] is True


def test_remote_runner_check_failure_masks_password(monkeypatch):
    monkeypatch.setenv("P1_PASSWORD", "secret-pass")

    def fake_run(args, *, capture_output, text, env):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="failed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    host = RemoteHost(
        "p1",
        "10.0.0.11",
        "root",
        RemoteAuth("password_env", env="P1_PASSWORD"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        RemoteDockerRunner(sshpass_path="/usr/bin/sshpass").run(
            host,
            ["docker", "ps"],
            check=True,
        )

    assert "secret-pass" not in str(exc_info.value)


def test_inspect_labels_parses_remote_docker_labels(monkeypatch):
    labels = {"bench.role": "prefill", "bench.run_id": "run123"}

    def fake_run(args, *, capture_output, text, env):
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps(labels),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("key"))

    assert RemoteDockerRunner().inspect_labels(host, "bench-p1") == labels
