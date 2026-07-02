import json
import shutil
import subprocess

import pytest

from auto_bench import ConfigError
from remote_docker import (
    RemoteDockerRunner,
    RemoteResourceReaders,
    build_ssh_base_command,
    mask_command,
)
from remote_topology import RemoteAuth, RemoteHost
from resource_monitor import NVIDIA_SMI_QUERY


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


def test_remote_resource_readers_capture_host_proc_and_gpu_commands():
    calls = []

    class FakeRunner:
        def capture(self, host, command):
            calls.append((host, list(command)))
            return f"value-{len(calls)}"

    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("key"))
    readers = RemoteResourceReaders(FakeRunner(), host)

    assert readers.proc_stat() == "value-1"
    assert readers.meminfo() == "value-2"
    assert readers.net_dev() == "value-3"
    assert readers.diskstats() == "value-4"
    assert readers.nvidia_smi() == "value-5"
    assert [call[0] for call in calls] == [host, host, host, host, host]
    assert [call[1] for call in calls] == [
        ["cat", "/proc/stat"],
        ["cat", "/proc/meminfo"],
        ["cat", "/proc/net/dev"],
        ["cat", "/proc/diskstats"],
        NVIDIA_SMI_QUERY,
    ]


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

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        captured["capture_output"] = kwargs["capture_output"]
        captured["input"] = kwargs.get("input")
        captured["text"] = kwargs["text"]
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
    assert captured["input"] == "exec docker ps\n"
    assert captured["args"][-2:] == ["sh", "-s"]
    assert "secret-pass" not in " ".join(captured["args"])
    assert captured["env"]["SSHPASS"] == "secret-pass"
    assert captured["capture_output"] is True
    assert captured["text"] is True


def test_remote_runner_does_not_put_api_key_in_process_args_or_completed_args(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["input"] = kwargs.get("input")
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("key"))

    result = RemoteDockerRunner().run(
        host,
        ["python", "serve.py", "--api-key", "api-secret"],
    )

    assert "api-secret" not in " ".join(captured["args"])
    assert "api-secret" not in " ".join(result.args)
    assert "api-secret" in captured["input"]
    assert captured["args"][-2:] == ["sh", "-s"]


def test_mask_command_redacts_flags_and_sensitive_env_assignments():
    masked = mask_command([
        "python",
        "serve.py",
        "--api-key",
        "api-secret",
        "--api-key=equal-api-secret",
        "--password",
        "password-secret",
        "--password=inline-secret",
        "--openai-api-key",
        "openai-secret",
        "--db-password=database-secret",
        "--service-token",
        "service-token-secret",
        "--tokenizer-path",
        "/models/tokenizer",
        "--tokenizer-path=/models/inline-tokenizer",
        "--host-tokenizer-path=/host/tokenizer",
        "--key-path=/keys/id_rsa",
        "OPENAI_API_KEY=env-secret",
        "PASSWORD=password-secret",
        "TOKEN=token-secret",
        "SECRET=secret-secret",
    ])

    rendered = " ".join(masked)
    assert "api-secret" not in rendered
    assert "equal-api-secret" not in rendered
    assert "password-secret" not in rendered
    assert "inline-secret" not in rendered
    assert "openai-secret" not in rendered
    assert "database-secret" not in rendered
    assert "service-token-secret" not in rendered
    assert "env-secret" not in rendered
    assert "token-secret" not in rendered
    assert "secret-secret" not in rendered
    assert masked == [
        "python",
        "serve.py",
        "--api-key",
        "***",
        "--api-key=***",
        "--password",
        "***",
        "--password=***",
        "--openai-api-key",
        "***",
        "--db-password=***",
        "--service-token",
        "***",
        "--tokenizer-path",
        "/models/tokenizer",
        "--tokenizer-path=/models/inline-tokenizer",
        "--host-tokenizer-path=/host/tokenizer",
        "--key-path=/keys/id_rsa",
        "OPENAI_API_KEY=***",
        "PASSWORD=***",
        "TOKEN=***",
        "SECRET=***",
    ]


def test_remote_runner_check_failure_masks_password(monkeypatch):
    monkeypatch.setenv("P1_PASSWORD", "secret-pass")

    def fake_run(args, **kwargs):
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


def test_remote_runner_check_failure_redacts_command_secret_from_error(monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            1,
            stdout="api-secret stdout",
            stderr="api-secret failed",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("key"))

    with pytest.raises(RuntimeError) as exc_info:
        RemoteDockerRunner().run(
            host,
            [
                "python",
                "serve.py",
                "--api-key",
                "api-secret",
                "--tokenizer-path=/models/tokenizer",
            ],
            check=True,
        )

    message = str(exc_info.value)
    assert "api-secret" not in message
    assert "--api-key ***" in message
    assert "--tokenizer-path=/models/tokenizer" in message


def test_remote_runner_supports_stdout_stderr_streams(monkeypatch):
    captured = {}
    stdout_stream = object()
    stderr_stream = object()

    def fake_run(args, **kwargs):
        captured["capture_output"] = kwargs["capture_output"]
        captured["stdout"] = kwargs["stdout"]
        captured["stderr"] = kwargs["stderr"]
        return subprocess.CompletedProcess(args, 0, stdout=None, stderr=None)

    monkeypatch.setattr(subprocess, "run", fake_run)
    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("key"))

    RemoteDockerRunner().run(
        host,
        ["docker", "ps"],
        stdout=stdout_stream,
        stderr=stderr_stream,
    )

    assert captured["capture_output"] is False
    assert captured["stdout"] is stdout_stream
    assert captured["stderr"] is stderr_stream


def test_remote_runner_uses_bytes_input_when_text_false(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["input"] = kwargs["input"]
        captured["text"] = kwargs["text"]
        return subprocess.CompletedProcess(args, 0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("key"))

    result = RemoteDockerRunner().run(host, ["docker", "ps"], text=False)

    assert captured["input"] == b"exec docker ps\n"
    assert captured["text"] is False
    assert result.stdout == b"ok"


def test_inspect_labels_parses_remote_docker_labels(monkeypatch):
    labels = {"bench.role": "prefill", "bench.run_id": "run123"}

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps(labels),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("key"))

    assert RemoteDockerRunner().inspect_labels(host, "bench-p1") == labels


def test_inspect_labels_nonzero_returns_none(monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="missing")

    monkeypatch.setattr(subprocess, "run", fake_run)
    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("key"))

    assert RemoteDockerRunner().inspect_labels(host, "bench-p1") is None


@pytest.mark.parametrize("payload", ["not json", "null", "[]", '"label"'])
def test_inspect_labels_invalid_or_non_dict_payload_returns_empty_dict(
    monkeypatch,
    payload,
):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    host = RemoteHost("p1", "10.0.0.11", "root", RemoteAuth("key"))

    assert RemoteDockerRunner().inspect_labels(host, "bench-p1") == {}
