from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Collection, Sequence
from typing import Any

from remote_topology import RemoteHost
from resource_monitor import NVIDIA_SMI_QUERY


_REDACTED = "***"
_SENSITIVE_ENV_TOKENS = ("API_KEY", "PASSWORD", "SECRET", "TOKEN")


class RemoteResourceReaders:
    def __init__(self, runner, host):
        self.runner = runner
        self.host = host

    def proc_stat(self):
        return self.runner.capture(self.host, ["cat", "/proc/stat"])

    def meminfo(self):
        return self.runner.capture(self.host, ["cat", "/proc/meminfo"])

    def net_dev(self):
        return self.runner.capture(self.host, ["cat", "/proc/net/dev"])

    def diskstats(self):
        return self.runner.capture(self.host, ["cat", "/proc/diskstats"])

    def nvidia_smi(self):
        return self.runner.capture(self.host, NVIDIA_SMI_QUERY)


def _is_sensitive_value_flag(arg: str) -> bool:
    if not arg.startswith("--"):
        return False
    flag = arg[2:].partition("=")[0].lower()
    if "tokenizer" in flag:
        return False
    if "api-key" in flag or "api_key" in flag:
        return True
    parts = [part for part in re.split(r"[^a-z0-9]+", flag) if part]
    return any(part in {"password", "secret", "token"} for part in parts)


def build_ssh_base_command(
    host: RemoteHost,
    sshpass_path: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    auth = host.auth
    target = f"{host.ssh_user}@{host.address}"
    if auth.type == "key":
        command = ["ssh", "-o", "BatchMode=yes"]
        if auth.key_path:
            command.extend(["-i", auth.key_path])
        command.append(target)
        return command, {}

    if auth.type in {"password_env", "password"}:
        password = _resolve_password(host)
        resolved_sshpass = sshpass_path or shutil.which("sshpass")
        if resolved_sshpass is None:
            raise _config_error("password SSH auth requires sshpass in PATH")
        return (
            [resolved_sshpass, "-e", "ssh", "-o", "BatchMode=no", target],
            {"SSHPASS": password},
        )

    raise _config_error(f"unsupported SSH auth type: {auth.type}")


def mask_command(
    argv: Sequence[str],
    secrets: Collection[str] = (),
) -> list[str]:
    masked: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            masked.append(_REDACTED)
            redact_next = False
            continue
        if _is_sensitive_value_flag(arg):
            flag, _sep, _value = arg.partition("=")
            if _sep:
                masked.append(f"{flag}={_REDACTED}")
            else:
                masked.append(arg)
                redact_next = True
            continue
        if arg.startswith("-") and "=" in arg:
            masked.append(arg)
            continue
        masked.append(_mask_env_assignment(arg))

    for secret in secrets:
        if not secret:
            continue
        masked = [arg.replace(secret, _REDACTED) for arg in masked]
    return masked


class RemoteDockerRunner:
    def __init__(self, *, sshpass_path: str | None = None) -> None:
        self.sshpass_path = sshpass_path

    def run(
        self,
        host: RemoteHost,
        command: list[str],
        *,
        check: bool = False,
        capture: bool = True,
        text: bool = True,
        stdout: Any = None,
        stderr: Any = None,
    ) -> "Completed":
        ssh_command, auth_env = build_ssh_base_command(
            host,
            sshpass_path=self.sshpass_path,
        )
        args = [*ssh_command, "sh", "-s"]
        script = f"exec {shlex.join(command)}\n"
        input_data: str | bytes = script if text else script.encode()
        run_env = os.environ.copy()
        run_env.update(auth_env)
        capture_output = capture if stdout is None and stderr is None else False
        completed = subprocess.run(
            args,
            check=False,
            capture_output=capture_output,
            text=text,
            input=input_data,
            env=run_env,
            stdout=stdout,
            stderr=stderr,
        )
        secrets = (*_auth_secrets(host), *_command_secrets(command))
        result = _completed(
            args=mask_command(args, secrets),
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
        if check and result.returncode != 0:
            ssh_display = shlex.join(mask_command(result.args, secrets))
            remote_display = " ".join(mask_command(command, secrets))
            stdout_text = _redact_text(result.stdout, secrets)
            stderr_text = _redact_text(result.stderr, secrets)
            details = [
                "remote command failed "
                f"({result.returncode}) on {host.name}: "
                f"{ssh_display} << {remote_display}"
            ]
            if stdout_text:
                details.append(stdout_text)
            if stderr_text:
                details.append(stderr_text)
            raise RuntimeError(
                "\n".join(details)
            )
        return result

    def capture(self, host: RemoteHost, command: list[str]) -> str:
        return self.run(host, command, check=True).stdout

    def inspect_labels(
        self,
        host: RemoteHost,
        container_name: str,
    ) -> dict[str, str] | None:
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
        try:
            labels = json.loads(result.stdout.strip() or "null")
        except json.JSONDecodeError:
            return {}
        if not isinstance(labels, dict):
            return {}
        return {str(key): str(value) for key, value in labels.items()}


def _resolve_password(host: RemoteHost) -> str:
    auth = host.auth
    if auth.type == "password_env":
        if auth.env is None:
            raise _config_error("password_env SSH auth requires auth.env")
        if auth.env not in os.environ:
            raise _config_error(f"password SSH auth environment variable is not set: {auth.env}")
        return os.environ[auth.env]
    if auth.type == "password":
        if auth.password is None:
            raise _config_error("password SSH auth requires auth.password")
        return auth.password
    raise _config_error(f"unsupported SSH auth type: {auth.type}")


def _auth_secrets(host: RemoteHost) -> tuple[str, ...]:
    auth = host.auth
    secrets: list[str] = []
    if auth.password:
        secrets.append(auth.password)
    if auth.env and auth.env in os.environ:
        secrets.append(os.environ[auth.env])
    return tuple(secrets)


def _command_secrets(command: Sequence[str]) -> tuple[str, ...]:
    secrets: list[str] = []
    redact_next = False
    for arg in command:
        if redact_next:
            if arg:
                secrets.append(arg)
            redact_next = False
            continue
        if _is_sensitive_value_flag(arg):
            _flag, separator, value = arg.partition("=")
            if not separator:
                redact_next = True
                continue
            if value:
                secrets.append(value)
            continue
        if arg.startswith("-") and "=" in arg:
            continue
        key, separator, value = arg.partition("=")
        if separator and value and _is_sensitive_env_key(key):
            secrets.append(value)
    return tuple(secrets)


def _mask_env_assignment(arg: str) -> str:
    key, separator, value = arg.partition("=")
    if not separator:
        return arg
    if _is_sensitive_env_key(key):
        return f"{key}={_REDACTED}"
    return f"{key}={value}"


def _is_sensitive_env_key(key: str) -> bool:
    upper_key = key.upper()
    return any(token in upper_key for token in _SENSITIVE_ENV_TOKENS)


def _redact_text(value: Any, secrets: Collection[str]) -> str:
    if value is None:
        redacted = ""
    elif isinstance(value, bytes):
        redacted = value.decode(errors="replace")
    else:
        redacted = str(value)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, _REDACTED)
    return redacted


def _completed(
    *,
    args: list[str],
    returncode: int,
    stdout: Any = "",
    stderr: Any = "",
) -> "Completed":
    from auto_bench import Completed

    return Completed(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _config_error(message: str) -> Exception:
    from auto_bench import ConfigError

    return ConfigError(message)
