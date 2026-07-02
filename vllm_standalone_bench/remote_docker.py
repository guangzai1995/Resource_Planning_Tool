from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Collection, Sequence

from auto_bench import Completed, ConfigError
from remote_topology import RemoteHost


_REDACTED = "***"
_SENSITIVE_VALUE_FLAGS = {"--password", "--api-key"}
_SENSITIVE_ENV_TOKENS = ("PASSWORD", "SECRET", "TOKEN")


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
            raise ConfigError("password SSH auth requires sshpass in PATH")
        return (
            [resolved_sshpass, "-e", "ssh", "-o", "BatchMode=no", target],
            {"SSHPASS": password},
        )

    raise ConfigError(f"unsupported SSH auth type: {auth.type}")


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
        if arg in _SENSITIVE_VALUE_FLAGS:
            masked.append(arg)
            redact_next = True
            continue
        if any(arg.startswith(f"{flag}=") for flag in _SENSITIVE_VALUE_FLAGS):
            flag, _sep, _value = arg.partition("=")
            masked.append(f"{flag}={_REDACTED}")
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
    ) -> Completed:
        ssh_command, auth_env = build_ssh_base_command(
            host,
            sshpass_path=self.sshpass_path,
        )
        args = [*ssh_command, shlex.join(command)]
        run_env = os.environ.copy()
        run_env.update(auth_env)
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            env=run_env,
        )
        result = Completed(
            args=args,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
        if check and result.returncode != 0:
            secrets = _auth_secrets(host)
            masked_args = [*ssh_command, shlex.join(mask_command(command, secrets))]
            raise RuntimeError(
                "remote command failed "
                f"({result.returncode}) on {host.name}: "
                f"{shlex.join(mask_command(masked_args, secrets))}\n"
                f"{_redact_text(result.stderr, secrets)}"
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
            raise ConfigError("password_env SSH auth requires auth.env")
        if auth.env not in os.environ:
            raise ConfigError(f"password SSH auth environment variable is not set: {auth.env}")
        return os.environ[auth.env]
    if auth.type == "password":
        if auth.password is None:
            raise ConfigError("password SSH auth requires auth.password")
        return auth.password
    raise ConfigError(f"unsupported SSH auth type: {auth.type}")


def _auth_secrets(host: RemoteHost) -> tuple[str, ...]:
    auth = host.auth
    secrets: list[str] = []
    if auth.password:
        secrets.append(auth.password)
    if auth.env and auth.env in os.environ:
        secrets.append(os.environ[auth.env])
    return tuple(secrets)


def _mask_env_assignment(arg: str) -> str:
    key, separator, value = arg.partition("=")
    if not separator:
        return arg
    upper_key = key.upper()
    if any(token in upper_key for token in _SENSITIVE_ENV_TOKENS):
        return f"{key}={_REDACTED}"
    return f"{key}={value}"


def _redact_text(text: str, secrets: Collection[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, _REDACTED)
    return redacted
