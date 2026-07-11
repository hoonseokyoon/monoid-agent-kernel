from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.reference.dbos import (
    DbosControlConfig,
    DbosControlEnvelope,
    DbosControlPlane,
    DbosDependencyError,
    DbosProcessOwnershipError,
)
from monoid_agent_kernel.reference.dbos.control_plane import (
    DbosDependencyError as LegacyDbosDependencyError,
)
from monoid_agent_kernel.reference.dbos.control_plane import (
    DbosProcessOwnershipError as LegacyDbosProcessOwnershipError,
)


def test_dbos_reference_models_import_without_loading_optional_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    code = """
import sys
from monoid_agent_kernel.reference.dbos import DbosControlEnvelope
assert DbosControlEnvelope
assert 'dbos' not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_control_plane_exception_module_paths_remain_compatible() -> None:
    assert LegacyDbosDependencyError is DbosDependencyError
    assert LegacyDbosProcessOwnershipError is DbosProcessOwnershipError


def test_dbos_envelope_removes_bearer_and_keeps_retry_identity_stable() -> None:
    first_token = "first-signed-bearer"
    second_token = "rotated-signed-bearer"

    first = DbosControlEnvelope.from_control_command(
        ControlCommand(
            type="resume",
            run_id="run_1",
            command_id="cmd_resume",
            args={"token": first_token, "password": first_token, "safe": "visible"},
            issuer=f"operator-{first_token}",
            reason=f"recover with {first_token}",
        ),
        tenant_id=f"tenant-{first_token}",
        user_id=f"user-{first_token}",
    )
    retry = DbosControlEnvelope.from_control_command(
        ControlCommand(
            type="resume",
            run_id="run_1",
            command_id="cmd_resume",
            args={"token": second_token, "password": second_token, "safe": "visible"},
            issuer=f"operator-{second_token}",
            reason=f"recover with {second_token}",
        ),
        tenant_id=f"tenant-{second_token}",
        user_id=f"user-{second_token}",
    )

    assert first.args == retry.args == {"password": "[redacted]", "safe": "visible"}
    assert first.principal.tenant_id == retry.principal.tenant_id == "tenant-[redacted]"
    assert first.principal.user_id == retry.principal.user_id == "user-[redacted]"
    assert first.principal.issuer == retry.principal.issuer == "operator-[redacted]"
    assert first.reason == retry.reason == "recover with [redacted]"
    assert first.token_sha256 != retry.token_sha256
    assert first.identity_sha256 == retry.identity_sha256
    assert first_token not in str(first.to_json())
    assert second_token not in str(retry.to_json())


def test_dbos_envelope_redacts_bearer_reintroduced_by_non_json_repr() -> None:
    token = "raw-bearer-in-repr"

    class OpaqueValue:
        def __repr__(self) -> str:
            return f"OpaqueValue({token})"

    envelope = DbosControlEnvelope.from_control_command(
        ControlCommand(
            type="status",
            run_id="run_1",
            command_id="cmd_1",
            args={
                "token": token,
                "bytes": token.encode(),
                "opaque": OpaqueValue(),
            },
        ),
        tenant_id="tenant",
        user_id="user",
    )

    assert token not in str(envelope.to_json())
    assert envelope.args == {
        "bytes": "b'[redacted]'",
        "opaque": "OpaqueValue([redacted])",
    }


@pytest.mark.parametrize("identifier", ("run_id", "command_id"))
def test_dbos_envelope_rejects_bearer_in_durable_identifiers(identifier: str) -> None:
    token = "signed-bearer"
    identifiers = {"run_id": "run_1", "command_id": "cmd_1"}
    identifiers[identifier] = f"prefix-{token}"

    with pytest.raises(NativeAgentError) as raised:
        DbosControlEnvelope.from_control_command(
            ControlCommand(
                type="resume",
                run_id=identifiers["run_id"],
                command_id=identifiers["command_id"],
                args={"token": token},
            ),
            tenant_id="tenant",
            user_id="user",
        )

    assert raised.value.error_code == "invalid_command_id"


def test_dbos_workflow_id_escapes_components_without_pair_collisions() -> None:
    first = DbosControlPlane.workflow_id("run/control/cmd", "tail")
    second = DbosControlPlane.workflow_id("run", "cmd/control/tail")

    assert first != second
    assert "%2F" in first


def test_dbos_queue_names_are_scoped_by_application_version() -> None:
    first = DbosControlPlane.versioned_queue_name("control", "version/one")
    second = DbosControlPlane.versioned_queue_name("control", "version/two")

    assert first == "monoid/control-queue/control/version/version%2Fone"
    assert second == "monoid/control-queue/control/version/version%2Ftwo"
    assert first != second
    assert DbosControlPlane.versioned_queue_name("control/version/one", "tail") != (
        DbosControlPlane.versioned_queue_name("control", "one/version/tail")
    )


@pytest.mark.parametrize("polling_interval_s", (0.0, float("nan"), float("inf")))
def test_dbos_config_rejects_invalid_polling_intervals(polling_interval_s: float) -> None:
    with pytest.raises(ValueError, match="queue settings"):
        DbosControlConfig(
            system_database_url="sqlite:///dbos.sqlite",
            polling_interval_s=polling_interval_s,
        )


@pytest.mark.parametrize("shutdown_grace_s", (0, -1, True, 0.5, float("inf")))
def test_dbos_config_requires_a_positive_whole_shutdown_grace(
    shutdown_grace_s: object,
) -> None:
    with pytest.raises(ValueError, match="whole number"):
        DbosControlConfig(
            system_database_url="sqlite:///dbos.sqlite",
            shutdown_grace_s=shutdown_grace_s,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(("tenant_id", "user_id"), (("", "user"), ("tenant", "")))
def test_dbos_envelope_requires_authenticated_principal(
    tenant_id: str,
    user_id: str,
) -> None:
    with pytest.raises(ValueError, match="authenticated"):
        DbosControlEnvelope.from_control_command(
            ControlCommand(type="status", run_id="run_1", command_id="cmd_1"),
            tenant_id=tenant_id,
            user_id=user_id,
        )


def test_dbos_envelope_generates_an_id_when_the_control_command_omits_it() -> None:
    envelope = DbosControlEnvelope.from_control_command(
        ControlCommand(type="status", run_id="run_1"),
        tenant_id="tenant",
        user_id="user",
    )

    assert envelope.command_id.startswith("control_")
    assert len(envelope.command_id) == len("control_") + 12
