"""Every reader of ``job.json`` publishes the same projection.

The defect this pins was not that a reader got the rules wrong. It was that there were five of
them and only the event sink had the rules at all: ``monoid jobs --json``, ``monoid job status
--json``, the reference backend's ``/v1/runs/<id>/jobs`` and Studio's ``/api/jobs`` re-read the
artifact off disk and published ``command``, ``cwd`` and ``changed_paths`` verbatim, and
``core.projections`` had a fourth answer that dropped ``command`` and redacted ``changed_paths``
but left ``cwd`` exact. Backgrounding a command was enough to route around ``redact_patterns``.

So the assertions are written once and run against every reader by name. A sixth reader added
without a row here is the failure mode, which is why ``test_every_reader_of_job_json_is_covered``
exists as well.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from monoid_agent_kernel.core.projections import project_run_status
from monoid_agent_kernel.core.schemas import validate_run_dir
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import REDACTED_PATH
from monoid_agent_kernel.reference.backend.jobs import JobService, JobServiceContext
from monoid_agent_kernel.tasks import (
    BackgroundJob,
    public_job_artifact_for,
    public_job_artifacts,
)

JOB_ID = "job_projection"

# Non-ASCII on purpose. `preview_value`'s byte budget is measured in bytes and its predecessor
# truncated by *character* count, so an all-ASCII fixture passed while a Korean one published the
# whole value. Asserted structurally rather than by substring, because `json.dumps` escapes
# non-ASCII by default and a grep for the sentinel finds `사` instead of the text.
# Both one segment under `secrets/`: `PurePosixPath.match` treats `secrets/**` as a single `*`,
# so `secrets/금고/creds.txt` is *not* redacted by that pattern. That is a real gap in
# `matches_path_patterns` and it is not this file's subject -- pinning it here would tie this
# suite to the behaviour rather than to the projection.
SECRET_DIR = "secrets/금고"
SECRET_FILE = "secrets/자격증명.txt"
LONG_COMMAND = "python -c '" + "비밀번호=1;" * 60 + "'"


def _write_job_artifact(run_dir: Path, *, redact: bool = True) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_projection",
                "permission_policy": {
                    "deny_patterns": [],
                    "redact_patterns": ["secrets/**"] if redact else [],
                },
            }
        ),
        encoding="utf-8",
    )
    job_dir = run_dir / "artifacts" / "jobs" / JOB_ID
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(_job(run_dir).to_json(run_dir), ensure_ascii=False),
        encoding="utf-8",
    )


def _job(run_dir: Path) -> BackgroundJob:
    job_dir = run_dir / "artifacts" / "jobs" / JOB_ID
    return BackgroundJob(
        job_id=JOB_ID,
        kind="shell",
        command=LONG_COMMAND,
        command_preview="python -c ...",
        cwd=SECRET_DIR,
        status="exited",
        started_at=1.0,
        timeout_s=10,
        max_output_bytes=1000,
        startup_wait_s=0,
        stdout_path=job_dir / "stdout.log",
        stderr_path=job_dir / "stderr.log",
        job_path=job_dir / "job.json",
        cancel_path=job_dir / "cancel",
        execution_workspace="isolated-copy",
        resume_on_exit=True,
        finished_at=2.0,
        exit_code=0,
        changed_paths=(SECRET_FILE,),
    )


class _Record:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.tenant_id = "tenant"


def _job_service(run_dir: Path) -> JobService:
    return JobService(
        JobServiceContext(
            authorize_run=lambda _run_id, _token: None,
            record=lambda _run_id: _Record(run_dir),  # type: ignore[arg-type,return-value]
        )
    )


# name -> (run_dir -> the projected job dict that reader publishes)
READERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "event_sink": lambda run_dir: _job(run_dir).public_payload(
        run_dir, PermissionPolicy(redact_patterns=("secrets/**",))
    ),
    "project_run_status": lambda run_dir: project_run_status(run_dir)["jobs"][0],
    "cli_jobs": lambda run_dir: public_job_artifacts(run_dir)[0],
    "cli_job_status": lambda run_dir: public_job_artifact_for(run_dir, JOB_ID),
    "backend_jobs": lambda run_dir: _job_service(run_dir).jobs("run_projection", "token")["jobs"][0],
    "backend_job_status": lambda run_dir: _job_service(run_dir).job_status(
        "run_projection", "token", JOB_ID
    )["job"],
}


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run_projection"
    path.mkdir()
    _write_job_artifact(path)
    return path


@pytest.mark.parametrize("reader", sorted(READERS))
def test_no_reader_publishes_the_raw_command(run_dir: Path, reader: str) -> None:
    job = READERS[reader](run_dir)
    assert "command" not in job
    # The bounded rendering stays, so dropping the field is not an information cliff.
    assert job["command_preview"] == "python -c ..."


@pytest.mark.parametrize("reader", sorted(READERS))
def test_every_reader_redacts_a_matched_cwd(run_dir: Path, reader: str) -> None:
    job = READERS[reader](run_dir)
    # The literal shape rather than `redacted_value(SECRET_DIR)`: building the expectation with the
    # function under test would keep passing if that function stopped redacting.
    assert job["cwd"] == {
        "redacted": True,
        "type": "str",
        "bytes": len(SECRET_DIR.encode("utf-8")),
    }


@pytest.mark.parametrize("reader", sorted(READERS))
def test_every_reader_redacts_matched_changed_paths(run_dir: Path, reader: str) -> None:
    job = READERS[reader](run_dir)
    assert job["changed_paths"] == [REDACTED_PATH]


@pytest.mark.parametrize("reader", sorted(READERS))
def test_no_reader_leaks_the_secret_as_bytes(run_dir: Path, reader: str) -> None:
    """The black-box check: serialize what the reader publishes and look for the input.

    Both encodings, because ``json.dumps`` escapes non-ASCII by default and the CLI passes
    ``ensure_ascii=False`` -- so a leak shows up as ``\\uc0ac`` on one surface and as the Korean
    text on the other, and asserting only one of them tests only one of them.
    """
    job = READERS[reader](run_dir)
    for text in (json.dumps(job), json.dumps(job, ensure_ascii=False)):
        assert SECRET_DIR not in text
        assert SECRET_FILE not in text
        assert "비밀번호" not in text
        assert json.dumps(SECRET_DIR)[1:-1] not in text


@pytest.mark.parametrize("reader", sorted(READERS))
def test_readers_agree_field_for_field(run_dir: Path, reader: str) -> None:
    """Not just "each is safe" -- each is the *same*. Two safe-but-different answers is the
    state this replaced, and it is what let one of them drift."""
    assert READERS[reader](run_dir) == READERS["event_sink"](run_dir)


def test_unredacted_run_still_publishes_the_path_but_never_the_command(tmp_path: Path) -> None:
    """With no `redact_patterns`, `cwd` is a path an operator asked to see. `command` is not:
    it is dropped on every run, because the artifact carries `command_preview` for that."""
    path = tmp_path / "run_open"
    path.mkdir()
    _write_job_artifact(path, redact=False)

    job = public_job_artifacts(path)[0]
    assert job["cwd"] == SECRET_DIR
    assert job["changed_paths"] == [SECRET_FILE]
    assert "command" not in job


def test_the_artifact_on_disk_is_untouched(run_dir: Path) -> None:
    """Only the projection changes. `JOB_SCHEMA` declares `command` and `cwd` required with
    `additionalProperties: false`, and `monoid validate` reads the file, not a reader."""
    payload = json.loads((run_dir / "artifacts" / "jobs" / JOB_ID / "job.json").read_text("utf-8"))
    assert payload["command"] == LONG_COMMAND
    assert payload["cwd"] == SECRET_DIR
    assert payload["changed_paths"] == [SECRET_FILE]
    assert [issue for issue in validate_run_dir(run_dir) if "job.json" in issue.path] == []


def test_every_key_of_the_artifact_is_classified(run_dir: Path) -> None:
    """The projection copies unknown keys through, so this is what stops a newly added field from
    being published by default: adding a key to ``BackgroundJob.to_json`` fails here until whoever
    added it decides which of the three treatments it gets.

    An allowlist in ``public_job_artifact`` would enforce it in code instead, and was rejected --
    it would drop the new field from every public reader without saying so, and a silent omission
    is the failure this release exists to close. The enforcement is loud here rather than silent
    there.
    """
    dropped = {"command"}
    previewed = {"cwd"}
    redacted_exactly = {"changed_paths"}
    copied_through = {
        "schema_version",
        "job_id",
        "kind",
        "command_preview",
        "status",
        "started_at",
        "finished_at",
        "duration_s",
        "exit_code",
        "timed_out",
        "output_truncated",
        "error",
        "stdout_path",
        "stderr_path",
        "stdout_bytes",
        "stderr_bytes",
        "requested_timeout_s",
        "effective_timeout_s",
        "requested_max_output_bytes",
        "effective_max_output_bytes",
        "requested_startup_wait_s",
        "effective_startup_wait_s",
        "execution_workspace",
        "resume_on_exit",
    }

    written = set(_job(run_dir).to_json(run_dir))
    classified = dropped | previewed | redacted_exactly | copied_through
    assert written == classified, "a field of job.json is not classified in the public projection"

    published = set(public_job_artifacts(run_dir)[0])
    assert published == written - dropped


def test_every_reader_of_job_json_is_covered() -> None:
    """``READERS`` is the point of this file, so a reader missing from it is the defect.

    Checked by import rather than by grep: the two disk readers are the only way to reach
    ``job.json`` outside the loop's own crash-recovery scan, and every surface named in the
    module docstring goes through one of them.
    """
    import monoid_agent_kernel.cli as cli
    import monoid_agent_kernel.core.projections as projections
    import monoid_agent_kernel.reference.backend.jobs as backend_jobs

    for module in (cli, projections, backend_jobs):
        assert not hasattr(module, "list_job_artifacts"), f"{module.__name__} kept a raw reader"
        assert not hasattr(module, "get_job_artifact"), f"{module.__name__} kept a raw reader"
    assert set(READERS) == {
        "event_sink",
        "project_run_status",
        "cli_jobs",
        "cli_job_status",
        "backend_jobs",
        "backend_job_status",
    }


def test_a_manifest_that_cannot_be_parsed_is_an_error_not_an_empty_policy(tmp_path: Path) -> None:
    """The fail-open case, pinned because the first draft of `run_permission_policy` had it.

    Collapsing "no manifest" and "unreadable manifest" to the default policy answers "no patterns
    were declared" for a file that might declare any. That is a redaction control failing open on
    exactly the input where an operator needs it not to.
    """
    from monoid_agent_kernel.tasks import run_permission_policy

    run_dir = tmp_path / "run_bad_manifest"
    run_dir.mkdir()

    # No manifest at all: nothing was declared, so honouring nothing is exact.
    assert run_permission_policy(run_dir) == PermissionPolicy()

    (run_dir / "manifest.json").write_text('{"permission_policy": "not-an-object"}', encoding="utf-8")
    with pytest.raises(ValueError):
        run_permission_policy(run_dir)

    (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        run_permission_policy(run_dir)

    (run_dir / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        run_permission_policy(run_dir)


def test_a_run_with_no_manifest_still_drops_the_command(tmp_path: Path) -> None:
    """The default policy redacts no paths, and `command` is dropped regardless -- it is not a
    redaction decision, it is a field that has a bounded replacement."""
    run_dir = tmp_path / "run_no_manifest"
    job_dir = run_dir / "artifacts" / "jobs" / JOB_ID
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(_job(run_dir).to_json(run_dir), ensure_ascii=False), encoding="utf-8"
    )

    job = public_job_artifacts(run_dir)[0]
    assert "command" not in job
    assert job["cwd"] == SECRET_DIR
