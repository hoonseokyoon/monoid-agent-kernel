from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from native_agent_runner.core._util import canonical_sha256, sha256_bytes, write_json_atomic
from native_agent_runner.core.events import AgentEvent, EventBus, EventSink, make_agent_event
from native_agent_runner.core.manifest import RunManifest
from native_agent_runner.core.result import AgentArtifact

if TYPE_CHECKING:
    from native_agent_runner.core.workspace import ChangedEntry, Workspace


@dataclass
class JsonlEventSink:
    path: Path
    _handle: TextIO = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def emit(self, event: AgentEvent) -> None:
        _write_jsonl(self._handle, event.to_json())

    def close(self) -> None:
        self._handle.close()


@dataclass
class StdoutJsonlSink:
    handle: TextIO = field(default_factory=lambda: sys.stdout)

    def emit(self, event: AgentEvent) -> None:
        _write_jsonl(self.handle, event.to_json())

    def close(self) -> None:
        self.handle.flush()


@dataclass
class StatusJsonSink:
    path: Path
    state: dict[str, Any] = field(default_factory=dict)

    def emit(self, event: AgentEvent) -> None:
        data = event.data
        if event.type == "run.started":
            self.state.update(
                {
                    "run_id": event.run_id,
                    "status": "running",
                    "started_at": event.timestamp,
                    "workspace": data.get("workspace"),
                    "workspace_backend": data.get("workspace_backend"),
                    "workspace_base_path": data.get("workspace_base_path"),
                    "manifest_path": data.get("manifest_path"),
                    "mode": data.get("mode"),
                    "model": data.get("model"),
                    "reasoning_effort": data.get("reasoning_effort"),
                }
            )
        elif event.type == "run.finished":
            self.state.update(
                {
                    "status": data.get("status", "completed"),
                    "finished_at": event.timestamp,
                    "final_text": data.get("final_text", ""),
                    "error": data.get("error", ""),
                    "error_code": data.get("error_code", ""),
                }
            )
        elif event.type == "run.failed":
            self.state.update(
                {
                    "status": "failed",
                    "error": data.get("error", ""),
                    "error_code": data.get("error_code", ""),
                    "error_type": data.get("type", ""),
                }
            )
        elif event.type == "run.waiting":
            self.state["status"] = "waiting_for_background_jobs"
            self.state["waiting_for_background_jobs"] = True
            self.state["waiting_jobs"] = data.get("jobs", [])
        elif event.type == "run.resumed":
            self.state["status"] = "running"
            self.state["waiting_for_background_jobs"] = False
            self.state["resumed_jobs"] = data.get("job_ids", [])
        elif event.type == "run.awaiting_input":
            self.state["status"] = "awaiting_input"
            self.state["awaiting_input"] = {
                "reason": data.get("reason"),
                "task_ids": data.get("task_ids", []),
                "prompt": data.get("prompt"),
            }
        elif event.type == "agent.config.updated":
            self.state["agent_config"] = {
                "definition_id": data.get("definition_id"),
                "config_version": data.get("config_version"),
                "config_hash": data.get("config_hash"),
            }
        elif event.type == "model.turn.started":
            self.state["current_turn_id"] = event.turn_id
            self.state["current_step"] = data.get("step")
            if self.state.get("status") == "awaiting_input":
                self.state["status"] = "running"
                self.state.pop("awaiting_input", None)
        elif event.type == "tool.call.started":
            self.state["current_tool"] = data.get("tool")
            self.state["current_tool_call_id"] = data.get("call_id")
        elif event.type in {"tool.call.finished", "tool.call.failed"}:
            self.state.pop("current_tool", None)
            self.state.pop("current_tool_call_id", None)
        elif event.type == "plan.updated":
            self.state["plan"] = data.get("items", [])
        elif event.type == "metrics.updated":
            self.state["metrics"] = data
        elif event.type == "workspace.proposal.updated":
            self.state["proposal"] = data
        elif event.type.startswith("job."):
            jobs = self.state.setdefault("jobs", {})
            if isinstance(jobs, dict):
                job_id = data.get("job_id")
                if isinstance(job_id, str) and job_id:
                    jobs[job_id] = data

        self.state.update(
            {
                "last_event_seq": event.seq,
                "last_event_type": event.type,
                "updated_at": event.timestamp,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def close(self) -> None:
        return None


@dataclass
class MemoryEventSink:
    events: list[AgentEvent] = field(default_factory=list)

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


@dataclass
class AgentRecorder:
    run_root: Path
    run_id: str
    extra_event_sinks: tuple[EventSink, ...] = ()
    status_file: bool = True
    run_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    event_bus: EventBus = field(init=False)
    _transcript_file: TextIO = field(init=False, repr=False)
    started_at: float = field(default_factory=time.time)
    artifacts: list[AgentArtifact] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_dir = self.run_root / self.run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=False)
        self._transcript_file = (self.run_dir / "transcript.jsonl").open("a", encoding="utf-8")
        sinks: list[EventSink] = [JsonlEventSink(self.run_dir / "events.jsonl")]
        if self.status_file:
            sinks.append(StatusJsonSink(self.run_dir / "status.json"))
        sinks.extend(self.extra_event_sinks)
        self.event_bus = EventBus(self.run_id, tuple(sinks))

    def emit(
        self,
        event_type: str,
        *,
        data: dict[str, Any] | None = None,
        level: str = "info",
        turn_id: str | None = None,
        parent_id: str | None = None,
    ) -> AgentEvent:
        return self.event_bus.emit(
            event_type,  # type: ignore[arg-type]
            data=data,
            level=level,  # type: ignore[arg-type]
            turn_id=turn_id,
            parent_id=parent_id,
        )

    def transcript(self, item: dict[str, Any]) -> None:
        _write_jsonl(self._transcript_file, item)

    def emit_artifact_bytes(
        self,
        *,
        workspace_path: str,
        content: bytes,
        kind: str,
        label: str | None,
    ) -> AgentArtifact:
        artifact_id = f"artifact_{len(self.artifacts) + 1:04d}"
        target = self.artifacts_dir / artifact_id / Path(workspace_path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        artifact = AgentArtifact(
            artifact_id=artifact_id,
            path=str(target.relative_to(self.run_dir).as_posix()),
            kind=kind,
            label=label,
        )
        self.artifacts.append(artifact)
        return artifact

    def write_diff(self, diff_text: str) -> Path:
        diff_path = self.run_dir / "diff.patch"
        diff_path.write_text(diff_text, encoding="utf-8")
        return diff_path

    def write_manifest(self, manifest: RunManifest) -> Path:
        manifest_path = self.run_dir / "manifest.json"
        write_json_atomic(manifest_path, manifest.to_json())
        return manifest_path

    def write_workspace_index(self, payload: dict[str, Any]) -> Path:
        path = self.run_dir / "workspace.index.json"
        write_json_atomic(path, payload)
        return path

    def write_workspace_base(self, payload: dict[str, Any]) -> Path:
        path = self.run_dir / "workspace.base.json"
        write_json_atomic(path, payload)
        return path

    def write_proposal_snapshot(self, workspace: Workspace, diff_path: Path) -> dict[str, Any]:
        proposal_path = self.run_dir / "proposal.json"
        files_dir = self.run_dir / "proposal" / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        for entry in workspace.changed_entries():
            files.append(self._write_proposal_entry(entry, files_dir))
        diff_data = diff_path.read_bytes() if diff_path.exists() else b""
        diff_bytes = diff_path.stat().st_size if diff_path.exists() else 0
        payload: dict[str, Any] = {
            "schema_version": "native-agent-runner.proposal.v2",
            "run_id": self.run_id,
            "updated_at": time.time(),
            "mode": workspace.mode,
            "diff_path": str(diff_path.relative_to(self.run_dir)),
            "diff_bytes": diff_bytes,
            "diff_sha256": sha256_bytes(diff_data),
            "changed_paths": [entry["path"] for entry in files],
            "files": files,
        }
        # updated_at is wall-clock metadata, not content; excluding it makes the
        # proposal_hash a stable content identifier so repeated settle checkpoints
        # with no workspace change produce the same hash.
        payload["proposal_hash"] = canonical_sha256(payload, drop=("proposal_hash", "updated_at"))
        write_json_atomic(proposal_path, payload)
        return payload

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.run_dir / "metrics.json"
        payload = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": time.time(),
            **metrics,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def close(self) -> None:
        self.event_bus.close()
        self._transcript_file.close()

    def _write_proposal_entry(self, entry: ChangedEntry, files_dir: Path) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": entry.path,
            "kind": entry.kind,
            "size": entry.size,
            "sha256": entry.sha256,
            "base_sha256": entry.base_sha256,
            "proposed_sha256": entry.proposed_sha256,
            "change_kind": entry.change_kind,
        }
        if entry.content is None:
            return payload
        target = files_dir.joinpath(*entry.path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entry.content)
        payload["snapshot_path"] = str(target.relative_to(self.run_dir).as_posix())
        payload["snapshot_sha256"] = sha256_bytes(entry.content)
        return payload


def _write_jsonl(handle: TextIO, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def append_event_to_run(
    run_dir: Path,
    event_type: str,
    *,
    data: dict[str, Any] | None = None,
    level: str = "info",
) -> AgentEvent:
    events_path = run_dir / "events.jsonl"
    run_id = _run_id_from_run_dir(run_dir)
    seq = _last_event_seq(events_path) + 1
    event = make_agent_event(
        run_id=run_id,
        seq=seq,
        event_type=event_type,  # type: ignore[arg-type]
        data=data,
        level=level,  # type: ignore[arg-type]
    )
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        _write_jsonl(handle, event.to_json())
    _update_status_last_event(run_dir, event)
    return event


def _run_id_from_run_dir(run_dir: Path) -> str:
    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(status, dict) and isinstance(status.get("run_id"), str):
                return status["run_id"]
        except json.JSONDecodeError:
            pass
    proposal_path = run_dir / "proposal.json"
    if proposal_path.exists():
        try:
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            if isinstance(proposal, dict) and isinstance(proposal.get("run_id"), str):
                return proposal["run_id"]
        except json.JSONDecodeError:
            pass
    return run_dir.name


def _last_event_seq(events_path: Path) -> int:
    if not events_path.exists():
        return 0
    last_seq = 0
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            last_seq = max(last_seq, int(event.get("seq") or 0))
    return last_seq


def _update_status_last_event(run_dir: Path, event: AgentEvent) -> None:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(status, dict):
        return
    status["last_event_seq"] = event.seq
    status["last_event_type"] = event.type
    status["updated_at"] = event.timestamp
    if event.type.startswith("proposal."):
        status["proposal_event"] = event.data
    write_json_atomic(status_path, status)
