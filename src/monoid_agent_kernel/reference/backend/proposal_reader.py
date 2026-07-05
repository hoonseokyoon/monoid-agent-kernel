from __future__ import annotations

import json
from typing import Any

from monoid_agent_kernel.reference.backend.ports import RunRecordPort


def read_proposal_snapshot(record: RunRecordPort) -> dict[str, Any] | None:
    proposal_path = record.run_dir / "proposal.json"
    if not proposal_path.exists():
        return None
    payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("proposal snapshot must be a JSON object")
    return payload
