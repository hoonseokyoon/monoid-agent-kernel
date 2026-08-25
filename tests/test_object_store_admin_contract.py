from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import get_type_hints

import pytest

from monoid_agent_kernel.hosting import (
    IncompleteMultipartPage,
    IncompleteMultipartUpload,
    MultipartAbortResult,
    ObjectDeleteResult,
    ObjectGcCandidate,
    ObjectGcPlan,
    ObjectGcReceipt,
    ObjectInventoryEntry,
    ObjectInventoryPage,
    ObjectStoreAdmin,
)


_NOW = datetime(2026, 8, 24, tzinfo=UTC)
_SHA = "a" * 64


def _entry() -> ObjectInventoryEntry:
    return ObjectInventoryEntry(
        sha256=_SHA,
        size_bytes=7,
        locator=f"s3://bucket/sha256/aa/{_SHA}",
        last_modified=_NOW - timedelta(days=2),
        delete_token='"etag"',
    )


def test_object_admin_values_are_immutable_bounded_and_closed() -> None:
    entry = _entry()
    page = ObjectInventoryPage(entries=(entry,), next_token="next")
    upload = IncompleteMultipartUpload(
        sha256=_SHA,
        upload_id="upload-1",
        initiated_at=_NOW,
    )
    multipart_page = IncompleteMultipartPage(uploads=(upload,))
    candidate = ObjectGcCandidate(
        sha256=entry.sha256,
        size_bytes=entry.size_bytes,
        locator=entry.locator,
        last_modified=entry.last_modified,
        delete_token=entry.delete_token,
        generation=0,
    )
    plan = ObjectGcPlan(
        plan_id="b" * 64,
        created_at=_NOW,
        grace_before=_NOW - timedelta(days=1),
        candidates=(candidate,),
        next_token=page.next_token,
    )
    receipt = ObjectGcReceipt(
        plan_id=plan.plan_id,
        sha256=_SHA,
        candidate_generation=0,
        observed_generation=0,
        status="deleted",
        recorded_at=_NOW,
    )

    assert multipart_page.uploads == (upload,)
    assert receipt.status == "deleted"
    assert ObjectDeleteResult(status="deleted").status == "deleted"
    assert MultipartAbortResult(status="aborted").status == "aborted"

    with pytest.raises(ValueError, match="timezone-aware"):
        ObjectInventoryEntry(
            sha256=_SHA,
            size_bytes=1,
            locator="memory:value",
            last_modified=datetime(2026, 8, 24),
            delete_token="token",
        )
    with pytest.raises(ValueError, match="after"):
        ObjectGcPlan(
            plan_id="b" * 64,
            created_at=_NOW,
            grace_before=_NOW + timedelta(seconds=1),
            candidates=(),
        )
    with pytest.raises(ValueError, match="vocabulary"):
        ObjectGcReceipt(
            plan_id="b" * 64,
            sha256=_SHA,
            candidate_generation=0,
            observed_generation=0,
            status="failed",  # type: ignore[arg-type]
            recorded_at=_NOW,
        )


def test_object_admin_protocol_annotations_resolve() -> None:
    inventory = get_type_hints(ObjectStoreAdmin.inventory_page)
    delete = get_type_hints(ObjectStoreAdmin.delete_if_match)
    multipart = get_type_hints(ObjectStoreAdmin.incomplete_multipart_page)
    abort = get_type_hints(ObjectStoreAdmin.abort_incomplete_multipart)

    assert inventory == {
        "continuation_token": str | None,
        "limit": int,
        "return": ObjectInventoryPage,
    }
    assert delete == {
        "sha256": str,
        "delete_token": str,
        "return": ObjectDeleteResult,
    }
    assert multipart == {
        "continuation_token": str | None,
        "limit": int,
        "return": IncompleteMultipartPage,
    }
    assert abort == {
        "upload": IncompleteMultipartUpload,
        "return": MultipartAbortResult,
    }
