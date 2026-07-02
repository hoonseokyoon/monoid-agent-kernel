from __future__ import annotations

from string import ascii_lowercase, digits

import pytest
from hypothesis import given, settings, strategies as st

from monoid_agent_kernel.errors import WorkspaceError
from monoid_agent_kernel.workspace.paths import normalize_workspace_path


SEGMENT = st.text(alphabet=ascii_lowercase + digits + "_.-", min_size=1, max_size=8).filter(
    lambda value: value not in {".", ".."}
)
SEPARATOR = st.sampled_from(("/", "\\"))
PROPERTY_SETTINGS = settings(max_examples=40, deadline=None)


@st.composite
def safe_workspace_paths(draw) -> str | None:
    if draw(st.booleans()):
        return draw(st.one_of(st.none(), st.sampled_from(("", "."))))

    separator = draw(SEPARATOR)
    parts = draw(st.lists(st.one_of(SEGMENT, st.just(".")), min_size=1, max_size=5))
    prefix = draw(st.sampled_from(("", f".{separator}")))
    return prefix + separator.join(parts)


@st.composite
def parent_traversal_paths(draw) -> str:
    separator = draw(SEPARATOR)
    before = draw(st.lists(SEGMENT, max_size=3))
    after = draw(st.lists(SEGMENT, max_size=3))
    return separator.join([*before, "..", *after])


@st.composite
def absolute_workspace_paths(draw) -> str:
    separator = draw(SEPARATOR)
    tail = separator.join(draw(st.lists(SEGMENT, min_size=1, max_size=3)))
    return draw(st.sampled_from((f"/{tail}", f"\\{tail}", f"C:/{tail}", f"z:\\{tail}")))


@PROPERTY_SETTINGS
@given(safe_workspace_paths())
def test_normalize_workspace_path_is_idempotent(raw: str | None) -> None:
    normalized = normalize_workspace_path(raw)

    assert normalize_workspace_path(normalized) == normalized


@PROPERTY_SETTINGS
@given(parent_traversal_paths())
def test_normalize_workspace_path_blocks_parent_traversal(raw: str) -> None:
    with pytest.raises(WorkspaceError):
        normalize_workspace_path(raw)


@PROPERTY_SETTINGS
@given(absolute_workspace_paths())
def test_normalize_workspace_path_blocks_absolute_paths(raw: str) -> None:
    with pytest.raises(WorkspaceError):
        normalize_workspace_path(raw)
