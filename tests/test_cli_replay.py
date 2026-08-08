"""``monoid run --replay-from``: the offline entry, its preflight, and its provenance.

W6-4b B5. The adapter's behavior is pinned beside this file; here the pins are about the
COMMAND: pure replay must bypass the live-adapter gates entirely (no gateway URL, no
``--allow-direct-provider-api``, no recognized provider name -- an offline replay needs none
of them, [P11]); the preflight refuses a run whose every lookup is doomed before the run
starts; a miss exits as the failed run_once shape; and the corpora that served a run land as
run-level provenance on every ledger line.

Red first: ``--replay-from`` was ``Error: No such option`` (exit 2) before this commit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from monoid_agent_kernel.cli import main
from monoid_agent_kernel.core.model_calls import MODEL_CALLS_FILENAME
from monoid_agent_kernel.core.model_payloads import MODEL_PAYLOADS_FILENAME
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter

from test_cli_and_openai import _write_config  # the shipped config helper, reused

_MARKER = "SECRET-CLI-REPLAY-7K"


def _record_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str = "recorded",
    instruction: str = f"Finish. {_MARKER}",
    model: ModelConfig | None = None,
) -> Path:
    """Record a corpus through the real CLI with a fake live adapter, then unpatch."""

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    config_file = _write_config(tmp_path / "runtime.json", "run.finish", model=model)
    monkeypatch.setattr(
        "monoid_agent_kernel.cli._model_adapter",
        lambda *_a, **_k: FakeModelAdapter(
            turns=[ModelTurn(response_id="r1", final_text="recorded answer")]
        ),
    )
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            instruction,
            "--run-root",
            str(tmp_path / "runs"),
            "--runtime-config-file",
            str(config_file),
            "--run-id",
            run_id,
            "--model-payload-file",
            "--model-calls-file",
        ],
    )
    assert result.exit_code == 0, result.output
    monkeypatch.undo()
    return tmp_path / "runs" / run_id


def _replay_args(tmp_path: Path, *sources: str, config: Path | None = None) -> list[str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    config_file = config or (tmp_path / "runtime.json")
    args = [
        "run",
        "--workspace",
        str(workspace),
        "--instruction",
        f"Finish. {_MARKER}",
        "--run-root",
        str(tmp_path / "runs"),
        "--runtime-config-file",
        str(config_file),
    ]
    for source in sources or ("recorded",):
        args += ["--replay-from", source]
    return args


def test_cli_replays_a_recorded_run_with_no_live_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[P11] The pure-replay path never builds the live adapter: no gateway URL, no token,
    no monkeypatch -- the same command that would need them live completes offline."""

    _record_run(tmp_path, monkeypatch)

    result = CliRunner().invoke(main, _replay_args(tmp_path) + ["--run-id", "replayed"])

    assert result.exit_code == 0, result.output
    assert "status: completed" in result.output
    assert "recorded answer" in result.output


def test_cli_warns_when_two_sources_can_answer_the_same_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The one property of a union an operator cannot read off their own command line.

    Two recordings of one conversation is the same prompt run twice, or the crash-and-rerun
    union `docs/CONTRACTS.md` calls the ordinary durable-resume shape. Across sources, "file
    order" is the order of the `--replay-from` flags, so reversing two arguments replays a
    different conversation. Both orders complete, both say `status: completed`, and nothing in
    the run afterwards records which source answered -- `attributes.replay_from` is the joined
    list and is identical either way.
    """

    _record_run(tmp_path, monkeypatch, run_id="first")
    _record_run(tmp_path, monkeypatch, run_id="second")

    result = CliRunner().invoke(
        main, _replay_args(tmp_path, "first", "second") + ["--run-id", "replayed"]
    )

    assert result.exit_code == 0, result.output
    # "key(s)", not "call(s)": the counter is over keys, and a key with five recordings is one.
    assert "1 recorded call key(s) can be answered by more than one --replay-from source" in (
        result.output
    )
    assert "the order the sources were named in decides" in result.output
    assert "children of one run" not in result.output, (
        "two independent runs are not one run's fan-out, and the spawn-order advice would "
        "send the operator after an order that does not exist"
    )


def test_cli_says_nothing_about_a_disjoint_union(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sources whose conversations differ share no key, so there is nothing to warn about --
    and warning anyway would train the operator to ignore the one above.

    Note what this does *not* say. Disjointness is a property of the prompts, not of the family
    shape: two children with the same definition and the same prompt record one key in two run
    directories, and that union does warn. See
    ``test_two_identical_children_of_one_run_cross_a_key``.
    """

    _record_run(tmp_path, monkeypatch, run_id="first")
    _record_run(tmp_path, monkeypatch, run_id="second", instruction=f"Something else. {_MARKER}")

    result = CliRunner().invoke(
        main, _replay_args(tmp_path, "first", "second") + ["--run-id", "replayed"]
    )

    assert result.exit_code == 0, result.output
    assert "more than one --replay-from source" not in result.output


def test_cli_pure_replay_skips_the_direct_provider_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[P11]+[D-h c] sharpened: a config naming provider=openai normally requires
    --allow-direct-provider-api before any run starts; under pure replay the gate is never
    consulted, and the key still hits because the adapter DECLARES the recorded provider
    (the key's provider term is independent of the run config's)."""

    _record_run(tmp_path, monkeypatch)
    openai_config = _write_config(
        tmp_path / "openai.json", "run.finish", model=ModelConfig(provider="openai")
    )

    result = CliRunner().invoke(
        main,
        _replay_args(tmp_path, config=openai_config) + ["--run-id", "replayed"],
    )

    assert result.exit_code == 0, result.output
    assert "status: completed" in result.output


def test_cli_preflight_rejects_a_config_that_can_only_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A wrong model identity is discoverable before the run starts; the rejection names
    expected and actual (config vocabulary, never conversation), and no run directory is
    created for a run that was never going to hit."""

    _record_run(tmp_path, monkeypatch)
    wrong = _write_config(
        tmp_path / "wrong.json", "run.finish", model=ModelConfig(model="elsewhere-9")
    )

    result = CliRunner().invoke(
        main, _replay_args(tmp_path, config=wrong) + ["--run-id", "rejected"]
    )

    assert result.exit_code != 0
    assert "preflight" in result.output
    assert "elsewhere-9" in result.output and "gpt-5.5" in result.output
    assert _MARKER not in result.output.replace(f"Finish. {_MARKER}", "")
    assert not (tmp_path / "runs" / "rejected").exists()


def test_cli_preflight_softens_to_a_warning_under_fallthrough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An all-live run is a valid run: with --replay-fallthrough the same divergence warns
    and the run proceeds against the live adapter."""

    _record_run(tmp_path, monkeypatch)
    wrong = _write_config(
        tmp_path / "wrong.json", "run.finish", model=ModelConfig(model="elsewhere-9")
    )
    monkeypatch.setattr(
        "monoid_agent_kernel.cli._model_adapter",
        lambda *_a, **_k: FakeModelAdapter(
            turns=[ModelTurn(response_id="live-1", final_text="live answer")]
        ),
    )

    result = CliRunner().invoke(
        main,
        _replay_args(tmp_path, config=wrong) + ["--run-id", "fallthrough", "--replay-fallthrough"],
    )

    assert result.exit_code == 0, result.output
    assert "warning: replay preflight" in result.output
    assert "live answer" in result.output


def test_cli_a_replay_miss_exits_as_the_failed_run_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _record_run(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "never recorded",
            "--run-root",
            str(tmp_path / "runs"),
            "--runtime-config-file",
            str(tmp_path / "runtime.json"),
            "--replay-from",
            "recorded",
            "--run-id",
            "missed",
        ],
    )

    assert result.exit_code != 0
    assert "replay miss" in result.output
    assert _MARKER not in result.output


def test_cli_replay_provenance_lands_on_every_ledger_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The corpora that served a run are run-level facts, recorded where run-level facts
    live: InvocationContext.attributes, verbatim on each ledger line (D-e). Repeatable
    sources join with commas, in corpus-envelope vocabulary."""

    _record_run(tmp_path, monkeypatch, run_id="recorded")
    second = _record_run(
        tmp_path, monkeypatch, run_id="recorded-2", instruction="another conversation"
    )
    del second

    result = CliRunner().invoke(
        main,
        _replay_args(tmp_path, "recorded", "recorded-2")
        + ["--run-id", "replayed", "--model-calls-file"],
    )

    assert result.exit_code == 0, result.output
    ledger_lines = [
        json.loads(line)
        for line in (tmp_path / "runs" / "replayed" / MODEL_CALLS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert ledger_lines
    for line in ledger_lines:
        assert line["context"]["attributes"]["replay_from"] == "recorded,recorded-2"


def test_cli_refuses_a_source_that_recorded_no_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Construction-time, surfaced as an ordinary CLI error -- not a traceback, and not a
    run that starts and misses on turn one."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")
    empty = tmp_path / "runs" / "no-corpus"
    empty.mkdir(parents=True)

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--runtime-config-file",
            str(config_file),
            "--replay-from",
            "no-corpus",
        ],
    )

    assert result.exit_code != 0
    assert "no readable corpus" in result.output
    assert "Traceback" not in result.output


def _noop_guard(*_a: Any, **_k: Any) -> None:
    raise AssertionError("the live-adapter branch must not be consulted under pure replay")


def test_cli_pure_replay_never_touches_the_live_adapter_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The structural half of [P11]: _model_adapter is not merely tolerant under pure
    replay, it is unreachable."""

    _record_run(tmp_path, monkeypatch)
    monkeypatch.setattr("monoid_agent_kernel.cli._model_adapter", _noop_guard)

    result = CliRunner().invoke(main, _replay_args(tmp_path) + ["--run-id", "guarded"])

    assert result.exit_code == 0, result.output


def test_the_damage_warning_does_not_promise_a_miss_that_need_not_happen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A corpus damaged only where this run asks nothing completes with no miss at all.

    The categorical "will miss" beside a clean exit 0 is how an operator learns to ignore the
    warning. The reader cannot know the key of a record whose key is what got damaged, so the
    conditional is the only honest form -- and the sentence had no pin, which is how the
    previous round's operator-facing wording regressed silently.
    """

    _record_run(tmp_path, monkeypatch, run_id="first")
    corpus = tmp_path / "runs" / "first" / MODEL_PAYLOADS_FILENAME
    corpus.write_text(corpus.read_text(encoding="utf-8") + "{ not json at all\n", encoding="utf-8")

    result = CliRunner().invoke(main, _replay_args(tmp_path, "first") + ["--run-id", "replayed"])

    assert result.exit_code == 0, result.output
    assert "1 unparseable line(s)" in result.output
    assert "any call that needed one of them will miss" in result.output
    assert "status: completed" in result.output, (
        "the run completed, so a categorical promise of a miss would have been false"
    )
