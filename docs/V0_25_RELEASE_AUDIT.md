# v0.25 Release Readiness Audit

Status on 2026-10-28: implementation integrated; documentation changes prepared and locally
qualified, with final review/integration tracked below. Release qualification remains pending.
Package metadata remains `0.24.0`. Version change, release artifacts,
release branch integration, tag, GitHub release, and package publication remain pending.

The [migration guide](migrations/v0.25.md) describes the upgrade and rollback requirements.
The [compatibility ledger](COMPATIBILITY.md) owns the reader/writer inventory. This audit records
the source and artifact identities behind evidence and keeps each pending gate visible.

## Source identity

| Item | Identity or state |
|---|---|
| Integrated implementation source | `b27a15c159465b978b50e89660fa5f7b7dd7d7b3` |
| Final implementation integration | [PR #170](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/170), merged into `develop` |
| Reviewed DX2 integration head | `07a84ae6dab971dd1ed13f554540f215b5af54dd` |
| Reviewed head and merge tree | `92a4c23201ce9fa4ca9fdf87b3cfdb23cf3e3436` |
| Current package/fallback version | `0.24.0` |
| Qualified v0.25 release source, wheel, and sdist | Pending |
| v0.25 tag and publication | Pending |

## Integrated behavior

| Work | User-visible result | Integration evidence |
|---|---|---|
| Call termination and usage facts (P1/P2) | Outcome, accepted stop requester, read end, explicit zero, and counter coverage carried through receipts and recovery; invocation/ledger v2 with retained v1 readers | [PR #161](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/161), merge `2914c4366f8dfd7ed537f854388836a12a05f078` |
| Host resources (P3) | Activation-owned typed resources, call-scoped authority checks, fresh restore factories, and opt-in child/fork propagation | [PR #155](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/155), merge `4e14618b3ba68ffd9e243a9bcad2e3c9d2f5ac48` |
| Termination budgets (P4) | Read-only process-local observations of call waits and stream-exit progress | [PR #165](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/165), merge `973e47492ab0c7ff0747af25776476f22704fcb5` |
| Stream authority (P5) | Authority checked before each streamed delta and buffered Unicode suffix reaches a consumer | [PR #148](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/148), merge `f19bbdcd18e66841a9cf4c5fee2da5bec4e6e89f` |
| Typed tools (DX1) | Validated Python model inputs and JSON-mode top-level model results | [PR #149](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/149), merge `5b08b10d82a948268d1a95b51112b1508e233e3d` |
| Integration recipes (DX2) | Installed first tool, request/passive streaming ownership, and approval resume in a fresh process | [PR #170](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/170), merge `b27a15c159465b978b50e89660fa5f7b7dd7d7b3` |

Historical implementation records remain in the [P1/P2](V0_25_P1_P2_PR_PLAN.md),
[P3](V0_25_P3_PR_PLAN.md), [P4](V0_25_P4_PR_PLAN.md),
[P5/DX1](V0_25_P5_DX1_PR_PLAN.md), and [DX2](V0_25_DX2_PR_PLAN.md) plans.
Their candidate results apply to the named source and inputs.

## Completed implementation qualification

The final DX2 integration head `07a84ae6dab971dd1ed13f554540f215b5af54dd` received a fresh cumulative
review with no blocking findings. The recorded automatic review completed with no findings.
Its required CI completed successfully:

| Check | Recorded result |
|---|---|
| [L1](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40613655101) | Success |
| [Required combined L2](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40613655177) | Success, including both core Python versions and combined services |
| [Full push CI](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40612881392) | All 14 jobs successful |
| Integration identity | Reviewed head and #170 merge have the same tree |

These results qualify the implementation source before documentation cleanup. Each later
documentation or release candidate needs review and CI for its final head.

The DX2 development wheel had SHA-256
`fea62fe2dd95ea5a7f61537287b451303c6f4156a0afa9682948cca4a912a0ab`, version `0.24.0`.
It was built for the DX2-03 implementation (`8d8277d`, [PR #169](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/169)).
The final DX2 integration preserved its packaged source, metadata, assets, examples, and installed
CI inputs, allowing that exact artifact's evidence to be reused at #170.

The wheel passed package audit. Installed consumer checks passed on Python 3.11.9 with Pydantic
2.14.1 and 2.6.4, outside the checkout with `PYTHONPATH` removed. Checks covered 9 parent and
2 gateway-child import provenances, 5 copied examples, 2 Stop variants, the existing host report
demo, and separate prepare/resume processes. Approval recovery retained the original IDs and
proposal, created one resource per process, changed approved reads from 0 to 1, and ignored
duplicate approval. Scripted cumulative usage changed from 5/3/8 to 9/6/15.

The [reader-preparation identity](migrations/v0.25.md#choose-a-deployment-path) has separate
historical qualification: actual binaries exercised v2 records, a v1-writing dual reader, and a
later v2 writer while preserving committed coordinates and hashes. A separately installed
released v0.24.0 reader rejected v2. Missing facts from new v1 revisions remained unavailable.
Those compatibility results support the migration choices; release artifacts still need their
own source and digest records.

## Documentation qualification

The documentation cleanup adds release/migration guidance, task-oriented entry points,
[tools and resources](guides/tools-and-resources.md), [streaming and Stop](guides/streaming-and-stop.md),
[approval and recovery](guides/approval-and-recovery.md), and [development history](history/README.md).
Existing handbook headings remain available. Current contribution and CI instructions are in
[Contributing](../CONTRIBUTING.md). Runtime and API inputs are unchanged by this work.

| Gate | Status |
|---|---|
| Release narrative, migration guide, and audit (DOC-01) | [PR #171](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/171), reviewed `f023600`, merged as `4aa189d`; [L1](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40620745652) and [required core L2](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40620745649) passed |
| User entry points and practical guides (DOC-02) | [PR #172](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/172), reviewed `39b7a67`, merged as `f41d95b`; [L1](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40623017325) and [required core L2](https://github.com/hoonseokyoon/monoid-agent-kernel/actions/runs/40623017335) passed |
| History navigation, contribution flow, and current-document validation (DOC-03) | Implemented at `5e1fcde`; final independent review and required CI pending |
| Documented CLI, compatibility ledger, packaging, and carriage checks | 118 passed at `5e1fcde` on Python 3.11.9 |
| Local/candidate-mapped links and code blocks | 503 links and 44 Python blocks passed across 45 documents; historical signature sketches are labelled as text |
| README rendering and exact development wheel | GitHub GFM and package-description rendering passed; wheel/installed evidence below |
| Final documentation integration | Pending final review and required CI for `codex/v0.25-release-docs` |

The discovery regression first failed because a frozen v0.24 command was treated as current.
The bounded repair excludes 14 exact historical paths. The same regression proves invalid commands
in new guides, migrations, current/future audits, the history index, README, and Reference docs
still reach the real Click tree. The parser and option validation are unchanged.

README edits required a fresh development wheel at `39b7a670d46b8c67419790c7022db4c80ffefce4`.
Its SHA-256 is `fd5040d191c4601bbcc09b292a096e22bc91b74ae25364939913b250db3e34f8`, version `0.24.0`.
Package audit passed and wheel metadata matched the candidate README. The exact wheel passed
the existing installed-consumer blocks on Python 3.11.9 with Pydantic 2.14.1 and 2.6.4 outside the
checkout, with `PYTHONPATH` removed: 9 parent and 2 gateway-child import provenances, 5 copied
examples and 2 Stop variants, and the packaged no-argument report plus separate prepare/resume.
Original IDs, resource lifetimes, approved-read count, duplicate handling, proposal, and usage
assertions passed. DOC-03 leaves these packaged source, metadata, example, and CI inputs unchanged.

Current tracked text and seven documentation images were inspected for private information.
Private integration descriptions, internal references, and workstation paths were removed through
ordinary commits. The inspected credential candidates were synthetic fixtures. Public gateway
audience values were preserved. This inspection covers the current tree; Git history is retained.
Release artifacts still need their own final source, digest, rendering, and installed checks.

## Release qualification and delivery

| Gate | Status and required evidence |
|---|---|
| Select final release source and update version metadata | Pending; record source SHA and verify all version locations |
| Build and audit exact wheel/sdist | Pending; record filenames, SHA-256, package contents, metadata, and README rendering |
| Install exact release wheel | Pending; verify imports and documented consumer paths in supported dependency environments |
| Compatibility and production qualification | Pending for release candidate; retain v1/v2, recovery, authority, and required service coverage |
| Final review and required CI | Pending on release source |
| Release branch integration, tag, and publication | Pending |
| Published artifact verification | Pending; match published artifacts and metadata to the qualified digests |

The release gate must retain the migration limits: released v0.24 rejects v2; rollback needs a
verified dual reader or coordinated backup; missing usage provenance stays unavailable; numeric
usage remains the accounting input. Stop snapshots cover process-local model waits, while tool,
child, resource cleanup, and billing reconciliation retain their own owners.
