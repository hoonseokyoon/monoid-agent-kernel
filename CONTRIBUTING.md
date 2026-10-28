# Contributing

Monoid Agent Kernel is a pre-1.0 (`0.x`) agent kernel. Contributions, issues, and design feedback
are welcome. Public behavior changes are recorded in the changelog and compatibility ledger.

## Ground rules

- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Contributions use the project's [license](LICENSE).
- Keep the layering intact: **core never imports `reference`**. Example services belong under
  `monoid_agent_kernel.reference.*`; the supported surface is `monoid_agent_kernel.contracts`.

## Development setup

Requires Python 3.11+. Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install ruff==0.15.6
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Ruff is configured in
`pyproject.toml`; CI uses the same pinned version.

Studio development uses the Node.js version in `.nvmrc` and the npm version in
`studio-ui/package.json`. From the repository root:

```bash
nvm install
nvm use
node --version  # v24.18.0
npm --version   # 11.16.0
```

With nvm-windows, use `nvm install 24.18.0` and `nvm use 24.18.0` from an elevated terminal.

## Choose focused checks

Use tests closest to the changed behavior during development. For documentation changes:

```bash
ruff check src tests tools/v023_ci.py
python -m pytest -q -n auto --dist=worksteal --timeout=120 --timeout-method=thread \
  tests/test_documented_cli_invocations.py tests/test_compatibility_ledger.py \
  tests/test_release_packaging.py
```

Set `PYTEST_XDIST_AUTO_NUM_WORKERS=4` to bound the parallel runner on a shared machine.
[tests/support/test_tiers.py](tests/support/test_tiers.py) assigns every test a primary tier:
`unit`, `contract`, or `integration`. `slow`, `live`, `serial`, and `service` are additional traits.
Run serial/integration selections without xdist. The workflow files below contain the exact
parallel and serial shard commands used in CI.

- Add or update tests for behavior changes. Custom adapters, workspaces, and stores should join
  existing parametrized contract suites such as `test_workspace_contract.py` and
  `test_checkpoint_store_contract.py`.
- Backend tests use `backend_factory` or `tests/support/backend_harness.py`; the fixture owns
  spawned futures and detects leaked runs.
- Add `--durations=30` when a change could affect suite runtime. Keep expensive and live-provider
  coverage in the corresponding explicit selections.
- Commands in README, current `docs/` files, and Reference Markdown resolve against the real
  Click tree. [Historical records](docs/history/README.md) have an exact exclusion list in the
  existing test; new guides, migrations, and versioned audits stay covered.
- README supplies the package description. After editing it, check rendered Markdown and built
  wheel metadata. Existing install-smoke CI exercises copied examples outside the checkout.
- Match the surrounding code style and keep changes focused. Describe breaking behavior in the
  PR and `CHANGELOG.md`.

## Before you open a PR

State the user-visible change, actual base revision, affected contracts, and validation results.
For related PRs, keep dependencies and integration order in a shared plan, refresh each part's
preflight against its actual base, and obtain independent review before integration. Use merge
commits for campaign integration. Retain valid evidence when its relevant inputs are unchanged.

The current CI entry points are:

| Gate | When it runs | Coverage |
|---|---|---|
| [L1](.github/workflows/pr-fast.yml) | PR opened, reopened, synchronized, or converted to draft | Lock validation, Ruff, deterministic unit/contract tests on Python 3.12 |
| [Required L2](.github/workflows/pr-full.yml) | Eligible non-draft open/reopen, Ready for review, or CI-profile label event | Core unit/contract tests on Python 3.11/3.12, serial core shard, selected services, required gate |
| [Full CI](.github/workflows/ci.yml) | Push to the workflow's listed branches, including `develop` and `main` | Integration, coverage, platforms, exact-wheel installs, services, and Studio build checks |

Ordinary branches default to the `core` L2 profile. Select at most one of `ci:core`, `ci:postgres`,
`ci:objectstore`, `ci:temporal`, or `ci:combined` when the changed boundary needs it. Named campaign
branches can require a particular profile; [tools/v023_ci.py](tools/v023_ci.py) resolves and validates
that choice. [Service test setup](tests/service/README.md) describes the local prerequisites.

A push to an already-ready PR refreshes L1. Refresh required L2 for the new head by returning the
PR to draft and marking it ready, or by changing the relevant CI-profile label. Manual workflow
dispatch creates diagnostic L2 evidence. Before merging, confirm independent review and required
checks cover the final head and its current base; a dormant or diagnostic run has its named scope.
Automatic review comments are additional evidence to assess against the actual changed behavior.

The [history index](docs/history/README.md) links versioned workflow records. This guide and the
current workflow sources define the contributor path. Version bumps, release artifacts, tags,
and publication follow the [current release audit](docs/V0_25_RELEASE_AUDIT.md).

## Reporting bugs / proposing features

Open a GitHub issue with a clear description and a minimal reproduction for bugs.
For security issues, follow [SECURITY.md](SECURITY.md).

## Commit messages

Use conventional, imperative messages (for example `feat(studio): ...` or `fix(tasks): ...`)
and call out breaking changes explicitly.
