# Contributing

Thanks for your interest in improving Monoid Agent Kernel. This is a pre-1.0
(`0.x`) agent kernel, so the public surface may still change. Contributions,
issues, and design feedback are welcome.

## Ground rules

- Be respectful. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- By contributing, you agree your contributions are licensed under the project's
  license (see `LICENSE`).
- Keep the layering intact: **core never imports `reference`**. New example
  services belong under `monoid_agent_kernel.reference.*`; the supported surface is
  `monoid_agent_kernel.contracts`.

## Development setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install ruff                                     # linter (configured in pyproject.toml)
```

## Before you open a PR

```bash
ruff check src tests       # lint (line-length 100, target py311)
python -m pytest -n 4 -q -m "(unit or contract) and not serial"  # parallel shard
python -m pytest -q -m "(integration or serial) and not live"  # serial shard
```

- Add or update tests for any behavior change. Custom adapters/workspaces/stores
  should plug into the existing parametrized contract suites in `tests/`
  (e.g. `test_workspace_contract.py`, `test_checkpoint_store_contract.py`).
- Backend tests that create `RunnerBackend` should use the `backend_factory`
  fixture or the helpers in `tests/support/backend_harness.py`; the fixture owns
  spawned futures and fails the test if a backend leaves live runs behind.
- Use `python -m pytest --durations=30` when a change could affect suite runtime.
- Every test receives exactly one primary tier from `tests/support/test_tiers.py`:
  `unit`, `contract`, or `integration`. Update that policy when a new module crosses
  a different boundary. `slow`, `live`, and `serial` are orthogonal traits.
- Mark deliberate timeout or live-provider coverage with `slow` or `live`. Integration
  and serial contract tests run in the required serial shard; worker-safe unit and
  contract tests run under xdist.
- CI requires both shards, branch coverage, minimal/all-extras install smoke, and a
  small Windows/macOS platform-sensitive smoke matrix.
- To profile without unrelated pytest plugins, run
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p xdist -n 4 -q -m "not slow and not live"`.
- Match the surrounding code style: typed, small functions, comment density and
  naming consistent with the file you are editing.
- Keep changes focused; note any breaking change to the public surface in the PR
  description and in `CHANGELOG.md`.

## Reporting bugs / proposing features

Open a GitHub issue with a clear description and, for bugs, a minimal reproduction.
For security issues, follow [SECURITY.md](SECURITY.md) instead of filing a public
issue.

## Commit messages

Conventional, imperative style is appreciated (e.g. `feat(studio): ...`,
`fix(tasks): ...`), with breaking changes called out explicitly.
