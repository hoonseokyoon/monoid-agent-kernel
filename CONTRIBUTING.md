# Contributing

Thanks for your interest in improving Native Agent Runner. This is a pre-1.0
(`0.x`) research package, so the public surface may still change — but
contributions, issues, and design feedback are very welcome.

## Ground rules

- Be respectful. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- By contributing, you agree your contributions are licensed under the project's
  license (see `LICENSE`).
- Keep the layering intact: **core never imports `reference`**. New example
  services belong under `native_agent_runner.reference.*`; the supported surface is
  `native_agent_runner.contracts`.

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
pytest -q                  # full suite
```

- Add or update tests for any behavior change. Custom adapters/workspaces/stores
  should plug into the existing parametrized contract suites in `tests/`
  (e.g. `test_workspace_contract.py`, `test_checkpoint_store_contract.py`).
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
