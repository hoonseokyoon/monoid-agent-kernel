---
name: code-review-checklist
description: Review a small code change for defects, missing tests, and behavioral regressions.
allowed-tools: fs.read text.search skill.read_file
context: fork
---

# Code review checklist

Review the requested change as a focused reviewer.

Prioritize findings over summaries. Report concrete issues with file paths, line numbers when
available, severity, and the specific behavior that can fail. Skip style preferences unless they
hide a bug or create maintenance risk.

## Review order

1. Identify the changed files or the files the user named.
2. Read the surrounding implementation, tests, and public contracts.
3. Look for incorrect behavior, missed edge cases, state leaks, race conditions, unsafe I/O,
   permission gaps, and missing tests.
4. Return findings first, ordered by severity.
5. Add a short residual-risk note only when useful.

Use `skill.read_file` for `references/review-checklist.md` when you need a compact checklist.
