---
name: commit-message
description: Write a clear, conventional git commit message from a short description of a change.
---

# Writing a good commit message

When the user asks for a commit message, produce a **Conventional Commits** subject line
followed by a short body.

## Subject line
- Format: `type(scope): summary`
- `type` is one of: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`.
- `scope` is the area touched (optional), e.g. `auth`, `api`, `ui`.
- `summary` is imperative mood, lower-case, no trailing period, ≤ 50 chars.

## Body (optional)
- One blank line after the subject.
- Explain *what* changed and *why*, wrapped at ~72 chars.
- Reference issues as `Refs #123` on their own line if relevant.

## Example
```
fix(auth): reject expired refresh tokens

The refresh endpoint accepted tokens past their expiry because the
clock-skew check used the wrong sign. Compare against issued_at + ttl.

Refs #482
```

Keep it concise — a reviewer should understand the change from the subject alone.
