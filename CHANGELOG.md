# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and this project is
pre-1.0 (`0.x`): minor versions may include breaking changes, which are called
out in commit messages and here.

## [Unreleased]

### Added
- Studio: agent-to-agent (A2A) demo over the durable outbox→inbox fabric — a
  one-click preset spins up two peer agents that message each other (lease-gated,
  idempotent, redrive-backed).
- Studio: inline image preview in the file viewer for both live workspace files
  and not-yet-applied proposal files.
- Open-source project files: contributing guide, code of conduct, security
  policy, CI workflow, environment template.

## [0.11.0]
- Baseline at first public preparation. See the git history for the full
  evolution of the contracts, session/control protocol, capability leases,
  inbox/outbox fabric, durable checkpoints, and the Studio reference app.
