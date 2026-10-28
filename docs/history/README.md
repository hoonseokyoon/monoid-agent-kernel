# Development history

These records preserve design choices, findings, commands, and qualification for their named
revisions. Their original paths and anchors remain available. Use the [documentation index](../README.md)
for current integration guidance and [Contributing](../../CONTRIBUTING.md) for the current PR workflow.

The [v0.25 release audit](../V0_25_RELEASE_AUDIT.md) remains a current document. It tracks source
integration, documentation qualification, and the pending release-artifact and publication gates.

## v0.25 implementation records

P1–P5, DX1, and DX2 were integrated through [PR #170](https://github.com/hoonseokyoon/monoid-agent-kernel/pull/170).
The investigation and proposal retain later ideas alongside the delivered scope. The task guides
and [migration guide](../migrations/v0.25.md) describe the current consumer API.

| Record | Purpose |
|---|---|
| [Original investigation](../V0_25_PRE_INVESTIGATION.md) | Priorities, implementation groups, and the original findings |
| [Developer experience proposal](../V0_25_DEVELOPER_EXPERIENCE_PROPOSAL.ko.md) | API choices, integration journeys, and later product directions |
| [P5 and DX1](../V0_25_P5_DX1_PR_PLAN.md) | Stream authority and typed tool implementation/review evidence |
| [P3](../V0_25_P3_PR_PLAN.md) | Host resource injection, authority, lifetime, restoration, and child inheritance |
| [P1 and P2](../V0_25_P1_P2_PR_PLAN.md) | Call facts, usage observations, durable formats, and binary rollback evidence |
| [P4](../V0_25_P4_PR_PLAN.md) | Termination budgets, clocks, stream-exit observations, and qualification |
| [DX2](../V0_25_DX2_PR_PLAN.md) | Executable developer journeys and installed-consumer qualification |

## Earlier plans and release records

| Period | Records |
|---|---|
| v0.24 | [Release audit](../V0_24_RELEASE_AUDIT.md) |
| v0.23 | [Implementation plan](../V0_23_IMPLEMENTATION_PLAN.md), [development workflow](../V0_23_DEVELOPMENT_WORKFLOW.md), [release audit](../V0_23_RELEASE_AUDIT.md) |
| v0.22 | [Implementation plan](../V0_22_IMPLEMENTATION_PLAN.md), [release audit](../V0_22_RELEASE_AUDIT.md) |
| v0.18 | [Release audit](../V0_18_RELEASE_AUDIT.md) |

Recorded candidate status and test results belong to the dates and revisions in each file.
Historical workflow instructions describe the procedure used then. Current CI triggers, profile
selection, and review requirements are described in [Contributing](../../CONTRIBUTING.md).

## Coverage and maintenance

[Phase 1S](../PHASE_1S_COVERAGE.md) points to the current operational matrix;
[Phase 4 closure](../PHASE_4_CLOSURE.md) records the backend structure pass.
[Operational rule coverage](../OPERATIONAL_RULE_COVERAGE.md), [subagents](../SUBAGENT_DESIGN.md),
and [skills](../SKILLS_DESIGN.md) remain current technical references.

The CLI documentation check excludes exactly the 14 versioned historical files listed above,
along with the changelog. It continues to discover README, this index, new guides and migrations,
current/future release audits, and Reference Markdown. Its negative regression checks that an
invalid command in a new guide reaches the real Click command tree.
