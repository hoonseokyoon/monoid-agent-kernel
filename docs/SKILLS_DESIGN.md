# Agent Skills (progressive disclosure) design

Status: P1 + P2 implemented (branch `feat/skills-progressive-disclosure`).

## Goal

Equip a run with **Agent Skills** — reusable *procedural knowledge* (how to do a
specific task) delivered to the model by **progressive disclosure**, the Anthropic
`SKILL.md` model. A skill library of dozens of entries should cost almost nothing
until a skill is actually relevant, then reveal exactly as much as is needed.

Skills are the **knowledge layer**, complementary to the two delegation features
already in the runner:

| Layer | Feature | What it provides |
|---|---|---|
| Knowledge | **Skills** (this doc) | procedural how-to, progressively disclosed |
| Execution | Subagents (agent-as-tool) | isolated child runs, parallelism |
| Integration | MCP | tools/data from external systems |

## Core idea: attach via existing seams, change the core not at all

A skill is not a new engine concept. It rides two extension seams that already exist,
exactly as MCP rides `ToolProvider`:

- **`ContextProvider`** (`core/context.py`) — `static_segment()` folds a fixed segment
  into the system prompt at bootstrap.
- **`ToolProvider`** (`tools/base.py`) — `get_tools()` yields `ToolSpec`s registered in
  the run's tool registry.

`SkillProvider` (`skills/provider.py`) implements **both**, so one instance, registered
in both `AgentLoop.context_providers` and `AgentLoop.tool_providers`, delivers the whole
feature. The pump / parking / reentry loop is untouched.

## Three levels of disclosure

```
L1  catalog        SkillProvider.static_segment()
    (~100 tok/skill, always resident)
    "# Available Skills
     - pdf-fill: Fill PDF forms
     - commit-msg: Write commit messages
     Call the `skill` tool with a name to load full instructions."

L2  instructions   skill(name) tool  ->  ToolResult
    (on trigger; model picks a skill by its description — model-native, no router)
    { name, instructions: <SKILL.md body>, allowed_tools?: [...], resources?: [...] }

L3  resources      skill.read_file(name, path) tool  ->  ToolResult
    (on demand; bundled references/assets, path relative to the skill dir)
    { name, path, content: <utf-8 text> }
```

Once a skill is activated at L2, its instructions live in tool-result history, so there
is nothing to re-inject per turn (`dynamic_segment` returns `None`).

### Why a tool for L2 (not a filesystem read or a router)

The model decides *which* skill to load by reading the L1 catalog and calling `skill` —
this is Claude's description-based, model-native triggering, with no separate matcher.
Delivering the body as a **tool result** keeps the feature self-contained: it does not
depend on mounting the skills directory into the agent workspace or on `fs.read` scope.
(Decision confirmed with the user; the alternative filesystem-read model is closer to
Claude's literal implementation but couples skills to workspace paths.)

## Definition and discovery

`SkillDefinition` (`skills/definition.py`) is pure data: `name`, `description`,
`instructions`, `allowed_tools`, `directory`, `metadata`. `from_frontmatter(meta, body,
*, directory)` builds one from a parsed `SKILL.md`, reusing the zero-dependency
`core/frontmatter.parse_frontmatter` written for subagents.

`load_skill_definitions(dir)` (`skills/loader.py`, CLI `--skills-directory`) scans
recursively for `SKILL.md` files (the `<skills>/<skill-name>/SKILL.md` convention). The
skill name is the frontmatter `name` (falling back to the directory name); the SKILL.md's
parent directory is the bundle root for L3. Duplicates: first sorted path wins. A missing
directory or unparseable file raises `ValueError` (fail loud), mirroring the subagent
loader.

## allowed-tools is advisory (Claude parity)

`allowed-tools` in the frontmatter is surfaced to the model (in the `skill` tool result)
as a hint about which tools the skill expects, but it does **not** restrict the tool
registry. This matches Claude's actual behavior (pre-approval hint, not a hard block).
Enforced per-turn gating is a possible later option (`DynamicToolProvider`), deliberately
out of scope here.

## Safety

- **Path traversal**: `skill.read_file` resolves `path` against the skill directory and
  rejects anything escaping it (`skill_path_invalid`). `SKILL.md` itself is not readable
  as a resource (it is the L2 payload).
- **Resource bounds**: non-utf8/binary resources and oversized files are rejected with a
  typed error rather than dumped into context; the resource manifest is capped.

## Wiring (CLI)

`--skills-directory` → `load_skill_definitions` → one `SkillProvider` → registered in
both `context_providers` and `tool_providers`, with `provider.tool_bindings()` merged
into the runtime config (provider tools are not auto-bound, same as MCP).

## Observability (P2)

Activating a skill is a normal `skill` tool call, so it is already covered by the
`tool.call.started`/`finished` events and their `execute_tool` span. P2 adds a typed
semantic signal on top of that, *without* changing the core ToolContext contract:

- The `skill` tool handler **duck-types** an optional `record_skill_activation` hook on
  the tool context (`getattr(context, "record_skill_activation", None)`). The engine's
  `AgentToolContext` implements it; bare test stubs simply don't, so the handler degrades
  to a no-op. This keeps skills decoupled from the core — the loop never imports skills.
- `record_skill_activation` emits a `skill.activated` event whose `parent_id` is the
  current skill tool-call event, and bumps a report-only counter.
- The OTel sink treats `skill.activated` as a point-in-time event and **enriches the
  already-open skill tool span** (looked up by `parent_id`) with `skill.name` /
  `skill.resource_count`, rather than opening an orphan span.
- Run metrics gain `skill_activation_count` + `skills_activated`, mirroring the subagent
  roll-up (report-only; skills don't consume the parent's context budget the way an
  inlined tool result does, so they are surfaced for visibility, not summed elsewhere).

`allowed_tools` is echoed in the `skill` tool result as an advisory hint (it is not added
to the L1 catalog, which stays at the ~100-token name+description budget).

## Scope

- **P1**: L1 + L2 + L3, directory discovery, CLI, exports, tests, docs.
- **P2** (this revision): `skill.activated` event + OTel span enrichment + activation
  metrics; advisory allowed-tools echoed in the tool result.
- **P3**: `context: fork` — run a skill's body as a *subagent* (reuse the merged subagent
  machine); optional `skill.run_script` (execute a bundled script, output-only, code never
  enters context); optional enforced allowed-tools gating.
