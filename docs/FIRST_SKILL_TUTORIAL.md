# Build Your First Skill

This tutorial creates a minimal Agent Skill, loads it from disk, and activates it with
the `skill` tool. The smoke test runs offline.

## Create the skill

From the repository root:

```bash
mkdir -p tmp-skills/polite-summary
cat > tmp-skills/polite-summary/SKILL.md <<'EOF'
---
name: polite-summary
description: Summarize notes in a concise, friendly tone.
allowed-tools: fs.read
---

# Polite summary

When the user asks for a summary:

1. Read the source text the user names.
2. Write three bullets at most.
3. Use direct wording and keep the tone friendly.
4. End with one concrete next step when the source implies one.
EOF
```

The frontmatter defines the Level 1 catalog entry:

- `name`: the identifier passed to the `skill` tool.
- `description`: the short catalog text the model sees before activation.
- `allowed-tools`: an inline-skill hint for the model. The parent tool surface stays
  unchanged. Fork skills use this field as an enforced subagent allowlist.

The Markdown body is the Level 2 instruction payload returned when the skill is activated.

## Load and activate it offline

```bash
python - <<'PY'
from pathlib import Path

from monoid_agent_kernel.skills import SkillProvider, load_skill_definitions

provider = SkillProvider(load_skill_definitions(Path("tmp-skills")))
tools = {tool.id: tool for tool in provider.get_tools()}
result = tools["skill"].handler(None, {"name": "polite-summary"})

print(result.ok)
print(result.content["name"])
print(result.content["instructions"])
print(result.content.get("allowed_tools"))
PY
```

Expected output includes:

```text
True
polite-summary
# Polite summary
['fs.read']
```

## Add a bundled resource

Skills can include small reference files that stay out of context until the model calls
`skill.read_file`.

```bash
mkdir -p tmp-skills/polite-summary/references
cat > tmp-skills/polite-summary/references/style.md <<'EOF'
# Style

- Prefer active voice.
- Keep bullets parallel.
- Replace vague labels with concrete nouns.
EOF
```

Read the resource:

```bash
python - <<'PY'
from pathlib import Path

from monoid_agent_kernel.skills import SkillProvider, load_skill_definitions

provider = SkillProvider(load_skill_definitions(Path("tmp-skills")))
tools = {tool.id: tool for tool in provider.get_tools()}
loaded = tools["skill"].handler(None, {"name": "polite-summary"})
resource = tools["skill.read_file"].handler(
    None,
    {"name": "polite-summary", "path": "references/style.md"},
)

print(loaded.content["resources"])
print(resource.content["content"])
PY
```

## Use it in a run

Programmatic users attach one `SkillProvider` as both a context provider and a tool provider,
then merge `provider.tool_bindings()` into the runtime config. The CLI does the same with:

```bash
monoid run \
  --workspace examples/workspaces/edit_markdown_notes \
  --runtime-config-file examples/runtime-config.json \
  --instruction "Use polite-summary on notes.md." \
  --skills-directory tmp-skills
```

Use your generated runtime config in place of `examples/runtime-config.json` when running a
custom profile. The offline snippets above verify the skill directory, frontmatter, activation
tool, and resource reader without a model.
