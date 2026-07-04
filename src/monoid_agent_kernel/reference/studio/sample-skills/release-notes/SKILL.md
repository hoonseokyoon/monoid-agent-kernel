---
name: release-notes
description: Draft concise release notes from a changelog, commit list, or user-provided change summary.
allowed-tools: fs.read text.search
---

# Release notes

Use this skill when the user needs release notes, upgrade notes, or a concise summary of
changes for a version.

## Workflow

1. Gather the source changes from the files or text the user names.
2. Group changes by user impact: Added, Changed, Fixed, Removed, Security.
3. Write only entries that a user or operator can act on.
4. Keep internal implementation detail out of the main notes unless it changes behavior.
5. Call out breaking changes and migrations before the grouped list.

Use `skill.read_file` for `references/release-note-template.md` when the user wants a
structured release note.

When the workspace has a plain text file of one change per line, read it directly and group
the entries in the response.
