---
name: incident-summary
description: Turn notes or logs into a clear incident timeline, impact statement, and follow-up list.
allowed-tools: fs.read text.search
---

# Incident summary

Use this skill when the user provides incident notes, logs, or a rough timeline and asks for a
postmortem-ready summary.

## Workflow

1. Extract absolute times, relative times, systems, user impact, detections, mitigations, and
   recovery signals.
2. Normalize the timeline into chronological order.
3. Separate observed facts from likely causes.
4. Write impact in user or operator terms.
5. Create follow-ups with an owner placeholder when the owner is unknown.

Use direct wording. State uncertainty explicitly with "Unknown" or "Likely" rather than hedging.

Use `skill.read_file` for `references/incident-template.md` when the user wants a structured
postmortem draft.
