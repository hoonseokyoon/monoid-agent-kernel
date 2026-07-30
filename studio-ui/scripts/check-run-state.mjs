import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import ts from "typescript";

const source = await readFile(new URL("../src/lib/run-state.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
  fileName: "run-state.ts",
});
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled.outputText).toString("base64")}`;
const { hydrateTranscript, initialRunState, isRunBusy, reduceRunEvent } = await import(moduleUrl);

function event(type, data, seq) {
  return {
    type,
    data,
    seq,
    event_id: `event-${seq}`,
    timestamp: "2026-01-01T00:00:00Z",
  };
}

let state = hydrateTranscript(initialRunState(), {
  schema_version: "studio.chat.v1",
  run_id: "v1-run",
  messages: [],
  event_cursor: -1,
});
assert.equal(state.eventLogError, null, "a supported v1 transcript has no degradation signal");

state = hydrateTranscript(initialRunState(), {
  schema_version: "studio.chat.v2",
  run_id: "clean-v2-run",
  messages: [],
  event_cursor: 3,
  event_log_error: "",
});
assert.equal(state.eventLogError, null, "a clean v2 transcript must not show a warning");

const degradedMessage = {
  id: "assistant:safe-prefix",
  role: "assistant",
  content: "safe prefix",
  attachments: [],
  created_at: 1,
};
state = hydrateTranscript(initialRunState(), {
  schema_version: "studio.chat.v2",
  run_id: "degraded-v2-run",
  messages: [degradedMessage],
  event_cursor: 7,
  event_log_error: "events.jsonl line 9 is not valid JSON",
});
assert.equal(state.eventLogError, "events.jsonl line 9 is not valid JSON");
assert.equal(state.messages[0], degradedMessage, "the readable transcript prefix must remain visible");
assert.equal(state.replayCursor, 7);

state = hydrateTranscript(
  {
    ...initialRunState("failed-run-before-hydration"),
    status: "failed",
    error: "provider failed",
    errorRetryable: true,
    manualRetryCandidate: true,
    manualRetryReady: true,
  },
  {
    schema_version: "studio.chat.v2",
    run_id: "failed-run-before-hydration",
    messages: [],
    event_cursor: 2,
    event_log_error: "event sequence is descending",
  },
);
assert.deepEqual(
  {
    status: state.status,
    error: state.error,
    errorRetryable: state.errorRetryable,
    manualRetryCandidate: state.manualRetryCandidate,
    manualRetryReady: state.manualRetryReady,
  },
  {
    status: "failed",
    error: "provider failed",
    errorRetryable: true,
    manualRetryCandidate: true,
    manualRetryReady: true,
  },
  "transcript degradation must not replace run failure or retry state",
);

state = reduceRunEvent(
  initialRunState("failed-run"),
  event("run.failed", { error: "provider failed" }, 1),
);
state = reduceRunEvent(
  state,
  event("run.finished", { status: "failed", error: "provider failed", error_code: "provider_error" }, 2),
);
assert.equal(state.status, "failed");
assert.equal(state.error, "provider failed");
assert.equal(state.messages.length, 1, "run.finished must not duplicate the prior failure message");

state = reduceRunEvent(
  initialRunState("limited-run"),
  event("run.finished", { status: "limited", error_code: "max_tool_calls_exceeded" }, 1),
);
assert.equal(state.status, "failed");
assert.equal(state.error, "max_tool_calls_exceeded");

state = reduceRunEvent(
  { ...initialRunState("completed-run"), error: "stale error" },
  event("run.finished", { status: "completed" }, 1),
);
assert.equal(state.status, "succeeded");
assert.equal(state.error, null);

state = reduceRunEvent(
  initialRunState("legacy-run"),
  event("run.finished", {}, 1),
);
assert.equal(state.status, "succeeded");

// --- turn.settled text, after v0.20 moved model-authored output off the event stream ---
//
// The reducer arm reads `data.final_text ?? next.activeResponse ?? ""` and appends nothing when
// that resolves empty. Nothing else in this repo asserts anything about event *content*, so a
// dropped answer would have shipped as a blank transcript with no throw, no log and no
// "undefined" — the frontend normalises absent to "" and hides on falsiness at every layer.
// These four scenarios pin each branch of that expression.

// 1. The shape a Studio consumer actually receives: hydration filled `final_text` back in
//    alongside the digest the kernel published. The answer must render.
state = reduceRunEvent(
  initialRunState("hydrated-run"),
  event("turn.settled", { status: "completed", final_text: "hydrated answer", final_text_digest: "abc123", final_text_len: 15 }, 1),
);
assert.equal(state.messages.length, 1, "a hydrated settle event must render its answer");
assert.equal(state.messages[0].content, "hydrated answer");

// 2. Kernel-authored text stays inline with no digest — the selectivity half. If the provenance
//    flag were stuck on, this branch would silently lose the one line explaining why a run stopped.
state = reduceRunEvent(
  initialRunState("limit-run"),
  event("turn.settled", { status: "limited", final_text: "Stopped after reaching max steps." }, 1),
);
assert.equal(state.messages.length, 1, "kernel-authored settle text must stay renderable");
assert.equal(state.messages[0].content, "Stopped after reaching max steps.");

// 3. Digest with no text and no preceding deltas — an unhydrated read. This appends NOTHING, and
//    that is the current, deliberate behaviour rather than an accident: asserted so a future
//    placeholder is a conscious change to this file, and so the silent-drop mode stays documented.
state = reduceRunEvent(
  initialRunState("unhydrated-run"),
  event("turn.settled", { status: "completed", final_text_digest: "abc123", final_text_len: 15 }, 1),
);
assert.equal(state.messages.length, 0, "an unresolved digest must not fabricate an empty message");

// 4. The `?? activeResponse` fallback, which only exists when token streaming is on. It is not a
//    general safety net for a missed hydration path — it is empty whenever deltas are disabled.
state = reduceRunEvent(
  initialRunState("streamed-run"),
  event("model.output.delta", { text: "streamed answer" }, 1),
);
state = reduceRunEvent(state, event("turn.settled", { status: "completed", final_text_digest: "abc123" }, 2));
assert.equal(state.messages.length, 1, "streamed deltas must still render when the event carries no text");
assert.equal(state.messages[0].content, "streamed answer");

assert.equal(isRunBusy("running"), true);
assert.equal(isRunBusy("queued"), true);
assert.equal(isRunBusy("awaiting-approval"), true);
assert.equal(isRunBusy("stopping"), true, "stop and pause requests must keep the composer busy");
assert.equal(isRunBusy("stopped"), false);

console.log("Run-state checks passed (17 scenarios).");
