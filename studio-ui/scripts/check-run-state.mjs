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
const { initialRunState, isRunBusy, reduceRunEvent } = await import(moduleUrl);

function event(type, data, seq) {
  return {
    type,
    data,
    seq,
    event_id: `event-${seq}`,
    timestamp: "2026-01-01T00:00:00Z",
  };
}

let state = reduceRunEvent(
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

assert.equal(isRunBusy("running"), true);
assert.equal(isRunBusy("queued"), true);
assert.equal(isRunBusy("awaiting-approval"), true);
assert.equal(isRunBusy("stopping"), true, "stop and pause requests must keep the composer busy");
assert.equal(isRunBusy("stopped"), false);

console.log("Run-state checks passed (9 scenarios).");
