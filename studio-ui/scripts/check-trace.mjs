import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import ts from "typescript";

const source = await readFile(new URL("../src/lib/trace.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
  fileName: "trace.ts",
});
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled.outputText).toString("base64")}`;
const {
  createCompactTraceExport,
  createRawTraceExport,
  isLegacyModelDeltaEvent,
  operationTraceEvents,
  summarizeTrace,
} = await import(moduleUrl);

const event = (type, data = {}, seq = 0) => ({
  seq,
  event_id: `event-${seq}`,
  type,
  data,
});

const events = [
  event("run.created", {}, 1),
  event("model.output.delta", { text: "A" }, 2),
  event("model.reasoning.delta", { text: "한🙂" }, 3),
  event("model.tool_call.delta", { text: "future operation" }, 4),
  event("model.output.delta", { text: "B", studio_scope: "subagent" }, 5),
  event("turn.settled", { final_text: "AB" }, 6),
];

assert.equal(isLegacyModelDeltaEvent(events[1]), true);
assert.equal(isLegacyModelDeltaEvent(events[2]), true);
assert.equal(isLegacyModelDeltaEvent(events[3]), false);

const operations = operationTraceEvents(events);
assert.deepEqual(operations.map(({ type }) => type), [
  "run.created",
  "model.tool_call.delta",
  "turn.settled",
]);
assert.equal(operations[0], events[0]);

const summary = summarizeTrace(events);
assert.equal(summary.sourceEventCount, 6);
assert.equal(summary.operationEvents.length, 3);
assert.equal(summary.omittedDeltaEventCount, 3);
assert.equal(summary.omittedDeltaTextBytes, 9);
assert.deepEqual(summary.omittedDeltaTypes["model.output.delta"], {
  eventCount: 2,
  textBytes: 2,
});
assert.deepEqual(summary.omittedDeltaTypes["model.reasoning.delta"], {
  eventCount: 1,
  textBytes: 7,
});

const raw = createRawTraceExport(events);
assert.equal(raw.schema_version, "studio.trace-export.v1");
assert.deepEqual(raw.events, events);
assert.notEqual(raw.events, events);

const compact = createCompactTraceExport(events);
assert.equal(compact.schema_version, "studio.trace-export.compact.v1");
assert.deepEqual(compact.summary, {
  source_event_count: 6,
  operation_event_count: 3,
  omitted_delta_event_count: 3,
  omitted_delta_text_bytes: 9,
  omitted_delta_types: {
    "model.output.delta": { event_count: 2, text_bytes: 2 },
    "model.reasoning.delta": { event_count: 1, text_bytes: 7 },
  },
});
assert.deepEqual(compact.events, operations);

const operationsOnly = [event("run.started", {}, 1)];
const operationsOnlySummary = summarizeTrace(operationsOnly);
assert.equal(operationsOnlySummary.omittedDeltaEventCount, 0);
assert.deepEqual(operationsOnlySummary.operationEvents, operationsOnly);

const deltasOnly = [
  event("model.output.delta", { text: null }, 1),
  event("model.reasoning.delta", { text: 7 }, 2),
];
const deltasOnlyCompact = createCompactTraceExport(deltasOnly);
assert.equal(deltasOnlyCompact.summary.operation_event_count, 0);
assert.equal(deltasOnlyCompact.summary.omitted_delta_event_count, 2);
assert.equal(deltasOnlyCompact.summary.omitted_delta_text_bytes, 0);
assert.deepEqual(deltasOnlyCompact.events, []);

console.log("Trace compaction checks passed.");
