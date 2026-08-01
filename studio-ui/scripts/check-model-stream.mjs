import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import ts from "typescript";

const source = await readFile(new URL("../src/lib/model-stream.ts", import.meta.url), "utf8");
const appSource = await readFile(new URL("../src/App.svelte", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
  fileName: "model-stream.ts",
});
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled.outputText).toString("base64")}`;
const {
  ModelStreamEventSource,
  decodeModelContentResponse,
  decodeModelStreamFrame,
  discardModelStreamAttempt,
  initialModelStreamState,
  markModelStreamPartialSuperseded,
  markModelStreamHydrated,
  modelStreamCallKey,
  projectModelStreamFrame,
  projectSubagentModelStream,
  reduceModelStreamFrame,
  restoreActiveModelContent,
  sealModelStreamTurn,
  seedModelStreamSnapshot,
  seedSubagentModelContent,
  projectModelContentSnapshot,
  projectSubagentStarted,
} = await import(moduleUrl);

// App-level ownership: a durable terminal boundary cancels pending hydration/reconnect work. A
// retry/recovery can reopen the same run. The sessions high-watermark, rather than the narrower
// chat cursor, fences the complete committed prefix for a process-lost recoverable run.
assert.match(appSource, /const replayedThrough = run\.lastSeq;/);
assert.match(appSource, /event\.seq <= Math\.max\(run\.replayCursor, replayedThrough\)/);
assert.match(appSource, /const initialReplayEvent = isInitialReplayEvent\(event, initialReplayBoundary\);/);
assert.match(
  appSource,
  /const live = \(await studioApi\.sessions\(\)\)\.sessions\.find\([\s\S]*?summary: live \?\? cached,\s*exact: live !== undefined,[\s\S]*?\} catch \{\s*return \{ summary: cached, exact: false \};/,
  "session open must prefer a live unfiltered summary and use the sidebar cache only on request failure",
);
assert.match(appSource, /modelStreamRunTerminal = summaryIsExact && summary\?\.terminal === true;/);
assert.match(
  appSource,
  /modelStreamRecoveryFenced = !summaryIsExact\s+\|\| summary\?\.recoverable === true\s+\|\| summary\?\.state === "paused";/,
);
assert.match(
  appSource,
  /summaryIsExact \? summary\?\.last_event_seq \?\? -1 : Number\.MAX_SAFE_INTEGER,/,
  "an unknown live summary must fence the whole operation replay until explicit reactivation",
);
assert.match(appSource, /const historicalTraceDrain = historical && !revive;/);
assert.match(appSource, /if \(\(!revive && !historicalTraceDrain\)/);
assert.match(appSource, /stopped: historicalTraceDrain,\s+historicalTraceDrain,/);
assert.match(
  appSource,
  /else if \(poller\.stopped && successfulPage && !needsFinalDrain\) \{\s*childPollers\.delete\(childRunId\);/,
  "a stopped historical child poller must exit after reaching the durable trace tail",
);
assert.match(
  appSource,
  /handleSubagentLifecycle\(\s*event,\s*parentRunId,\s*epoch,\s*childRunId,\s*poller\.historicalTraceDrain,\s*poller\.historicalTraceDrain,/,
  "a recovered child trace drain must keep nested child lifecycle events behind the recovery fence",
);
assert.match(appSource, /if \(!eventAlreadyProjected && runTerminal\)/);
assert.match(appSource, /modelStreamRunTerminal = true;\s+modelStreamHydrationKey = null;/);
assert.match(appSource, /if \(modelStreamRunTerminal\s+\|\| modelStreamRecoveryFenced\s+\|\| modelStreamHydrationKey === key/);
assert.match(appSource, /modelStreamRunTerminal = false;\s+modelStreamRecoveryFenced = false;\s+modelStreamState = initialModelStreamState\(runId\);/);
assert.match(appSource, /response\.retry_of_turn_id/);
assert.match(appSource, /discardModelStreamAttempt\(run, modelStreamState, runId, retryOfTurnId\)/);
assert.match(appSource, /markModelStreamPartialSuperseded\(modelStreamState, retryOfTurnId\)/);
assert.match(
  appSource,
  /event\.type === "run\.resumed"[\s\S]*?event\.data\.reason === "studio-retry"[\s\S]*?event\.turn_id/,
);

const ROOT = "run-root";
const GENERATION = "generation-a";

function common(kind, sequence, overrides = {}) {
  return {
    schema_version: "monoid.model-stream.live.v1",
    cursor: `${GENERATION}:${sequence}`,
    sequence,
    kind,
    root_run_id: ROOT,
    run_id: ROOT,
    turn_id: "turn-1",
    stream_id: "stream-1",
    step: 1,
    started_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

const opened = (sequence, overrides = {}) => common("opened", sequence, {
  provider: "test",
  model: "test-model",
  started_at: "2026-01-01T00:00:00Z",
  ...overrides,
});
const byteLength = (text) => new TextEncoder().encode(text).byteLength;
const delta = (sequence, channel, text, overrides = {}) => {
  const start = overrides.start_offset ?? 0;
  return common("delta", sequence, {
    channel,
    text,
    start_offset: start,
    end_offset: overrides.end_offset ?? start + byteLength(text),
    ...overrides,
  });
};
const closed = (sequence, status = "completed", overrides = {}) => common("closed", sequence, {
  finished_at: "2026-01-01T00:00:01Z",
  status,
  partial: status !== "completed",
  ...overrides,
});

function runState() {
  return {
    runId: ROOT,
    status: "running",
    messages: [],
    activeResponse: "",
    reasoning: "",
    events: [{ type: "model.turn.started", data: {}, event_id: "durable-1" }],
    lastSeq: 0,
    replayCursor: -1,
    approvals: [],
    usage: { input: 0, output: 0, total: 0 },
    plan: [],
    planTruncatedItems: 0,
    eventLogError: null,
    error: null,
    errorRetryable: false,
    manualRetryCandidate: false,
    manualRetryReady: false,
    proposalDirty: false,
    lastUserMessage: "",
  };
}

assert.equal(decodeModelStreamFrame(opened(1))?.kind, "opened");
assert.equal(decodeModelStreamFrame(delta(2, "output", "hello"))?.kind, "delta");
assert.equal(decodeModelStreamFrame(closed(3, "failed", {
  final_text: "hello",
  retryable: true,
}))?.retryable, true);
assert.equal(decodeModelStreamFrame({
  schema_version: "monoid.model-stream.live.v1",
  cursor: `${GENERATION}:4`,
  sequence: 4,
  kind: "reset",
  root_run_id: ROOT,
  reason: "cursor_gap",
  oldest_available_cursor: `${GENERATION}:3`,
  latest_cursor: `${GENERATION}:8`,
})?.kind, "reset");
assert.equal(decodeModelStreamFrame({ ...delta(2, "output", "x"), sequence: 7 }), null);
assert.equal(decodeModelStreamFrame({ ...delta(2, "tool", "secret fragment") }), null);
assert.equal(decodeModelStreamFrame({ ...delta(2, "output", "x"), step: 0 }), null);
assert.equal(decodeModelStreamFrame({ ...delta(2, "output", "🙂"), end_offset: 2 }), null);
assert.equal(decodeModelStreamFrame({ ...delta(2, "output", "x"), schema_version: "future.v2" }), null);
assert.equal(
  decodeModelStreamFrame(closed(3, "failed", { retryable: 1 })),
  null,
  "closed-frame retryability must be a boolean",
);
assert.equal(decodeModelStreamFrame({
  schema_version: "monoid.model-stream.live.v1",
  cursor: "generation-recreated:0",
  sequence: 0,
  kind: "reset",
  root_run_id: ROOT,
  reason: "generation_changed",
  latest_cursor: "generation-recreated:0",
})?.kind, "reset", "an LRU-recreated root may have no oldest retained cursor");

let live = initialModelStreamState(ROOT);
let run = runState();
let before = live;
live = reduceModelStreamFrame(live, opened(1));
run = projectModelStreamFrame(run, before, live, opened(1));
before = live;
live = reduceModelStreamFrame(live, delta(2, "reasoning", "think "));
run = projectModelStreamFrame(run, before, live, delta(2, "reasoning", "think "));
before = live;
live = reduceModelStreamFrame(live, delta(3, "output", "answer"));
run = projectModelStreamFrame(run, before, live, delta(3, "output", "answer"));
assert.equal(run.reasoning, "think ");
assert.equal(run.activeResponse, "answer");
assert.equal(run.events.length, 1, "live model content must stay out of the durable Trace event list");

const duplicate = reduceModelStreamFrame(live, delta(3, "output", "answer"));
assert.equal(duplicate, live, "a replayed generation:sequence cursor must be deduplicated");

const fallbackState = initialModelStreamState(ROOT);
const fallbackOpened = opened(1, { stream_id: "fallback-stream" });
const fallbackAfter = reduceModelStreamFrame(fallbackState, fallbackOpened);
const fallbackRun = projectModelStreamFrame({
  ...runState(),
  messages: [{
    id: "assistant:model-stream:fallback-stream:active",
    role: "assistant",
    content: "hydrated prefix",
    attachments: [],
    created_at: 1,
    source: { kind: "model_stream_active", stream_id: "fallback-stream" },
  }],
}, fallbackState, fallbackAfter, fallbackOpened);
assert.equal(
  fallbackRun.messages.length,
  0,
  "a valid live root frame must replace a transcript-only active placeholder",
);
const childOpened = opened(4, {
  run_id: `${ROOT}.sub.child`,
  stream_id: "child-stream",
  turn_id: "child-turn",
});
before = live;
live = reduceModelStreamFrame(live, childOpened);
run = projectModelStreamFrame(run, before, live, childOpened);
let childActivity = projectSubagentModelStream(undefined, before, live, childOpened);
const childDelta = delta(5, "output", "child answer", {
  run_id: `${ROOT}.sub.child`,
  stream_id: "child-stream",
  turn_id: "child-turn",
});
before = live;
live = reduceModelStreamFrame(live, childDelta);
run = projectModelStreamFrame(run, before, live, childDelta);
childActivity = projectSubagentModelStream(childActivity, before, live, childDelta);
assert.equal(live.calls[modelStreamCallKey(`${ROOT}.sub.child`, "child-stream")].output, "child answer");
assert.equal(run.activeResponse, "answer", "descendant content must not enter the root chat bubble");
assert.equal(childActivity.childRunId, `${ROOT}.sub.child`);
assert.equal(childActivity.liveOutput, "child answer");
assert.equal(childActivity.events.length, 0, "child content must not become pseudo trace events");

const rootClose = closed(6, "completed", { final_text: "answer" });
before = live;
live = reduceModelStreamFrame(live, rootClose);
run = projectModelStreamFrame(run, before, live, rootClose);
assert.equal(
  run.activeResponse,
  "answer",
  "a current completed close must stay visible until the durable settle arrives",
);
assert.equal(run.messages.length, 0, "the hydrated/durable transcript owns completed answers");
assert.equal(run.events.length, 1);

// The durable settle owns the committed answer and clears the live projection. A close replayed
// after that operation is terminal already and cannot bring the bubble back.
live = sealModelStreamTurn(live, "turn-1");
run = { ...run, status: "idle", activeResponse: "", reasoning: "" };
run = projectModelStreamFrame(run, live, live, rootClose);
assert.equal(run.activeResponse, "", "settle-after-close must commit without a live-bubble gap");

const late = delta(7, "output", " late");
before = live;
live = reduceModelStreamFrame(live, late);
run = projectModelStreamFrame(run, before, live, late);
assert.equal(live.calls[modelStreamCallKey(ROOT, "stream-1")].output, "answer", "late deltas after close must be ignored");
assert.equal(run.activeResponse, "");

live = reduceModelStreamFrame(live, opened(8, {
  turn_id: "turn-2",
  stream_id: "stream-2",
  step: 2,
}));
live = reduceModelStreamFrame(live, delta(9, "output", "partial", {
  turn_id: "turn-2",
  stream_id: "stream-2",
  step: 2,
}));
const interrupted = closed(10, "interrupted", {
  turn_id: "turn-2",
  stream_id: "stream-2",
  step: 2,
  final_text: "partial",
});
before = live;
live = reduceModelStreamFrame(live, interrupted);
run = projectModelStreamFrame(run, before, live, interrupted);
assert.equal(run.messages.length, 1);
assert.equal(run.messages[0].id, "assistant:model-stream:stream-2:partial");
assert.equal(run.messages[0].content, "partial");
assert.equal(run.messages[0].source.kind, "model_stream_partial");
const projectedAgain = projectModelStreamFrame(run, live, live, interrupted);
assert.equal(projectedAgain.messages.length, 1, "replayed partial closes must use a stable id");

// A retry starts a fresh kernel turn/stream while reissuing the same committed message log. The
// failed provider attempt is ephemeral and must leave chat when that replacement stream opens.
let retryState = initialModelStreamState(ROOT);
let retryRun = runState();
const failedOpen = opened(1, { turn_id: "failed-turn", stream_id: "failed-stream" });
before = retryState;
retryState = reduceModelStreamFrame(retryState, failedOpen);
retryRun = projectModelStreamFrame(retryRun, before, retryState, failedOpen);
const failedDelta = delta(2, "output", "abandoned prefix", {
  turn_id: "failed-turn",
  stream_id: "failed-stream",
});
before = retryState;
retryState = reduceModelStreamFrame(retryState, failedDelta);
retryRun = projectModelStreamFrame(retryRun, before, retryState, failedDelta);
const failedClose = closed(3, "failed", {
  turn_id: "failed-turn",
  stream_id: "failed-stream",
  final_text: "abandoned prefix",
  retryable: true,
});
before = retryState;
retryState = reduceModelStreamFrame(retryState, failedClose);
retryRun = projectModelStreamFrame(retryRun, before, retryState, failedClose);
assert.equal(retryRun.messages.at(-1).source.status, "failed");
retryRun = {
  ...retryRun,
  messages: [run.messages[0], ...retryRun.messages],
};
const staleRetryOpen = opened(4, {
  turn_id: "stale-retry-turn",
  stream_id: "stale-retry-stream",
  step: 1,
  started_at: "2025-12-31T23:59:59Z",
});
before = retryState;
retryState = reduceModelStreamFrame(retryState, staleRetryOpen);
retryRun = projectModelStreamFrame(retryRun, before, retryState, staleRetryOpen);
assert.equal(
  retryRun.messages.at(-1).id,
  "assistant:model-stream:failed-stream:partial",
  "a stale opened frame must not supersede a retryable failed partial",
);
const retryOpen = opened(5, {
  turn_id: "replacement-turn",
  stream_id: "replacement-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
});
before = retryState;
retryState = reduceModelStreamFrame(retryState, retryOpen);
retryRun = projectModelStreamFrame(retryRun, before, retryState, retryOpen);
assert.deepEqual(
  retryRun.messages.map((message) => message.id),
  ["assistant:model-stream:stream-2:partial"],
  "a replacement stream must discard failed partials while preserving interrupted partials",
);
const retryDelta = delta(6, "output", "recovered answer", {
  turn_id: "replacement-turn",
  stream_id: "replacement-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
});
before = retryState;
retryState = reduceModelStreamFrame(retryState, retryDelta);
retryRun = projectModelStreamFrame(retryRun, before, retryState, retryDelta);
const retryClose = closed(7, "completed", {
  turn_id: "replacement-turn",
  stream_id: "replacement-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
  final_text: "recovered answer",
});
before = retryState;
retryState = reduceModelStreamFrame(retryState, retryClose);
retryRun = projectModelStreamFrame(retryRun, before, retryState, retryClose);
assert.equal(retryRun.activeResponse, "recovered answer");
assert.deepEqual(
  retryRun.messages.map((message) => message.id),
  ["assistant:model-stream:stream-2:partial"],
  "a successful retry must not resurrect the abandoned provider attempt",
);

let staleState = initialModelStreamState(ROOT);
let staleRun = runState();
const latestOpen = opened(1, {
  turn_id: "latest-failed-turn",
  stream_id: "latest-failed-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
});
before = staleState;
staleState = reduceModelStreamFrame(staleState, latestOpen);
staleRun = projectModelStreamFrame(staleRun, before, staleState, latestOpen);
const latestDelta = delta(2, "output", "latest failed prefix", {
  turn_id: "latest-failed-turn",
  stream_id: "latest-failed-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
});
before = staleState;
staleState = reduceModelStreamFrame(staleState, latestDelta);
staleRun = projectModelStreamFrame(staleRun, before, staleState, latestDelta);
const latestClose = closed(3, "failed", {
  turn_id: "latest-failed-turn",
  stream_id: "latest-failed-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
  final_text: "latest failed prefix",
  retryable: false,
});
before = staleState;
staleState = reduceModelStreamFrame(staleState, latestClose);
staleRun = projectModelStreamFrame(staleRun, before, staleState, latestClose);
const staleOpen = opened(4, {
  turn_id: "stale-turn",
  stream_id: "stale-stream",
  step: 1,
  started_at: "2026-01-01T00:00:01Z",
});
before = staleState;
staleState = reduceModelStreamFrame(staleState, staleOpen);
staleRun = projectModelStreamFrame(staleRun, before, staleState, staleOpen);
assert.equal(
  staleRun.messages.at(-1).id,
  "assistant:model-stream:latest-failed-stream:partial",
  "a stale opened replay must not discard the latest failed partial",
);
// The durable failure event can lag the independent live SSE channel. Retryability on the
// provider close must preserve a non-retryable partial even when the next user turn opens first.
let nextUserState = before;
let nextUserRun = staleRun;
const nextUserOpen = opened(4, {
  turn_id: "next-user-turn",
  stream_id: "next-user-stream",
  step: 3,
  started_at: "2026-01-01T00:00:03Z",
});
before = nextUserState;
nextUserState = reduceModelStreamFrame(nextUserState, nextUserOpen);
nextUserRun = projectModelStreamFrame(nextUserRun, before, nextUserState, nextUserOpen);
assert.equal(
  nextUserRun.messages.at(-1).id,
  "assistant:model-stream:latest-failed-stream:partial",
  "a new user turn racing a durable non-retryable failure must preserve its partial",
);
nextUserState = sealModelStreamTurn(nextUserState, "latest-failed-turn");
assert.equal(
  nextUserRun.messages.at(-1).id,
  "assistant:model-stream:latest-failed-stream:partial",
  "a late durable failure must not retroactively remove the non-retryable partial",
);

let manualRetryRun = discardModelStreamAttempt(
  nextUserRun,
  nextUserState,
  ROOT,
  "latest-failed-turn",
);
assert.equal(
  manualRetryRun.messages.some((message) => (
    message.id === "assistant:model-stream:latest-failed-stream:partial"
  )),
  false,
  "an explicit retry must discard the exact non-retryable attempt it reissues",
);
let manualRetryState = markModelStreamPartialSuperseded(
  initialModelStreamState(ROOT),
  "latest-failed-turn",
);
const replayedManualOpen = opened(1, {
  turn_id: "latest-failed-turn",
  stream_id: "latest-failed-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
});
before = manualRetryState;
manualRetryState = reduceModelStreamFrame(manualRetryState, replayedManualOpen);
manualRetryRun = projectModelStreamFrame(manualRetryRun, before, manualRetryState, replayedManualOpen);
const replayedManualDelta = delta(2, "output", "latest failed prefix", {
  turn_id: "latest-failed-turn",
  stream_id: "latest-failed-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
});
before = manualRetryState;
manualRetryState = reduceModelStreamFrame(manualRetryState, replayedManualDelta);
manualRetryRun = projectModelStreamFrame(manualRetryRun, before, manualRetryState, replayedManualDelta);
const replayedManualClose = closed(3, "failed", {
  turn_id: "latest-failed-turn",
  stream_id: "latest-failed-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
  final_text: "latest failed prefix",
  retryable: false,
});
before = manualRetryState;
manualRetryState = reduceModelStreamFrame(manualRetryState, replayedManualClose);
manualRetryRun = projectModelStreamFrame(manualRetryRun, before, manualRetryState, replayedManualClose);
assert.equal(
  manualRetryRun.messages.some((message) => (
    message.id === "assistant:model-stream:latest-failed-stream:partial"
  )),
  false,
  "retained-ring replay must not resurrect an explicitly superseded attempt",
);

// Durable operation events and passive model frames use independent SSE connections. A retry
// identity can therefore arrive while the abandoned turn still owns the active chat bubble.
let overtakenRetryState = initialModelStreamState(ROOT);
let overtakenRetryRun = runState();
const overtakenOpen = opened(1, {
  turn_id: "overtaken-turn",
  stream_id: "overtaken-stream",
  step: 1,
});
before = overtakenRetryState;
overtakenRetryState = reduceModelStreamFrame(overtakenRetryState, overtakenOpen);
overtakenRetryRun = projectModelStreamFrame(
  overtakenRetryRun,
  before,
  overtakenRetryState,
  overtakenOpen,
);
const overtakenDelta = delta(2, "output", "abandoned live prefix", {
  turn_id: "overtaken-turn",
  stream_id: "overtaken-stream",
  step: 1,
});
before = overtakenRetryState;
overtakenRetryState = reduceModelStreamFrame(overtakenRetryState, overtakenDelta);
overtakenRetryRun = projectModelStreamFrame(
  overtakenRetryRun,
  before,
  overtakenRetryState,
  overtakenDelta,
);
assert.equal(overtakenRetryRun.activeResponse, "abandoned live prefix");
overtakenRetryRun = discardModelStreamAttempt(
  overtakenRetryRun,
  overtakenRetryState,
  ROOT,
  "overtaken-turn",
);
overtakenRetryState = markModelStreamPartialSuperseded(
  overtakenRetryState,
  "overtaken-turn",
);
assert.equal(overtakenRetryRun.activeResponse, "");
assert.equal(overtakenRetryState.output, "");
assert.equal(overtakenRetryState.activeRootTurnId, null);

const overtakenLateClose = closed(3, "failed", {
  turn_id: "overtaken-turn",
  stream_id: "overtaken-stream",
  step: 1,
  final_text: "abandoned live prefix",
  retryable: false,
});
before = overtakenRetryState;
overtakenRetryState = reduceModelStreamFrame(overtakenRetryState, overtakenLateClose);
overtakenRetryRun = projectModelStreamFrame(
  overtakenRetryRun,
  before,
  overtakenRetryState,
  overtakenLateClose,
);
assert.equal(
  overtakenRetryRun.messages.length,
  0,
  "a late passive close must not restore the exact attempt superseded by durable retry",
);

const overtakenReplacementOpen = opened(4, {
  turn_id: "overtaken-replacement-turn",
  stream_id: "overtaken-replacement-stream",
  step: 2,
});
before = overtakenRetryState;
overtakenRetryState = reduceModelStreamFrame(overtakenRetryState, overtakenReplacementOpen);
overtakenRetryRun = projectModelStreamFrame(
  overtakenRetryRun,
  before,
  overtakenRetryState,
  overtakenReplacementOpen,
);
const overtakenReplacementDelta = delta(5, "output", "replacement prefix", {
  turn_id: "overtaken-replacement-turn",
  stream_id: "overtaken-replacement-stream",
  step: 2,
});
before = overtakenRetryState;
overtakenRetryState = reduceModelStreamFrame(overtakenRetryState, overtakenReplacementDelta);
overtakenRetryRun = projectModelStreamFrame(
  overtakenRetryRun,
  before,
  overtakenRetryState,
  overtakenReplacementDelta,
);
assert.equal(overtakenRetryRun.activeResponse, "replacement prefix");
overtakenRetryState = {
  ...overtakenRetryState,
  supersededRootTurnIds: new Set(),
};
overtakenRetryRun = discardModelStreamAttempt(
  overtakenRetryRun,
  overtakenRetryState,
  ROOT,
  "overtaken-turn",
);
overtakenRetryState = markModelStreamPartialSuperseded(
  overtakenRetryState,
  "overtaken-turn",
);
assert.equal(
  overtakenRetryRun.activeResponse,
  "replacement prefix",
  "a delayed durable retry signal must preserve the active replacement turn",
);
assert.equal(overtakenRetryState.output, "replacement prefix");
assert.equal(overtakenRetryState.supersededRootTurnIds.has("overtaken-turn"), true);

live = reduceModelStreamFrame(live, opened(11, {
  turn_id: "turn-3",
  stream_id: "stream-3",
  step: 3,
}));
live = sealModelStreamTurn(live, "turn-3");
live = reduceModelStreamFrame(live, delta(12, "output", "too late", {
  turn_id: "turn-3",
  stream_id: "stream-3",
  step: 3,
}));
assert.equal(live.calls[modelStreamCallKey(ROOT, "stream-3")].output, "", "durable terminal state must fence late deltas");
const closedAfterDurable = closed(13, "interrupted", {
  turn_id: "turn-3",
  stream_id: "stream-3",
  step: 3,
  final_text: "provider partial",
});
before = live;
live = reduceModelStreamFrame(live, closedAfterDurable);
run = projectModelStreamFrame(run, before, live, closedAfterDurable);
assert.equal(
  run.messages.at(-1).content,
  "provider partial",
  "a terminal close may preserve its partial even when the durable interruption arrived first",
);

// A transcript snapshot can be ahead of the durable EventSource, which always replays from zero.
// Old terminal events must neither seal nor erase the active turn restored by model-content.
let replay = initialModelStreamState(ROOT);
replay = reduceModelStreamFrame(replay, opened(1, {
  turn_id: "turn-3",
  stream_id: "replay-stream",
  step: 3,
}));
replay = reduceModelStreamFrame(replay, delta(2, "output", "prefix", {
  turn_id: "turn-3",
  stream_id: "replay-stream",
  step: 3,
}));
const replayBeforeOldTerminal = replay;
replay = sealModelStreamTurn(replay, "turn-1");
replay = sealModelStreamTurn(replay, "turn-2");
assert.equal(replay, replayBeforeOldTerminal, "historical turns must not seal the active snapshot");
let replayRun = restoreActiveModelContent(
  { ...runState(), activeResponse: "", reasoning: "" },
  replay,
);
assert.equal(replayRun.activeResponse, "prefix", "historical replay must preserve snapshot content");
assert.equal(replayRun.status, "running", "an active snapshot must keep historical replay from idling the run");
replay = reduceModelStreamFrame(replay, delta(3, "output", " plus", {
  turn_id: "turn-3",
  stream_id: "replay-stream",
  step: 3,
  start_offset: byteLength("prefix"),
}));
assert.equal(replay.output, "prefix plus", "the active turn must continue after old terminal replay");
replay = sealModelStreamTurn(replay, "turn-3");
replay = reduceModelStreamFrame(replay, delta(4, "output", " fenced", {
  turn_id: "turn-3",
  stream_id: "replay-stream",
  step: 3,
  start_offset: byteLength("prefix plus"),
}));
assert.equal(replay.output, "prefix plus", "the correlated terminal event must fence later deltas");
replayRun = restoreActiveModelContent({ ...replayRun, activeResponse: "" }, replay);
assert.equal(replayRun.activeResponse, "", "sealed content must not be resurrected");

// The durable EventSource can win before the passive opened frame is dispatched. Keep an explicit
// per-turn fence so that queued frames for that terminal operation cannot resurrect it.
let earlyFence = sealModelStreamTurn(initialModelStreamState(ROOT), "terminal-before-open");
earlyFence = reduceModelStreamFrame(earlyFence, opened(1, {
  turn_id: "terminal-before-open",
  stream_id: "terminal-before-open-stream",
}));
earlyFence = reduceModelStreamFrame(earlyFence, delta(2, "output", "must stay hidden", {
  turn_id: "terminal-before-open",
  stream_id: "terminal-before-open-stream",
}));
assert.equal(earlyFence.output, "");
assert.equal(earlyFence.rootTurnSealed, true);
earlyFence = reduceModelStreamFrame(earlyFence, opened(3, {
  turn_id: "legitimate-next-turn",
  stream_id: "legitimate-next-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
}));
earlyFence = reduceModelStreamFrame(earlyFence, delta(4, "output", "new answer", {
  turn_id: "legitimate-next-turn",
  stream_id: "legitimate-next-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
}));
assert.equal(earlyFence.output, "new answer", "a later turn must clear the prior turn fence");

let runFence = sealModelStreamTurn(initialModelStreamState(ROOT));
runFence = reduceModelStreamFrame(runFence, opened(1, {
  turn_id: "after-run-terminal",
  stream_id: "after-run-terminal-stream",
}));
runFence = reduceModelStreamFrame(runFence, delta(2, "output", "never revive", {
  turn_id: "after-run-terminal",
  stream_id: "after-run-terminal-stream",
}));
assert.equal(runFence.output, "", "a run-terminal fence must cover every later queued frame");

// Each provider call is independently hydrated. A later model call in the same user turn must
// replace the previous operation's projection, matching the one-latest-call snapshot contract.
let sameTurn = initialModelStreamState(ROOT);
sameTurn = reduceModelStreamFrame(sameTurn, opened(1, {
  turn_id: "multi-call-turn",
  stream_id: "multi-call-1",
}));
sameTurn = reduceModelStreamFrame(sameTurn, delta(2, "output", "first operation", {
  turn_id: "multi-call-turn",
  stream_id: "multi-call-1",
}));
sameTurn = reduceModelStreamFrame(sameTurn, closed(3, "completed", {
  turn_id: "multi-call-turn",
  stream_id: "multi-call-1",
  final_text: "first operation",
}));
sameTurn = reduceModelStreamFrame(sameTurn, opened(4, {
  turn_id: "multi-call-turn",
  stream_id: "multi-call-2",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
}));
assert.equal(sameTurn.output, "", "a new operation must not concatenate the prior call");
sameTurn = reduceModelStreamFrame(sameTurn, delta(5, "output", "second operation", {
  turn_id: "multi-call-turn",
  stream_id: "multi-call-2",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
}));
assert.equal(sameTurn.output, "second operation");

// Reverse ordering: the durable settle can win the cross-EventSource race. Its seal fences deltas,
// and the later completed close must leave the committed UI untouched.
let reverse = initialModelStreamState(ROOT);
let reverseRun = runState();
reverse = reduceModelStreamFrame(reverse, opened(1, {
  turn_id: "reverse-turn",
  stream_id: "reverse-stream",
}));
reverse = reduceModelStreamFrame(reverse, delta(2, "output", "complete", {
  turn_id: "reverse-turn",
  stream_id: "reverse-stream",
}));
reverse = sealModelStreamTurn(reverse, "reverse-turn");
reverseRun = { ...reverseRun, status: "idle", activeResponse: "", reasoning: "" };
const reverseClose = closed(3, "completed", {
  turn_id: "reverse-turn",
  stream_id: "reverse-stream",
  final_text: "complete",
});
before = reverse;
reverse = reduceModelStreamFrame(reverse, reverseClose);
reverseRun = projectModelStreamFrame(reverseRun, before, reverse, reverseClose);
assert.equal(reverseRun.activeResponse, "", "close-after-settle must not resurrect live content");
assert.equal(reverseRun.messages.length, 0, "the durable settle remains the sole completed message");

const gapFrame = {
  schema_version: "monoid.model-stream.live.v1",
  cursor: `${GENERATION}:15`,
  sequence: 15,
  kind: "delta",
  root_run_id: ROOT,
  run_id: ROOT,
  turn_id: "turn-4",
  stream_id: "stream-4",
  step: 4,
  started_at: "2026-01-01T00:00:04Z",
  channel: "output",
  text: "missed fourteen",
  start_offset: 99,
  end_offset: 99 + byteLength("missed fourteen"),
};
const beforeGapCursor = live.cursor;
live = reduceModelStreamFrame(live, gapFrame);
assert.equal(live.needsHydration, true);
assert.equal(live.resetReason, "sequence_gap");
assert.equal(live.resumeCursor, beforeGapCursor, "a local gap must resume from the last accepted cursor");

const reset = {
  schema_version: "monoid.model-stream.live.v1",
  cursor: `${GENERATION}:13`,
  sequence: 13,
  kind: "reset",
  root_run_id: ROOT,
  reason: "cursor_gap",
  oldest_available_cursor: `${GENERATION}:14`,
  latest_cursor: `${GENERATION}:20`,
};
live = reduceModelStreamFrame(live, reset);
assert.equal(live.resumeCursor, `${GENERATION}:13`, "reset hydration must resume at the replay baseline");
live = markModelStreamHydrated(live);
assert.equal(live.needsHydration, false);
assert.equal(live.sequence, 13);
assert.equal(live.output, "");
live = reduceModelStreamFrame(live, opened(14, {
  turn_id: "turn-5",
  stream_id: "stream-5",
  step: 5,
}));
assert.equal(live.sequence, 14, "retained frames after the hydrated baseline must replay normally");

let ahead = {
  ...initialModelStreamState(ROOT),
  generation: GENERATION,
  sequence: 99,
  cursor: `${GENERATION}:99`,
  resumeCursor: `${GENERATION}:99`,
};
ahead = reduceModelStreamFrame(ahead, {
  schema_version: "monoid.model-stream.live.v1",
  cursor: `${GENERATION}:14`,
  sequence: 14,
  kind: "reset",
  root_run_id: ROOT,
  reason: "cursor_ahead",
  latest_cursor: `${GENERATION}:14`,
});
assert.equal(ahead.needsHydration, true, "a lower cursor-ahead reset must bypass normal dedupe");
assert.equal(ahead.resumeCursor, `${GENERATION}:14`);

let regenerated = {
  ...initialModelStreamState(ROOT),
  generation: GENERATION,
  sequence: 8,
  cursor: `${GENERATION}:8`,
  resumeCursor: `${GENERATION}:8`,
};
regenerated = reduceModelStreamFrame(regenerated, {
  schema_version: "monoid.model-stream.live.v1",
  cursor: "generation-b:0",
  sequence: 0,
  kind: "reset",
  root_run_id: ROOT,
  reason: "generation_changed",
  latest_cursor: "generation-b:0",
});
assert.equal(regenerated.generation, "generation-b");
assert.equal(regenerated.needsHydration, true);

const activeSnapshotPayload = {
  schema_version: "studio.model-content.v1",
  root_run_id: ROOT,
  streams: [
    {
      root_run_id: ROOT,
      run_id: ROOT,
      turn_id: "unicode-turn",
      stream_id: "unicode-stream",
      step: 1,
      provider: "test",
      model: "test-model",
      started_at: "2026-01-01T00:00:00Z",
      status: "running",
      output_text: "A🙂",
      output_end_offset: 5,
      reasoning_text: "생각",
      reasoning_end_offset: 6,
      partial: false,
    },
    {
      root_run_id: ROOT,
      run_id: `${ROOT}.sub.snapshot-child`,
      turn_id: "child-snapshot-turn",
      // Stream ids are activation-local and may collide across root/descendant loops.
      stream_id: "unicode-stream",
      step: 2,
      provider: null,
      model: null,
      started_at: "2026-01-01T00:00:00Z",
      status: "running",
      output_text: "child prefix",
      output_end_offset: 12,
      reasoning_text: "",
      reasoning_end_offset: 0,
      partial: false,
    },
  ],
};
const activeSnapshot = decodeModelContentResponse(activeSnapshotPayload);
assert.ok(activeSnapshot);
assert.equal(decodeModelContentResponse({
  ...activeSnapshotPayload,
  streams: [{ ...activeSnapshotPayload.streams[0], output_end_offset: 2 }],
}), null, "snapshot UTF-8 offsets must match their channel text");
assert.equal(decodeModelContentResponse({
  ...activeSnapshotPayload,
  streams: [{ ...activeSnapshotPayload.streams[0], retryable: "yes" }],
}), null, "snapshot retryability must be a boolean");
assert.equal(decodeModelContentResponse({
  ...activeSnapshotPayload,
  streams: [
    activeSnapshotPayload.streams[0],
    {
      ...activeSnapshotPayload.streams[0],
      stream_id: "second-root-stream",
      turn_id: "second-root-turn",
    },
  ],
}), null, "the bounded hydration contract permits only one latest call per run");

let hydrated = {
  ...initialModelStreamState(ROOT),
  resumeCursor: `${GENERATION}:30`,
  needsHydration: true,
};
hydrated = seedModelStreamSnapshot(hydrated, activeSnapshot);
assert.equal(hydrated.output, "A🙂");
assert.equal(hydrated.reasoning, "생각");
assert.equal(hydrated.calls[modelStreamCallKey(ROOT, "unicode-stream")].outputBytes, 5);
assert.equal(hydrated.sequence, 30);

let hydratedRun = {
  ...runState(),
  messages: [
    {
      id: "assistant:model-stream:unicode-stream:active",
      role: "assistant",
      content: "A🙂",
      attachments: [],
      created_at: 1,
      source: { kind: "model_stream_active", stream_id: "unicode-stream" },
    },
  ],
};
hydratedRun = projectModelContentSnapshot(hydratedRun, hydrated);
assert.equal(hydratedRun.messages.length, 0, "active sidecar rows must merge into the live bubble");
assert.equal(hydratedRun.activeResponse, "A🙂");
assert.equal(hydratedRun.events.length, 1);

const failedRetrySnapshot = decodeModelContentResponse({
  ...activeSnapshotPayload,
  streams: [{
    ...activeSnapshotPayload.streams[0],
    turn_id: "snapshot-failed-turn",
    stream_id: "snapshot-failed-stream",
    status: "failed",
    output_text: "snapshot failed prefix",
    output_end_offset: byteLength("snapshot failed prefix"),
    reasoning_text: "",
    reasoning_end_offset: 0,
    final_text: "snapshot failed prefix",
    partial: true,
    retryable: true,
  }],
});
assert.ok(failedRetrySnapshot);
let snapshotRetryState = seedModelStreamSnapshot(
  { ...initialModelStreamState(ROOT), resumeCursor: `${GENERATION}:50` },
  failedRetrySnapshot,
);
let snapshotRetryRun = projectModelContentSnapshot(runState(), snapshotRetryState);
assert.equal(snapshotRetryRun.messages.at(-1).source.status, "failed");
const snapshotRetryOpen = opened(51, {
  turn_id: "snapshot-replacement-turn",
  stream_id: "snapshot-replacement-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
});
before = snapshotRetryState;
snapshotRetryState = reduceModelStreamFrame(snapshotRetryState, snapshotRetryOpen);
snapshotRetryRun = projectModelStreamFrame(
  snapshotRetryRun,
  before,
  snapshotRetryState,
  snapshotRetryOpen,
);
assert.equal(
  snapshotRetryRun.messages.length,
  0,
  "a live retry must supersede a failed partial restored from the private snapshot",
);

const nonRetryableSnapshot = decodeModelContentResponse({
  ...activeSnapshotPayload,
  streams: [{
    ...activeSnapshotPayload.streams[0],
    turn_id: "snapshot-non-retryable-turn",
    stream_id: "snapshot-non-retryable-stream",
    status: "failed",
    output_text: "snapshot retained prefix",
    output_end_offset: byteLength("snapshot retained prefix"),
    reasoning_text: "",
    reasoning_end_offset: 0,
    final_text: "snapshot retained prefix",
    partial: true,
    retryable: false,
  }],
});
assert.ok(nonRetryableSnapshot);
let nonRetryableState = seedModelStreamSnapshot(
  { ...initialModelStreamState(ROOT), resumeCursor: `${GENERATION}:60` },
  nonRetryableSnapshot,
);
let nonRetryableRun = projectModelContentSnapshot(runState(), nonRetryableState);
const postFailureUserOpen = opened(61, {
  turn_id: "post-failure-user-turn",
  stream_id: "post-failure-user-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
});
before = nonRetryableState;
nonRetryableState = reduceModelStreamFrame(nonRetryableState, postFailureUserOpen);
nonRetryableRun = projectModelStreamFrame(
  nonRetryableRun,
  before,
  nonRetryableState,
  postFailureUserOpen,
);
assert.equal(
  nonRetryableRun.messages.at(-1).id,
  "assistant:model-stream:snapshot-non-retryable-stream:partial",
  "a new user turn must preserve a non-retryable failed partial restored after reload",
);

const legacyFailedSnapshot = decodeModelContentResponse({
  ...activeSnapshotPayload,
  streams: [{
    ...activeSnapshotPayload.streams[0],
    turn_id: "legacy-failed-turn",
    stream_id: "legacy-failed-stream",
    status: "failed",
    output_text: "legacy failed prefix",
    output_end_offset: byteLength("legacy failed prefix"),
    reasoning_text: "",
    reasoning_end_offset: 0,
    final_text: "legacy failed prefix",
    partial: true,
  }],
});
assert.ok(legacyFailedSnapshot);
let legacyFailedState = seedModelStreamSnapshot(
  { ...initialModelStreamState(ROOT), resumeCursor: `${GENERATION}:70` },
  legacyFailedSnapshot,
);
let legacyFailedRun = projectModelContentSnapshot(runState(), legacyFailedState);
const legacyNextOpen = opened(71, {
  turn_id: "legacy-next-turn",
  stream_id: "legacy-next-stream",
  step: 2,
  started_at: "2026-01-01T00:00:02Z",
});
before = legacyFailedState;
legacyFailedState = reduceModelStreamFrame(legacyFailedState, legacyNextOpen);
legacyFailedRun = projectModelStreamFrame(
  legacyFailedRun,
  before,
  legacyFailedState,
  legacyNextOpen,
);
assert.equal(
  legacyFailedRun.messages.at(-1).id,
  "assistant:model-stream:legacy-failed-stream:partial",
  "an old failed snapshot without retryability must be preserved conservatively",
);

let suppressedSnapshotState = markModelStreamPartialSuperseded(
  { ...initialModelStreamState(ROOT), resumeCursor: `${GENERATION}:80` },
  "snapshot-non-retryable-turn",
);
suppressedSnapshotState = seedModelStreamSnapshot(
  suppressedSnapshotState,
  nonRetryableSnapshot,
);
const suppressedSnapshotRun = projectModelContentSnapshot(runState(), suppressedSnapshotState);
assert.equal(
  suppressedSnapshotRun.messages.length,
  0,
  "hydration must not resurrect a failed partial superseded by an explicit retry",
);

// A reload can hydrate the failed sidecar before its durable Studio retry marker is replayed.
// Applying that event must remove the row immediately and fence every later hydration attempt.
let lostResponseState = seedModelStreamSnapshot(
  { ...initialModelStreamState(ROOT), resumeCursor: `${GENERATION}:81` },
  nonRetryableSnapshot,
);
let lostResponseRun = projectModelContentSnapshot(runState(), lostResponseState);
assert.equal(lostResponseRun.messages.length, 1);
lostResponseState = markModelStreamPartialSuperseded(
  lostResponseState,
  "snapshot-non-retryable-turn",
);
lostResponseRun = discardModelStreamAttempt(
  lostResponseRun,
  lostResponseState,
  ROOT,
  "snapshot-non-retryable-turn",
);
assert.equal(
  lostResponseRun.messages.length,
  0,
  "durable retry replay must remove a failed partial hydrated after a lost HTTP response",
);
lostResponseState = seedModelStreamSnapshot(lostResponseState, nonRetryableSnapshot);
lostResponseRun = projectModelContentSnapshot(lostResponseRun, lostResponseState);
assert.equal(
  lostResponseRun.messages.length,
  0,
  "later hydration must preserve durable retry suppression after reload",
);

const abandonedSnapshot = decodeModelContentResponse({
  ...activeSnapshotPayload,
  streams: [{
    ...activeSnapshotPayload.streams[0],
    stream_id: "abandoned-stream",
    status: "abandoned",
    partial: true,
  }],
});
assert.ok(abandonedSnapshot);
const abandonedState = seedModelStreamSnapshot(initialModelStreamState(ROOT), abandonedSnapshot);
const abandonedRun = projectModelContentSnapshot(runState(), abandonedState);
assert.equal(abandonedState.activeRootStreamId, null, "unproven abandoned work is terminal");
assert.equal(abandonedRun.activeResponse, "", "an abandoned snapshot must not revive a live bubble");

const abandonedChildSnapshot = decodeModelContentResponse({
  ...activeSnapshotPayload,
  streams: [{
    ...activeSnapshotPayload.streams[1],
    stream_id: "abandoned-child-stream",
    status: "abandoned",
    partial: true,
  }],
});
assert.ok(abandonedChildSnapshot);
const abandonedChildren = seedSubagentModelContent({}, abandonedChildSnapshot);
assert.equal(
  abandonedChildren[`${ROOT}.sub.snapshot-child`].status,
  "running",
  "a model-call outcome must not fabricate the child lifecycle outcome",
);
assert.equal(abandonedChildren[`${ROOT}.sub.snapshot-child`].liveStreamStatus, "abandoned");
const historicalChildStart = projectSubagentStarted(
  abandonedChildren[`${ROOT}.sub.snapshot-child`],
  {
    child_run_id: `${ROOT}.sub.snapshot-child`,
    subagent_type: "researcher",
    task_id: "task-1",
  },
  ROOT,
  true,
  true,
);
assert.equal(historicalChildStart.revive, false);
assert.equal(historicalChildStart.activity.status, "failed");
assert.equal(historicalChildStart.activity.subagentType, "researcher");
assert.equal(historicalChildStart.activity.taskId, "task-1");
const crashChildStart = projectSubagentStarted(
  undefined,
  { child_run_id: `${ROOT}.sub.crashed`, subagent_type: "researcher" },
  ROOT,
  true,
  true,
);
assert.equal(crashChildStart.revive, false);
assert.equal(crashChildStart.activity.status, "failed");

const completedChildSnapshot = decodeModelContentResponse({
  ...activeSnapshotPayload,
  streams: [{
    ...activeSnapshotPayload.streams[1],
    stream_id: "completed-child-model-call",
    status: "completed",
    final_text: "model call finished before child tools",
    partial: false,
  }],
});
assert.ok(completedChildSnapshot);
const completedChildren = seedSubagentModelContent({}, completedChildSnapshot);
const activeHistoricalChildStart = projectSubagentStarted(
  completedChildren[`${ROOT}.sub.snapshot-child`],
  { child_run_id: `${ROOT}.sub.snapshot-child`, subagent_type: "researcher" },
  ROOT,
  true,
  false,
);
assert.equal(activeHistoricalChildStart.revive, true);
assert.equal(activeHistoricalChildStart.activity.status, "running");
const fencedHistoricalChildStart = projectSubagentStarted(
  completedChildren[`${ROOT}.sub.snapshot-child`],
  { child_run_id: `${ROOT}.sub.snapshot-child`, subagent_type: "researcher" },
  ROOT,
  true,
  true,
);
assert.equal(fencedHistoricalChildStart.revive, false);
assert.equal(fencedHistoricalChildStart.activity.status, "failed");

const floorSnapshot = decodeModelContentResponse({
  schema_version: "studio.model-content.v1",
  root_run_id: ROOT,
  streams: [
    {
      ...activeSnapshotPayload.streams[0],
      turn_id: "latest-root-turn",
      stream_id: "latest-root-stream",
      step: 5,
      started_at: "2026-01-01T00:00:05Z",
      output_text: "latest root prefix",
      output_end_offset: 18,
      reasoning_text: "",
      reasoning_end_offset: 0,
    },
    {
      ...activeSnapshotPayload.streams[1],
      turn_id: "latest-child-turn",
      stream_id: "latest-child-stream",
      step: 5,
      started_at: "2026-01-01T00:00:05Z",
      output_text: "latest child prefix",
      output_end_offset: 19,
    },
  ],
});
assert.ok(floorSnapshot);
let floored = {
  ...initialModelStreamState(ROOT),
  resumeCursor: `${GENERATION}:40`,
};
floored = seedModelStreamSnapshot(floored, floorSnapshot);
let flooredChildren = seedSubagentModelContent({}, floorSnapshot);
const staleRoot = opened(41, {
  turn_id: "older-root-turn",
  stream_id: "older-root-stream",
  step: 4,
  started_at: "2026-01-01T00:00:04Z",
});
before = floored;
floored = reduceModelStreamFrame(floored, staleRoot);
assert.equal(floored.sequence, 41, "stale replay still advances the global cursor");
assert.equal(floored.output, "latest root prefix");
assert.equal(floored.activeRootStreamId, "latest-root-stream");
const staleChild = opened(42, {
  run_id: `${ROOT}.sub.snapshot-child`,
  turn_id: "older-child-turn",
  stream_id: "older-child-stream",
  step: 4,
  started_at: "2026-01-01T00:00:04Z",
});
before = floored;
floored = reduceModelStreamFrame(floored, staleChild);
flooredChildren[staleChild.run_id] = projectSubagentModelStream(
  flooredChildren[staleChild.run_id],
  before,
  floored,
  staleChild,
);
assert.equal(flooredChildren[staleChild.run_id].liveOutput, "latest child prefix");
assert.equal(flooredChildren[staleChild.run_id].liveStreamId, "latest-child-stream");

let bounded = initialModelStreamState(ROOT);
let boundedSequence = 0;
for (let step = 1; step <= 80; step += 1) {
  boundedSequence += 1;
  bounded = reduceModelStreamFrame(bounded, opened(boundedSequence, {
    turn_id: `bounded-turn-${step}`,
    stream_id: `bounded-stream-${step}`,
    step,
    started_at: `2026-01-01T00:00:${String(step).padStart(2, "0")}Z`,
  }));
  boundedSequence += 1;
  bounded = reduceModelStreamFrame(bounded, closed(boundedSequence, "completed", {
    turn_id: `bounded-turn-${step}`,
    stream_id: `bounded-stream-${step}`,
    step,
    started_at: `2026-01-01T00:00:${String(step).padStart(2, "0")}Z`,
    final_text: `answer ${step}`,
  }));
}
assert.equal(
  Object.values(bounded.calls).filter((call) => call.runId === ROOT).length,
  1,
  "long sessions must retain only the latest model call per run",
);

let hydratedChildren = seedSubagentModelContent({}, activeSnapshot);
assert.equal(hydratedChildren[`${ROOT}.sub.snapshot-child`].liveOutput, "child prefix");
assert.equal(hydratedChildren[`${ROOT}.sub.snapshot-child`].events.length, 0);

// Snapshot ended after the first byte ('A') of this retained frame's coordinate range. The
// reducer drops the overlapping four-byte emoji and appends only the exact UTF-8 suffix.
const unicodeOverlap = delta(31, "output", "🙂B", {
  turn_id: "unicode-turn",
  stream_id: "unicode-stream",
  start_offset: 1,
  end_offset: 6,
});
before = hydrated;
hydrated = reduceModelStreamFrame(hydrated, unicodeOverlap);
hydratedRun = projectModelStreamFrame(hydratedRun, before, hydrated, unicodeOverlap);
assert.equal(hydrated.output, "A🙂B");
assert.equal(hydrated.calls[modelStreamCallKey(ROOT, "unicode-stream")].outputBytes, 6);
assert.equal(hydratedRun.activeResponse, "A🙂B");

// An oversized final value is absent only from the public marker. Hydration resumes after that
// marker and replaces the incomplete running call with the terminal private-sidecar snapshot.
let omitted = initialModelStreamState(ROOT);
omitted = reduceModelStreamFrame(omitted, opened(1, {
  turn_id: "omitted-turn",
  stream_id: "omitted-stream",
}));
omitted = reduceModelStreamFrame(omitted, delta(2, "output", "prefix", {
  turn_id: "omitted-turn",
  stream_id: "omitted-stream",
}));
const omittedClose = closed(3, "completed", {
  turn_id: "omitted-turn",
  stream_id: "omitted-stream",
  content_omitted: true,
});
const beforeOmittedClose = omitted;
omitted = reduceModelStreamFrame(omitted, omittedClose);
assert.equal(omitted.needsHydration, true);
assert.equal(omitted.resetReason, "content_omitted");
assert.equal(omitted.resumeCursor, `${GENERATION}:3`, "hydration must resume after the marker");
assert.equal(omitted.sequence, 3, "the omitted marker still advances the public cursor");
assert.equal(
  omitted.calls[modelStreamCallKey(ROOT, "omitted-stream")].status,
  "running",
  "an incomplete public prefix must not be finalized before hydration",
);
const fullTerminalText = `prefix ${"complete ".repeat(64)}`;
const omittedSnapshot = decodeModelContentResponse({
  schema_version: "studio.model-content.v1",
  root_run_id: ROOT,
  streams: [{
    ...activeSnapshotPayload.streams[0],
    turn_id: "omitted-turn",
    stream_id: "omitted-stream",
    status: "completed",
    output_text: fullTerminalText,
    output_end_offset: byteLength(fullTerminalText),
    reasoning_text: "",
    reasoning_end_offset: 0,
    final_text: fullTerminalText,
  }],
});
assert.ok(omittedSnapshot);
omitted = seedModelStreamSnapshot(omitted, omittedSnapshot);
const omittedCall = omitted.calls[modelStreamCallKey(ROOT, "omitted-stream")];
assert.equal(omittedCall.status, "completed");
assert.equal(omittedCall.output, fullTerminalText);
assert.equal(omitted.activeRootStreamId, null, "a terminal snapshot cannot revive a live bubble");
const committedMessage = {
  id: "assistant:durable-omitted",
  role: "assistant",
  content: fullTerminalText,
  attachments: [],
  created_at: 1,
  source: { kind: "event", event_type: "turn.settled", event_id: "durable-omitted", seq: 3 },
};
const omittedRun = projectModelContentSnapshot({
  ...runState(),
  activeResponse: beforeOmittedClose.output,
  messages: [committedMessage],
}, omitted);
assert.equal(omittedRun.activeResponse, "");
assert.deepEqual(omittedRun.messages, [committedMessage]);
assert.equal(
  reduceModelStreamFrame(omitted, omittedClose),
  omitted,
  "replaying the consumed omitted marker must not start a hydration loop",
);

const hydratedInterrupted = seedModelStreamSnapshot(
  { ...initialModelStreamState(ROOT), resumeCursor: `${GENERATION}:4` },
  decodeModelContentResponse({
    schema_version: "studio.model-content.v1",
    root_run_id: ROOT,
    streams: [{
      ...activeSnapshotPayload.streams[0],
      turn_id: "omitted-partial-turn",
      stream_id: "omitted-partial-stream",
      status: "interrupted",
      output_text: "",
      output_end_offset: 0,
      reasoning_text: "",
      reasoning_end_offset: 0,
      final_text: "private final-only partial",
      partial: true,
    }],
  }),
);
assert.equal(
  hydratedInterrupted.calls[modelStreamCallKey(ROOT, "omitted-partial-stream")].output,
  "private final-only partial",
  "terminal snapshots must prefer their authoritative final_text",
);
const interruptedRun = projectModelContentSnapshot(runState(), hydratedInterrupted);
assert.equal(interruptedRun.activeResponse, "");
assert.equal(interruptedRun.messages.at(-1).id, "assistant:model-stream:omitted-partial-stream:partial");
assert.equal(interruptedRun.messages.at(-1).content, "private final-only partial");

const snapshotDuplicate = delta(32, "output", "A🙂", {
  turn_id: "unicode-turn",
  stream_id: "unicode-stream",
  start_offset: 0,
  end_offset: 5,
});
hydrated = reduceModelStreamFrame(hydrated, snapshotDuplicate);
assert.equal(hydrated.output, "A🙂B", "replay wholly covered by a newer snapshot must be dropped");

const channelGap = delta(33, "output", "gap", {
  turn_id: "unicode-turn",
  stream_id: "unicode-stream",
  start_offset: 9,
  end_offset: 12,
});
hydrated = reduceModelStreamFrame(hydrated, channelGap);
assert.equal(hydrated.needsHydration, true, "a channel byte-offset gap must re-enter hydration");

class FakeEventSource {
  static instances = [];
  listeners = new Map();
  closed = false;
  onopen = null;
  onerror = null;

  constructor(url) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(name, listener) {
    this.listeners.set(name, listener);
  }

  emit(name, data) {
    this.listeners.get(name)?.({ data: JSON.stringify(data), lastEventId: data.cursor });
  }

  close() {
    this.closed = true;
  }
}

globalThis.EventSource = FakeEventSource;
const received = [];
const connections = [];
const transport = new ModelStreamEventSource({
  rootRunId: ROOT,
  cursor: `${GENERATION}:20`,
  onFrame: (frame) => received.push(frame),
  onConnectionChange: (connected) => connections.push(connected),
});
transport.open();
const sourceInstance = FakeEventSource.instances.at(-1);
assert.equal(sourceInstance.url, `/api/model-stream?run_id=${ROOT}&cursor=generation-a%3A20`);
assert.equal(sourceInstance.listeners.has("model-stream"), true, "the named SSE event must be consumed");
sourceInstance.onopen();
sourceInstance.emit("model-stream", opened(21));
sourceInstance.emit("model-stream", { malformed: true });
assert.equal(received.length, 1, "malformed frames must be isolated from the live stream");
assert.deepEqual(connections, [false, true], "opening first closes any previous passive source");
transport.open();
const replacementSource = FakeEventSource.instances.at(-1);
assert.notEqual(replacementSource, sourceInstance);
assert.equal(sourceInstance.closed, true);
sourceInstance.emit("model-stream", opened(22));
sourceInstance.onerror();
assert.equal(received.length, 1, "queued messages from a replaced EventSource must be fenced");
assert.deepEqual(connections, [false, true, false], "stale errors cannot change connection state");
replacementSource.onopen();
replacementSource.emit("model-stream", opened(22));
assert.equal(received.length, 2);
transport.close();
assert.equal(replacementSource.closed, true);
assert.deepEqual(connections, [false, true, false, true, false]);

console.log("Live model-stream checks passed (13 protocol/reducer/transport/integration groups).");
