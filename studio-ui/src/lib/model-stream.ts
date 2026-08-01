import type { ChatMessage, RunViewState, SubagentActivity } from "./types";

export const MODEL_STREAM_SCHEMA_VERSION = "monoid.model-stream.live.v1" as const;

export type ModelStreamChannel = "output" | "reasoning";
export type ModelStreamStatus =
  | "completed"
  | "interrupted"
  | "failed"
  | "cancelled"
  | "timed_out";
export type ModelStreamResetReason = "generation_changed" | "cursor_gap" | "cursor_ahead";
export type ModelContentStatus = "running" | ModelStreamStatus | "abandoned";

export interface ModelContentSnapshot {
  root_run_id: string;
  run_id: string;
  turn_id: string;
  stream_id: string;
  step: number;
  provider: string | null;
  model: string | null;
  started_at: string;
  status: ModelContentStatus;
  output_text: string;
  output_end_offset: number;
  reasoning_text: string;
  reasoning_end_offset: number;
  partial: boolean;
  final_text?: string;
  usage?: Record<string, unknown>;
  error_code?: string;
}

export interface ModelContentResponse {
  schema_version: "studio.model-content.v1";
  root_run_id: string;
  streams: ModelContentSnapshot[];
}

interface ModelStreamFrameBase {
  schema_version: typeof MODEL_STREAM_SCHEMA_VERSION;
  cursor: string;
  sequence: number;
  root_run_id: string;
}

interface ModelStreamCallFrameBase extends ModelStreamFrameBase {
  run_id: string;
  turn_id: string;
  stream_id: string;
  step: number;
  started_at: string;
}

export interface ModelStreamOpenedFrame extends ModelStreamCallFrameBase {
  kind: "opened";
  provider?: string;
  model?: string;
}

export interface ModelStreamDeltaFrame extends ModelStreamCallFrameBase {
  kind: "delta";
  channel: ModelStreamChannel;
  text: string;
  start_offset: number;
  end_offset: number;
}

export interface ModelStreamClosedFrame extends ModelStreamCallFrameBase {
  kind: "closed";
  finished_at: string;
  status: ModelStreamStatus;
  final_text?: string;
  usage?: Record<string, unknown>;
  error_code?: string;
  partial?: boolean;
  content_omitted?: boolean;
}

export interface ModelStreamResetFrame extends ModelStreamFrameBase {
  kind: "reset";
  reason: ModelStreamResetReason;
  oldest_available_cursor?: string;
  latest_cursor: string;
}

export type ModelStreamFrame =
  | ModelStreamOpenedFrame
  | ModelStreamDeltaFrame
  | ModelStreamClosedFrame
  | ModelStreamResetFrame;

interface ParsedCursor {
  generation: string;
  sequence: number;
}

export interface LiveModelCallState {
  rootRunId: string;
  runId: string;
  turnId: string;
  streamId: string;
  step: number;
  provider?: string;
  model?: string;
  startedAt?: string;
  finishedAt?: string;
  status: "running" | ModelStreamStatus | "abandoned";
  output: string;
  reasoning: string;
  outputBytes: number;
  reasoningBytes: number;
  finalText?: string;
  partial: boolean;
}

export interface ModelStreamViewState {
  rootRunId: string | null;
  generation: string | null;
  sequence: number;
  cursor: string | null;
  resumeCursor: string | null;
  calls: Record<string, LiveModelCallState>;
  floors: Record<string, ModelStreamCallFloor>;
  activeRootStreamId: string | null;
  activeRootTurnId: string | null;
  output: string;
  reasoning: string;
  rootTurnSealed: boolean;
  sealedRootTurnId: string | null;
  needsHydration: boolean;
  resetReason: ModelStreamResetReason | "sequence_gap" | "content_omitted" | null;
  resetLatestCursor: string | null;
}

export interface ModelStreamCallFloor {
  step: number;
  startedAt: string;
  streamId: string;
}

/** Stream ids are activation-local; run lineage is part of their UI identity. */
export function modelStreamCallKey(runId: string, streamId: string): string {
  return JSON.stringify([runId, streamId]);
}

export function initialModelStreamState(rootRunId: string | null = null): ModelStreamViewState {
  return {
    rootRunId,
    generation: null,
    sequence: -1,
    cursor: null,
    resumeCursor: null,
    calls: {},
    floors: {},
    activeRootStreamId: null,
    activeRootTurnId: null,
    output: "",
    reasoning: "",
    rootTurnSealed: false,
    sealedRootTurnId: null,
    needsHydration: false,
    resetReason: null,
    resetLatestCursor: null,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseCursor(value: unknown): ParsedCursor | null {
  if (typeof value !== "string") return null;
  const separator = value.lastIndexOf(":");
  if (separator <= 0) return null;
  const generation = value.slice(0, separator);
  const rawSequence = value.slice(separator + 1);
  if (!/^[A-Za-z0-9._-]+$/.test(generation) || !/^(0|[1-9]\d*)$/.test(rawSequence)) {
    return null;
  }
  const sequence = Number(rawSequence);
  if (!Number.isSafeInteger(sequence)) return null;
  return { generation, sequence };
}

const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });

function utf8Length(text: string): number {
  return UTF8_ENCODER.encode(text).byteLength;
}

function utf8Suffix(text: string, skippedBytes: number): string | null {
  if (skippedBytes === 0) return text;
  try {
    return UTF8_DECODER.decode(UTF8_ENCODER.encode(text).slice(skippedBytes));
  } catch {
    return null;
  }
}

function hasCallIdentity(payload: Record<string, unknown>): boolean {
  return typeof payload.run_id === "string"
    && payload.run_id.length > 0
    && typeof payload.turn_id === "string"
    && payload.turn_id.length > 0
    && typeof payload.stream_id === "string"
    && payload.stream_id.length > 0
    && Number.isSafeInteger(payload.step)
    && Number(payload.step) >= 1
    && typeof payload.started_at === "string"
    && payload.started_at.length > 0;
}

const MODEL_STREAM_STATUSES = new Set<ModelStreamStatus>([
  "completed",
  "interrupted",
  "failed",
  "cancelled",
  "timed_out",
]);
const RESET_REASONS = new Set<ModelStreamResetReason>([
  "generation_changed",
  "cursor_gap",
  "cursor_ahead",
]);
const MODEL_CONTENT_STATUSES = new Set<ModelContentStatus>([
  "running",
  "completed",
  "interrupted",
  "failed",
  "cancelled",
  "timed_out",
  "abandoned",
]);

/** Validate the renderer-required live protocol fields while allowing additive fields. */
export function decodeModelStreamFrame(payload: unknown): ModelStreamFrame | null {
  if (!isRecord(payload)
    || payload.schema_version !== MODEL_STREAM_SCHEMA_VERSION
    || typeof payload.root_run_id !== "string"
    || payload.root_run_id.length === 0
    || !Number.isSafeInteger(payload.sequence)) return null;
  const cursor = parseCursor(payload.cursor);
  if (!cursor || cursor.sequence !== payload.sequence) return null;

  if (payload.kind === "reset") {
    if (!RESET_REASONS.has(payload.reason as ModelStreamResetReason)
      || (payload.oldest_available_cursor !== undefined
        && !parseCursor(payload.oldest_available_cursor))
      || !parseCursor(payload.latest_cursor)) return null;
    return payload as unknown as ModelStreamResetFrame;
  }
  if (!hasCallIdentity(payload)) return null;
  if (payload.kind === "opened") {
    if ((payload.provider !== undefined && typeof payload.provider !== "string")
      || (payload.model !== undefined && typeof payload.model !== "string")) return null;
    return payload as unknown as ModelStreamOpenedFrame;
  }
  if (payload.kind === "delta") {
    if ((payload.channel !== "output" && payload.channel !== "reasoning")
      || typeof payload.text !== "string"
      || !Number.isSafeInteger(payload.start_offset)
      || !Number.isSafeInteger(payload.end_offset)
      || Number(payload.start_offset) < 0
      || Number(payload.end_offset) < Number(payload.start_offset)
      || Number(payload.end_offset) - Number(payload.start_offset) !== utf8Length(payload.text)) {
      return null;
    }
    return payload as unknown as ModelStreamDeltaFrame;
  }
  if (payload.kind === "closed") {
    if (typeof payload.finished_at !== "string"
      || !MODEL_STREAM_STATUSES.has(payload.status as ModelStreamStatus)
      || (payload.final_text !== undefined && typeof payload.final_text !== "string")
      || (payload.usage !== undefined && !isRecord(payload.usage))
      || (payload.error_code !== undefined && typeof payload.error_code !== "string")
      || (payload.partial !== undefined && typeof payload.partial !== "boolean")
      || (payload.content_omitted !== undefined && typeof payload.content_omitted !== "boolean")) {
      return null;
    }
    return payload as unknown as ModelStreamClosedFrame;
  }
  return null;
}

/** Validate the private root+descendant hydration projection before it reaches chat state. */
export function decodeModelContentResponse(payload: unknown): ModelContentResponse | null {
  if (!isRecord(payload)
    || payload.schema_version !== "studio.model-content.v1"
    || typeof payload.root_run_id !== "string"
    || payload.root_run_id.length === 0
    || !Array.isArray(payload.streams)) return null;
  const seen = new Set<string>();
  const seenRuns = new Set<string>();
  for (const stream of payload.streams) {
    if (!isRecord(stream)
      || stream.root_run_id !== payload.root_run_id
      || typeof stream.run_id !== "string"
      || stream.run_id.length === 0
      || typeof stream.turn_id !== "string"
      || stream.turn_id.length === 0
      || typeof stream.stream_id !== "string"
      || stream.stream_id.length === 0
      || seen.has(modelStreamCallKey(stream.run_id, stream.stream_id))
      || seenRuns.has(stream.run_id)
      || !Number.isSafeInteger(stream.step)
      || Number(stream.step) < 1
      || (stream.provider !== null && typeof stream.provider !== "string")
      || (stream.model !== null && typeof stream.model !== "string")
      || typeof stream.started_at !== "string"
      || !MODEL_CONTENT_STATUSES.has(stream.status as ModelContentStatus)
      || typeof stream.output_text !== "string"
      || !Number.isSafeInteger(stream.output_end_offset)
      || Number(stream.output_end_offset) !== utf8Length(stream.output_text)
      || typeof stream.reasoning_text !== "string"
      || !Number.isSafeInteger(stream.reasoning_end_offset)
      || Number(stream.reasoning_end_offset) !== utf8Length(stream.reasoning_text)
      || typeof stream.partial !== "boolean"
      || (stream.final_text !== undefined && typeof stream.final_text !== "string")
      || (stream.usage !== undefined && !isRecord(stream.usage))
      || (stream.error_code !== undefined && typeof stream.error_code !== "string")) {
      return null;
    }
    seen.add(modelStreamCallKey(stream.run_id, stream.stream_id));
    seenRuns.add(stream.run_id);
  }
  return payload as unknown as ModelContentResponse;
}

function callFromFrame(
  frame: ModelStreamOpenedFrame | ModelStreamDeltaFrame | ModelStreamClosedFrame,
): LiveModelCallState {
  return {
    rootRunId: frame.root_run_id,
    runId: frame.run_id,
    turnId: frame.turn_id,
    streamId: frame.stream_id,
    step: frame.step,
    provider: frame.kind === "opened" ? frame.provider : undefined,
    model: frame.kind === "opened" ? frame.model : undefined,
    startedAt: frame.started_at,
    status: "running",
    output: "",
    reasoning: "",
    outputBytes: 0,
    reasoningBytes: 0,
    partial: false,
  };
}

function withCursor(
  state: ModelStreamViewState,
  frame: ModelStreamFrame,
  cursor: ParsedCursor,
): ModelStreamViewState {
  return {
    ...state,
    generation: cursor.generation,
    sequence: cursor.sequence,
    cursor: frame.cursor,
    resumeCursor: frame.cursor,
  };
}

function requireHydration(
  state: ModelStreamViewState,
  reason: ModelStreamViewState["resetReason"],
  resumeCursor = state.cursor,
): ModelStreamViewState {
  return {
    ...state,
    needsHydration: true,
    resetReason: reason,
    resumeCursor,
  };
}

/**
 * Reduce one passive frame. Content is projected only for the root run; descendant calls remain
 * independently tracked so their lifecycle cannot corrupt the root chat bubble.
 */
export function reduceModelStreamFrame(
  state: ModelStreamViewState,
  frame: ModelStreamFrame,
): ModelStreamViewState {
  if (state.rootRunId !== null && frame.root_run_id !== state.rootRunId) return state;
  const cursor = parseCursor(frame.cursor);
  if (!cursor || cursor.sequence !== frame.sequence) return state;
  if (frame.kind === "reset") {
    if (!parseCursor(frame.latest_cursor)) return state;
    return requireHydration(
      {
        ...state,
        generation: cursor.generation,
        sequence: cursor.sequence,
        cursor: frame.cursor,
        resetLatestCursor: frame.latest_cursor,
      },
      frame.reason,
      // Reconnect at the broker-provided baseline after hydration. Frames retained after that
      // baseline rebuild the live suffix; jumping to latest would silently discard the backlog.
      frame.cursor,
    );
  }
  if (state.needsHydration) return state;

  if (state.generation !== null) {
    if (cursor.generation === state.generation && cursor.sequence <= state.sequence) return state;
    if (cursor.generation !== state.generation) {
      return requireHydration(state, "generation_changed");
    }
    if (cursor.generation === state.generation
      && state.sequence >= 0
      && cursor.sequence !== state.sequence + 1) {
      return requireHydration(state, "sequence_gap");
    }
  }

  let base = withCursor(
    state.rootRunId === null ? { ...state, rootRunId: frame.root_run_id } : state,
    frame,
    cursor,
  );
  if (frame.kind === "closed" && frame.content_omitted) {
    // The broker deliberately retained this small terminal marker while the complete content was
    // flushed to the private sidecar. Resume after the marker so hydration cannot replay-loop.
    return requireHydration(base, "content_omitted", frame.cursor);
  }
  const incomingFloor: ModelStreamCallFloor = {
    step: frame.step,
    startedAt: frame.started_at,
    streamId: frame.stream_id,
  };
  const existingFloor = base.floors[frame.run_id];
  const floorOrder = compareCallFloor(incomingFloor, existingFloor);
  if (floorOrder < 0) return base;
  if (floorOrder > 0) {
    const calls = Object.fromEntries(
      Object.entries(base.calls).filter(([, call]) => call.runId !== frame.run_id),
    );
    base = {
      ...base,
      calls,
      floors: { ...base.floors, [frame.run_id]: incomingFloor },
    };
  }
  const rootCall = frame.run_id === frame.root_run_id;
  const identity = modelStreamCallKey(frame.run_id, frame.stream_id);
  let call = base.calls[identity];

  if (frame.kind === "opened") {
    if (call) return base;
    const output = rootCall ? "" : base.output;
    const reasoning = rootCall ? "" : base.reasoning;
    const remainsSealed = rootCall
      && base.rootTurnSealed
      && (base.sealedRootTurnId === null || base.sealedRootTurnId === frame.turn_id);
    call = callFromFrame(frame);
    return {
      ...base,
      calls: { ...base.calls, [identity]: call },
      activeRootStreamId: rootCall ? frame.stream_id : base.activeRootStreamId,
      activeRootTurnId: rootCall ? frame.turn_id : base.activeRootTurnId,
      output,
      reasoning,
      rootTurnSealed: rootCall ? remainsSealed : base.rootTurnSealed,
      sealedRootTurnId: rootCall && !remainsSealed ? null : base.sealedRootTurnId,
    };
  }

  call ??= callFromFrame(frame);
  const calls = { ...base.calls };
  if (frame.kind === "delta") {
    if (call.status !== "running"
      || (rootCall
        && base.rootTurnSealed
        && (base.sealedRootTurnId === null || frame.turn_id === base.sealedRootTurnId))) {
      return base;
    }
    const knownBytes = frame.channel === "output" ? call.outputBytes : call.reasoningBytes;
    if (frame.start_offset > knownBytes) {
      // The global cursor can remain contiguous while an oversized content frame is omitted. A
      // channel offset gap therefore carries the same hydration requirement as a cursor gap.
      return requireHydration(state, "sequence_gap");
    }
    const suffix = frame.end_offset <= knownBytes
      ? ""
      : utf8Suffix(frame.text, knownBytes - frame.start_offset);
    if (suffix === null) return requireHydration(state, "sequence_gap");
    const updated = frame.channel === "output"
      ? {
          ...call,
          output: call.output + suffix,
          outputBytes: Math.max(call.outputBytes, frame.end_offset),
        }
      : {
          ...call,
          reasoning: call.reasoning + suffix,
          reasoningBytes: Math.max(call.reasoningBytes, frame.end_offset),
        };
    calls[identity] = updated;
    const activatesRoot = rootCall && base.activeRootTurnId === null;
    const projectsRoot = rootCall
      && (activatesRoot || frame.turn_id === base.activeRootTurnId)
      && (!base.activeRootStreamId || base.activeRootStreamId === frame.stream_id);
    return {
      ...base,
      calls,
      activeRootStreamId: activatesRoot ? frame.stream_id : base.activeRootStreamId,
      activeRootTurnId: activatesRoot ? frame.turn_id : base.activeRootTurnId,
      output: projectsRoot && frame.channel === "output" ? base.output + suffix : base.output,
      reasoning: projectsRoot && frame.channel === "reasoning"
        ? base.reasoning + suffix
        : base.reasoning,
    };
  }

  if (call.status !== "running") return base;
  const finalText = frame.final_text;
  const updated: LiveModelCallState = {
    ...call,
    status: frame.status,
    finishedAt: frame.finished_at,
    finalText,
    output: finalText ?? call.output,
    outputBytes: finalText === undefined ? call.outputBytes : utf8Length(finalText),
    partial: frame.partial === true,
  };
  calls[identity] = updated;
  return {
    ...base,
    calls,
    // A retained-ring replay must never leave an already closed operation looking live. The
    // durable turn event or hydrated transcript owns completed text; partial closes are projected
    // separately with a deterministic message id.
    activeRootStreamId: rootCall && base.activeRootStreamId === frame.stream_id
      ? null
      : base.activeRootStreamId,
    activeRootTurnId: rootCall && base.activeRootStreamId === frame.stream_id
      ? frame.status === "completed" ? base.activeRootTurnId : null
      : base.activeRootTurnId,
    output: rootCall && base.activeRootStreamId === frame.stream_id
      ? frame.status === "completed" ? updated.output : ""
      : base.output,
    reasoning: rootCall && base.activeRootStreamId === frame.stream_id ? "" : base.reasoning,
  };
}

function appendMessage(messages: ChatMessage[], message: ChatMessage): ChatMessage[] {
  if (messages.some((item) => item.id === message.id)) return messages;
  return [...messages, message];
}

function compareCallFloor(
  incoming: ModelStreamCallFloor,
  existing: ModelStreamCallFloor | undefined,
): number {
  if (!existing) return 1;
  if (incoming.step !== existing.step) return incoming.step > existing.step ? 1 : -1;
  if (incoming.startedAt !== existing.startedAt) {
    return incoming.startedAt > existing.startedAt ? 1 : -1;
  }
  if (incoming.streamId === existing.streamId) return 0;
  return incoming.streamId > existing.streamId ? 1 : -1;
}

/** Apply root-call live content to chat state without ever inserting it into RunViewState.events. */
export function projectModelStreamFrame(
  run: RunViewState,
  before: ModelStreamViewState,
  after: ModelStreamViewState,
  frame: ModelStreamFrame,
): RunViewState {
  if (after.needsHydration || frame.kind === "reset" || frame.root_run_id !== run.runId) return run;
  if (frame.run_id !== frame.root_run_id) return run;
  if (run.messages.some((message) => message.source?.kind === "model_stream_active")) {
    // The initial private snapshot request may fail while the transcript and live SSE still work.
    // Once a root frame is valid, its bubble owns the ephemeral row and must not render beside it.
    run = {
      ...run,
      messages: run.messages.filter((message) => message.source?.kind !== "model_stream_active"),
    };
  }

  if (frame.kind === "closed") {
    const identity = modelStreamCallKey(frame.run_id, frame.stream_id);
    const prior = before.calls[identity];
    const call = after.calls[identity];
    const firstClose = prior?.status === "running" || (!prior && call?.status !== "running");
    if (firstClose && call?.partial && call.output) {
      const parsedStarted = call.startedAt ? Date.parse(call.startedAt) : Number.NaN;
      const parsedFinished = Date.parse(frame.finished_at);
      const createdAt = Number.isFinite(parsedFinished)
        ? parsedFinished / 1000
        : Number.isFinite(parsedStarted)
          ? parsedStarted / 1000
          : Date.now() / 1000;
      const partial: ChatMessage = {
        id: `assistant:model-stream:${frame.stream_id}:partial`,
        role: "assistant",
        content: call.output,
        attachments: [],
        created_at: createdAt,
        source: {
          kind: "model_stream_partial",
          root_run_id: frame.root_run_id,
          run_id: frame.run_id,
          turn_id: frame.turn_id,
          stream_id: frame.stream_id,
          status: frame.status,
          partial: true,
        },
      };
      return {
        ...run,
        activeResponse: "",
        reasoning: "",
        messages: appendMessage(run.messages, partial),
      };
    }
    if (firstClose && frame.status === "completed") {
      const operationStillLive = !after.rootTurnSealed
        && ["queued", "running", "awaiting-approval", "stopping"].includes(run.status);
      return operationStillLive
        ? { ...run, activeResponse: call?.output || run.activeResponse, reasoning: "" }
        : { ...run, activeResponse: "", reasoning: "" };
    }
    if (firstClose) return { ...run, activeResponse: "", reasoning: "" };
    return run;
  }

  const projectionChanged = before.output !== after.output
    || before.reasoning !== after.reasoning
    || before.activeRootTurnId !== after.activeRootTurnId;
  return projectionChanged
    ? { ...run, activeResponse: after.output, reasoning: after.reasoning }
    : run;
}

/**
 * Project descendant content onto its activity card without fabricating durable trace events.
 * A placeholder keeps frames that race the durable `subagent.started` event; that event later
 * enriches the same activity with task/type metadata.
 */
export function projectSubagentModelStream(
  activity: SubagentActivity | undefined,
  before: ModelStreamViewState,
  after: ModelStreamViewState,
  frame: ModelStreamFrame,
): SubagentActivity | undefined {
  if (after.needsHydration || frame.kind === "reset" || frame.run_id === frame.root_run_id) {
    return activity;
  }
  const identity = modelStreamCallKey(frame.run_id, frame.stream_id);
  const priorCall = before.calls[identity];
  const call = after.calls[identity];
  if (!call || priorCall === call) return activity;
  const current: SubagentActivity = activity ?? {
    childRunId: frame.run_id,
    subagentType: "delegate",
    parentRunId: frame.root_run_id,
    taskId: "",
    depth: Math.max(1, frame.run_id.split(".sub.").length - 1),
    status: "running",
    events: [],
  };
  const newStream = current.liveStreamId !== frame.stream_id;
  return {
    ...current,
    liveStreamId: frame.stream_id,
    liveTurnId: frame.turn_id,
    liveOutput: call.output,
    liveReasoning: call.reasoning,
    liveStreamStatus: call.status,
    // A later durable failure/finish owns the card's final status. The passive opened frame only
    // revives a placeholder that has not yet received that lifecycle.
    status: frame.kind === "opened" && newStream && current.events.length === 0
      ? "running"
      : current.status,
  };
}

/** Stop accepting root deltas after the correlated durable turn (or the whole run) is terminal. */
export function sealModelStreamTurn(
  state: ModelStreamViewState,
  turnId?: string,
): ModelStreamViewState {
  if (turnId !== undefined
    && state.activeRootTurnId !== null
    && turnId !== state.activeRootTurnId) return state;
  const sealedRootTurnId = turnId ?? null;
  if (state.rootTurnSealed && state.sealedRootTurnId === sealedRootTurnId) return state;
  return { ...state, rootTurnSealed: true, sealedRootTurnId };
}

/** Reapply an authoritative active snapshot after historical durable events are replayed. */
export function restoreActiveModelContent(
  run: RunViewState,
  state: ModelStreamViewState,
): RunViewState {
  if (state.rootTurnSealed || !state.activeRootTurnId) return run;
  const status = state.activeRootStreamId ? "running" as const : run.status;
  if (run.activeResponse === state.output
    && run.reasoning === state.reasoning
    && run.status === status) return run;
  return { ...run, status, activeResponse: state.output, reasoning: state.reasoning };
}

/** Clear speculative state after authoritative chat hydration and resume at its high-watermark. */
export function markModelStreamHydrated(state: ModelStreamViewState): ModelStreamViewState {
  const cursor = parseCursor(state.resumeCursor);
  return {
    ...initialModelStreamState(state.rootRunId),
    generation: cursor?.generation ?? null,
    sequence: cursor?.sequence ?? -1,
    cursor: state.resumeCursor,
    resumeCursor: state.resumeCursor,
  };
}

function callFromSnapshot(snapshot: ModelContentSnapshot): LiveModelCallState {
  const output = snapshot.status === "running"
    ? snapshot.output_text
    : snapshot.final_text ?? snapshot.output_text;
  return {
    rootRunId: snapshot.root_run_id,
    runId: snapshot.run_id,
    turnId: snapshot.turn_id,
    streamId: snapshot.stream_id,
    step: snapshot.step,
    provider: snapshot.provider ?? undefined,
    model: snapshot.model ?? undefined,
    startedAt: snapshot.started_at,
    status: snapshot.status,
    output,
    reasoning: snapshot.reasoning_text,
    outputBytes: snapshot.output_end_offset,
    reasoningBytes: snapshot.reasoning_end_offset,
    finalText: snapshot.final_text,
    partial: snapshot.partial,
  };
}

/** Seed exact channel prefixes from the private sidecar before retained SSE replay resumes. */
export function seedModelStreamSnapshot(
  state: ModelStreamViewState,
  response: ModelContentResponse,
): ModelStreamViewState {
  if (state.rootRunId !== null && state.rootRunId !== response.root_run_id) return state;
  const cursor = parseCursor(state.resumeCursor);
  const calls: Record<string, LiveModelCallState> = {};
  const floors: Record<string, ModelStreamCallFloor> = {};
  let activeRoot: LiveModelCallState | null = null;
  for (const snapshot of response.streams) {
    const call = callFromSnapshot(snapshot);
    calls[modelStreamCallKey(call.runId, call.streamId)] = call;
    floors[call.runId] = {
      step: call.step,
      startedAt: call.startedAt ?? "",
      streamId: call.streamId,
    };
    if (call.runId === response.root_run_id && call.status === "running") activeRoot = call;
  }
  return {
    ...initialModelStreamState(response.root_run_id),
    generation: cursor?.generation ?? null,
    sequence: cursor?.sequence ?? -1,
    cursor: state.resumeCursor,
    resumeCursor: state.resumeCursor,
    calls,
    floors,
    activeRootStreamId: activeRoot?.streamId ?? null,
    activeRootTurnId: activeRoot?.turnId ?? null,
    output: activeRoot?.output ?? "",
    reasoning: activeRoot?.reasoning ?? "",
  };
}

/** Replace ephemeral active snapshot rows with the single live bubble seeded above. */
export function projectModelContentSnapshot(
  run: RunViewState,
  state: ModelStreamViewState,
): RunViewState {
  let messages = run.messages.filter((message) => message.source?.kind !== "model_stream_active");
  const rootFloor = state.rootRunId ? state.floors[state.rootRunId] : undefined;
  const terminal = state.rootRunId && rootFloor
    ? state.calls[modelStreamCallKey(state.rootRunId, rootFloor.streamId)]
    : undefined;
  if (terminal
    && terminal.status !== "running"
    && terminal.status !== "completed"
    && terminal.status !== "abandoned"
    && terminal.partial
    && terminal.output) {
    const parsedStarted = terminal.startedAt ? Date.parse(terminal.startedAt) : Number.NaN;
    messages = appendMessage(messages, {
      id: `assistant:model-stream:${terminal.streamId}:partial`,
      role: "assistant",
      content: terminal.output,
      attachments: [],
      created_at: Number.isFinite(parsedStarted) ? parsedStarted / 1000 : Date.now() / 1000,
      source: {
        kind: "model_stream_partial",
        root_run_id: terminal.rootRunId,
        run_id: terminal.runId,
        turn_id: terminal.turnId,
        stream_id: terminal.streamId,
        status: terminal.status,
        partial: true,
      },
    });
  }
  const withoutEphemeralRow = { ...run, messages };
  return state.activeRootStreamId
    ? restoreActiveModelContent(withoutEphemeralRow, state)
    : { ...withoutEphemeralRow, activeResponse: "", reasoning: "" };
}

/** Seed child cards from the same authorized snapshot without creating trace events. */
export function seedSubagentModelContent(
  activities: Record<string, SubagentActivity>,
  response: ModelContentResponse,
): Record<string, SubagentActivity> {
  let next = activities;
  for (const snapshot of response.streams) {
    if (snapshot.run_id === response.root_run_id) continue;
    const existing = next[snapshot.run_id];
    const activity: SubagentActivity = {
      ...(existing ?? {
        childRunId: snapshot.run_id,
        subagentType: "delegate",
        parentRunId: response.root_run_id,
        taskId: "",
        depth: Math.max(1, snapshot.run_id.split(".sub.").length - 1),
        // A model call can settle before the child performs tools or starts another turn. Durable
        // subagent lifecycle events own the card status; this private snapshot is content only.
        status: "running" as const,
        events: [],
      }),
      liveStreamId: snapshot.stream_id,
      liveTurnId: snapshot.turn_id,
      liveOutput: snapshot.final_text ?? snapshot.output_text,
      liveReasoning: snapshot.reasoning_text,
      liveStreamStatus: snapshot.status,
    };
    if (next === activities) next = { ...activities };
    next[snapshot.run_id] = activity;
  }
  return next;
}

export interface SubagentStartedProjection {
  activity: SubagentActivity;
  revive: boolean;
}

/** Enrich a child lifecycle start while keeping process-lost recovery prefixes fenced. */
export function projectSubagentStarted(
  existing: SubagentActivity | undefined,
  data: Record<string, unknown>,
  parentRunId: string,
  historical: boolean,
  recoveryFenced: boolean,
): SubagentStartedProjection {
  const childRunId = String(data.child_run_id ?? "");
  const revive = !(historical && recoveryFenced);
  return {
    revive,
    activity: {
      ...(existing ?? {
        childRunId,
        status: "running" as const,
        events: [],
      }),
      childRunId,
      subagentType: String(data.subagent_type ?? existing?.subagentType ?? "delegate"),
      parentRunId: String(data.parent_run_id ?? existing?.parentRunId ?? parentRunId),
      taskId: String(data.task_id ?? existing?.taskId ?? ""),
      status: revive
        ? "running"
        : existing?.status === "succeeded" || existing?.status === "failed"
          ? existing.status
          : "failed",
      depth: Number.isFinite(Number(data.depth))
        ? Number(data.depth)
        : existing?.depth ?? childRunId.split(".sub.").length - 1,
    },
  };
}

export interface ModelStreamEventSourceOptions {
  rootRunId: string;
  cursor?: string | null;
  onFrame: (frame: ModelStreamFrame) => void;
  onConnectionChange?: (connected: boolean) => void;
}

/** Independent passive transport. Closing it never touches the execution-owned run stream. */
export class ModelStreamEventSource {
  #source: EventSource | null = null;
  #options: ModelStreamEventSourceOptions;

  constructor(options: ModelStreamEventSourceOptions) {
    this.#options = options;
  }

  open(): void {
    this.close();
    const query = new URLSearchParams({ run_id: this.#options.rootRunId });
    if (this.#options.cursor) query.set("cursor", this.#options.cursor);
    const source = new EventSource(`/api/model-stream?${query}`);
    this.#source = source;
    source.onopen = () => {
      if (this.#source === source) this.#options.onConnectionChange?.(true);
    };
    source.onerror = () => {
      if (this.#source === source) this.#options.onConnectionChange?.(false);
    };
    source.addEventListener("model-stream", (message) => {
      if (this.#source !== source) return;
      try {
        const frame = decodeModelStreamFrame(JSON.parse((message as MessageEvent).data));
        if (frame) this.#options.onFrame(frame);
      } catch {
        // One malformed/private frame cannot affect the durable operation stream or the run.
      }
    });
  }

  close(): void {
    const source = this.#source;
    this.#source = null;
    source?.close();
    this.#options.onConnectionChange?.(false);
  }
}
