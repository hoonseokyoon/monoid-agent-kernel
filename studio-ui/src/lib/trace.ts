import type { RunEvent } from "./types";

export const LEGACY_MODEL_DELTA_TYPES = [
  "model.output.delta",
  "model.reasoning.delta",
] as const;

type LegacyModelDeltaType = (typeof LEGACY_MODEL_DELTA_TYPES)[number];

export interface DeltaTypeSummary {
  eventCount: number;
  textBytes: number;
}

export interface TraceSummary {
  sourceEventCount: number;
  operationEvents: RunEvent[];
  omittedDeltaEventCount: number;
  omittedDeltaTextBytes: number;
  omittedDeltaTypes: Record<LegacyModelDeltaType, DeltaTypeSummary>;
}

const legacyModelDeltaTypes: ReadonlySet<string> = new Set(LEGACY_MODEL_DELTA_TYPES);
const utf8Encoder = new TextEncoder();

export function isLegacyModelDeltaEvent(event: Pick<RunEvent, "type">): boolean {
  return legacyModelDeltaTypes.has(event.type);
}

export function operationTraceEvents(events: readonly RunEvent[]): RunEvent[] {
  return events.filter((event) => !isLegacyModelDeltaEvent(event));
}

export function summarizeTrace(events: readonly RunEvent[]): TraceSummary {
  const operationEvents: RunEvent[] = [];
  const omittedDeltaTypes: TraceSummary["omittedDeltaTypes"] = {
    "model.output.delta": { eventCount: 0, textBytes: 0 },
    "model.reasoning.delta": { eventCount: 0, textBytes: 0 },
  };

  for (const event of events) {
    if (!isLegacyModelDeltaEvent(event)) {
      operationEvents.push(event);
      continue;
    }

    const type = event.type as LegacyModelDeltaType;
    const text = event.data?.text;
    const textBytes = typeof text === "string" ? utf8Encoder.encode(text).byteLength : 0;
    omittedDeltaTypes[type].eventCount += 1;
    omittedDeltaTypes[type].textBytes += textBytes;
  }

  const omittedDeltaEventCount = LEGACY_MODEL_DELTA_TYPES.reduce(
    (total, type) => total + omittedDeltaTypes[type].eventCount,
    0,
  );
  const omittedDeltaTextBytes = LEGACY_MODEL_DELTA_TYPES.reduce(
    (total, type) => total + omittedDeltaTypes[type].textBytes,
    0,
  );

  return {
    sourceEventCount: events.length,
    operationEvents,
    omittedDeltaEventCount,
    omittedDeltaTextBytes,
    omittedDeltaTypes,
  };
}

export function createRawTraceExport(events: readonly RunEvent[]) {
  return {
    schema_version: "studio.trace-export.v1" as const,
    events: [...events],
  };
}

export function createCompactTraceExport(events: readonly RunEvent[]) {
  const summary = summarizeTrace(events);
  return {
    schema_version: "studio.trace-export.compact.v1" as const,
    summary: {
      source_event_count: summary.sourceEventCount,
      operation_event_count: summary.operationEvents.length,
      omitted_delta_event_count: summary.omittedDeltaEventCount,
      omitted_delta_text_bytes: summary.omittedDeltaTextBytes,
      omitted_delta_types: {
        "model.output.delta": {
          event_count: summary.omittedDeltaTypes["model.output.delta"].eventCount,
          text_bytes: summary.omittedDeltaTypes["model.output.delta"].textBytes,
        },
        "model.reasoning.delta": {
          event_count: summary.omittedDeltaTypes["model.reasoning.delta"].eventCount,
          text_bytes: summary.omittedDeltaTypes["model.reasoning.delta"].textBytes,
        },
      },
    },
    events: summary.operationEvents,
  };
}
