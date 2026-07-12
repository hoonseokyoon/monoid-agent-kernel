import type {
  ApprovalRequest,
  ChatMessage,
  ChatTranscriptResponse,
  PlanItem,
  RunEvent,
  RunStatus,
  RunViewState,
} from "./types";

export function isRunBusy(status: RunStatus): boolean {
  return status === "queued"
    || status === "running"
    || status === "awaiting-approval"
    || status === "stopping";
}

export function initialRunState(runId: string | null = null): RunViewState {
  return {
    runId,
    status: "idle",
    messages: [],
    activeResponse: "",
    reasoning: "",
    events: [],
    lastSeq: -1,
    replayCursor: -1,
    approvals: [],
    usage: { input: 0, output: 0, total: 0 },
    plan: [],
    error: null,
    errorRetryable: false,
    manualRetryCandidate: false,
    manualRetryReady: false,
    proposalDirty: false,
    lastUserMessage: "",
  };
}

export function hydrateTranscript(
  state: RunViewState,
  transcript: ChatTranscriptResponse,
): RunViewState {
  return {
    ...state,
    runId: transcript.run_id,
    messages: transcript.messages ?? [],
    replayCursor: Number.isInteger(transcript.event_cursor) ? transcript.event_cursor : -1,
  };
}

function appendMessage(state: RunViewState, message: ChatMessage): ChatMessage[] {
  if (state.messages.some((item) => item.id === message.id)) return state.messages;
  return [...state.messages, message];
}

function eventMessage(event: RunEvent, role: "assistant" | "error", content: string): ChatMessage {
  const id = event.event_id || `seq:${event.seq ?? Date.now()}`;
  return {
    id: `${role}:${id}`,
    role,
    content,
    attachments: [],
    created_at: event.timestamp ? Date.parse(event.timestamp) / 1000 : Date.now() / 1000,
    source: { kind: "event", event_type: event.type, event_id: event.event_id, seq: event.seq },
  };
}

function approvalFrom(event: RunEvent): ApprovalRequest {
  const data = event.data;
  const request = typeof data.request === "object" && data.request !== null
    ? data.request as Record<string, unknown>
    : {};
  const receivedChoices = Array.isArray(data.choices) ? data.choices.map(String) : [];
  const choices = receivedChoices.length || data.kind !== "tool_approval"
    ? receivedChoices
    : ["Approve", "Deny"];
  const argumentsPreview = request.arguments_preview ?? data.arguments_preview;
  return {
    taskId: String(data.task_id ?? ""),
    kind: data.kind === "tool_approval" ? "tool_approval" : "hitl",
    prompt: String(data.prompt ?? request.prompt ?? "This action needs your approval."),
    choices,
    tool: String(
      request.call_name ?? request.model_name ?? request.tool_id ?? data.tool ?? "",
    ) || undefined,
    argumentsPreview: argumentsPreview === undefined
      ? undefined
      : typeof argumentsPreview === "string"
        ? argumentsPreview
        : JSON.stringify(argumentsPreview, null, 2),
    requestedAt: event.timestamp,
    status: "pending",
  };
}

export function reduceRunEvent(state: RunViewState, event: RunEvent): RunViewState {
  const seq = typeof event.seq === "number" ? event.seq : -1;
  if (seq >= 0 && seq <= state.lastSeq) return state;

  const data = event.data ?? {};
  let next: RunViewState = {
    ...state,
    lastSeq: seq >= 0 ? seq : state.lastSeq,
    events: event.event_id ? [...state.events, event] : state.events,
  };
  const projectedByTranscript = seq >= 0 && seq <= state.replayCursor;

  switch (event.type) {
    case "run.queued":
    case "run.created":
      return { ...next, status: "queued" };
    case "model.turn.started":
      return {
        ...next,
        status: "running",
        activeResponse: "",
        reasoning: "",
        error: null,
        errorRetryable: false,
        manualRetryCandidate: false,
        manualRetryReady: false,
      };
    case "model.reasoning.delta":
      return { ...next, reasoning: next.reasoning + String(data.text ?? "") };
    case "model.output.delta":
      return { ...next, activeResponse: next.activeResponse + String(data.text ?? "") };
    case "task.started": {
      if (data.kind !== "hitl" && data.kind !== "tool_approval") return next;
      const approval = approvalFrom(event);
      if (!approval.taskId || next.approvals.some((item) => item.taskId === approval.taskId)) {
        return { ...next, status: "awaiting-approval" };
      }
      return { ...next, status: "awaiting-approval", approvals: [...next.approvals, approval] };
    }
    case "task.finished":
    case "task.cancelled":
    case "task.timed_out":
    case "task.failed": {
      const taskId = String(data.task_id ?? "");
      const result = typeof data.result === "object" && data.result !== null
        ? data.result as Record<string, unknown>
        : {};
      const denied = event.type !== "task.finished" || result.approved === false || /deny|reject/i.test(String(result.answer ?? ""));
      const approvals = next.approvals.map((item) => item.taskId === taskId
        ? { ...item, status: denied ? "denied" as const : "approved" as const }
        : item);
      const hasPending = approvals.some((item) => item.status === "pending" || item.status === "submitting");
      return { ...next, approvals, status: hasPending ? "awaiting-approval" : event.type === "task.finished" ? "queued" : "idle" };
    }
    case "turn.interrupted":
      return { ...next, status: "stopped", activeResponse: "", manualRetryCandidate: false, manualRetryReady: false };
    case "run.paused":
    case "run.pause.completed":
      return { ...next, status: "stopped", activeResponse: "" };
    case "session.state.changed":
      if (data.state === "paused") {
        return { ...next, status: "stopped", activeResponse: "" };
      }
      if (data.state === "running") {
        return { ...next, status: "running" };
      }
      return next;
    case "run.resumed":
    case "run.resume.completed":
    case "run.retrying":
      return { ...next, status: "queued", error: null, errorRetryable: false, manualRetryCandidate: false, manualRetryReady: false };
    case "turn.failed": {
      const retryable = Boolean(data.retryable);
      if (retryable) return { ...next, status: "running" };
      const content = String(data.error ?? "The model turn failed.");
      return {
        ...next,
        status: "failed",
        error: content,
        errorRetryable: false,
        manualRetryCandidate: true,
        manualRetryReady: false,
        messages: projectedByTranscript ? next.messages : appendMessage(next, eventMessage(event, "error", content)),
      };
    }
    case "turn.settled": {
      const content = String(data.final_text ?? next.activeResponse ?? "");
      return {
        ...next,
        status: "idle",
        activeResponse: "",
        reasoning: "",
        manualRetryCandidate: false,
        manualRetryReady: false,
        messages:
          !content || projectedByTranscript
            ? next.messages
            : appendMessage(next, eventMessage(event, "assistant", content)),
      };
    }
    case "run.awaiting_input":
      if (next.approvals.some((item) => item.status === "pending" || item.status === "submitting")) {
        return { ...next, status: "awaiting-approval" };
      }
      if (next.manualRetryCandidate) {
        return { ...next, status: "failed", manualRetryReady: true, errorRetryable: true };
      }
      return { ...next, status: "idle" };
    case "run.finished": {
      const terminalStatus = String(data.status ?? "completed").toLowerCase();
      const succeeded = ["completed", "succeeded", "success", "ok"].includes(terminalStatus);
      if (succeeded) {
        return {
          ...next,
          status: "succeeded",
          activeResponse: "",
          reasoning: "",
          error: null,
          manualRetryCandidate: false,
          manualRetryReady: false,
          errorRetryable: false,
        };
      }
      const hadError = Boolean(next.error);
      const content = String(
        data.error
        || next.error
        || data.error_code
        || `Run finished with status ${terminalStatus}.`,
      );
      return {
        ...next,
        status: "failed",
        activeResponse: "",
        reasoning: "",
        error: content,
        errorRetryable: false,
        manualRetryCandidate: false,
        manualRetryReady: false,
        messages: hadError || projectedByTranscript
          ? next.messages
          : appendMessage(next, eventMessage(event, "error", content)),
      };
    }
    case "run.failed":
    case "ModelAdapterError": {
      const content = String(data.error ?? data.message ?? "The run failed.");
      return {
        ...next,
        status: "failed",
        error: content,
        errorRetryable: Boolean(data.retryable),
        manualRetryCandidate: false,
        manualRetryReady: false,
        activeResponse: "",
        messages: projectedByTranscript ? next.messages : appendMessage(next, eventMessage(event, "error", content)),
      };
    }
    case "metrics.updated":
      return {
        ...next,
        usage: {
          input: Number(data.input_tokens ?? next.usage.input),
          output: Number(data.output_tokens ?? next.usage.output),
          total: Number(data.total_tokens ?? next.usage.total),
        },
      };
    case "plan.updated":
      return { ...next, plan: Array.isArray(data.items) ? (data.items as unknown as PlanItem[]) : [] };
    case "proposal.ready":
    case "workspace.diff.updated":
    case "workspace.proposal.updated":
      return { ...next, proposalDirty: true };
    default:
      return next;
  }
}

export function appendOptimisticUserMessage(
  state: RunViewState,
  id: string,
  content: string,
  attachments: Array<{ name: string; mime: string }> = [],
): RunViewState {
  return {
    ...state,
    lastUserMessage: content,
    status: "queued",
    messages: appendMessage(state, {
      id,
      role: "user",
      content,
      attachments,
      created_at: Date.now() / 1000,
      source: { kind: "client", client_message_id: id },
    }),
  };
}
