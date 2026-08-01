import type {
  ApplyResponse,
  AttachmentInput,
  ChatMessage,
  ChatResponse,
  ChatTranscriptResponse,
  EventsResponse,
  FilePreviewResponse,
  FilesResponse,
  JobsResponse,
  PackageReceipt,
  Profile,
  ProfilePreviewResponse,
  ProfilesResponse,
  ProposalResponse,
  PauseResponse,
  ResumeResponse,
  RetryResponse,
  RunEvent,
  SessionsResponse,
  SettingsResponse,
  SettingsUpdateResponse,
  StudioConfig,
} from "./types";

const CHAT_V1_KEYS = ["schema_version", "run_id", "messages", "event_cursor"] as const;
const CHAT_V2_KEYS = [...CHAT_V1_KEYS, "event_log_error"] as const;
const CHAT_TRANSCRIPT_PROTOCOL_ERROR =
  "Studio returned an unsupported or malformed chat transcript response.";

function isRecord(payload: unknown): payload is Record<string, unknown> {
  return typeof payload === "object" && payload !== null && !Array.isArray(payload);
}

function hasExactKeys(payload: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(payload);
  return actual.length === expected.length
    && expected.every((key) => Object.prototype.hasOwnProperty.call(payload, key));
}

function areChatAttachments(payload: unknown): payload is ChatMessage["attachments"] {
  if (!Array.isArray(payload)) return false;
  for (const attachment of payload) {
    if (!isRecord(attachment)
      || typeof attachment.name !== "string"
      || typeof attachment.mime !== "string") return false;
  }
  return true;
}

function isChatMessage(payload: unknown): payload is ChatMessage {
  if (!isRecord(payload)) return false;
  const roleIsValid = payload.role === "user"
    || payload.role === "assistant"
    || payload.role === "error";
  return typeof payload.id === "string"
    && payload.id.trim().length > 0
    && roleIsValid
    && typeof payload.content === "string"
    && areChatAttachments(payload.attachments)
    && typeof payload.created_at === "number"
    && Number.isFinite(payload.created_at)
    && (payload.source === undefined || isRecord(payload.source));
}

function areChatMessages(payload: unknown): payload is ChatMessage[] {
  if (!Array.isArray(payload)) return false;
  const ids = new Set<string>();
  for (const message of payload) {
    if (!isChatMessage(message) || ids.has(message.id)) return false;
    ids.add(message.id);
  }
  return true;
}

export function isChatTranscriptResponse(payload: unknown): payload is ChatTranscriptResponse {
  if (!isRecord(payload)) return false;
  const commonFieldsAreValid = typeof payload.run_id === "string"
    && areChatMessages(payload.messages)
    && Number.isInteger(payload.event_cursor);
  if (!commonFieldsAreValid) return false;
  if (payload.schema_version === "studio.chat.v1") {
    return hasExactKeys(payload, CHAT_V1_KEYS);
  }
  if (payload.schema_version === "studio.chat.v2") {
    return typeof payload.event_log_error === "string" && hasExactKeys(payload, CHAT_V2_KEYS);
  }
  return false;
}

export class StudioApiError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "StudioApiError";
    this.status = status;
    this.payload = payload;
  }
}

export function decodeChatTranscriptResponse(payload: unknown): ChatTranscriptResponse {
  if (!isChatTranscriptResponse(payload)) {
    throw new StudioApiError(CHAT_TRANSCRIPT_PROTOCOL_ERROR, 502, payload);
  }
  return payload;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  const payload: unknown = await response.json().catch(() => ({}));
  const error = isRecord(payload) ? payload.error : undefined;
  if (!response.ok || typeof error === "string") {
    throw new StudioApiError(
      typeof error === "string" ? error : `${response.status} ${response.statusText}`,
      response.status,
      payload,
    );
  }
  return payload as T;
}

const post = <T>(path: string, body: unknown): Promise<T> =>
  json<T>(path, { method: "POST", body: JSON.stringify(body) });

export const studioApi = {
  config: () => json<StudioConfig>("/api/config"),
  settings: () => json<SettingsResponse>("/api/settings"),
  updateSettings: (settings: Partial<Pick<SettingsResponse, "capabilities" | "model" | "effort" | "summary" | "otel">>) =>
    post<SettingsUpdateResponse>("/api/settings", settings),
  profiles: () => json<ProfilesResponse>("/api/profiles"),
  saveProfile: (profile: Profile) => post<{ profile: Profile }>("/api/profiles", profile),
  previewProfile: (profile: Profile) => post<ProfilePreviewResponse>("/api/profile-preview", profile),
  sessions: (profileId?: string) =>
    json<SessionsResponse>(`/api/sessions${profileId ? `?profile_id=${encodeURIComponent(profileId)}` : ""}`),
  transcript: async (runId: string) =>
    decodeChatTranscriptResponse(
      await json<unknown>(`/api/chat-transcript?run_id=${encodeURIComponent(runId)}`),
    ),
  modelContent: (runId: string) =>
    json<unknown>(`/api/model-content?run_id=${encodeURIComponent(runId)}`),
  subagentEvents: (childRunId: string, from = 0, signal?: AbortSignal) =>
    json<EventsResponse>(
      `/api/subagent-events?run_id=${encodeURIComponent(childRunId)}&from=${Math.max(0, from)}`,
      { signal },
    ),
  send: (input: {
    runId?: string | null;
    message: string;
    attachments?: AttachmentInput[];
    profileId: string;
    clientMessageId: string;
  }) =>
    post<ChatResponse>("/api/chat", {
      run_id: input.runId ?? undefined,
      message: input.message,
      attachments: input.attachments ?? [],
      profile_id: input.profileId,
      client_message_id: input.clientMessageId,
    }),
  interrupt: (runId: string) => post<ChatResponse>("/api/interrupt", { run_id: runId }),
  pause: (runId: string) => post<PauseResponse>("/api/pause", { run_id: runId }),
  resume: (runId: string) => post<ResumeResponse>("/api/resume", { run_id: runId }),
  retry: (runId: string) => post<RetryResponse>("/api/retry", { run_id: runId }),
  cancel: (runId: string) => post<ChatResponse>("/api/cancel", { run_id: runId }),
  answerApproval: (runId: string, taskId: string, answer: string) =>
    post<Record<string, unknown>>("/api/hitl", { run_id: runId, task_id: taskId, answer }),
  files: () => json<FilesResponse>("/api/files"),
  file: (path: string) =>
    json<FilePreviewResponse>(`/api/file?path=${encodeURIComponent(path)}`),
  fileRawUrl: (path: string) => `/api/file-raw?path=${encodeURIComponent(path)}`,
  jobs: (runId: string) => json<JobsResponse>(`/api/jobs?run_id=${encodeURIComponent(runId)}`),
  proposal: (runId: string) => json<ProposalResponse>(`/api/proposal?run_id=${encodeURIComponent(runId)}`),
  proposalFileRawUrl: (runId: string, path: string, proposalHash: string, retry = 0) => {
    const query = new URLSearchParams({
      run_id: runId,
      path,
      expected_proposal_hash: proposalHash,
      retry: String(retry),
    });
    return `/api/proposal-file-raw?${query}`;
  },
  applyProposal: (runId: string, approvedPaths: string[], expectedProposalHash: string) =>
    post<ApplyResponse>("/api/apply", {
      run_id: runId,
      approved_paths: approvedPaths,
      expected_proposal_hash: expectedProposalHash,
    }),
  exportPackage: (runId: string, expectedProposalHash: string) =>
    post<PackageReceipt>("/api/export-package", {
      run_id: runId,
      expected_proposal_hash: expectedProposalHash,
    }),
  artifactUrl: (runId: string, digest: string) =>
    `/api/artifact?run_id=${encodeURIComponent(runId)}&digest=${encodeURIComponent(digest)}`,
};

export async function downloadProposalPackage(runId: string, expectedProposalHash: string): Promise<PackageReceipt> {
  const receipt = await studioApi.exportPackage(runId, expectedProposalHash);
  if (receipt.proposal_hash !== expectedProposalHash) {
    throw new StudioApiError("The exported package does not match the reviewed proposal revision.", 409, receipt);
  }
  const response = await fetch(studioApi.artifactUrl(runId, receipt.digest));
  if (!response.ok) {
    throw new StudioApiError(`Artifact download failed (${response.status})`, response.status, null);
  }
  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const name = disposition.match(/filename="([^"]+)"/)?.[1] ?? receipt.name ?? "proposal.tar";
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(href);
  return { ...receipt, name };
}

export interface RunEventStreamOptions {
  runId: string;
  from?: number;
  onEvent: (event: RunEvent) => void;
  onConnectionChange?: (connected: boolean) => void;
}

export class RunEventStream {
  #source: EventSource | null = null;
  #options: RunEventStreamOptions;

  constructor(options: RunEventStreamOptions) {
    this.#options = options;
  }

  open(): void {
    this.close();
    const from = Math.max(0, this.#options.from ?? 0);
    this.#source = new EventSource(
      `/api/events?run_id=${encodeURIComponent(this.#options.runId)}&from=${from}`,
    );
    this.#source.onopen = () => this.#options.onConnectionChange?.(true);
    this.#source.onerror = () => this.#options.onConnectionChange?.(false);
    this.#source.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data) as RunEvent;
        this.#options.onEvent(event);
        if (event.type === "studio.stream.end") {
          this.close();
        }
      } catch {
        // One malformed frame cannot invalidate the replay-safe stream.
      }
    };
  }

  close(): void {
    this.#source?.close();
    this.#source = null;
    this.#options.onConnectionChange?.(false);
  }
}

export function clientMessageId(): string {
  return `studio_${Date.now().toString(36)}_${crypto.randomUUID().slice(0, 8)}`;
}
