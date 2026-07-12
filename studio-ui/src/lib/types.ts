export type RunStatus =
  | "idle"
  | "queued"
  | "running"
  | "awaiting-approval"
  | "stopping"
  | "stopped"
  | "failed"
  | "succeeded";

export type StudioMode = "run" | "profile" | "review" | "trace";
export type InspectorMode = "workspace" | "config" | "trace";

export interface StudioConfig {
  workspace: string;
  provider: string;
  offline: boolean;
}

export interface CapabilityOption {
  key: string;
  label: string;
}

export interface SettingsResponse {
  provider: string;
  offline: boolean;
  capabilities: string[];
  available: CapabilityOption[];
  model: string;
  effort: string;
  efforts: string[];
  summary: string;
  summaries: string[];
  otel: boolean;
  applied_runs?: number;
}

export type SettingsUpdateResponse = Pick<
  SettingsResponse,
  "capabilities" | "model" | "effort" | "summary" | "otel" | "applied_runs"
>;

export interface Profile {
  id: string;
  name: string;
  description: string;
  instructions: string;
  capabilities: string[];
  model: string;
  effort: string;
  summary: string;
  built_in?: boolean;
}

export interface ProfilesResponse {
  profiles: Profile[];
  default_profile_id: string;
  system_prompt_base: string;
  available_capabilities: CapabilityOption[];
  efforts: string[];
  summaries: string[];
}

export interface ToolSchema {
  type?: string;
  name?: string;
  description?: string;
  parameters?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ProfilePreviewResponse {
  schema_version: "studio.model-request-preview.v1";
  snapshot_kind: "initial_new_chat_turn";
  input_bound: boolean;
  unbound_fields: string[];
  model_request: Record<string, unknown>;
  system_prompt: string;
  tools: ToolSchema[];
  tool_count: number;
  tool_surface: Record<string, unknown>;
  request_config: {
    model: string;
    reasoning: { effort: string; summary: string };
    tool_schema_format: string;
    turn: string;
  };
  notes: string[];
}

export interface SessionSummary {
  run_id: string;
  title: string;
  state: string;
  terminal: boolean;
  created_at: number;
  recoverable: boolean;
  profile_id: string;
  profile_name: string;
}

export interface SessionsResponse {
  sessions: SessionSummary[];
  profile_id: string | null;
}

export interface AttachmentInput {
  name: string;
  mime: string;
  data_b64: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  attachments: Array<{ name: string; mime: string }>;
  created_at: number;
  source?: Record<string, unknown>;
}

export interface ChatTranscriptResponse {
  schema_version: string;
  run_id: string;
  messages: ChatMessage[];
  event_cursor: number;
}

export interface ChatResponse {
  run_id?: string;
  state?: string;
  terminal?: boolean;
  message_id?: string;
  [key: string]: unknown;
}

export interface PauseResponse extends ChatResponse {
  pause_requested?: boolean;
}

export interface ResumeResponse extends ChatResponse {
  resumed: boolean;
  recovery_resumed: boolean;
  turn_resumed: boolean;
  resume_kind: "checkpoint_and_paused_turn" | "checkpoint" | "paused_turn" | "already_live";
}

export interface RetryResponse extends ChatResponse {
  retried?: boolean;
  retry_of_event_seq?: number;
}

export interface RunEvent<T extends Record<string, unknown> = Record<string, unknown>> {
  seq?: number;
  event_id?: string;
  parent_id?: string;
  type: string;
  level?: string;
  timestamp?: string;
  studio_activity?: string;
  data: T;
}

export interface ApprovalRequest {
  taskId: string;
  kind: "hitl" | "tool_approval";
  prompt: string;
  choices: string[];
  tool?: string;
  argumentsPreview?: string;
  requestedAt?: string;
  status: "pending" | "submitting" | "approved" | "denied" | "expired";
}

export interface UsageSummary {
  input: number;
  output: number;
  total: number;
}

export interface PlanItem {
  step: string;
  status: "pending" | "in_progress" | "completed";
}

export interface ProposalFile {
  path: string;
  status?: string;
  size?: number;
}

export interface ProposalResponse {
  ready?: boolean;
  proposal_hash: string;
  diff?: string;
  changed_paths?: string[];
  files?: ProposalFile[];
  [key: string]: unknown;
}

export interface ApplyResponse {
  status?: string;
  applied_paths?: string[];
  skipped_paths?: string[];
  conflicts?: string[];
  [key: string]: unknown;
}

export interface PackageReceipt {
  digest: string;
  proposal_hash?: string;
  name?: string;
  size?: number;
}

export interface WorkspaceFile {
  path: string;
  is_dir?: boolean;
  type?: "file" | "directory";
  size?: number;
  children?: WorkspaceFile[];
  [key: string]: unknown;
}

export interface FilesResponse {
  workspace: string;
  files: WorkspaceFile[];
}

export interface FilePreviewResponse {
  path: string;
  binary: boolean;
  image: boolean;
  mime?: string;
  truncated: boolean;
  content: string;
}

export interface JobSummary {
  job_id: string;
  status?: string;
  command?: string;
  [key: string]: unknown;
}

export interface JobsResponse {
  jobs: JobSummary[];
}

export interface RunViewState {
  runId: string | null;
  status: RunStatus;
  messages: ChatMessage[];
  activeResponse: string;
  reasoning: string;
  events: RunEvent[];
  lastSeq: number;
  replayCursor: number;
  approvals: ApprovalRequest[];
  usage: UsageSummary;
  plan: PlanItem[];
  error: string | null;
  errorRetryable: boolean;
  manualRetryCandidate: boolean;
  manualRetryReady: boolean;
  proposalDirty: boolean;
  lastUserMessage: string;
}
