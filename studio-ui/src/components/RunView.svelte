<script lang="ts">
  import type { ApprovalRequest, RunViewState, SubagentActivity } from "../lib/types";
  import { MODEL_USAGE_COUNTERS, type LiveModelCallState } from "../lib/model-stream";
  import ApprovalCard from "./ApprovalCard.svelte";
  import ErrorPanel from "./ErrorPanel.svelte";
  import Icon from "./Icon.svelte";
  import RichText from "./RichText.svelte";
  import StatusBadge from "./StatusBadge.svelte";
  import SubagentActivityCard from "./SubagentActivityCard.svelte";

  let { state, subagents, modelCalls, connected, onStop, onPause, onResume, onRetry, onApproval, onOpenConfig } = $props<{
    state: RunViewState;
    subagents: SubagentActivity[];
    modelCalls: LiveModelCallState[];
    connected: boolean;
    onStop: () => Promise<void>;
    onPause: () => Promise<void>;
    onResume: () => Promise<void>;
    onRetry: () => Promise<void>;
    onApproval: (request: ApprovalRequest, answer: string) => Promise<void>;
    onOpenConfig: () => void;
  }>();

  const activeApproval = $derived(state.approvals.find((item: ApprovalRequest) => item.status === "pending" || item.status === "submitting"));
  const elapsedLabel = $derived(state.status === "running" ? "Running now" : state.status === "stopped" ? "Checkpoint saved" : "Session open");
  const runAnnouncement = $derived(
    activeApproval
      ? `Approval required${activeApproval.tool ? ` for ${activeApproval.tool}` : ""}.`
      : state.status === "succeeded"
        ? "Run complete."
        : state.status === "failed"
          ? "Run failed."
          : state.status === "stopped"
            ? "Run paused at a checkpoint."
            : "",
  );

  function counterLabel(counter: string): string {
    return counter[0].toUpperCase() + counter.slice(1).replaceAll("_", " ");
  }
</script>

<section class="run-view">
  <header class="mode-header run-header">
    <div>
      <div class="breadcrumb"><span>Sessions</span><Icon name="chevron" size={12} /><span>{state.runId ? state.runId.slice(0, 8) : "New session"}</span></div>
      <div class="title-line"><h1>{state.messages[0]?.content.slice(0, 54) || "New agent session"}</h1><StatusBadge status={state.status} /></div>
      <p><span class:connected class="connection-dot"></span>{connected ? "Live event stream" : "Reconnecting"} · {elapsedLabel}</p>
    </div>
    <div class="run-control-bar" aria-label="Run controls">
      {#if state.status === "running" || state.status === "queued" || state.status === "awaiting-approval"}
        <button class="secondary-button" onclick={() => void onPause()}><Icon name="stop" size={13} />Pause</button>
        <button class="danger-outline" onclick={() => void onStop()}><Icon name="x" size={13} />Stop turn</button>
      {:else if state.status === "stopped"}
        <button class="primary-button" onclick={() => void onResume()}><Icon name="play" size={13} />Resume</button>
      {:else if state.status === "failed" && state.manualRetryReady}
        <button class="primary-button" onclick={() => void onRetry()}><Icon name="retry" size={13} />Retry request</button>
      {/if}
      <button class="icon-button" title="Run settings" onclick={onOpenConfig}><Icon name="settings" size={15} /></button>
    </div>
  </header>

  <div class="sr-only" role="status" aria-live="polite">{runAnnouncement}</div>
  <div class="chat-scroll" data-testid="chat-log" aria-label="Session transcript" aria-busy={state.status === "running" || state.status === "queued"}>
    <div class="conversation">
      {#if state.eventLogError}
        <article class="error-panel" role="alert" data-testid="transcript-event-log-error">
          <span class="error-mark"><Icon name="alert" size={18} /></span>
          <div class="min-w-0 flex-1">
            <div class="flex items-start justify-between gap-4">
              <div>
                <div class="eyebrow">Transcript degraded</div>
                <h3>Transcript is incomplete</h3>
              </div>
              <span class="error-code">EVENT LOG</span>
            </div>
            <p>{state.eventLogError}</p>
          </div>
        </article>
      {/if}

      {#each state.messages as message (message.id)}
        <article class:from-user={message.role === "user"} class:error-message={message.role === "error"} class="message-row">
          <span class="message-avatar">{#if message.role === "user"}<Icon name="profile" size={14} />{:else if message.role === "error"}<Icon name="alert" size={14} />{:else}<Icon name="spark" size={14} />{/if}</span>
          <div class="message-content">
            <header><strong>{message.role === "user" ? "You" : message.role === "error" ? "Studio error" : "Monoid"}</strong><time>{new Date(message.created_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></header>
            <RichText content={message.content} />
            {#if message.attachments?.length}<div class="message-attachments">{#each message.attachments as file}<span><Icon name="paperclip" size={11} />{file.name}</span>{/each}</div>{/if}
          </div>
        </article>
      {/each}

      {#if state.reasoning}
        <details class="reasoning-card" open={!state.activeResponse}>
          <summary><span><Icon name="spark" size={14} />Thinking</span><small>Live reasoning summary</small></summary>
          <p>{state.reasoning}</p>
        </details>
      {/if}

      {#if state.activeResponse}
        <article class="message-row streaming">
          <span class="message-avatar"><Icon name="spark" size={14} /></span>
          <div class="message-content"><header><strong>Monoid</strong><span class="typing-dot"><i></i><i></i><i></i></span></header><RichText content={state.activeResponse} /></div>
        </article>
      {/if}

      {#each subagents as activity (activity.childRunId)}
        <SubagentActivityCard {activity} />
      {/each}

      {#each modelCalls as call (JSON.stringify([call.runId, call.turnId, call.streamId, call.step, call.startedAt]))}
        <details class="model-call-details" data-testid="model-call-details">
          <summary>
            <span>{call.runId === call.rootRunId ? "Root call" : "Subagent call"} · {call.model ?? call.provider ?? "Model"} · step {call.step}</span>
            <small>{call.status.replaceAll("_", " ")}</small>
          </summary>
          <div class="call-facts">
            <p class="call-identity">{call.runId}</p>
            {#if call.termination || call.usageEvidence}<p>Facts received live</p>{/if}
            {#if call.termination}
              <dl>
                <dt>Outcome</dt><dd>{call.termination.outcome.replaceAll("_", " ")}</dd>
                <dt>Requested by</dt><dd>{call.termination.requested_by?.replaceAll("_", " ") ?? "None"}</dd>
                <dt>Read ended by</dt><dd>{call.termination.ended_by.replaceAll("_", " ")}</dd>
              </dl>
            {:else}
              <p>Termination facts unavailable</p>
            {/if}
            {#if call.usageEvidence}
              <table>
                <caption>Usage across {call.usageEvidence.attempts.length} kernel dispatches</caption>
                <thead><tr><th scope="col">Counter</th><th scope="col">Reported</th><th scope="col">Coverage</th></tr></thead>
                <tbody>
                  {#each MODEL_USAGE_COUNTERS as counter (counter)}
                    <tr>
                      <th scope="row">{counterLabel(counter)}</th>
                      <td>{call.usageEvidence.reported[counter]?.toLocaleString() ?? "Unreported"}</td>
                      <td>{call.usageEvidence.completeness[counter].replaceAll("_", " ")}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            {:else}
              <p>Usage evidence unavailable</p>
            {/if}
          </div>
        </details>
      {/each}

      {#if activeApproval}
        <div class="approval-live">
          <ApprovalCard request={activeApproval} onAnswer={(answer) => onApproval(activeApproval, answer)} />
        </div>
      {/if}

      {#if state.error}
        <ErrorPanel message={state.error} retryable={state.manualRetryReady} onRetry={() => void onRetry()} {onOpenConfig} />
      {/if}

      {#if state.status === "queued" && !state.activeResponse}
        <div class="run-pending"><span class="spinner"></span><span>Preparing the effective request…</span></div>
      {/if}
    </div>
  </div>
</section>

<style>
  .model-call-details {
    margin: 0 0 20px 39px;
    border: 1px solid var(--line-strong);
    border-radius: var(--radius-md);
    background: var(--surface);
    color: var(--fg-secondary);
    font-size: 12px;
  }
  .model-call-details summary {
    min-height: 44px;
    padding: 12px;
    cursor: pointer;
    overflow-wrap: anywhere;
  }
  .model-call-details summary small { margin-left: 8px; color: var(--fg-muted); }
  .call-facts { padding: 0 12px 12px; }
  .call-facts p { margin: 0 0 10px; }
  .call-identity { color: var(--fg-muted); overflow-wrap: anywhere; }
  .call-facts dl { display: grid; grid-template-columns: 1fr 2fr; gap: 6px 12px; margin: 0 0 16px; }
  .call-facts dd { margin: 0; overflow-wrap: anywhere; }
  .call-facts table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .call-facts caption { margin-bottom: 8px; text-align: left; color: var(--fg-primary); }
  .call-facts th, .call-facts td { padding: 5px 4px; border-bottom: 1px solid var(--line); text-align: left; }
  .call-facts th { font-weight: 500; }
</style>
