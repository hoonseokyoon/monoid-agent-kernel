<script lang="ts">
  import type { ApprovalRequest, RunViewState, SubagentActivity } from "../lib/types";
  import ApprovalCard from "./ApprovalCard.svelte";
  import ErrorPanel from "./ErrorPanel.svelte";
  import Icon from "./Icon.svelte";
  import RichText from "./RichText.svelte";
  import StatusBadge from "./StatusBadge.svelte";
  import SubagentActivityCard from "./SubagentActivityCard.svelte";

  let { state, subagents, connected, onStop, onPause, onResume, onRetry, onApproval, onOpenConfig } = $props<{
    state: RunViewState;
    subagents: SubagentActivity[];
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
