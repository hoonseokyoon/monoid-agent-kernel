<script lang="ts">
  import type { ApprovalRequest } from "../lib/types";
  import Icon from "./Icon.svelte";

  let { request, onAnswer } = $props<{
    request: ApprovalRequest;
    onAnswer: (answer: string) => Promise<void>;
  }>();

  let submitting = $state(false);
  let failure = $state("");
  let customAnswer = $state("");

  async function answer(choice: string): Promise<void> {
    submitting = true;
    failure = "";
    try {
      await onAnswer(choice);
    } catch (error) {
      failure = error instanceof Error ? error.message : String(error);
    } finally {
      submitting = false;
    }
  }

  const destructive = $derived(/deny|delete|remove|overwrite|reject/i.test(request.prompt));
</script>

<article class="approval-card" data-testid="approval-card" aria-busy={submitting || request.status === "submitting"}>
  <header>
    <span class="approval-icon" class:danger={destructive}><Icon name="shield" size={17} /></span>
    <div>
      <div class="eyebrow">Human approval</div>
      <h3>{request.kind === "tool_approval" ? "Tool action is waiting" : "Your decision is required"}</h3>
    </div>
    <span class="pending-label"><span></span>Waiting</span>
  </header>

  <p class="approval-prompt">{request.prompt}</p>

  {#if request.tool || request.argumentsPreview}
    <div class="approval-scope">
      {#if request.tool}
        <div><span>Tool</span><code>{request.tool}</code></div>
      {/if}
      {#if request.argumentsPreview}
        <div><span>Scope</span><code>{request.argumentsPreview}</code></div>
      {/if}
    </div>
  {/if}

  <div class="approval-note">
    <Icon name="clock" size={14} />
    <span>The run is waiting. Your decision will be recorded in the trace.</span>
  </div>

  {#if failure}<p class="inline-error" role="alert">{failure}</p>{/if}

  {#if request.choices.length}
    <footer>
      {#each request.choices as choice, index}
        <button
          class:primary={index === 0 && !/deny|reject|cancel/i.test(choice)}
          class:danger-button={/deny|reject/i.test(choice)}
          disabled={submitting || request.status !== "pending"}
          onclick={() => answer(choice)}
        >
          {#if submitting && index === 0}<span class="spinner"></span>{/if}
          {choice}
        </button>
      {/each}
    </footer>
  {:else}
    <form class="approval-answer" onsubmit={(event) => { event.preventDefault(); if (customAnswer.trim()) void answer(customAnswer.trim()); }}>
      <label for={`approval-answer-${request.taskId}`}>Your response</label>
      <textarea id={`approval-answer-${request.taskId}`} bind:value={customAnswer} rows="3" placeholder="Type the information the agent requested"></textarea>
      <button class="primary-button" type="submit" disabled={!customAnswer.trim() || submitting || request.status !== "pending"}>{#if submitting}<span class="spinner"></span>{/if}Submit response</button>
    </form>
  {/if}
</article>
