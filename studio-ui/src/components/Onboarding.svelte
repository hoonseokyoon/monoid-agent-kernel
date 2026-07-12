<script lang="ts">
  import Icon from "./Icon.svelte";

  let { offline, workspaceName, onPrompt, onOpenProfile } = $props<{
    offline: boolean;
    workspaceName: string;
    onPrompt: (prompt: string) => void;
    onOpenProfile: () => void;
  }>();

  const prompts = [
    { icon: "search", title: "Understand this workspace", text: "Map the architecture and identify the highest-risk areas." },
    { icon: "code", title: "Plan a focused change", text: "Review the repository and draft a safe implementation plan." },
    { icon: "shield", title: "Audit before release", text: "Run the release checklist and report anything that needs attention." },
  ];
</script>

<section class="onboarding" data-testid="onboarding">
  <div class="onboarding-hero">
    <div class="hero-orbit"><span><Icon name="spark" size={24} /></span></div>
    <div class="eyebrow">Workspace ready</div>
    <h1>What should we work on?</h1>
    <p>Start with a goal. Studio will keep the run, evidence, and proposed changes in one calm workspace.</p>
    {#if offline}
      <div class="offline-notice"><span></span>Offline preview uses an echo model. Connect a provider for tool-driven work.</div>
    {/if}
  </div>

  <div class="onboarding-grid">
    {#each prompts as prompt}
      <button class="prompt-card" onclick={() => onPrompt(prompt.text)}>
        <span><Icon name={prompt.icon} size={17} /></span>
        <strong>{prompt.title}</strong>
        <small>{prompt.text}</small>
        <i><Icon name="chevron" size={14} /></i>
      </button>
    {/each}
  </div>

  <div class="setup-checklist">
    <div class="setup-heading">
      <div>
        <div class="eyebrow">First-run checklist</div>
        <h2>Ready in three steps</h2>
      </div>
      <span>2 of 3 complete</span>
    </div>
    <div class="setup-steps">
      <div class="complete"><b><Icon name="check" size={13} /></b><span><strong>Workspace opened</strong><small>{workspaceName || "Local workspace"}</small></span></div>
      <button class="complete" onclick={onOpenProfile}><b><Icon name="check" size={13} /></b><span><strong>Agent profile selected</strong><small>Default · balanced reasoning</small></span><Icon name="chevron" size={14} /></button>
      <button onclick={onOpenProfile}><b>3</b><span><strong>Review capabilities</strong><small>Confirm what the agent may access</small></span><Icon name="chevron" size={14} /></button>
    </div>
  </div>
</section>
