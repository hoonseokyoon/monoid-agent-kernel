<script lang="ts">
  import type { ProfilePreviewResponse } from "../lib/types";
  import Icon from "./Icon.svelte";

  let { preview, loading = false, error = "" } = $props<{
    preview: ProfilePreviewResponse | null;
    loading?: boolean;
    error?: string;
  }>();

  let tab = $state<"formatted" | "raw" | "diff">("formatted");
  let copied = $state(false);
  const tabs = ["formatted", "raw", "diff"] as const;
  const rawPayload = $derived(preview?.model_request ?? null);

  async function copyPayload(): Promise<void> {
    if (!rawPayload) return;
    await navigator.clipboard.writeText(JSON.stringify(rawPayload, null, 2));
    copied = true;
    window.setTimeout(() => (copied = false), 1200);
  }

  function moveTab(event: KeyboardEvent, index: number): void {
    const keys = ["ArrowRight", "ArrowLeft", "Home", "End"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    tab = tabs[next];
    const items = (event.currentTarget as HTMLElement).parentElement?.querySelectorAll<HTMLElement>("[role=tab]");
    items?.[next]?.focus();
  }
</script>

<aside class="request-preview" data-testid="profile-request-preview" aria-busy={loading}>
  <header class="preview-header">
    <div>
      <div class="eyebrow">Resolved by backend</div>
      <h2>ModelRequest preview</h2>
    </div>
    <span class="snapshot-pill"><span></span>Draft</span>
  </header>

  <div class="preview-meta">
    <div><span>Turn</span><strong>{preview?.request_config.turn ?? "Initial new-chat turn"}</strong></div>
    <div><span>Tools</span><strong>{preview?.tool_count ?? 0} bound</strong></div>
  </div>

  <div class="preview-tabbar">
    <div class="preview-tabs" role="tablist" aria-label="Request preview format">
      {#each tabs as item, index}
        <button
          id={`request-preview-tab-${item}`}
          role="tab"
          aria-selected={tab === item}
          aria-controls="request-preview-panel"
          tabindex={tab === item ? 0 : -1}
          class:active={tab === item}
          onclick={() => (tab = item)}
          onkeydown={(event) => moveTab(event, index)}
        >{item === "formatted" ? "Formatted" : item === "raw" ? "Raw JSON" : "Changes"}</button>
      {/each}
    </div>
    <button class="copy-action" aria-label={copied ? "Request copied" : "Copy exact request JSON"} onclick={copyPayload}><Icon name={copied ? "check" : "copy"} size={14} /></button>
  </div>

  {#if loading}
    <div class="preview-loading" role="status" aria-label="Resolving ModelRequest preview"><span></span><span></span><span></span></div>
  {:else if error}
    <div class="preview-error" role="alert"><Icon name="alert" size={16} /><span>{error}</span></div>
  {:else if preview}
    <div class="preview-content" id="request-preview-panel" role="tabpanel" aria-labelledby={`request-preview-tab-${tab}`}>
      {#if tab === "formatted"}
        <section>
          <div class="block-label"><span>Model</span><em>Profile</em></div>
          <code class="model-value">{preview.request_config.model}</code>
          <div class="setting-pairs">
            <span>Reasoning <b>{preview.request_config.reasoning.effort}</b></span>
            <span>Summary <b>{preview.request_config.reasoning.summary}</b></span>
          </div>
        </section>
        <section>
          <div class="block-label"><span>System prompt</span><em>Profile + runtime</em></div>
          <pre>{preview.system_prompt}</pre>
        </section>
        <section>
          <div class="block-label"><span>Tool schemas</span><em>{preview.tool_count} exact</em></div>
          <div class="tool-list">
            {#each preview.tools.slice(0, 8) as tool}
              <div><Icon name="terminal" size={13} /><code>{String(tool.name ?? tool.type ?? "tool")}</code></div>
            {/each}
            {#if preview.tools.length > 8}<small>+{preview.tools.length - 8} more schemas</small>{/if}
          </div>
        </section>
        {#if preview.unbound_fields.length}
          <section class="preview-boundary-note">
            <div class="block-label"><span>Unbound input</span><em>Awaiting composer</em></div>
            <p>{preview.unbound_fields.join(", ")} will be populated by the first user message.</p>
          </section>
        {/if}
      {:else if tab === "raw"}
        <pre class="raw-json">{JSON.stringify(rawPayload, null, 2)}</pre>
      {:else}
        <div class="preview-diff-empty">
          <span><Icon name="code" size={18} /></span>
          <strong>No sent snapshot yet</strong>
          <p>After the first request, this view compares the immutable sent snapshot with the current draft.</p>
        </div>
      {/if}
    </div>
  {:else}
    <div class="preview-empty">Edit the profile to resolve its first-turn request.</div>
  {/if}

  <footer class="preview-footnote">
    <Icon name="shield" size={14} />
    <span>Secrets stay server-side. Tool schemas reflect the effective capability policy.</span>
  </footer>
</aside>
