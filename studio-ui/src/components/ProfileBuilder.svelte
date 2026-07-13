<script lang="ts">
  import { untrack } from "svelte";
  import type { CapabilityOption, Profile, ProfilePreviewResponse } from "../lib/types";
  import Icon from "./Icon.svelte";
  import RequestPreview from "./RequestPreview.svelte";

  let { activeProfile, capabilities, efforts, summaries, onSave, onPreview } = $props<{
    activeProfile: Profile;
    capabilities: CapabilityOption[];
    efforts: string[];
    summaries: string[];
    onSave: (profile: Profile) => Promise<Profile>;
    onPreview: (profile: Profile) => Promise<ProfilePreviewResponse>;
  }>();

  let draft = $state<Profile>(untrack(() => ({ ...activeProfile, capabilities: [...activeProfile.capabilities] })));
  let section = $state<"identity" | "instructions" | "model" | "capabilities">("identity");
  let preview = $state<ProfilePreviewResponse | null>(null);
  let previewLoading = $state(false);
  let previewError = $state("");
  let saving = $state(false);
  let saved = $state(false);

  const sections = [
    { id: "identity", label: "Identity", icon: "profile" },
    { id: "instructions", label: "Instructions", icon: "message" },
    { id: "model", label: "Model & reasoning", icon: "spark" },
    { id: "capabilities", label: "Capabilities", icon: "shield" },
  ] as const;

  const signature = $derived(JSON.stringify(draft));

  $effect(() => {
    signature;
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      previewLoading = true;
      previewError = "";
      try {
        const resolved = await onPreview({ ...draft, capabilities: [...draft.capabilities] });
        if (!cancelled) preview = resolved;
      } catch (error) {
        if (!cancelled) previewError = error instanceof Error ? error.message : String(error);
      } finally {
        if (!cancelled) previewLoading = false;
      }
    }, 320);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  });

  function toggleCapability(key: string): void {
    draft.capabilities = draft.capabilities.includes(key)
      ? draft.capabilities.filter((item) => item !== key)
      : [...draft.capabilities, key];
  }

  function moveTab(event: KeyboardEvent, index: number): void {
    const keys = ["ArrowRight", "ArrowLeft", "Home", "End"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % sections.length;
    if (event.key === "ArrowLeft") next = (index - 1 + sections.length) % sections.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = sections.length - 1;
    section = sections[next].id;
    const tabs = (event.currentTarget as HTMLElement).parentElement?.querySelectorAll<HTMLElement>("[role=tab]");
    tabs?.[next]?.focus();
  }

  async function save(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    saving = true;
    saved = false;
    try {
      draft = { ...(await onSave({ ...draft, capabilities: [...draft.capabilities] })) };
      saved = true;
      window.setTimeout(() => (saved = false), 1600);
    } finally {
      saving = false;
    }
  }
</script>

<div class="profile-builder" data-testid="profile-editor-popup">
  <form class="profile-form" onsubmit={save}>
    <header class="mode-header">
      <div>
        <div class="breadcrumb"><span>Profiles</span><Icon name="chevron" size={12} /><span>{draft.name}</span></div>
        <div class="title-line">
          <h1>Agent Profile Builder</h1>
          {#if draft.built_in}<span class="subtle-badge">Built in</span>{/if}
        </div>
        <p>Shape the defaults that every new session starts from.</p>
      </div>
      <div class="header-actions">
        <span class="save-state" aria-live="polite">{saved ? "Saved" : "Draft autosaved locally"}</span>
        <button class="primary-button" type="submit" disabled={saving}>
          {#if saving}<span class="spinner"></span>{:else}<Icon name="check" size={14} />{/if}
          Save profile
        </button>
      </div>
    </header>

    <div class="profile-body">
      <div class="profile-section-tabs" role="tablist" aria-label="Profile sections">
        {#each sections as item, index}
          <button
            type="button"
            id={`profile-tab-${item.id}`}
            role="tab"
            aria-selected={section === item.id}
            aria-controls="profile-panel"
            tabindex={section === item.id ? 0 : -1}
            class:active={section === item.id}
            onclick={() => (section = item.id)}
            onkeydown={(event) => moveTab(event, index)}
          >
            <Icon name={item.icon} size={15} />
            <span>{item.label}</span>
            {#if item.id === "capabilities"}<small>{draft.capabilities.length}</small>{/if}
          </button>
        {/each}
      </div>

      <div class="profile-fields">
        {#if section === "identity"}
          <div class="profile-panel" id="profile-panel" role="tabpanel" aria-labelledby="profile-tab-identity">
            <div class="section-heading">
              <div class="section-icon"><Icon name="profile" size={17} /></div>
              <div><h2>Identity</h2><p>Give this profile a clear role in the workspace.</p></div>
            </div>
            <div class="field-grid two-columns">
              <label><span>Profile name</span><input bind:value={draft.name} autocomplete="off" /></label>
              <label><span>Profile ID</span><input value={draft.id} disabled aria-describedby="profile-id-note" /></label>
            </div>
            <label class="field"><span>Description</span><input bind:value={draft.description} placeholder="When should someone choose this profile?" /></label>
            <p id="profile-id-note" class="field-note">The stable ID keeps sessions grouped after a restart.</p>
            <div class="identity-preview">
              <span class="avatar large"><Icon name="spark" size={20} /></span>
              <div><strong>{draft.name || "Untitled profile"}</strong><p>{draft.description || "Add a concise purpose for this profile."}</p></div>
              <span class="subtle-badge">{draft.model}</span>
            </div>
          </div>
        {:else if section === "instructions"}
          <div class="profile-panel" id="profile-panel" role="tabpanel" aria-labelledby="profile-tab-instructions">
            <div class="section-heading">
              <div class="section-icon"><Icon name="message" size={17} /></div>
              <div><h2>Instructions</h2><p>Describe behavior, priorities, and output expectations.</p></div>
            </div>
            <label class="field">
              <span>Profile instructions <em>Appended to the Studio system prompt</em></span>
              <textarea bind:value={draft.instructions} rows="14" placeholder="You are a careful implementation agent…"></textarea>
            </label>
            <div class="instruction-tip"><Icon name="spark" size={14} /><span>Use direct, testable instructions. Runtime context and available tools are resolved by the backend.</span></div>
          </div>
        {:else if section === "model"}
          <div class="profile-panel" id="profile-panel" role="tabpanel" aria-labelledby="profile-tab-model">
            <div class="section-heading">
              <div class="section-icon"><Icon name="spark" size={17} /></div>
              <div><h2>Model & reasoning</h2><p>Choose a profile default. Live Config can update the runtime defaults and active sessions.</p></div>
            </div>
            <label class="field"><span>Model</span><input bind:value={draft.model} spellcheck="false" /></label>
            <div class="field-grid two-columns">
              <label><span>Reasoning effort</span><select bind:value={draft.effort}>{#each efforts as effort}<option value={effort}>{effort}</option>{/each}</select></label>
              <label><span>Reasoning summary</span><select bind:value={draft.summary}>{#each summaries as summary}<option value={summary}>{summary}</option>{/each}</select></label>
            </div>
            <div class="setting-summary">
              <div><span>Effective on</span><strong>New chats</strong></div>
              <div><span>Live override</span><strong>Allowed</strong></div>
              <div><span>Preview source</span><strong>Backend resolved</strong></div>
            </div>
          </div>
        {:else}
          <div class="profile-panel" id="profile-panel" role="tabpanel" aria-labelledby="profile-tab-capabilities">
            <div class="section-heading">
              <div class="section-icon"><Icon name="shield" size={17} /></div>
              <div><h2>Capabilities</h2><p>Control which tool schemas appear in the effective ModelRequest.</p></div>
            </div>
            <div class="capability-list" data-testid="capability-toggles">
              {#each capabilities as capability}
                <label class="capability-row" class:enabled={draft.capabilities.includes(capability.key)}>
                  <span class="capability-glyph"><Icon name={capability.key === "shell" ? "terminal" : capability.key === "write" ? "code" : "shield"} size={15} /></span>
                  <span class="capability-copy"><strong>{capability.label}</strong><small>{capability.key} · inherited by new sessions</small></span>
                  <input
                    data-testid={`capability-toggle-${capability.key}`}
                    type="checkbox"
                    checked={draft.capabilities.includes(capability.key)}
                    onchange={() => toggleCapability(capability.key)}
                  />
                  <span class="switch" aria-hidden="true"><i></i></span>
                </label>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    </div>
  </form>

  <RequestPreview {preview} loading={previewLoading} error={previewError} />
</div>
