<script lang="ts">
  import { untrack } from "svelte";
  import type { SettingsResponse } from "../lib/types";
  import Icon from "./Icon.svelte";

  let { settings, onApply } = $props<{
    settings: SettingsResponse;
    onApply: (settings: Partial<SettingsResponse>) => Promise<void>;
  }>();

  let model = $state(untrack(() => settings.model));
  let effort = $state(untrack(() => settings.effort));
  let summary = $state(untrack(() => settings.summary));
  let otel = $state(untrack(() => settings.otel));
  let capabilities = $state(untrack(() => [...settings.capabilities]));
  let saving = $state(false);
  let notice = $state("");

  const dirty = $derived(
    model !== settings.model ||
      effort !== settings.effort ||
      summary !== settings.summary ||
      otel !== settings.otel ||
      capabilities.join("|") !== settings.capabilities.join("|"),
  );

  function toggleCapability(key: string): void {
    capabilities = capabilities.includes(key)
      ? capabilities.filter((item) => item !== key)
      : [...capabilities, key];
  }

  async function apply(): Promise<void> {
    saving = true;
    notice = "";
    try {
      await onApply({ model, effort, summary, otel, capabilities });
      notice = "Runtime defaults applied";
    } catch (error) {
      notice = error instanceof Error ? error.message : String(error);
    } finally {
      saving = false;
    }
  }
</script>

<div class="live-config" data-testid="settings-config-popup">
  <div data-testid="settings-popup">
    <header class="inspector-heading">
      <div><div class="eyebrow">Runtime defaults</div><h2>Live Config</h2></div>
      {#if dirty}<span class="modified-pill">Modified</span>{/if}
    </header>
    <p class="inspector-intro">Changes become the defaults and update active sessions at their next model turn.</p>

    <section class="config-section">
      <div class="config-section-title"><span><Icon name="spark" size={14} />Model</span><small>Runtime default</small></div>
      <label><span>Model</span><input bind:value={model} spellcheck="false" /></label>
      <div class="field-grid two-columns compact">
        <label><span>Effort</span><select bind:value={effort}>{#each settings.efforts as item}<option value={item}>{item}</option>{/each}</select></label>
        <label><span>Summary</span><select bind:value={summary}>{#each settings.summaries as item}<option value={item}>{item}</option>{/each}</select></label>
      </div>
    </section>

    <section class="config-section">
      <div class="config-section-title"><span><Icon name="shield" size={14} />Capabilities</span><small>{capabilities.length} enabled</small></div>
      <div class="compact-capabilities" data-testid="capability-toggles">
        {#each settings.available as item}
          <label>
            <span><strong>{item.label}</strong><small>{capabilities.includes(item.key) ? "Allowed" : "Off"}</small></span>
            <input
              data-testid={`capability-toggle-${item.key}`}
              type="checkbox"
              checked={capabilities.includes(item.key)}
              onchange={() => toggleCapability(item.key)}
            />
            <span class="switch" aria-hidden="true"><i></i></span>
          </label>
        {/each}
      </div>
    </section>

    <section class="config-section">
      <div class="config-section-title"><span><Icon name="trace" size={14} />Observability</span><small>Runtime</small></div>
      <label class="otel-row">
        <span><strong>OpenTelemetry export</strong><small>Send GenAI spans to the configured collector</small></span>
        <input type="checkbox" bind:checked={otel} />
        <span class="switch" aria-hidden="true"><i></i></span>
      </label>
    </section>
  </div>

  <footer class="config-footer">
    <div aria-live="polite">{notice || (dirty ? "Unsaved runtime changes" : "Using effective settings")}</div>
    <button class="primary-button" disabled={!dirty || saving} onclick={apply}>
      {#if saving}<span class="spinner"></span>{:else}<Icon name="check" size={14} />{/if}
      Apply changes
    </button>
  </footer>
</div>
