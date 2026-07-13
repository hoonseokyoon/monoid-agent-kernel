<script lang="ts">
  import type { RunEvent } from "../lib/types";
  import Icon from "./Icon.svelte";

  let { events } = $props<{ events: RunEvent[] }>();
  let selectedId = $state<string | null>(null);
  const selected = $derived(events.find((event: RunEvent) => event.event_id === selectedId) ?? events.at(-1) ?? null);

  function label(event: RunEvent): string {
    const data = event.data;
    const scope = data.studio_scope === "subagent" ? `Subagent · ${String(data.subagent_type ?? "delegate")} · ` : "";
    if (event.type === "tool.call.started") return `${scope}Tool · ${String(data.tool ?? "call")}`;
    if (event.type.startsWith("subagent.")) return `Subagent · ${String(data.subagent_type ?? data.status ?? "task")}`;
    if (event.type === "plan.updated") return `${scope}Plan · ${Array.isArray(data.items) ? data.items.length : 0} steps`;
    return scope + event.type.replaceAll(".", " · ");
  }

  function exportTrace(): void {
    const blob = new Blob([JSON.stringify({ schema_version: "studio.trace-export.v1", events }, null, 2)], {
      type: "application/json",
    });
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = "monoid-studio-trace.json";
    anchor.click();
    URL.revokeObjectURL(href);
  }
</script>

<section class="trace-workbench">
  <header class="mode-header">
    <div><div class="breadcrumb"><span>Session</span><Icon name="chevron" size={12} /><span>Trace</span></div><div class="title-line"><h1>Trace & Tasks</h1><span class="subtle-badge">{events.length} events</span></div><p>Inspect the operational story behind the current run.</p></div>
    <div class="header-actions"><button class="secondary-button" disabled={!events.length} onclick={exportTrace}><Icon name="download" size={14} />Export trace</button></div>
  </header>

  <div class="trace-layout">
    <div class="trace-list" aria-label="Run events">
      <div class="trace-list-head"><span>Timeline</span><small>Live</small></div>
      {#if events.length === 0}
        <div class="trace-empty"><Icon name="trace" size={21} /><strong>No trace yet</strong><span>Events appear here when a run starts.</span></div>
      {:else}
        {#each events as event}
          <button
            aria-current={selected?.event_id === event.event_id ? "true" : undefined}
            class:active={selected?.event_id === event.event_id}
            class:error={event.level === "error" || event.data.ok === false}
            onclick={() => (selectedId = event.event_id ?? null)}
          >
            <span class="trace-node"><i></i></span>
            <span><strong>{label(event)}</strong><small>seq {event.seq ?? "—"} · {event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "live"}</small></span>
            <Icon name="chevron" size={13} />
          </button>
        {/each}
      {/if}
    </div>

    <aside class="trace-detail">
      {#if selected}
        <header><div><div class="eyebrow">Selected span</div><h2>{label(selected)}</h2></div><span class:error={selected.level === "error"}>{selected.level ?? "info"}</span></header>
        <dl>
          <div><dt>Event ID</dt><dd>{selected.event_id ?? "—"}</dd></div>
          <div><dt>Parent</dt><dd>{selected.parent_id ?? "Root"}</dd></div>
          <div><dt>Sequence</dt><dd>{selected.seq ?? "—"}</dd></div>
          <div><dt>Timestamp</dt><dd>{selected.timestamp ?? "—"}</dd></div>
        </dl>
        <section><div class="block-label"><span>Attributes</span><em>{Object.keys(selected.data).length} fields</em></div><pre>{JSON.stringify(selected.data, null, 2)}</pre></section>
      {:else}
        <div class="trace-empty"><Icon name="eye" size={21} /><strong>Select an event</strong><span>Attributes and relationships appear here.</span></div>
      {/if}
    </aside>
  </div>
</section>
