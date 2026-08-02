<script lang="ts">
  import {
    createCompactTraceExport,
    createRawTraceExport,
    isLegacyModelDeltaEvent,
    summarizeTrace,
  } from "../lib/trace";
  import type { RunEvent } from "../lib/types";
  import Icon from "./Icon.svelte";

  let { events } = $props<{ events: RunEvent[] }>();
  let selectedId = $state<string | null>(null);
  let showTokenDeltas = $state(false);
  const summary = $derived(summarizeTrace(events));
  const visibleEvents = $derived(showTokenDeltas ? events : summary.operationEvents);
  const timelineStatus = $derived(
    showTokenDeltas
      ? `${visibleEvents.length.toLocaleString()} events shown`
      : summary.omittedDeltaEventCount
        ? `${summary.omittedDeltaEventCount.toLocaleString()} deltas hidden`
        : `${visibleEvents.length.toLocaleString()} operations shown`,
  );
  const selected = $derived(
    visibleEvents.find((event: RunEvent) => event.event_id === selectedId)
      ?? visibleEvents.at(-1)
      ?? null,
  );

  function label(event: RunEvent): string {
    const data = event.data;
    const scope = data.studio_scope === "subagent" ? `Subagent · ${String(data.subagent_type ?? "delegate")} · ` : "";
    if (event.type === "tool.call.started") return `${scope}Tool · ${String(data.tool ?? "call")}`;
    if (event.type.startsWith("subagent.")) return `Subagent · ${String(data.subagent_type ?? data.status ?? "task")}`;
    if (event.type === "plan.updated") {
      const shown = Array.isArray(data.items) ? data.items.length : 0;
      const omitted = typeof data.truncated_items === "number"
        && Number.isSafeInteger(data.truncated_items)
        && data.truncated_items >= 0
        ? data.truncated_items
        : 0;
      return `${scope}Plan · ${shown + omitted} steps${omitted ? ` · ${omitted} omitted` : ""}`;
    }
    return scope + event.type.replaceAll(".", " · ");
  }

  function downloadTrace(payload: unknown, filename: string): void {
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(href);
  }

  function exportCompactTrace(): void {
    downloadTrace(createCompactTraceExport(events), "monoid-studio-trace-compact.json");
  }

  function exportRawTrace(): void {
    downloadTrace(createRawTraceExport(events), "monoid-studio-trace.json");
  }

  function toggleTokenDeltas(): void {
    const next = !showTokenDeltas;
    if (!next && selected && isLegacyModelDeltaEvent(selected)) selectedId = null;
    showTokenDeltas = next;
  }
</script>

<section class="trace-workbench">
  <header class="mode-header">
    <div><div class="breadcrumb"><span>Session</span><Icon name="chevron" size={12} /><span>Trace</span></div><div class="title-line"><h1>Trace & Tasks</h1><span class="subtle-badge">{summary.operationEvents.length.toLocaleString()} operation{summary.operationEvents.length === 1 ? "" : "s"}</span></div><p>Inspect the operational story behind the current run.</p></div>
    <div class="header-actions">
      {#if summary.omittedDeltaEventCount}
        <button
          type="button"
          class="secondary-button trace-delta-toggle"
          aria-pressed={showTokenDeltas}
          onclick={toggleTokenDeltas}
        ><Icon name="eye" size={14} />{showTokenDeltas ? "Hide token deltas" : `Show ${summary.omittedDeltaEventCount.toLocaleString()} token deltas`}</button>
      {/if}
      <button type="button" class="secondary-button" disabled={!events.length} onclick={exportCompactTrace}><Icon name="download" size={14} />Export compact</button>
      <button type="button" class="secondary-button" disabled={!events.length} onclick={exportRawTrace}><Icon name="download" size={14} />Export raw</button>
    </div>
  </header>

  <div class="trace-layout">
    <div class="trace-list" aria-label="Run trace events">
      <div class="trace-list-head"><span>Timeline</span><small>{timelineStatus}</small></div>
      {#if events.length === 0}
        <div class="trace-empty"><Icon name="trace" size={21} /><strong>No trace yet</strong><span>Events appear here when a run starts.</span></div>
      {:else if visibleEvents.length === 0}
        <div class="trace-empty"><Icon name="trace" size={21} /><strong>No operation events</strong><span>{summary.omittedDeltaEventCount.toLocaleString()} token delta events are hidden. Use the toggle above to inspect them.</span></div>
      {:else}
        {#each visibleEvents as event}
          <button
            type="button"
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
