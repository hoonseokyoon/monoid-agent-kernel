<script lang="ts">
  import type { RunEvent, SubagentActivity } from "../lib/types";
  import Icon from "./Icon.svelte";
  import RichText from "./RichText.svelte";

  let { activity } = $props<{ activity: SubagentActivity }>();

  const reasoning = $derived(
    activity.events
      .filter((event: RunEvent) => event.type === "model.reasoning.delta")
      .map((event: RunEvent) => String(event.data.text ?? ""))
      .join(""),
  );
  const streamedOutput = $derived(
    activity.events
      .filter((event: RunEvent) => event.type === "model.output.delta")
      .map((event: RunEvent) => String(event.data.text ?? ""))
      .join(""),
  );
  const settledOutput = $derived(String(
    activity.events.filter((event: RunEvent) => event.type === "turn.settled").at(-1)?.data.final_text ?? "",
  ));
  const output = $derived(
    activity.status === "running" ? streamedOutput : settledOutput || streamedOutput,
  );
  const toolCalls = $derived(
    activity.events.filter((event: RunEvent) => event.type === "tool.call.started"),
  );
  const statusLabel = $derived(
    activity.status === "running" ? "Running" : activity.status === "failed" ? "Failed" : "Complete",
  );

  function toolLabel(event: RunEvent): string {
    const data = event.data;
    const preview = typeof data.args_preview === "object" && data.args_preview !== null
      ? data.args_preview as Record<string, unknown>
      : {};
    const target = (Array.isArray(data.paths) ? data.paths[0] : undefined)
      ?? preview.path
      ?? preview.query_preview
      ?? preview.url_preview
      ?? preview.query;
    return `${String(data.tool ?? "tool").replaceAll("_", ".")}${target ? ` · ${String(target)}` : ""}`;
  }
</script>

<section class:error={activity.status === "failed"} class="subagent-activity" aria-label={`${activity.subagentType} subagent ${statusLabel.toLowerCase()}`}>
  <header>
    <span class="subagent-mark"><Icon name="spark" size={13} /></span>
    <div><strong>{activity.subagentType} subagent{activity.depth > 1 ? ` · depth ${activity.depth}` : ""}</strong><small>{activity.childRunId}</small></div>
    <span class="subagent-state" aria-live="polite"><i></i>{statusLabel}</span>
  </header>
  <div class="subagent-body">
    {#if reasoning}<details open={!output}><summary>Reasoning summary</summary><p>{reasoning}</p></details>{/if}
    {#if toolCalls.length}
      <ul aria-label="Subagent tool activity">
        {#each toolCalls as event (event.event_id)}<li><Icon name="terminal" size={11} /><span>{toolLabel(event)}</span></li>{/each}
      </ul>
    {/if}
    {#if output}<div class="subagent-output"><RichText content={output} /></div>{/if}
    {#if activity.liveTraceUnavailable}
      <div class="subagent-trace-warning" role="status"><Icon name="alert" size={13} />The final child trace could not be loaded after three retries.</div>
    {/if}
    {#if activity.status === "running" && !reasoning && !toolCalls.length && !output}
      <div class="subagent-waiting"><span class="spinner"></span>Starting delegated work…</div>
    {/if}
  </div>
</section>
