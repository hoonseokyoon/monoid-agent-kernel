<script lang="ts">
  import type { RunStatus } from "../lib/types";

  let { status, compact = false } = $props<{ status: RunStatus; compact?: boolean }>();

  const labels: Record<RunStatus, string> = {
    idle: "Ready",
    queued: "Queued",
    running: "Running",
    "awaiting-approval": "Approval needed",
    stopping: "Stopping",
    stopped: "Stopped",
    failed: "Failed",
    succeeded: "Complete",
  };
  const label = $derived(labels[status as RunStatus]);
  const compactLabels: Record<RunStatus, string> = {
    idle: "Ready",
    queued: "Queue",
    running: "Run",
    "awaiting-approval": "Approve",
    stopping: "Pause…",
    stopped: "Paused",
    failed: "Failed",
    succeeded: "Done",
  };
  const compactLabel = $derived(compactLabels[status as RunStatus]);
</script>

<span class:compact class="status-badge status-{status}" title={label}>
  <span class:animate-pulse={status === "running" || status === "queued"} class="status-dot"></span>
  <span>{compact ? compactLabel : label}</span>
</span>
