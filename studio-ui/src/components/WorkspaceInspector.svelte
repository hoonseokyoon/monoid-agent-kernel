<script lang="ts">
  import { studioApi } from "../lib/api";
  import type { FilePreviewResponse, JobSummary, PlanItem, WorkspaceFile } from "../lib/types";
  import Icon from "./Icon.svelte";

  let { workspaceName, files, jobs, plan, onRefresh } = $props<{
    workspaceName: string;
    files: WorkspaceFile[];
    jobs: JobSummary[];
    plan: PlanItem[];
    onRefresh: () => Promise<void>;
  }>();

  let query = $state("");
  let preview = $state<FilePreviewResponse | null>(null);
  let loading = $state(false);
  let error = $state("");
  const visibleFiles = $derived(files.filter((file: WorkspaceFile) => file.path.toLowerCase().includes(query.toLowerCase())).slice(0, 80));

  async function openFile(file: WorkspaceFile): Promise<void> {
    if (file.is_dir || file.type === "directory") return;
    loading = true;
    error = "";
    try {
      preview = await studioApi.file(file.path);
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    } finally {
      loading = false;
    }
  }

  async function refresh(): Promise<void> {
    error = "";
    try {
      await onRefresh();
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    }
  }
</script>

<div class="workspace-inspector">
  <header class="inspector-heading"><div><div class="eyebrow">Current context</div><h2>Workspace</h2></div><button class="icon-button" title="Refresh workspace" onclick={() => void refresh()}><Icon name="retry" size={14} /></button></header>
  <div class="workspace-card"><span class="workspace-icon"><Icon name="files" size={16} /></span><div><strong>{workspaceName || "Workspace"}</strong><small>Local · propose mode</small></div><span class="healthy"><i></i>Ready</span></div>

  {#if plan.length}
    <section class="inspector-section">
      <div class="inspector-section-title"><span>Active plan</span><small>{plan.filter((item: PlanItem) => item.status === "completed").length}/{plan.length}</small></div>
      <div class="mini-plan">
        {#each plan as item}
          <div class:active={item.status === "in_progress"} class:complete={item.status === "completed"}><span>{#if item.status === "completed"}<Icon name="check" size={10} />{:else}<i></i>{/if}</span><strong>{item.step}</strong></div>
        {/each}
      </div>
    </section>
  {/if}

  {#if jobs.length}
    <section class="inspector-section">
      <div class="inspector-section-title"><span>Background jobs</span><small>{jobs.length}</small></div>
      <div class="mini-jobs">
        {#each jobs.slice(0, 8) as job}
          <div><span class:running={job.status === "running"}></span><strong>{job.command || job.job_id}</strong><small>{job.status || "unknown"}</small></div>
        {/each}
      </div>
    </section>
  {/if}

  <section class="inspector-section file-section">
    <div class="inspector-section-title"><span>Files</span><small>{files.length}</small></div>
    <label class="file-search"><Icon name="search" size={13} /><input bind:value={query} aria-label="Filter workspace files" placeholder="Filter files" /></label>
    {#if loading}
      <div class="file-preview loading" role="status">Loading file preview…</div>
    {:else if error}
      <div class="file-preview error" role="alert">{error}</div>
    {:else if preview}
      <section class="file-preview" aria-labelledby="file-preview-title">
        <header><strong id="file-preview-title">{preview.path}</strong><button class="icon-button" aria-label="Close file preview" onclick={() => (preview = null)}><Icon name="x" size={13} /></button></header>
        {#if preview.image}
          <img src={studioApi.fileRawUrl(preview.path)} alt={`Preview of ${preview.path}`} />
        {:else if preview.binary}
          <p>Binary preview is unavailable.</p>
        {:else}
          <pre>{preview.content}</pre>
          {#if preview.truncated}<small>Preview truncated at the Studio read limit.</small>{/if}
        {/if}
      </section>
    {/if}
    <div class="mini-file-tree">
      {#if visibleFiles.length}
        {#each visibleFiles as file}
          <button disabled={file.is_dir || file.type === "directory"} aria-current={preview?.path === file.path ? "true" : undefined} onclick={() => void openFile(file)}><Icon name={file.is_dir || file.type === "directory" ? "files" : "file"} size={13} /><span>{file.path}</span></button>
        {/each}
      {:else}
        <div class="inspector-empty">No files match this filter.</div>
      {/if}
    </div>
  </section>
</div>
