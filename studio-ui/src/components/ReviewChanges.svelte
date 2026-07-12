<script lang="ts">
  import type { ApplyResponse, PackageReceipt, ProposalResponse } from "../lib/types";
  import { studioApi } from "../lib/api";
  import Icon from "./Icon.svelte";

  let { proposalRunId, proposal, onApply, onExport } = $props<{
    proposalRunId: string | null;
    proposal: ProposalResponse | null;
    onApply: (approvedPaths: string[]) => Promise<ApplyResponse>;
    onExport: () => Promise<PackageReceipt>;
  }>();

  type Decision = "pending" | "approved" | "excluded";
  let selectedPath = $state("");
  let decisions = $state<Record<string, Decision>>({});
  let checked = $state<Record<string, boolean>>({});
  let filter = $state<"all" | Decision>("all");
  let actionStatus = $state("");
  let applying = $state(false);
  let exporting = $state(false);
  let dialogError = $state("");
  let proposalIdentity = $state("");
  let imageIdentity = $state("");
  let imageLoaded = $state(false);
  let imageError = $state("");
  let imageRetry = $state(0);
  let packageDialog: HTMLDialogElement;

  const paths: string[] = $derived(
    proposal?.changed_paths ?? proposal?.files?.map((file: { path: string }) => file.path) ?? [],
  );
  const visiblePaths: string[] = $derived(paths.filter((path: string) => filter === "all" || (decisions[path] ?? "pending") === filter));
  const approvedPaths: string[] = $derived(paths.filter((path: string) => decisions[path] === "approved"));
  const pendingCount = $derived(paths.filter((path: string) => !decisions[path] || decisions[path] === "pending").length);
  const approvedCount = $derived(approvedPaths.length);
  const excludedCount = $derived(paths.filter((path: string) => decisions[path] === "excluded").length);
  const selectedCount = $derived(Object.values(checked).filter(Boolean).length);
  const visibleSelectedCount = $derived(visiblePaths.filter((path: string) => checked[path]).length);
  const selectedSegment = $derived(fileDiff(proposal?.diff ?? "", selectedPath));
  const diffLines = $derived((selectedSegment ?? proposal?.diff ?? "").split("\n"));
  const diffLabel = $derived(selectedSegment ? `Diff for ${selectedPath}` : "Unified diff for all proposed files");
  const selectedImageUrl = $derived(
    proposalRunId && proposal?.proposal_hash && selectedPath && isImagePath(selectedPath)
      ? studioApi.proposalFileRawUrl(
          proposalRunId,
          selectedPath,
          proposal.proposal_hash,
          imageRetry,
        )
      : "",
  );

  $effect(() => {
    const identity = `${proposal?.proposal_hash ?? ""}:${proposal?.diff ?? ""}`;
    if (identity !== proposalIdentity) {
      proposalIdentity = identity;
      decisions = {};
      checked = {};
      selectedPath = paths[0] ?? "";
    } else if (!selectedPath && paths.length) {
      selectedPath = paths[0];
    }
  });

  $effect(() => {
    const identity = `${proposal?.proposal_hash ?? ""}:${proposalRunId ?? ""}:${selectedPath}`;
    if (identity !== imageIdentity) {
      imageIdentity = identity;
      imageLoaded = false;
      imageError = "";
      imageRetry = 0;
    }
  });

  function isImagePath(path: string): boolean {
    return /\.(png|jpe?g|gif|webp|bmp|ico|svg)$/i.test(path);
  }

  function retryImage(): void {
    imageLoaded = false;
    imageError = "";
    imageRetry += 1;
  }

  function fileDiff(diff: string, path: string): string | null {
    if (!diff || !path) return null;
    const normalized = path.replaceAll("\\", "/");
    const sections = diff.split(/(?=^diff --git )/m).filter(Boolean);
    return sections.find((section) => {
      const firstLine = section.split("\n", 1)[0] ?? "";
      return firstLine.includes(` a/${normalized} b/${normalized}`)
        || section.includes(`\n+++ b/${normalized}\n`)
        || section.includes(`\n--- a/${normalized}\n`);
    }) ?? null;
  }

  function setDecision(path: string, decision: Decision): void {
    decisions = { ...decisions, [path]: decision };
  }

  function applyBatch(decision: Decision): void {
    const targets = paths.filter((path: string) => checked[path]);
    const next = { ...decisions };
    for (const path of targets) next[path] = decision;
    decisions = next;
    checked = {};
  }

  function toggleAll(): void {
    const allChecked = visiblePaths.length > 0 && visiblePaths.every((path: string) => checked[path]);
    checked = Object.fromEntries(visiblePaths.map((path: string) => [path, !allChecked]));
  }

  async function applyApproved(): Promise<void> {
    if (approvedPaths.length === 0) {
      actionStatus = "Approve at least one file before applying.";
      return;
    }
    applying = true;
    actionStatus = "Applying approved files…";
    try {
      const result = await onApply(approvedPaths);
      actionStatus = result.status === "conflict"
        ? "Workspace conflict detected. Review the changed files again."
        : `Applied ${(result.applied_paths ?? []).length} file(s); skipped ${(result.skipped_paths ?? []).length}.`;
    } catch (error) {
      actionStatus = error instanceof Error ? error.message : String(error);
    } finally {
      applying = false;
    }
  }

  async function exportPackage(): Promise<void> {
    exporting = true;
    dialogError = "";
    actionStatus = "Building portable package…";
    try {
      const receipt = await onExport();
      actionStatus = `Exported ${receipt.name ?? "proposal package"} · ${receipt.digest.slice(0, 12)}`;
      packageDialog.close();
    } catch (error) {
      dialogError = error instanceof Error ? error.message : String(error);
      actionStatus = dialogError;
    } finally {
      exporting = false;
    }
  }

  function lineClass(line: string): string {
    if (line.startsWith("+") && !line.startsWith("+++")) return "add";
    if (line.startsWith("-") && !line.startsWith("---")) return "delete";
    if (line.startsWith("@@") || line.startsWith("diff ") || line.startsWith("index ")) return "meta";
    return "context";
  }
</script>

<section class="review-workbench">
  <header class="mode-header review-header">
    <div>
      <div class="breadcrumb"><span>Session</span><Icon name="chevron" size={12} /><span>Changes</span></div>
      <div class="title-line"><h1>Review Changes</h1><span class="review-count">{paths.length} files</span></div>
      <p>Approve files deliberately, then apply or export a portable review package.</p>
    </div>
    <div class="header-actions">
      <button class="secondary-button" onclick={() => packageDialog.showModal()} disabled={!proposal?.ready}><Icon name="package" size={14} />Export package</button>
      <button class="primary-button" onclick={() => void applyApproved()} disabled={applying || approvedCount === 0}>{#if applying}<span class="spinner"></span>{:else}<Icon name="check" size={14} />{/if}Apply approved ({approvedCount})</button>
    </div>
  </header>

  <div class="review-layout">
    <aside class="review-files">
      <div class="review-summary">
        <div><strong>{pendingCount}</strong><span>Pending</span></div>
        <div><strong>{approvedCount}</strong><span>Approved</span></div>
        <div><strong>{excludedCount}</strong><span>Excluded</span></div>
      </div>
      <div class="filter-tabs" aria-label="File review filters">
        {#each ["all", "pending", "approved", "excluded"] as item}
          <button class:active={filter === item} aria-pressed={filter === item} onclick={() => (filter = item as typeof filter)}>{item}</button>
        {/each}
      </div>
      <div class="batch-bar">
        <button class="checkbox-button" role="checkbox" aria-label="Select all visible files" aria-checked={visibleSelectedCount === 0 ? "false" : visibleSelectedCount === visiblePaths.length ? "true" : "mixed"} onclick={toggleAll}><span><Icon name="check" size={11} /></span></button>
        <span>{selectedCount ? `${selectedCount} selected` : "Select for batch action"}</span>
        {#if selectedCount}<button title="Approve selected files" onclick={() => applyBatch("approved")}><Icon name="check" size={13} /></button><button title="Exclude selected files" onclick={() => applyBatch("excluded")}><Icon name="x" size={13} /></button>{/if}
      </div>
      <fieldset class="file-review-list">
        <legend class="sr-only">Files proposed for review</legend>
        {#if paths.length === 0}
          <div class="review-empty"><Icon name="files" size={21} /><strong>No proposed changes</strong><span>Run the agent in propose mode to review a diff.</span></div>
        {:else}
          {#each visiblePaths as path}
            <div class:active={selectedPath === path} class="file-review-row">
              <label title={`Select ${path} for batch action`}><input type="checkbox" aria-label={`Select ${path} for batch action`} checked={Boolean(checked[path])} onchange={() => (checked = { ...checked, [path]: !checked[path] })} /><span class="checkmark"><Icon name="check" size={10} /></span></label>
              <button class="file-open" onclick={() => (selectedPath = path)} aria-current={selectedPath === path ? "true" : undefined}>
                <Icon name="file" size={14} /><span><strong>{path.split(/[\\/]/).at(-1)}</strong><small>{path}</small></span>
              </button>
              <span class="decision decision-{decisions[path] ?? 'pending'}">{decisions[path] ?? "pending"}</span>
            </div>
          {/each}
        {/if}
      </fieldset>
    </aside>

    <div class="diff-panel">
      <div class="diff-toolbar">
        <div><Icon name="file" size={14} /><strong>{selectedSegment ? selectedPath : "Unified diff"}</strong></div>
        {#if selectedPath}
          <div class="file-decisions">
            <button class:active={decisions[selectedPath] === "excluded"} onclick={() => setDecision(selectedPath, "excluded")}><Icon name="x" size={12} />Exclude</button>
            <button class:active={decisions[selectedPath] === "approved"} onclick={() => setDecision(selectedPath, "approved")}><Icon name="check" size={12} />Approve file</button>
          </div>
        {/if}
      </div>
      {#if selectedImageUrl}
        <div class="proposal-image-preview" role="region" aria-label={`Proposed image preview for ${selectedPath}`}>
          <div class="proposal-image-stage" aria-busy={!imageLoaded && !imageError}>
            {#if !imageLoaded && !imageError}<div class="image-preview-status" role="status"><span class="spinner"></span>Loading proposed image…</div>{/if}
            {#key selectedImageUrl}
              <img
                class:loaded={imageLoaded}
                src={selectedImageUrl}
                alt={`Proposed image preview: ${selectedPath}`}
                aria-hidden={!imageLoaded}
                onload={() => (imageLoaded = true)}
                onerror={() => (imageError = "The proposed image could not be loaded from this revision.")}
              />
            {/key}
            {#if imageError}<div class="image-preview-error" role="alert"><Icon name="alert" size={17} /><span>{imageError}</span><button type="button" onclick={retryImage}>Retry preview</button></div>{/if}
          </div>
          <p><Icon name="shield" size={12} />Previewed from the proposal snapshot before approval.</p>
          {#if selectedSegment}
            <details class="proposal-image-patch">
              <summary>Patch metadata</summary>
              <pre>{selectedSegment}</pre>
            </details>
          {/if}
        </div>
      {:else}
        <!-- svelte-ignore a11y_no_noninteractive_tabindex -- keyboard focus exposes both scroll axes -->
        <div class="diff-code" role="region" aria-label={diffLabel} tabindex="0">
          {#if proposal?.diff}
            {#each diffLines as line, index}
              <div class={lineClass(line)}><span>{index + 1}</span><code>{line || " "}</code></div>
            {/each}
          {:else}
            <div class="review-empty"><Icon name="code" size={21} /><strong>Diff unavailable</strong><span>The proposal may still be preparing.</span></div>
          {/if}
        </div>
      {/if}
      <footer class="review-status" aria-live="polite"><span>{actionStatus || `${pendingCount} files still need a decision.`}</span><span>Approval and apply are recorded separately</span></footer>
    </div>
  </div>
</section>

<dialog bind:this={packageDialog} class="package-dialog" aria-labelledby="package-dialog-title">
  <form method="dialog">
    <header><span class="package-mark"><Icon name="package" size={19} /></span><div><div class="eyebrow">Portable artifact</div><h2 id="package-dialog-title">Export review package</h2></div><button class="icon-button" value="cancel" aria-label="Close package dialog"><Icon name="x" size={15} /></button></header>
    <div class="dialog-body">
      <p>Build a content-addressed TAR from the current proposal. The BFF returns a receipt, then Studio downloads the bytes by digest.</p>
      <div class="package-option selected"><span><Icon name="package" size={16} /></span><div><strong>Portable proposal bundle</strong><small>Manifest, proposed files, and integrity metadata</small></div><Icon name="check" size={15} /></div>
      <dl class="package-manifest"><div><dt>Files</dt><dd>{paths.length} proposed</dd></div><div><dt>Format</dt><dd>application/x-tar</dd></div><div><dt>Transfer</dt><dd>Digest-addressed</dd></div></dl>
      <div class="security-note"><Icon name="shield" size={14} /><span>No run-directory path crosses the browser boundary.</span></div>
      {#if dialogError}<div class="dialog-error" role="alert"><Icon name="alert" size={14} /><span>{dialogError}</span></div>{/if}
    </div>
    <footer><button class="secondary-button" value="cancel">Cancel</button><button type="button" class="primary-button" disabled={exporting} onclick={() => void exportPackage()}>{#if exporting}<span class="spinner"></span>{:else}<Icon name="download" size={14} />{/if}Create & download</button></footer>
  </form>
</dialog>
