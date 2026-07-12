<script lang="ts">
  import { onMount } from "svelte";

  import Composer from "./components/Composer.svelte";
  import Icon from "./components/Icon.svelte";
  import LiveConfig from "./components/LiveConfig.svelte";
  import Onboarding from "./components/Onboarding.svelte";
  import ProfileBuilder from "./components/ProfileBuilder.svelte";
  import ReviewChanges from "./components/ReviewChanges.svelte";
  import RunView from "./components/RunView.svelte";
  import StatusBadge from "./components/StatusBadge.svelte";
  import TraceView from "./components/TraceView.svelte";
  import WorkspaceInspector from "./components/WorkspaceInspector.svelte";
  import {
    RunEventStream,
    clientMessageId,
    downloadProposalPackage,
    studioApi,
  } from "./lib/api";
  import {
    appendOptimisticUserMessage,
    hydrateTranscript,
    initialRunState,
    reduceRunEvent,
  } from "./lib/run-state";
  import type {
    ApplyResponse,
    ApprovalRequest,
    AttachmentInput,
    CapabilityOption,
    PackageReceipt,
    Profile,
    ProposalResponse,
    RunEvent,
    SessionSummary,
    SettingsResponse,
    StudioConfig,
    StudioMode,
    InspectorMode,
    JobSummary,
    WorkspaceFile,
  } from "./lib/types";

  const fallbackSettings: SettingsResponse = {
    provider: "offline",
    offline: true,
    capabilities: ["read", "write", "hitl", "shell", "artifact"],
    available: [
      { key: "read", label: "Read files" },
      { key: "write", label: "Write files (staged as a proposal)" },
      { key: "hitl", label: "Ask the human for approval" },
      { key: "shell", label: "Run shell commands + background jobs" },
      { key: "web", label: "Search & fetch the web" },
      { key: "delegate", label: "Delegate subtasks to a subagent" },
      { key: "artifact", label: "Emit run artifacts" },
    ],
    model: "gpt-5.5",
    effort: "medium",
    efforts: ["none", "low", "medium", "high", "xhigh"],
    summary: "auto",
    summaries: ["off", "auto"],
    otel: false,
  };

  const fallbackProfile: Profile = {
    id: "default",
    name: "Default",
    description: "A balanced agent for everyday workspace tasks.",
    instructions: "Work carefully, keep a live plan, and verify changes before reporting completion.",
    capabilities: [...fallbackSettings.capabilities],
    model: fallbackSettings.model,
    effort: fallbackSettings.effort,
    summary: fallbackSettings.summary,
    built_in: true,
  };

  let config = $state<StudioConfig>({ workspace: "studio-workspace", provider: "offline", offline: true });
  let settings = $state<SettingsResponse>({ ...fallbackSettings });
  let profiles = $state<Profile[]>([fallbackProfile]);
  let profileCapabilities = $state<CapabilityOption[]>(fallbackSettings.available);
  let activeProfileId = $state("default");
  let sessions = $state<SessionSummary[]>([]);
  let files = $state<WorkspaceFile[]>([]);
  let jobs = $state<JobSummary[]>([]);
  let proposal = $state<ProposalResponse | null>(null);
  let proposalRunId = $state<string | null>(null);
  let run = $state(initialRunState());
  let mode = $state<StudioMode>("run");
  let inspector = $state<InspectorMode>("workspace");
  let streamConnected = $state(false);
  let leftOpen = $state(false);
  let inspectorOpen = $state(false);
  let booting = $state(true);
  let toast = $state("");
  let bootWarning = $state("");
  let wideNavigation = $state(false);
  let wideInspector = $state(false);
  let sessionEpoch = $state(0);
  let stream: RunEventStream | null = null;
  let navigationButton = $state<HTMLButtonElement>();
  let navigationDrawer = $state<HTMLElement>();
  let inspectorButton = $state<HTMLButtonElement>();
  let inspectorDrawer = $state<HTMLElement>();

  const activeProfile = $derived(profiles.find((profile) => profile.id === activeProfileId) ?? profiles[0] ?? fallbackProfile);
  const workspaceName = $derived(config.workspace.replace(/[\\/]+$/, "").split(/[\\/]/).at(-1) || config.workspace);
  const sessionStatus = $derived(run.status);
  const topModes: Array<{ id: StudioMode; label: string; icon: string }> = [
    { id: "run", label: "Sessions", icon: "message" },
    { id: "profile", label: "Profiles", icon: "profile" },
    { id: "review", label: "Changes", icon: "code" },
    { id: "trace", label: "Trace & tasks", icon: "trace" },
  ];
  const inspectorTabs: Array<{ id: InspectorMode; label: string; icon: string }> = [
    { id: "workspace", label: "Workspace", icon: "files" },
    { id: "config", label: "Config", icon: "settings" },
    { id: "trace", label: "Trace", icon: "trace" },
  ];

  onMount(() => {
    const navigationMedia = window.matchMedia("(min-width: 64rem)");
    const inspectorMedia = window.matchMedia("(min-width: 80rem)");
    const syncMedia = () => {
      wideNavigation = navigationMedia.matches;
      wideInspector = inspectorMedia.matches;
    };
    syncMedia();
    navigationMedia.addEventListener("change", syncMedia);
    inspectorMedia.addEventListener("change", syncMedia);
    void bootstrap();
    return () => {
      stream?.close();
      navigationMedia.removeEventListener("change", syncMedia);
      inspectorMedia.removeEventListener("change", syncMedia);
    };
  });

  async function bootstrap(): Promise<void> {
    booting = true;
    const [configResult, settingsResult, profilesResult, filesResult] = await Promise.allSettled([
      studioApi.config(),
      studioApi.settings(),
      studioApi.profiles(),
      studioApi.files(),
    ]);
    if (configResult.status === "fulfilled") config = configResult.value;
    if (settingsResult.status === "fulfilled") settings = settingsResult.value;
    if (profilesResult.status === "fulfilled") {
      profiles = profilesResult.value.profiles;
      profileCapabilities = profilesResult.value.available_capabilities;
      activeProfileId = profilesResult.value.default_profile_id;
    }
    if (filesResult.status === "fulfilled") files = flattenFiles(filesResult.value.files);
    if ([configResult, settingsResult, profilesResult].some((item) => item.status === "rejected")) {
      bootWarning = "Studio is showing local defaults while the BFF reconnects.";
    }
    await refreshSessions();
    const params = new URLSearchParams(location.search);
    if (location.pathname === "/settings") {
      mode = "run";
      inspector = "config";
      inspectorOpen = true;
    }
    const runId = params.get("run");
    if (runId) await openSession(runId);
    if (params.get("panel") === "trace") {
      inspectorOpen = false;
      mode = "trace";
    }
    booting = false;
  }

  function flattenFiles(items: WorkspaceFile[], prefix = ""): WorkspaceFile[] {
    const output: WorkspaceFile[] = [];
    for (const item of items) {
      const path = item.path || prefix;
      output.push({ ...item, path });
      if (item.children) output.push(...flattenFiles(item.children, path));
    }
    return output;
  }

  async function refreshSessions(): Promise<void> {
    const profileId = activeProfileId;
    const epoch = sessionEpoch;
    try {
      const next = (await studioApi.sessions(profileId)).sessions;
      if (profileId === activeProfileId && epoch === sessionEpoch) sessions = next;
    } catch {
      // The current view remains usable while the BFF reconnects.
    }
  }

  function announce(message: string): void {
    toast = message;
    window.setTimeout(() => {
      if (toast === message) toast = "";
    }, 2800);
  }

  function switchMode(next: StudioMode): void {
    if (next !== "run") closeInspector();
    mode = next;
    closeNavigation();
    if (next === "review" && run.runId) void refreshProposal();
  }

  function openFullTrace(): void {
    closeInspector();
    switchMode("trace");
  }

  function focusFirst(container: HTMLElement | undefined): void {
    container?.querySelector<HTMLElement>("button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), [tabindex='0']")?.focus();
  }

  function openNavigation(): void {
    leftOpen = true;
    if (!wideInspector) inspectorOpen = false;
    queueMicrotask(() => focusFirst(navigationDrawer));
  }

  function closeNavigation(restoreFocus = true): void {
    if (!leftOpen) return;
    leftOpen = false;
    if (restoreFocus && !wideNavigation) queueMicrotask(() => navigationButton?.focus());
  }

  function openInspector(next: InspectorMode = inspector): void {
    inspector = next;
    inspectorOpen = true;
    if (!wideNavigation) leftOpen = false;
    queueMicrotask(() => focusFirst(inspectorDrawer));
  }

  function closeInspector(restoreFocus = true): void {
    if (!inspectorOpen) return;
    inspectorOpen = false;
    if (restoreFocus && !wideInspector) queueMicrotask(() => inspectorButton?.focus());
  }

  function cycleDrawerFocus(event: KeyboardEvent, drawer: HTMLElement | undefined): void {
    if (!drawer) return;
    const focusable = [...drawer.querySelectorAll<HTMLElement>("button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), a[href], [tabindex='0']")];
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable.at(-1) ?? first;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function handleGlobalKeydown(event: KeyboardEvent): void {
    if (event.key === "Escape") {
      if (leftOpen && !wideNavigation) closeNavigation();
      else if (inspectorOpen && !wideInspector) closeInspector();
      return;
    }
    if (event.key !== "Tab") return;
    if (leftOpen && !wideNavigation) cycleDrawerFocus(event, navigationDrawer);
    else if (inspectorOpen && !wideInspector) cycleDrawerFocus(event, inspectorDrawer);
  }

  async function copyWorkspacePath(): Promise<void> {
    try {
      await navigator.clipboard.writeText(config.workspace);
      announce("Workspace path copied.");
    } catch {
      announce("Clipboard access is unavailable. The workspace path is shown in the button tooltip.");
    }
  }

  function moveInspectorTab(event: KeyboardEvent, index: number): void {
    const keys = ["ArrowRight", "ArrowLeft", "Home", "End"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % inspectorTabs.length;
    if (event.key === "ArrowLeft") next = (index - 1 + inspectorTabs.length) % inspectorTabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = inspectorTabs.length - 1;
    inspector = inspectorTabs[next].id;
    const tabs = (event.currentTarget as HTMLElement).parentElement?.querySelectorAll<HTMLElement>("[role=tab]");
    tabs?.[next]?.focus();
  }

  function isCurrentSession(runId: string | null, epoch: number): boolean {
    return sessionEpoch === epoch && run.runId === runId;
  }

  function openStream(runId: string, epoch = sessionEpoch): void {
    stream?.close();
    stream = new RunEventStream({
      runId,
      from: 0,
      onConnectionChange: (connected) => {
        if (isCurrentSession(runId, epoch)) streamConnected = connected;
      },
      onEvent: (event: RunEvent) => {
        if (!isCurrentSession(runId, epoch)) return;
        run = reduceRunEvent(run, event);
        if (["proposal.ready", "workspace.diff.updated", "workspace.proposal.updated"].includes(event.type)) {
          void refreshProposal(runId);
        }
        if (event.type.startsWith("job.")) void refreshJobs(runId);
        if (["turn.settled", "turn.interrupted", "run.awaiting_input", "run.finished", "run.failed", "session.state.changed"].includes(event.type)) {
          void refreshSessions();
        }
      },
    });
    stream.open();
  }

  async function openSession(runId: string): Promise<void> {
    const epoch = ++sessionEpoch;
    stream?.close();
    proposal = null;
    proposalRunId = null;
    jobs = [];
    run = initialRunState(runId);
    mode = "run";
    inspectorOpen = false;
    leftOpen = false;
    try {
      const transcript = await studioApi.transcript(runId);
      if (!isCurrentSession(runId, epoch)) return;
      run = hydrateTranscript(run, transcript);
      const summary = sessions.find((session) => session.run_id === runId);
      if (summary?.recoverable || summary?.state === "paused") {
        run = { ...run, status: "stopped" };
      }
      openStream(runId, epoch);
      history.replaceState({}, "", `/?run=${encodeURIComponent(runId)}`);
      await Promise.all([refreshProposal(runId), refreshJobs(runId)]);
    } catch (error) {
      if (isCurrentSession(runId, epoch)) {
        run = { ...run, status: "failed", error: error instanceof Error ? error.message : String(error) };
      }
    }
  }

  function newSession(): void {
    sessionEpoch += 1;
    stream?.close();
    stream = null;
    streamConnected = false;
    run = initialRunState();
    proposal = null;
    proposalRunId = null;
    jobs = [];
    mode = "run";
    inspectorOpen = false;
    history.replaceState({}, "", "/");
  }

  async function sendMessage(message: string, attachments: AttachmentInput[] = []): Promise<void> {
    const epoch = sessionEpoch;
    const targetRunId = run.runId;
    const id = clientMessageId();
    run = appendOptimisticUserMessage(
      run,
      id,
      message,
      attachments.map(({ name, mime }) => ({ name, mime })),
    );
    try {
      const response = await studioApi.send({
        runId: targetRunId,
        message,
        attachments,
        profileId: activeProfileId,
        clientMessageId: id,
      });
      if (!isCurrentSession(targetRunId, epoch)) return;
      if (!run.runId && response.run_id) {
        run = { ...run, runId: response.run_id };
        openStream(response.run_id, epoch);
        history.replaceState({}, "", `/?run=${encodeURIComponent(response.run_id)}`);
        void refreshSessions();
      }
    } catch (error) {
      if (!isCurrentSession(targetRunId, epoch)) return;
      run = {
        ...run,
        status: "failed",
        error: error instanceof Error ? error.message : String(error),
        errorRetryable: false,
        manualRetryCandidate: false,
        manualRetryReady: false,
      };
    }
  }

  async function stopTurn(): Promise<void> {
    const runId = run.runId;
    if (!runId) return;
    const epoch = sessionEpoch;
    run = { ...run, status: "stopping" };
    try {
      await studioApi.interrupt(runId);
      if (!isCurrentSession(runId, epoch)) return;
      announce("Current turn stopped. The session remains open.");
    } catch (error) {
      if (!isCurrentSession(runId, epoch)) return;
      run = { ...run, status: "failed", error: error instanceof Error ? error.message : String(error) };
    }
  }

  async function pauseRun(): Promise<void> {
    const runId = run.runId;
    if (!runId) return;
    const epoch = sessionEpoch;
    run = { ...run, status: "stopping" };
    try {
      const response = await studioApi.pause(runId);
      if (!isCurrentSession(runId, epoch)) return;
      if (response.state === "paused") {
        run = { ...run, status: "stopped" };
        announce("Run paused at a resumable checkpoint.");
      } else if (response.pause_requested) {
        announce("Pause requested. Studio will stop at the next clean checkpoint.");
      } else {
        run = { ...run, status: response.state === "running" ? "running" : run.status };
        announce("The run could not pause at its current boundary.");
      }
    } catch (error) {
      if (!isCurrentSession(runId, epoch)) return;
      run = { ...run, status: "failed", error: error instanceof Error ? error.message : String(error) };
    }
  }

  async function resumeRun(): Promise<void> {
    const runId = run.runId;
    if (!runId) return;
    const epoch = sessionEpoch;
    try {
      const response = await studioApi.resume(runId);
      if (!isCurrentSession(runId, epoch)) return;
      run = {
        ...run,
        status: response.state === "paused" ? "stopped" : response.state === "running" ? "running" : response.resumed ? "queued" : run.status,
        error: null,
      };
      openStream(runId, epoch);
      announce(response.resumed
        ? response.resume_kind === "paused_turn"
          ? "Paused turn resumed."
          : response.resume_kind === "checkpoint_and_paused_turn"
            ? "Checkpoint restored and paused turn resumed."
            : "Run restored from its checkpoint."
        : "Run is already live; Studio reconnected to its event stream.");
    } catch (error) {
      if (!isCurrentSession(runId, epoch)) return;
      run = { ...run, status: "failed", error: error instanceof Error ? error.message : String(error) };
    }
  }

  async function retryRun(): Promise<void> {
    const runId = run.runId;
    if (!runId) return;
    const epoch = sessionEpoch;
    try {
      const response = await studioApi.retry(runId);
      if (!isCurrentSession(runId, epoch)) return;
      if (response.retried === false) throw new Error("The failed request was not reissued.");
      run = {
        ...run,
        status: "queued",
        error: null,
        errorRetryable: false,
        manualRetryCandidate: false,
        manualRetryReady: false,
      };
      openStream(runId, epoch);
      announce("Request reissued with the current runtime configuration.");
    } catch (error) {
      if (!isCurrentSession(runId, epoch)) return;
      run = { ...run, status: "failed", error: error instanceof Error ? error.message : String(error) };
    }
  }

  async function answerApproval(request: ApprovalRequest, answer: string): Promise<void> {
    const runId = run.runId;
    if (!runId) return;
    const epoch = sessionEpoch;
    run = { ...run, approvals: run.approvals.map((item) => item.taskId === request.taskId ? { ...item, status: "submitting" } : item) };
    try {
      await studioApi.answerApproval(runId, request.taskId, answer);
      if (!isCurrentSession(runId, epoch)) return;
      const denied = /deny|reject|cancel/i.test(answer);
      run = {
        ...run,
        status: "queued",
        approvals: run.approvals.map((item) => item.taskId === request.taskId ? { ...item, status: denied ? "denied" : "approved" } : item),
      };
      announce(denied ? "Action denied and recorded." : "Approval recorded. Run continuing.");
      queueMicrotask(() => document.querySelector<HTMLTextAreaElement>(".composer textarea")?.focus());
    } catch (error) {
      if (!isCurrentSession(runId, epoch)) return;
      run = { ...run, status: "awaiting-approval", approvals: run.approvals.map((item) => item.taskId === request.taskId ? { ...item, status: "pending" } : item) };
      throw error;
    }
  }

  async function updateSettings(next: Partial<SettingsResponse>): Promise<void> {
    const updated = await studioApi.updateSettings({
      model: next.model,
      effort: next.effort,
      summary: next.summary,
      capabilities: next.capabilities,
      otel: next.otel,
    });
    settings = { ...settings, ...updated };
    announce(`Live Config applied to ${updated.applied_runs ?? 0} active run(s).`);
  }

  async function saveProfile(profile: Profile): Promise<Profile> {
    const epoch = sessionEpoch;
    const response = await studioApi.saveProfile(profile);
    profiles = profiles.some((item) => item.id === response.profile.id)
      ? profiles.map((item) => item.id === response.profile.id ? response.profile : item)
      : [...profiles, response.profile];
    if (sessionEpoch !== epoch) return response.profile;
    activeProfileId = response.profile.id;
    sessionEpoch += 1;
    stream?.close();
    run = initialRunState();
    proposal = null;
    proposalRunId = null;
    jobs = [];
    history.replaceState({}, "", "/");
    await refreshSessions();
    announce(`Saved ${response.profile.name}.`);
    return response.profile;
  }

  function addProfile(): void {
    const id = `profile-${Date.now().toString(36)}`;
    const profile: Profile = {
      ...fallbackProfile,
      id,
      name: "New profile",
      description: "",
      built_in: false,
      capabilities: [...activeProfile.capabilities],
      model: activeProfile.model,
      effort: activeProfile.effort,
      summary: activeProfile.summary,
    };
    profiles = [...profiles, profile];
    activeProfileId = id;
    switchMode("profile");
  }

  async function refreshProposal(targetRunId: string | null = run.runId): Promise<void> {
    if (!targetRunId) {
      proposal = null;
      proposalRunId = null;
      return;
    }
    const epoch = sessionEpoch;
    try {
      const next = await studioApi.proposal(targetRunId);
      if (!isCurrentSession(targetRunId, epoch)) return;
      proposal = next;
      proposalRunId = targetRunId;
      run = { ...run, proposalDirty: false };
    } catch {
      if (isCurrentSession(targetRunId, epoch)) {
        proposal = null;
        proposalRunId = null;
      }
    }
  }

  async function refreshFiles(): Promise<void> {
    files = flattenFiles((await studioApi.files()).files);
  }

  async function refreshJobs(targetRunId: string | null = run.runId): Promise<void> {
    if (!targetRunId) {
      jobs = [];
      return;
    }
    const epoch = sessionEpoch;
    try {
      const next = (await studioApi.jobs(targetRunId)).jobs ?? [];
      if (isCurrentSession(targetRunId, epoch)) jobs = next;
    } catch {
      if (isCurrentSession(targetRunId, epoch)) jobs = [];
    }
  }

  async function applyApproved(paths: string[]): Promise<ApplyResponse> {
    const runId = run.runId;
    const epoch = sessionEpoch;
    const proposalHash = proposal?.proposal_hash;
    if (!runId) throw new Error("Start or open a run before applying changes.");
    if (proposalRunId !== runId || !proposalHash) throw new Error("The displayed proposal does not belong to the active run. Refresh Changes before applying.");
    if (!paths.length) throw new Error("Approve at least one reviewed file before applying.");
    const result = await studioApi.applyProposal(runId, paths, proposalHash);
    const nextFiles = flattenFiles((await studioApi.files()).files);
    if (isCurrentSession(runId, epoch)) files = nextFiles;
    return result;
  }

  async function exportPackage(): Promise<PackageReceipt> {
    const runId = run.runId;
    const proposalHash = proposal?.proposal_hash;
    if (!runId) throw new Error("Start or open a run before exporting a package.");
    if (proposalRunId !== runId || !proposalHash) throw new Error("The displayed proposal does not belong to the active run. Refresh Changes before exporting.");
    return downloadProposalPackage(runId, proposalHash);
  }

  function switchProfile(id: string): void {
    sessionEpoch += 1;
    activeProfileId = id;
    run = initialRunState();
    proposal = null;
    proposalRunId = null;
    jobs = [];
    stream?.close();
    history.replaceState({}, "", "/");
    void refreshSessions();
  }
</script>

<svelte:head><title>{mode === "run" ? "Studio" : mode === "profile" ? "Profiles" : mode === "review" ? "Review Changes" : "Trace"} · Monoid</title></svelte:head>
<svelte:window onkeydown={handleGlobalKeydown} />

<div class="studio-shell" data-testid="studio-shell">
  <header class="app-topbar">
    <div class="topbar-left">
      <button bind:this={navigationButton} class="mobile-menu" aria-label={leftOpen ? "Close navigation" : "Open navigation"} aria-controls="studio-navigation" aria-expanded={leftOpen} onclick={() => leftOpen ? closeNavigation() : openNavigation()}><Icon name="menu" size={18} /></button>
      <div class="wordmark"><span><Icon name="spark" size={17} /></span><strong>MONOID</strong><small>STUDIO</small></div>
      <span class="topbar-divider"></span>
      <button class="workspace-switcher" title={`Copy workspace path: ${config.workspace}`} onclick={() => void copyWorkspacePath()}><Icon name="files" size={14} /><span>{workspaceName}</span><Icon name="copy" size={11} /></button>
    </div>

    <div class="topbar-center">
      <button data-testid="profile-switcher" class="profile-switcher" onclick={() => switchMode("profile")}>
        <span class="avatar"><Icon name="spark" size={13} /></span>
        <span><strong>{activeProfile.name}</strong><small>{activeProfile.model} · {activeProfile.effort}</small></span>
        <Icon name="chevron" size={11} />
      </button>
    </div>

    <div class="topbar-right">
      {#if run.runId}
        <div class="token-meter" title="Token usage"><span>{run.usage.total.toLocaleString()}</span><small>tokens</small></div>
        <StatusBadge status={sessionStatus} compact />
      {/if}
      <button bind:this={inspectorButton} class="icon-button inspector-toggle" title="Toggle inspector" aria-label={inspectorOpen ? "Close run inspector" : "Open run inspector"} aria-controls="run-inspector" aria-expanded={inspectorOpen} onclick={() => inspectorOpen ? closeInspector() : openInspector()}><Icon name="settings" size={15} /></button>
      <button class="new-session-button" aria-label="New session" onclick={newSession}><Icon name="plus" size={14} /><span>New session</span></button>
    </div>
  </header>

  <div class="app-body">
    {#if leftOpen}<button class="drawer-scrim" aria-label="Close navigation" onclick={() => closeNavigation()}></button>{/if}
    <aside
      bind:this={navigationDrawer}
      id="studio-navigation"
      class:open={leftOpen}
      class="left-sidebar"
      data-testid="left-config-panel"
      inert={!leftOpen && !wideNavigation}
      aria-hidden={!leftOpen && !wideNavigation}
      role={leftOpen && !wideNavigation ? "dialog" : undefined}
      aria-modal={leftOpen && !wideNavigation ? "true" : undefined}
      aria-label={leftOpen && !wideNavigation ? "Studio navigation" : undefined}
    >
      <nav class="primary-nav" aria-label="Studio">
        <div class="nav-label">Work</div>
        {#each topModes as item}
          <button
            class:active={mode === item.id}
            aria-current={mode === item.id ? "page" : undefined}
            onclick={() => switchMode(item.id)}
          >
            <span class="nav-rail"></span><Icon name={item.icon} size={16} /><span>{item.label}</span>
            {#if item.id === "review" && proposal?.ready}<small>{proposal.changed_paths?.length ?? proposal.files?.length ?? 0}</small>{/if}
          </button>
        {/each}
      </nav>

      <section class="sidebar-section session-section">
        <header><span>Recent sessions</span><button title="New session" onclick={newSession}><Icon name="plus" size={14} /></button></header>
        <div class="session-list">
          {#if sessions.length === 0}
            <div class="sidebar-empty">No sessions for this profile yet.</div>
          {:else}
            {#each sessions as session}
              <button class:active={run.runId === session.run_id} aria-current={run.runId === session.run_id ? "true" : undefined} onclick={() => void openSession(session.run_id)}>
                <span class:live={!session.terminal} class:failed={session.state === "failed"} class="session-dot"></span>
                <span><strong>{session.title || "Untitled run"}</strong><small>{session.recoverable ? "Checkpoint available" : session.state}</small></span>
                {#if session.recoverable}<Icon name="retry" size={12} />{/if}
              </button>
            {/each}
          {/if}
        </div>
      </section>

      <section class="sidebar-section profile-section" data-testid="profile-list">
        <header><span>Agent profiles</span><button data-testid="profile-add" title="Add profile" onclick={addProfile}><Icon name="plus" size={14} /></button></header>
        {#each profiles.slice(0, 5) as profile}
          <button class:active={profile.id === activeProfileId} aria-pressed={profile.id === activeProfileId} onclick={() => switchProfile(profile.id)}>
            <span class="avatar small"><Icon name="spark" size={11} /></span><span><strong>{profile.name}</strong><small>{profile.model}</small></span>
          </button>
        {/each}
      </section>

      <footer class="sidebar-footer"><span class="connection-dot" class:connected={!bootWarning}></span><span>{bootWarning ? "BFF reconnecting" : "Local BFF connected"}</span><small>reference</small></footer>
    </aside>

    <main class:wide={mode !== "run"} class="main-stage" inert={(leftOpen && !wideNavigation) || (mode === "run" && inspectorOpen && !wideInspector)}>
      {#if booting}
        <div class="app-loading"><span class="brand-loader"><Icon name="spark" size={21} /></span><strong>Opening Studio</strong><small>Restoring profiles and sessions…</small></div>
      {:else if mode === "profile"}
        {#key activeProfile.id}
          <ProfileBuilder
            {activeProfile}
            capabilities={profileCapabilities}
            efforts={settings.efforts}
            summaries={settings.summaries}
            onSave={saveProfile}
            onPreview={studioApi.previewProfile}
          />
        {/key}
      {:else if mode === "review"}
        {#key `${run.runId ?? "none"}:${proposalRunId ?? "none"}`}
          <ReviewChanges {proposal} onApply={applyApproved} onExport={exportPackage} />
        {/key}
      {:else if mode === "trace"}
        <TraceView events={run.events} />
      {:else}
        <div class="run-stage">
          {#if run.runId || run.messages.length}
            <RunView
              state={run}
              connected={streamConnected}
              onStop={stopTurn}
              onPause={pauseRun}
              onResume={resumeRun}
              onRetry={retryRun}
              onApproval={answerApproval}
              onOpenConfig={() => openInspector("config")}
            />
          {:else}
            <Onboarding
              offline={config.offline}
              {workspaceName}
              onPrompt={(prompt) => void sendMessage(prompt)}
              onOpenProfile={() => switchMode("profile")}
            />
          {/if}
          <Composer status={run.status} onSend={sendMessage} onStop={stopTurn} />
        </div>
      {/if}
    </main>

    {#if mode === "run"}
      {#if inspectorOpen}<button class="inspector-scrim" aria-label="Close inspector" onclick={() => closeInspector()}></button>{/if}
      <aside
        bind:this={inspectorDrawer}
        id="run-inspector"
        class:open={inspectorOpen}
        class="right-inspector"
        inert={!inspectorOpen && !wideInspector}
        aria-hidden={!inspectorOpen && !wideInspector}
        role={inspectorOpen && !wideInspector ? "dialog" : undefined}
        aria-modal={inspectorOpen && !wideInspector ? "true" : undefined}
        aria-label={inspectorOpen && !wideInspector ? "Run inspector" : undefined}
      >
        <div class="inspector-tabs" data-testid="right-panel-tabs" role="tablist" aria-label="Run inspector">
          {#each inspectorTabs as item, index}
            <button
              role="tab"
              id={`inspector-tab-${item.id}`}
              aria-selected={inspector === item.id}
              aria-controls="inspector-panel"
              tabindex={inspector === item.id ? 0 : -1}
              class:active={inspector === item.id}
              onclick={() => (inspector = item.id)}
              onkeydown={(event) => moveInspectorTab(event, index)}
            ><Icon name={item.icon} size={14} /><span>{item.label}</span></button>
          {/each}
        </div>
        <div class="inspector-content" id="inspector-panel" role="tabpanel" aria-labelledby={`inspector-tab-${inspector}`}>
          {#if inspector === "workspace"}
            <WorkspaceInspector {workspaceName} {files} {jobs} plan={run.plan} onRefresh={refreshFiles} />
          {:else if inspector === "config"}
            {#key `${settings.model}:${settings.effort}:${settings.capabilities.join(',')}:${settings.otel}`}
              <LiveConfig {settings} onApply={updateSettings} />
            {/key}
          {:else}
            <div class="mini-trace-inspector">
              <header class="inspector-heading"><div><div class="eyebrow">Live events</div><h2>Trace</h2></div><button class="icon-button" title="Open full trace" onclick={openFullTrace}><Icon name="chevron" size={14} /></button></header>
              {#if run.events.length}
                {#each run.events.slice(-30).reverse() as event}
                  <button onclick={openFullTrace}><span class:error={event.level === "error"}></span><div><strong>{event.type}</strong><small>seq {event.seq ?? "—"}</small></div></button>
                {/each}
              {:else}<div class="inspector-empty large"><Icon name="trace" size={20} />Trace events appear after the run starts.</div>{/if}
            </div>
          {/if}
        </div>
      </aside>
    {/if}
  </div>

  {#if toast}<div class="toast" role="status" aria-live="polite"><Icon name="check" size={14} />{toast}</div>{/if}
</div>
