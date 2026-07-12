<script lang="ts">
  import type { AttachmentInput, RunStatus } from "../lib/types";
  import { isRunBusy } from "../lib/run-state";
  import Icon from "./Icon.svelte";

  let { status, disabled = false, onSend, onStop } = $props<{
    status: RunStatus;
    disabled?: boolean;
    onSend: (message: string, attachments: AttachmentInput[]) => Promise<void>;
    onStop: () => Promise<void>;
  }>();

  let message = $state("");
  let attachments = $state<AttachmentInput[]>([]);
  let submitting = $state(false);
  let fileInput: HTMLInputElement;

  const busy = $derived(isRunBusy(status));

  function readBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const value = String(reader.result ?? "");
        resolve(value.slice(value.indexOf(",") + 1));
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }

  async function addFiles(event: Event): Promise<void> {
    const input = event.currentTarget as HTMLInputElement;
    for (const file of Array.from(input.files ?? [])) {
      attachments = [
        ...attachments,
        { name: file.name, mime: file.type || "application/octet-stream", data_b64: await readBase64(file) },
      ];
    }
    input.value = "";
  }

  async function submit(): Promise<void> {
    const clean = message.trim();
    if ((!clean && attachments.length === 0) || submitting || disabled) return;
    const pending = attachments;
    message = "";
    attachments = [];
    submitting = true;
    try {
      await onSend(clean, pending);
    } finally {
      submitting = false;
    }
  }

  function keydown(event: KeyboardEvent): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  }
</script>

<div class="composer-wrap" data-testid="composer">
  {#if attachments.length}
    <div class="attachment-list" aria-label="Attached files">
      {#each attachments as attachment, index}
        <span><Icon name="paperclip" size={12} />{attachment.name}<button title={`Remove ${attachment.name}`} onclick={() => (attachments = attachments.filter((_, i) => i !== index))}><Icon name="x" size={11} /></button></span>
      {/each}
    </div>
  {/if}
  <div class="composer">
    <textarea
      bind:value={message}
      rows="1"
      placeholder={busy ? "Add context for the next turn…" : "Ask Studio to work on something…"}
      aria-label="Message"
      onkeydown={keydown}
    ></textarea>
    <div class="composer-toolbar">
      <div>
        <input bind:this={fileInput} type="file" multiple hidden onchange={addFiles} />
        <button class="icon-button" title="Attach files" onclick={() => fileInput.click()}><Icon name="paperclip" size={16} /></button>
        <span class="composer-hint"><kbd>Enter</kbd> send · <kbd>Shift Enter</kbd> newline</span>
      </div>
      {#if busy}
        <button class="stop-button" onclick={() => void onStop()}><Icon name="stop" size={13} />Stop turn</button>
      {:else}
        <button class="send-button" disabled={submitting || (!message.trim() && attachments.length === 0)} onclick={() => void submit()} aria-label="Send message"><Icon name="send" size={15} /></button>
      {/if}
    </div>
  </div>
</div>
