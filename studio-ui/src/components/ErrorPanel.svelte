<script lang="ts">
  import Icon from "./Icon.svelte";

  let { title = "The run needs attention", message, retryable = true, onRetry, onOpenConfig } = $props<{
    title?: string;
    message: string;
    retryable?: boolean;
    onRetry?: () => void;
    onOpenConfig?: () => void;
  }>();
</script>

<article class="error-panel" role="alert" data-testid="run-error">
  <span class="error-mark"><Icon name="alert" size={18} /></span>
  <div class="min-w-0 flex-1">
    <div class="flex items-start justify-between gap-4">
      <div>
        <div class="eyebrow">Run interrupted</div>
        <h3>{title}</h3>
      </div>
      <span class="error-code">RECOVERABLE</span>
    </div>
    <p>{message}</p>
    <div class="mt-4 flex flex-wrap gap-2">
      {#if retryable && onRetry}
        <button class="secondary-button" onclick={onRetry}><Icon name="retry" size={14} />Retry request</button>
      {/if}
      {#if onOpenConfig}
        <button class="ghost-button" onclick={onOpenConfig}><Icon name="settings" size={14} />Review configuration</button>
      {/if}
    </div>
  </div>
</article>
