<script lang="ts">
  let { name, size = 16, strokeWidth = 1.8, label = "" } = $props<{
    name: string;
    size?: number;
    strokeWidth?: number;
    label?: string;
  }>();
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width={strokeWidth}
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden={label ? undefined : "true"}
  aria-label={label || undefined}
>
  {#if name === "spark"}
    <path d="m12 3 1.2 4.3L17.5 9l-4.3 1.5L12 15l-1.3-4.5L6.5 9l4.2-1.7L12 3Z" />
    <path d="m18.5 14 .7 2.3 2.3.7-2.3.8-.7 2.2-.7-2.2-2.3-.8 2.3-.7.7-2.3Z" />
  {:else if name === "plus"}
    <path d="M12 5v14M5 12h14" />
  {:else if name === "message"}
    <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4v8Z" />
  {:else if name === "profile"}
    <circle cx="12" cy="8" r="3.5" /><path d="M5 21a7 7 0 0 1 14 0" />
  {:else if name === "files"}
    <path d="M4 4h6l2 2h8v14H4z" /><path d="M4 9h16" />
  {:else if name === "file"}
    <path d="M6 2h8l4 4v16H6z" /><path d="M14 2v5h5" />
  {:else if name === "trace"}
    <circle cx="6" cy="5" r="2" /><circle cx="18" cy="12" r="2" /><circle cx="8" cy="19" r="2" />
    <path d="M8 6.5 16 11M16.5 13.5 9.5 18" />
  {:else if name === "settings"}
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3A1.7 1.7 0 0 0 10 3V2.8h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z" />
  {:else if name === "stop"}
    <rect x="6" y="6" width="12" height="12" rx="2" />
  {:else if name === "play"}
    <path d="m8 5 11 7-11 7V5Z" />
  {:else if name === "retry"}
    <path d="M20 7v5h-5" /><path d="M19 12a7 7 0 1 0-2 5" />
  {:else if name === "shield"}
    <path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z" /><path d="m9 12 2 2 4-4" />
  {:else if name === "check"}
    <path d="m5 12 4 4L19 6" />
  {:else if name === "x" || name === "close"}
    <path d="m6 6 12 12M18 6 6 18" />
  {:else if name === "alert"}
    <path d="M10.3 3.7 2.5 18a2 2 0 0 0 1.8 3h15.4a2 2 0 0 0 1.8-3L13.7 3.7a2 2 0 0 0-3.4 0Z" />
    <path d="M12 9v4M12 17h.01" />
  {:else if name === "package"}
    <path d="m21 8-9 5-9-5 9-5 9 5Z" /><path d="m3 8 9 5v9l9-5V8M12 13v9" />
  {:else if name === "code"}
    <path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14" />
  {:else if name === "copy"}
    <rect x="9" y="9" width="11" height="11" rx="2" /><path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3" />
  {:else if name === "terminal"}
    <rect x="3" y="4" width="18" height="16" rx="2" /><path d="m7 9 3 3-3 3M13 15h4" />
  {:else if name === "search"}
    <circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" />
  {:else if name === "clock"}
    <circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />
  {:else if name === "bot"}
    <rect x="4" y="7" width="16" height="13" rx="3" /><path d="M12 3v4M8 12h.01M16 12h.01M8 16h8" />
  {:else if name === "send"}
    <path d="m22 2-7 20-4-9-9-4 20-7Z" /><path d="M22 2 11 13" />
  {:else if name === "paperclip"}
    <path d="m20.5 11.5-8.8 8.8a6 6 0 0 1-8.5-8.5l9.5-9.5a4 4 0 0 1 5.7 5.7l-9.5 9.5a2 2 0 0 1-2.9-2.9l8.8-8.8" />
  {:else if name === "download"}
    <path d="M12 3v12M7 10l5 5 5-5M5 21h14" />
  {:else if name === "eye"}
    <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z" /><circle cx="12" cy="12" r="3" />
  {:else if name === "menu"}
    <path d="M4 6h16M4 12h16M4 18h16" />
  {:else if name === "chevron"}
    <path d="m9 18 6-6-6-6" />
  {:else}
    <circle cx="12" cy="12" r="9" />
  {/if}
</svg>
