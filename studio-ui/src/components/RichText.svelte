<script lang="ts">
  import { onMount } from "svelte";

  let { content } = $props<{ content: string }>();
  let root: HTMLDivElement;

  function escapeHtml(value: string): string {
    return value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function inlineMarkup(value: string): string {
    return escapeHtml(value)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  }

  function renderMarkdown(value: string): string {
    const blocks: Array<{ language: string; code: string }> = [];
    let source = value.replaceAll("\u0000", "\uFFFD").replace(/```([\w.+-]*)\n?([\s\S]*?)```/g, (_match, language, code) => {
      blocks.push({ language, code });
      return `\u0000STUDIO_CODE_${blocks.length - 1}\u0000`;
    });
    const openFence = source.indexOf("```");
    if (openFence >= 0) {
      const rest = source.slice(openFence + 3);
      const newline = rest.indexOf("\n");
      blocks.push({
        language: (newline >= 0 ? rest.slice(0, newline) : rest).trim(),
        code: newline >= 0 ? rest.slice(newline + 1) : "",
      });
      source = `${source.slice(0, openFence)}\u0000STUDIO_CODE_${blocks.length - 1}\u0000`;
    }

    const rendered = source.split("\n").map((line) => {
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        const level = Math.min(heading[1].length + 2, 6);
        return `<h${level}>${inlineMarkup(heading[2])}</h${level}>`;
      }
      if (line.startsWith("> ")) return `<blockquote>${inlineMarkup(line.slice(2))}</blockquote>`;
      if (/^[-*]\s+/.test(line)) {
        return `<div class="md-list-item"><span aria-hidden="true">•</span><span>${inlineMarkup(line.replace(/^[-*]\s+/, ""))}</span></div>`;
      }
      if (/^\u0000STUDIO_CODE_\d+\u0000$/.test(line)) return line;
      return line ? `<p>${inlineMarkup(line)}</p>` : "<br />";
    }).join("");

    return rendered.replace(/\u0000STUDIO_CODE_(\d+)\u0000/g, (_match, index) => {
      const block = blocks[Number(index)];
      if (!block) return escapeHtml(_match);
      const language = block.language
        ? `<span class="code-lang">${escapeHtml(block.language)}</span>`
        : "";
      return `<div class="codeblock">${language}<button type="button" data-copy-code>Copy code</button><pre><code>${escapeHtml(block.code)}</code></pre></div>`;
    });
  }

  const html = $derived(renderMarkdown(content));

  function typeset(): void {
    if (!root || !window.renderMathInElement) return;
    window.renderMathInElement(root, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }

  $effect(() => {
    html;
    queueMicrotask(typeset);
  });

  onMount(() => {
    root.addEventListener("click", handleClick);
    window.addEventListener("studio:katex-ready", typeset);
    return () => {
      root.removeEventListener("click", handleClick);
      window.removeEventListener("studio:katex-ready", typeset);
    };
  });

  async function handleClick(event: Event): Promise<void> {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-copy-code]");
    if (!button) return;
    const code = button.parentElement?.querySelector("code")?.textContent ?? "";
    await navigator.clipboard.writeText(code);
    button.textContent = "Copied";
    window.setTimeout(() => (button.textContent = "Copy code"), 1200);
  }
</script>

<!-- The renderer escapes all model text before adding a small, fixed markup vocabulary. -->
<div class="rich-text" bind:this={root}>{@html html}</div>
