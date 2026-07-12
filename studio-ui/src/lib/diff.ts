function normalizedHeaderPath(line: string): string | null {
  const raw = line.slice(4).split("\t", 1)[0];
  if (!raw || raw === "/dev/null") return null;
  return raw.startsWith("a/") || raw.startsWith("b/") ? raw.slice(2) : raw;
}

function isUnifiedHeader(lines: string[], index: number): boolean {
  const from = lines[index] ?? "";
  const to = lines[index + 1] ?? "";
  return (from.startsWith("--- a/") || from === "--- /dev/null")
    && (to.startsWith("+++ b/") || to === "+++ /dev/null");
}

export function fileDiff(diff: string, path: string): string | null {
  if (!diff || !path) return null;
  const normalized = path.replaceAll("\\", "/");
  const lines = diff.split("\n");
  const gitStarts = lines
    .map((line, index) => line.startsWith("diff --git ") ? index : -1)
    .filter((index) => index >= 0);
  const starts = gitStarts.length > 0
    ? gitStarts
    : lines.map((_, index) => isUnifiedHeader(lines, index) ? index : -1).filter((index) => index >= 0);

  for (let index = 0; index < starts.length; index += 1) {
    const start = starts[index];
    const end = starts[index + 1] ?? lines.length;
    const section = lines.slice(start, end);
    const matchesPath = section.some((line) =>
      (line.startsWith("--- ") || line.startsWith("+++ "))
      && normalizedHeaderPath(line) === normalized
    );
    if (matchesPath) return section.join("\n");
  }
  return null;
}
