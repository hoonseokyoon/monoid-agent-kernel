import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../src/app.css", import.meta.url), "utf8");
const colors = Object.fromEntries(
  [...css.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6})\s*;/gi)].map((match) => [match[1], match[2]]),
);

function luminance(color) {
  const channels = [1, 3, 5].map((offset) => Number.parseInt(color.slice(offset, offset + 2), 16) / 255);
  const linear = channels.map((channel) => channel <= 0.04045
    ? channel / 12.92
    : ((channel + 0.055) / 1.055) ** 2.4);
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

function contrast(foreground, background) {
  const values = [luminance(foreground), luminance(background)].sort((a, b) => b - a);
  return (values[0] + 0.05) / (values[1] + 0.05);
}

const checks = [
  ["muted on canvas", "fg-muted", "canvas", 4.5],
  ["muted on panel", "fg-muted", "panel", 4.5],
  ["muted on surface", "fg-muted", "surface", 4.5],
  ["muted on raised", "fg-muted", "raised", 4.5],
  ["muted on selected", "fg-muted", "selected", 4.5],
  ["secondary text on user bubble", "on-user-secondary", "user", 4.5],
  ["success status", "status-success-fg", "status-success-bg", 4.5],
  ["warning status", "status-warning-fg", "status-warning-bg", 4.5],
  ["danger status", "status-danger-fg", "status-danger-bg", 4.5],
  ["paused status", "status-paused-fg", "status-paused-bg", 4.5],
  ["disabled text", "fg-muted", "disabled-bg", 4.5],
  ["control boundary", "control-border", "surface", 3],
];

const failures = [];
for (const [label, foregroundName, backgroundName, minimum] of checks) {
  const foreground = colors[foregroundName];
  const background = colors[backgroundName];
  if (!foreground || !background) {
    failures.push(`${label}: missing token`);
    continue;
  }
  const value = contrast(foreground, background);
  if (value < minimum) failures.push(`${label}: ${value.toFixed(2)}:1 < ${minimum}:1`);
}

if (failures.length) {
  console.error(`Contrast checks failed:\n${failures.map((failure) => `- ${failure}`).join("\n")}`);
  process.exitCode = 1;
} else {
  console.log(`Contrast checks passed (${checks.length} pairs).`);
}
