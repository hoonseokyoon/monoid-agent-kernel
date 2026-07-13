import { mount } from "svelte";

import App from "./App.svelte";
import "./app.css";

const target = document.getElementById("app");

if (!target) {
  throw new Error("Studio mount target was not found");
}

mount(App, { target });

const katexStyles = document.createElement("link");
katexStyles.rel = "stylesheet";
katexStyles.href = "/vendor/katex/katex.min.css";
document.head.appendChild(katexStyles);

function loadScript(source: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = source;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Unable to load ${source}`));
    document.head.appendChild(script);
  });
}

void loadScript("/vendor/katex/katex.min.js")
  .then(() => loadScript("/vendor/katex/auto-render.min.js"))
  .then(() => window.dispatchEvent(new Event("studio:katex-ready")))
  .catch(() => {
    // Math rendering is an enhancement; chat remains readable when the local vendor asset fails.
  });
