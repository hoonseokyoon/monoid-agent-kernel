import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [tailwindcss(), svelte()],
  build: {
    emptyOutDir: true,
    outDir: "../src/monoid_agent_kernel/reference/studio/web/dist",
    sourcemap: false,
    target: "es2022",
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8799",
        changeOrigin: false,
      },
      "/healthz": "http://127.0.0.1:8799",
      "/vendor": "http://127.0.0.1:8799",
    },
  },
});
