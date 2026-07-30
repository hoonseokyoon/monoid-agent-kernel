import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import ts from "typescript";

const source = await readFile(new URL("../src/lib/api.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
  fileName: "api.ts",
});
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled.outputText).toString("base64")}`;
const { StudioApiError, studioApi } = await import(moduleUrl);

let responsePayload;
globalThis.fetch = async () => ({
  ok: true,
  status: 200,
  statusText: "OK",
  json: async () => responsePayload,
});

const base = {
  run_id: "run-1",
  // Message records have their own permissive compatibility contract. This boundary validates
  // the response envelope without turning nested message expansion into a breaking change.
  messages: [{ legacy_or_future_member: true }],
  event_cursor: -1,
};

async function accepts(payload) {
  responsePayload = payload;
  assert.equal(await studioApi.transcript("run-1"), payload);
}

async function rejects(payload) {
  responsePayload = payload;
  await assert.rejects(
    () => studioApi.transcript("run-1"),
    (error) => {
      assert.ok(error instanceof StudioApiError);
      assert.equal(error.status, 502);
      assert.equal(
        error.message,
        "Studio returned an unsupported or malformed chat transcript response.",
      );
      assert.equal(error.payload, payload);
      return true;
    },
  );
}

await accepts({ schema_version: "studio.chat.v1", ...base });
await accepts({ schema_version: "studio.chat.v2", ...base, event_log_error: "" });
await accepts({
  schema_version: "studio.chat.v2",
  ...base,
  event_log_error: "events.jsonl is corrupt",
});

for (const malformed of [
  { schema_version: "studio.chat.v3", ...base, event_log_error: "" },
  { schema_version: "studio.chat.v1", ...base, event_log_error: "" },
  { schema_version: "studio.chat.v2", ...base },
  { schema_version: "studio.chat.v2", ...base, event_log_error: null },
  { schema_version: "studio.chat.v2", ...base, event_log_error: 7 },
  { schema_version: "studio.chat.v2", ...base, event_log_error: "", extra: true },
  { schema_version: "studio.chat.v2", ...base, run_id: 7, event_log_error: "" },
  { schema_version: "studio.chat.v2", ...base, messages: {}, event_log_error: "" },
  { schema_version: "studio.chat.v2", ...base, event_cursor: true, event_log_error: "" },
  { schema_version: "studio.chat.v2", ...base, event_cursor: 1.5, event_log_error: "" },
  { schema_version: "studio.chat.v2", ...base, event_cursor: "1", event_log_error: "" },
  null,
  [],
  "studio.chat.v2",
]) {
  await rejects(malformed);
}

console.log("Studio chat-transcript boundary checks passed (17 scenarios).");
