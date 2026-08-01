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
let accepted = 0;
let rejected = 0;
globalThis.fetch = async () => ({
  ok: true,
  status: 200,
  statusText: "OK",
  json: async () => responsePayload,
});

const message = (overrides = {}) => ({
  id: "message-1",
  role: "assistant",
  content: "done",
  attachments: [{ name: "result.txt", mime: "text/plain" }],
  // A fractional value catches an accidental Number.isInteger check.
  created_at: 1.25,
  ...overrides,
});
const validMessage = message({
  attachments: [{ name: "result.txt", mime: "text/plain", future_attachment_member: true }],
  source: { kind: "event", future_source_member: true },
  // Message records remain version- and extension-permissive even though the renderer-required
  // fields are validated. This unknown identifier and the additive members must remain accepted.
  schema_version: "studio.chat.message.v999",
  future_message_member: true,
});
const base = {
  run_id: "run-1",
  messages: [validMessage],
  event_cursor: -1,
};

async function accepts(payload) {
  responsePayload = payload;
  assert.equal(await studioApi.transcript("run-1"), payload);
  accepted += 1;
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
  rejected += 1;
}

await accepts({ schema_version: "studio.chat.v1", ...base });
await accepts({ schema_version: "studio.chat.v2", ...base, event_log_error: "" });
await accepts({
  schema_version: "studio.chat.v2",
  ...base,
  event_log_error: "events.jsonl is corrupt",
});
await accepts({ schema_version: "studio.chat.v1", ...base, messages: [] });
await accepts({
  schema_version: "studio.chat.v2",
  ...base,
  messages: [
    message({ id: "message-user", role: "user", content: "question" }),
    message({ id: "message-assistant" }),
    message({ id: "message-error", role: "error", content: "failure" }),
  ],
  event_log_error: "",
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

function withoutRequiredField(field) {
  const candidate = message();
  delete candidate[field];
  return candidate;
}

const malformedMessages = [
  null,
  [],
  { legacy_or_future_member: true },
  withoutRequiredField("id"),
  withoutRequiredField("role"),
  withoutRequiredField("content"),
  withoutRequiredField("attachments"),
  withoutRequiredField("created_at"),
  message({ id: "" }),
  message({ id: "   " }),
  message({ id: 7 }),
  message({ role: "system" }),
  message({ role: 7 }),
  message({ content: null }),
  message({ attachments: {} }),
  message({ attachments: Array(1) }),
  message({ attachments: [null] }),
  message({ attachments: [[]] }),
  message({ attachments: [{ mime: "text/plain" }] }),
  message({ attachments: [{ name: 7, mime: "text/plain" }] }),
  message({ attachments: [{ name: "result.txt" }] }),
  message({ attachments: [{ name: "result.txt", mime: 7 }] }),
  message({ created_at: "1" }),
  message({ created_at: true }),
  message({ created_at: Number.NaN }),
  message({ created_at: Number.POSITIVE_INFINITY }),
  message({ created_at: Number.NEGATIVE_INFINITY }),
  message({ source: null }),
  message({ source: [] }),
  message({ source: "event" }),
];

for (const malformedMessage of malformedMessages) {
  await rejects({
    schema_version: "studio.chat.v2",
    ...base,
    messages: [malformedMessage],
    event_log_error: "",
  });
}

await rejects({
  schema_version: "studio.chat.v2",
  ...base,
  messages: Array(1),
  event_log_error: "",
});

await rejects({
  schema_version: "studio.chat.v2",
  ...base,
  messages: [message(), message()],
  event_log_error: "",
});

console.log(`Studio chat-transcript boundary checks passed (${accepted + rejected} scenarios).`);
