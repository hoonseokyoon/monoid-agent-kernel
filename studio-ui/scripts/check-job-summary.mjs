import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const types = await readFile(new URL("../src/lib/types.ts", import.meta.url), "utf8");
const inspector = await readFile(
  new URL("../src/components/WorkspaceInspector.svelte", import.meta.url),
  "utf8",
);

const jobSummaryMatch = types.match(/export interface JobSummary\s*\{([^}]*)\}/);
assert.ok(jobSummaryMatch, "the Studio job-summary type must exist");
const jobSummaryBody = jobSummaryMatch[1];

assert.match(
  jobSummaryBody,
  /command_preview:\s*string;/,
  "the Studio type must require the public job command preview",
);
assert.doesNotMatch(
  jobSummaryBody,
  /\bcommand\??:\s*string;/,
  "the Studio type must not reintroduce the private command field",
);
assert.doesNotMatch(
  jobSummaryBody,
  /\[key:\s*string\]:\s*unknown;/,
  "the Studio type must not permit undeclared private fields through an index signature",
);
assert.match(
  inspector,
  /\{job\.command_preview\s*\|\|\s*job\.job_id\}/,
  "the background-job label must prefer command_preview and retain an id fallback",
);
assert.doesNotMatch(
  inspector,
  /\bjob\.command\b/,
  "the background-job renderer must not consume the private command field",
);

console.log("Public job-summary checks passed (6 assertions).");
