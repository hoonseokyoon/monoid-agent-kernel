import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import ts from "typescript";

const source = await readFile(new URL("../src/lib/diff.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
  fileName: "diff.ts",
});
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled.outputText).toString("base64")}`;
const { fileDiff } = await import(moduleUrl);

const plainDiff = [
  "--- a/one.txt",
  "+++ b/one.txt",
  "@@ -1 +1 @@",
  "-old one",
  "+new one",
  "--- a/two.txt",
  "+++ b/two.txt",
  "@@ -1 +1 @@",
  "-old two",
  "+new two",
  "",
].join("\n");
const one = fileDiff(plainDiff, "one.txt");
const two = fileDiff(plainDiff, "two.txt");
assert.match(one, /new one/);
assert.doesNotMatch(one, /two\.txt|new two/);
assert.match(two, /new two/);
assert.doesNotMatch(two, /one\.txt|new one/);

const gitDiff = [
  "diff --git a/keep.txt b/keep.txt",
  "--- a/keep.txt",
  "+++ b/keep.txt",
  "@@ -1 +1 @@",
  "-before",
  "+after",
  "diff --git a/gone.txt b/gone.txt",
  "--- a/gone.txt",
  "+++ /dev/null",
  "@@ -1 +0,0 @@",
  "-gone",
  "",
].join("\n");
assert.match(fileDiff(gitDiff, "keep.txt"), /\+after/);
assert.doesNotMatch(fileDiff(gitDiff, "keep.txt"), /gone\.txt/);
assert.match(fileDiff(gitDiff, "gone.txt"), /\+\+\+ \/dev\/null/);
assert.equal(fileDiff(plainDiff, "missing.txt"), null);

console.log("Per-file diff checks passed (7 assertions).");
