import test from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

// Guard: ContextSift must only ever mutate the OUTGOING request payload.
// It must never intercept or rewrite a model response.
const SRC = join(dirname(fileURLToPath(import.meta.url)), "..", "src")
const read = (file: string): string => readFileSync(join(SRC, file), "utf8")

test("plugin registers only the outgoing-request session hooks", () => {
  const names = [...read("index.ts").matchAll(/\.hook\(\s*"([^"]+)"/g)].map((match) => match[1])
  assert.deepEqual(names.sort(), ["compaction", "context"])
})

test("plugin never touches model responses", () => {
  const source = read("index.ts") + read("compact.ts")
  assert.doesNotMatch(source, /http\.(response|request)/)
  assert.doesNotMatch(source, /experimental\.ws/)
  assert.doesNotMatch(source, /\.result\s*=\s*(?!=)/) // e.g. event.result = <summary>
})
