import test from "node:test"
import assert from "node:assert/strict"

import { compactEvent, type Compactor } from "../src/compact.ts"
import { DEFAULTS, type PolicyConfig } from "../src/policy.ts"

const engine: Compactor = { compact: async (text) => `«${text}»` }

function config(overrides: Partial<PolicyConfig> = {}): PolicyConfig {
  return { ...DEFAULTS, keepRecent: 0, minChars: 1, ...overrides }
}

const text = (value: string) => ({ type: "text", text: value })
const withContent = (content: unknown[]) => ({ role: "assistant", content })

test("compacts text parts", async () => {
  const message = withContent([text("hello")])
  await compactEvent({ system: [], messages: [message] }, engine, config(), undefined)
  assert.equal((message.content[0] as { text: string }).text, "«hello»")
})

test("keeps the most recent messages verbatim", async () => {
  const old = withContent([text("old")])
  const recent = withContent([text("recent")])
  await compactEvent({ system: [], messages: [old, recent] }, engine, config({ keepRecent: 1 }), undefined)
  assert.equal((old.content[0] as { text: string }).text, "«old»")
  assert.equal((recent.content[0] as { text: string }).text, "recent")
})

test("compacts tool-result content and string values, never errors", async () => {
  const contentResult = {
    type: "tool-result",
    result: { type: "content", value: [text("log line"), { type: "file", uri: "x" }] },
  }
  const jsonResult = { type: "tool-result", result: { type: "json", value: "{}" } }
  const errorResult = { type: "tool-result", result: { type: "error", value: "boom" } }
  const message = withContent([contentResult, jsonResult, errorResult])

  await compactEvent({ system: [], messages: [message] }, engine, config(), undefined)

  const value = contentResult.result.value as { type: string; text?: string }[]
  assert.equal(value[0].text, "«log line»")
  assert.equal(value[1].type, "file")
  assert.equal(jsonResult.result.value, "«{}»")
  assert.equal(errorResult.result.value, "boom")
})

test("never touches tool-call input or reasoning", async () => {
  const call = { type: "tool-call", id: "1", name: "read", input: { path: "big" } }
  const reasoning = { type: "reasoning", text: "thinking" }
  const message = withContent([call, reasoning])

  await compactEvent({ system: [], messages: [message] }, engine, config(), undefined)

  assert.deepEqual(call.input, { path: "big" })
  assert.equal(reasoning.text, "thinking")
})

test("system parts only compact when compactSystem is set", async () => {
  const part = text("system rules")
  await compactEvent({ system: [part], messages: [] }, engine, config(), undefined)
  assert.equal(part.text, "system rules")

  await compactEvent({ system: [part], messages: [] }, engine, config({ compactSystem: true }), undefined)
  assert.equal(part.text, "«system rules»")
})

test("mode off is a no-op", async () => {
  const message = withContent([text("hello")])
  await compactEvent({ system: [], messages: [message] }, engine, config({ mode: "off" }), undefined)
  assert.equal((message.content[0] as { text: string }).text, "hello")
})

test("size gate and failure guard skip untouched", async () => {
  const short = withContent([text("tiny")])
  const failureText = "ERROR: " + "x".repeat(200)
  const failure = withContent([text(failureText)])
  await compactEvent(
    { system: [], messages: [short, failure] },
    engine,
    config({ minChars: 100 }),
    undefined,
  )
  assert.equal((short.content[0] as { text: string }).text, "tiny")
  assert.equal((failure.content[0] as { text: string }).text, failureText)
})
