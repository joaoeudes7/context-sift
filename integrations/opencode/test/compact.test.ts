import test from "node:test"
import assert from "node:assert/strict"

import { compactEvent, type Compactor } from "../src/compact.ts"
import { DEFAULTS, type PolicyConfig } from "../src/policy.ts"

const engine: Compactor = { compact: async () => "x" }

function config(overrides: Partial<PolicyConfig> = {}): PolicyConfig {
  return { ...DEFAULTS, keepRecent: 0, minChars: 1, ...overrides }
}

const text = (value: string) => ({ type: "text", text: value })
const withContent = (content: unknown[]) => ({ role: "assistant", content })

test("compacts text parts", async () => {
  const message = withContent([text("hello")])
  await compactEvent({ system: [], messages: [message] }, engine, config(), undefined)
  assert.equal((message.content[0] as { text: string }).text, "x")
})

test("does not resend unchanged or already compacted text", async () => {
  let calls = 0
  const spy: Compactor = {
    compact: async (value) => {
      calls++
      return "x"
    },
  }
  const cache = new Map<string, Promise<string>>()
  const first = withContent([text("same context")])
  const fresh = withContent([text("same context")])

  await compactEvent({ system: [], messages: [first] }, spy, config(), undefined, {}, cache)
  await compactEvent({ system: [], messages: [fresh] }, spy, config(), undefined, {}, cache)
  await compactEvent({ system: [], messages: [first] }, spy, config(), undefined, {}, cache)

  assert.equal(calls, 1)
  assert.equal((fresh.content[0] as { text: string }).text, "x")
})

test("keeps original when compaction would not save tokens", async () => {
  const message = withContent([text("same")])
  const noSaving: Compactor = { compact: async () => "longer output" }
  await compactEvent({ system: [], messages: [message] }, noSaving, config(), undefined)
  assert.equal((message.content[0] as { text: string }).text, "same")
})

test("shares a cache entry for different messages that trim to one payload", async () => {
  let calls = 0
  const spy: Compactor = {
    compact: async () => {
      calls++
      return "x"
    },
  }
  const cache = new Map<string, Promise<string>>()
  // With maxChars 4 the payload keeps 2 head + 2 tail chars, so both messages
  // trim to the same string even though their originals differ.
  const first = withContent([text(`ab${"1".repeat(6)}yz`)])
  const second = withContent([text(`ab${"2".repeat(6)}yz`)])

  await compactEvent({ system: [], messages: [first] }, spy, config({ maxChars: 4 }), undefined, {}, cache)
  await compactEvent({ system: [], messages: [second] }, spy, config({ maxChars: 4 }), undefined, {}, cache)

  assert.equal(calls, 1)
  assert.equal((second.content[0] as { text: string }).text, "x")
})

test("never hands a message another message's text when it keeps the original", async () => {
  const cache = new Map<string, Promise<string>>()
  const noSaving: Compactor = { compact: async () => "longer output" }
  const first = withContent([text(`ab${"1".repeat(6)}yz`)])
  const second = withContent([text(`ab${"2".repeat(6)}yz`)])

  await compactEvent({ system: [], messages: [first] }, noSaving, config({ maxChars: 4 }), undefined, {}, cache)
  await compactEvent({ system: [], messages: [second] }, noSaving, config({ maxChars: 4 }), undefined, {}, cache)

  assert.equal((first.content[0] as { text: string }).text, `ab${"1".repeat(6)}yz`)
  assert.equal((second.content[0] as { text: string }).text, `ab${"2".repeat(6)}yz`)
})

test("keeps the most recent messages verbatim", async () => {
  const old = withContent([text("old")])
  const recent = withContent([text("recent")])
  await compactEvent({ system: [], messages: [old, recent] }, engine, config({ keepRecent: 1 }), undefined)
  assert.equal((old.content[0] as { text: string }).text, "old")
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
  assert.equal(value[0].text, "x")
  assert.equal(value[1].type, "file")
  assert.equal(jsonResult.result.value, "{}")
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
  assert.equal(part.text, "x")
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

test("engine failure passes the payload through untouched", async () => {
  const failing: Compactor = {
    compact: async () => {
      throw new Error("engine down")
    },
  }
  const message = withContent([text("keep me exactly")])
  const originalError = console.error
  console.error = () => {}
  try {
    await compactEvent({ system: [], messages: [message] }, failing, config(), undefined)
  } finally {
    console.error = originalError
  }
  assert.equal((message.content[0] as { text: string }).text, "keep me exactly")
})

test("budget gate leaves everything untouched when the context is small", async () => {
  const message = withContent([text("hello")])
  await compactEvent(
    { system: [], messages: [message] },
    engine,
    config({ budgetRatio: 0.5 }),
    1000,
  )
  assert.equal((message.content[0] as { text: string }).text, "hello")
})

test("compacts tool-result text values", async () => {
  const part = { type: "tool-result", result: { type: "text", value: "bash log" } }
  await compactEvent({ system: [], messages: [withContent([part])] }, engine, config(), undefined)
  assert.equal((part.result as { value: string }).value, "x")
})

test("reports the original size and trims before the engine", async () => {
  const seen: { text: string; meta?: Record<string, unknown> }[] = []
  const spy: Compactor = {
    compact: async (value, meta) => {
      seen.push({ text: value, meta })
      return "x"
    },
  }
  const message = withContent([text("abcdefghij")])
  await compactEvent({ system: [], messages: [message] }, spy, config({ maxChars: 4 }), undefined)

  assert.equal(seen[0].meta?.in_chars, 10) // original, before the head/tail trim
  assert.ok(seen[0].text.includes("chars trimmed"))
  assert.ok(seen[0].text.startsWith("ab"))
})
