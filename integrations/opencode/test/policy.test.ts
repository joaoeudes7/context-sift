import test from "node:test"
import assert from "node:assert/strict"

import {
  DEFAULTS,
  budgetOpen,
  estimateTokens,
  isFailure,
  shouldConsider,
  trimHeadTail,
} from "../src/policy.ts"

test("estimateTokens uses the chars/4 heuristic", () => {
  assert.equal(estimateTokens(""), 0)
  assert.equal(estimateTokens("abcd"), 1)
  assert.equal(estimateTokens("abcde"), 2)
})

test("isFailure flags errors and warnings", () => {
  assert.equal(isFailure("Traceback (most recent call last):"), true)
  assert.equal(isFailure("FAILED tests/test_x.py::test_y"), true)
  assert.equal(isFailure("Segmentation fault"), true)
  assert.equal(isFailure("all checks passed, 42 files"), false)
})

test("shouldConsider gates by mode, size, and failures", () => {
  const long = "x".repeat(3000)
  assert.equal(shouldConsider(long, DEFAULTS), true)
  assert.equal(shouldConsider("short", DEFAULTS), false)
  assert.equal(shouldConsider(long, { ...DEFAULTS, mode: "off" }), false)
  assert.equal(shouldConsider(`ERROR ${long}`, DEFAULTS), false)
})

test("budgetOpen respects mode, ratio, and unknown limits", () => {
  const base = { ...DEFAULTS, budgetRatio: 0.6 }
  assert.equal(budgetOpen(10, 100, { ...base, budgetRatio: 0 }), true)
  assert.equal(budgetOpen(50, 100, base), false)
  assert.equal(budgetOpen(61, 100, base), true)
  assert.equal(budgetOpen(10, undefined, base), true)
  assert.equal(budgetOpen(10, 100, { ...base, mode: "aggressive" }), true)
})

test("trimHeadTail keeps both ends and drops the middle", () => {
  const text = "A".repeat(500) + "MIDDLE" + "B".repeat(500)
  assert.equal(trimHeadTail("small", 100), "small")

  const trimmed = trimHeadTail(text, 200)
  assert.ok(trimmed.startsWith("A"))
  assert.ok(trimmed.endsWith("B"))
  assert.ok(trimmed.includes(`${text.length - 200} chars trimmed`))
  assert.ok(!trimmed.includes("MIDDLE"))
})
