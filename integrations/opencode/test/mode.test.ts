import test, { after } from "node:test"
import assert from "node:assert/strict"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

import { readMode, writeMode } from "../src/mode.ts"

const dir = mkdtempSync(join(tmpdir(), "cs-mode-"))
process.env.CONTEXT_SIFT_MODE_FILE = join(dir, "mode")

after(() => rmSync(dir, { recursive: true, force: true }))

test("readMode is empty before any write", () => {
  assert.equal(readMode(), "")
})

test("writeMode then readMode round-trips and toggles", () => {
  writeMode("max")
  assert.equal(readMode(), "max")
  writeMode("off") // must be seen despite a same-millisecond write
  assert.equal(readMode(), "off")
})

test("invalid stored values are ignored", () => {
  writeMode("bogus")
  assert.equal(readMode(), "")
})
