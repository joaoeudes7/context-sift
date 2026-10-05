import test from "node:test"
import assert from "node:assert/strict"

import { Engine } from "../src/engine.ts"

const FAKE_SERVER = `
const readline = require("node:readline");
const rl = readline.createInterface({ input: process.stdin });
process.stdout.write(JSON.stringify({ ready: true }) + "\\n");
rl.on("line", (line) => {
  const q = JSON.parse(line);
  if (q.text === "boom") {
    process.stdout.write(JSON.stringify({ id: q.id, error: "nope" }) + "\\n");
  } else {
    process.stdout.write(JSON.stringify({ id: q.id, text: q.text.toUpperCase() }) + "\\n");
  }
});
`

function engine(): Engine {
  return new Engine({ command: "node", args: ["-e", FAKE_SERVER] })
}

test("engine round-trips requests in order", async () => {
  const e = engine()
  await e.start()
  assert.equal(await e.compact("hello"), "HELLO")
  assert.equal(await e.compact("world"), "WORLD")
  await e.stop()
})

test("engine surfaces engine-side errors per request", async () => {
  const e = engine()
  await e.start()
  await assert.rejects(() => e.compact("boom"), /nope/)
  // the engine keeps serving after an error line
  assert.equal(await e.compact("still"), "STILL")
  await e.stop()
})

test("compact before start rejects", async () => {
  await assert.rejects(() => engine().compact("x"), /not started/)
})
