import test, { after } from "node:test"
import assert from "node:assert/strict"
import { spawn } from "node:child_process"
import { existsSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

import { Engine } from "../src/engine.ts"

const FAKE_SERVER = `
const net = require("node:net");
const fs = require("node:fs");
const path = process.env.SOCK;
try { fs.unlinkSync(path); } catch {}
const server = net.createServer((socket) => {
  socket.setEncoding("utf8");
  let buffer = "";
  socket.on("data", (chunk) => {
    buffer += chunk;
    let index;
    while ((index = buffer.indexOf("\\n")) >= 0) {
      const line = buffer.slice(0, index);
      buffer = buffer.slice(index + 1);
      const request = JSON.parse(line);
      const prefix = request.meta && request.meta.source ? request.meta.source + ":" : "";
      const response = request.text === "boom"
        ? { id: request.id, error: "nope" }
        : { id: request.id, text: prefix + request.text.toUpperCase() };
      socket.write(JSON.stringify(response) + "\\n");
    }
  });
});
server.listen(path);
`

const socketPath = join(tmpdir(), `context-sift-test-${process.pid}.sock`)
let server: ReturnType<typeof spawn> | undefined

function startServer(): Promise<void> {
  server = spawn(process.execPath, ["-e", FAKE_SERVER], {
    env: { ...process.env, SOCK: socketPath },
    stdio: "ignore",
  })
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + 5000
    const poll = setInterval(() => {
      if (existsSync(socketPath)) {
        clearInterval(poll)
        resolve()
      } else if (Date.now() > deadline) {
        clearInterval(poll)
        reject(new Error("fake server did not start"))
      }
    }, 20)
  })
}

function engine(): Engine {
  return new Engine({ socketPath, spawn: false })
}

await startServer()

after(() => {
  server?.kill()
  rmSync(socketPath, { force: true })
})

test("engine round-trips requests in order", async () => {
  const e = engine()
  await e.start()
  assert.equal(await e.compact("hello"), "HELLO")
  assert.equal(await e.compact("world"), "WORLD")
  e.stop()
})

test("engine forwards per-request metadata", async () => {
  const e = engine()
  await e.start()
  assert.equal(await e.compact("x", { source: "opencode", in_chars: 42 }), "opencode:X")
  e.stop()
})

test("engine surfaces engine-side errors per request", async () => {
  const e = engine()
  await e.start()
  await assert.rejects(() => e.compact("boom"), /nope/)
  assert.equal(await e.compact("still"), "STILL")
  e.stop()
})

test("two engines reuse the same daemon", async () => {
  const a = engine()
  const b = engine()
  await a.start()
  await b.start()
  assert.equal(await a.compact("one"), "ONE")
  assert.equal(await b.compact("two"), "TWO")
  a.stop()
  b.stop()
})

test("compact before start rejects", async () => {
  await assert.rejects(() => engine().compact("x"), /not started/)
})
