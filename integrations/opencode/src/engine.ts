import { spawn } from "node:child_process"
import net from "node:net"
import { tmpdir } from "node:os"
import { join } from "node:path"

export interface EngineOptions {
  /** Unix socket the shared daemon listens on. Defaults to a per-user path in the temp dir. */
  socketPath?: string
  /** Executable used to spawn the daemon when it is absent. Defaults to `context-sift`. */
  command?: string
  /** Arguments used to spawn the daemon. Defaults to `--serve-socket <path> --idle-timeout <n>`. */
  args?: string[]
  /** Seconds the daemon stays alive with no client. Defaults to 60. */
  idleTimeout?: number
  /** Per-request timeout in ms. Defaults to 30s. */
  timeoutMs?: number
  /** Set false to connect only (never spawn). Used by tests. */
  spawn?: boolean
}

interface Pending {
  resolve: (text: string) => void
  reject: (error: Error) => void
  timer: ReturnType<typeof setTimeout>
}

export function defaultSocketPath(): string {
  const uid = typeof process.getuid === "function" ? process.getuid() : "0"
  return join(tmpdir(), `context-sift-${uid}.sock`)
}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/**
 * Client for the single shared ContextSift daemon. Connects to its Unix socket;
 * spawns one detached daemon only when the socket is absent. Never owns the
 * daemon's lifetime — it exits on its own after an idle period, so every
 * plugin runtime and CLI invocation can connect to the same warm process.
 */
export class Engine {
  private socket?: net.Socket
  private buffer = ""
  private nextId = 1
  private readonly pending = new Map<number, Pending>()
  private started?: Promise<void>
  private disposed = false
  private readonly options: EngineOptions
  private readonly socketPath: string

  constructor(options: EngineOptions = {}) {
    this.options = options
    this.socketPath = options.socketPath ?? defaultSocketPath()
  }

  start(): Promise<void> {
    if (!this.started) this.started = this.connectOrSpawn()
    return this.started
  }

  compact(text: string): Promise<string> {
    const socket = this.socket
    if (!socket || this.disposed) return Promise.reject(new Error("engine not started"))
    const id = this.nextId++
    const timeoutMs = this.options.timeoutMs ?? 30_000
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`context-sift timed out after ${timeoutMs}ms`))
      }, timeoutMs)
      this.pending.set(id, { resolve, reject, timer })
      socket.write(JSON.stringify({ id, text }) + "\n")
    })
  }

  stop(): void {
    this.disposed = true
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer)
      reject(new Error("engine stopped"))
    }
    this.pending.clear()
    this.socket?.end()
    this.socket?.destroy()
    this.socket = undefined
  }

  private async connectOrSpawn(): Promise<void> {
    if (await this.tryConnect()) return
    if (this.options.spawn === false) throw new Error(`no context-sift daemon at ${this.socketPath}`)

    const args = this.options.args ?? [
      "--serve-socket",
      this.socketPath,
      "--idle-timeout",
      String(this.options.idleTimeout ?? 60),
    ]
    const child = spawn(this.options.command ?? "context-sift", args, {
      detached: true,
      stdio: "ignore",
    })
    child.unref()

    const deadline = Date.now() + 15_000
    while (Date.now() < deadline) {
      await delay(50)
      if (await this.tryConnect()) return
    }
    throw new Error(`context-sift daemon did not start at ${this.socketPath}`)
  }

  private tryConnect(): Promise<boolean> {
    return new Promise((resolve) => {
      const socket = net.connect(this.socketPath)
      const onError = (): void => {
        socket.destroy()
        resolve(false)
      }
      socket.once("error", onError)
      socket.once("connect", () => {
        socket.off("error", onError)
        socket.setEncoding("utf8")
        socket.on("data", (chunk: string) => this.onData(chunk))
        socket.on("error", (error) => this.failAll(error))
        socket.on("close", () => {
          if (!this.disposed) this.failAll(new Error("context-sift daemon connection closed"))
        })
        this.socket = socket
        resolve(true)
      })
    })
  }

  private onData(chunk: string): void {
    this.buffer += chunk
    let index: number
    while ((index = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, index).trim()
      this.buffer = this.buffer.slice(index + 1)
      if (line) this.onLine(line)
    }
  }

  private onLine(line: string): void {
    let message: { id?: number; text?: string; error?: string }
    try {
      message = JSON.parse(line)
    } catch {
      return
    }
    if (typeof message.id !== "number") return
    const pending = this.pending.get(message.id)
    if (!pending) return
    this.pending.delete(message.id)
    clearTimeout(pending.timer)
    if (message.error) pending.reject(new Error(message.error))
    else pending.resolve(message.text ?? "")
  }

  private failAll(error: Error): void {
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer)
      reject(error)
    }
    this.pending.clear()
  }
}
