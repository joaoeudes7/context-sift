import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process"

export interface EngineOptions {
  /** Executable to run. Defaults to `context-sift` on PATH. */
  command?: string
  /** Arguments. Defaults to `["--serve"]`. */
  args?: string[]
  cwd?: string
  /** Per-request timeout in ms. Defaults to 30s. */
  timeoutMs?: number
  /** Optional log sink for engine stderr. */
  onStderr?: (line: string) => void
}

interface Pending {
  resolve: (text: string) => void
  reject: (error: Error) => void
  timer: ReturnType<typeof setTimeout>
}

/**
 * Supervises one warm `context-sift --serve` child and speaks its JSON-lines
 * protocol: `{id, text}` out, `{id, text|error}` back, one `{ready:true}` after
 * the model loads. Calls are serialized by the child (one model, one lock).
 */
export class Engine {
  private child?: ChildProcessWithoutNullStreams
  private buffer = ""
  private nextId = 1
  private readonly pending = new Map<number, Pending>()
  private started?: Promise<void>
  private markReady?: () => void
  private markFailed?: (error: Error) => void
  private disposed = false
  private readonly options: EngineOptions

  constructor(options: EngineOptions = {}) {
    this.options = options
  }

  start(): Promise<void> {
    if (this.started) return this.started
    const command = this.options.command ?? "context-sift"
    const args = this.options.args ?? ["--serve"]
    this.started = new Promise<void>((resolve, reject) => {
      this.markReady = resolve
      this.markFailed = reject
    })

    const child = spawn(command, args, { cwd: this.options.cwd, stdio: ["pipe", "pipe", "pipe"] })
    this.child = child
    child.stdout.setEncoding("utf8")
    child.stdout.on("data", (chunk: string) => this.onData(chunk))
    child.stderr.setEncoding("utf8")
    child.stderr.on("data", (chunk: string) => this.options.onStderr?.(chunk.trimEnd()))
    child.on("error", (error) => this.failAll(error))
    child.on("exit", (code) => {
      if (!this.disposed) this.failAll(new Error(`context-sift exited with code ${code}`))
    })
    return this.started
  }

  compact(text: string): Promise<string> {
    const child = this.child
    if (!child) return Promise.reject(new Error("engine not started"))
    const id = this.nextId++
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`context-sift timed out after ${this.options.timeoutMs ?? 30_000}ms`))
      }, this.options.timeoutMs ?? 30_000)
      this.pending.set(id, { resolve, reject, timer })
      child.stdin.write(JSON.stringify({ id, text }) + "\n")
    })
  }

  async stop(): Promise<void> {
    this.disposed = true
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer)
      reject(new Error("engine stopped"))
    }
    this.pending.clear()
    const child = this.child
    this.child = undefined
    if (!child) return
    child.stdin.end()
    child.kill()
    await new Promise<void>((resolve) => child.once("exit", () => resolve()))
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
    let message: { ready?: boolean; id?: number; text?: string; error?: string }
    try {
      message = JSON.parse(line)
    } catch {
      return
    }
    if (message.ready) {
      this.markReady?.()
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
    this.markFailed?.(error)
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer)
      reject(error)
    }
    this.pending.clear()
  }
}
