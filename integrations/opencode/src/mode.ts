import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs"
import { homedir } from "node:os"
import { dirname, join } from "node:path"

// Runtime mode override shared by the server plugin (reads it per request) and
// the TUI command `/sift <mode>` (writes it). Kept out of config so it can change
// without editing opencode.json or reloading.

export const MODES = ["off", "auto", "max"] as const
export type Mode = (typeof MODES)[number]

export function modeFile(): string {
  // Deterministic per-user path so the OpenCode server and the TUI always agree,
  // even when one process does not inherit XDG_DATA_HOME. Override for tests.
  const override = process.env.CONTEXT_SIFT_MODE_FILE
  if (override) return override
  return join(homedir(), ".local", "share", "context-sift", "mode")
}

let cache: { mtime: bigint; value: string } = { mtime: -1n, value: "" }

/** The override mode, or "" when unset/invalid. Cached by nanosecond mtime. */
export function readMode(): string {
  try {
    const mtime = statSync(modeFile(), { bigint: true }).mtimeNs
    if (mtime !== cache.mtime) {
      cache = { mtime, value: readFileSync(modeFile(), "utf8").trim() }
    }
  } catch {
    cache = { mtime: -1n, value: "" }
  }
  return (MODES as readonly string[]).includes(cache.value) ? cache.value : ""
}

export function writeMode(mode: string): void {
  mkdirSync(dirname(modeFile()), { recursive: true })
  writeFileSync(modeFile(), mode + "\n")
}
