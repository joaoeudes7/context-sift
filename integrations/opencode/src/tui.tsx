import { execFile } from "node:child_process"
import { promisify } from "node:util"

import { Plugin } from "@opencode/plugin/tui"
import { MODES, readMode, writeMode } from "./mode.ts"

// CLI plugin: a ContextSift savings line in the TUI footer, scoped to the active
// session/conversation (falls back to the global total when none is in view),
// plus a `/sift <mode>` command to switch modes at runtime.
// Reads `context-sift gain --oneline [--session <id>]` (no model load — the gain
// command never imports torch), refreshed after each completed session execution.

const run = promisify(execFile)
const PLACEHOLDER = "sift  …"

async function gain(args: string[]): Promise<string> {
  try {
    const { stdout } = await run("context-sift", args, { timeout: 5000 })
    return stdout.trim()
  } catch {
    return ""
  }
}

async function readGainLine(sessionID: string): Promise<string> {
  const args = sessionID ? ["gain", "--oneline", "--session", sessionID] : ["gain", "--oneline"]
  // A fresh conversation legitimately has no rows yet → "no data".
  return (await gain(args)) || "sift  unavailable"
}

function activeSession(context: { ui: { router: { current(): unknown } } }, props?: unknown): string {
  // Slots pass `sessionID`; the route is the authoritative fallback.
  const p = props as { sessionID?: string; session_id?: string } | undefined
  const fromProps = p?.sessionID ?? p?.session_id
  if (fromProps) return fromProps
  const route = context.ui.router.current() as
    | { sessionID?: string; session_id?: string; params?: { sessionID?: string; session_id?: string } }
    | undefined
  return route?.params?.sessionID ?? route?.params?.session_id ?? route?.sessionID ?? route?.session_id ?? ""
}

export default Plugin.define({
  id: "context-sift.cli",
  async setup(context) {
    // Bumped key: memory survives plugin reloads, and the old shape was { line }.
    const [state, update] = context.storage.memory("gain.v2", {
      initial: { byId: {} as Record<string, string>, global: PLACEHOLDER },
    })
    const inflight = new Set<string>()
    let currentSession = ""

    const refresh = async (sessionID: string): Promise<void> => {
      const line = await readGainLine(sessionID)
      update((draft) => {
        if (sessionID) draft.byId[sessionID] = line
        else draft.global = line
      })
    }

    // Fetch a scope once; the execution event refreshes it afterwards.
    const ensure = (sessionID: string): void => {
      const known = sessionID ? state.byId?.[sessionID] : state.global
      if ((known && known !== PLACEHOLDER) || inflight.has(sessionID)) return
      inflight.add(sessionID)
      void refresh(sessionID).finally(() => inflight.delete(sessionID))
    }

    const render = (props?: unknown) => {
      currentSession = activeSession(context, props)
      ensure(currentSession)
      return (
        <text fg={context.theme.text.base}>
          {currentSession ? state.byId?.[currentSession] ?? PLACEHOLDER : state.global}
        </text>
      )
    }

    context.ui.slot({ append: "home.footer.status", render })
    context.ui.slot({ append: "prompt.footer.status", render })

    let keymapWarned = false
    try {
      context.keymap.layer(() => ({
        mode: "global",
        commands: [
          {
            id: "context-sift.info",
            title: "ContextSift: status & modes",
            group: "ContextSift",
            palette: true,
            slash: { name: "sift", aliases: ["gain"], arguments: true },
            run: async (input?: string) => {
              const options = (context.options ?? {}) as Record<string, unknown>
              const fallback = typeof options.mode === "string" ? options.mode : "auto"
              const mode = (input ?? "").trim().toLowerCase()

              if ((MODES as readonly string[]).includes(mode)) {
                try {
                  writeMode(mode)
                  context.ui.toast.show({ message: `ContextSift mode: ${mode}`, variant: "success" })
                  void refresh(currentSession)
                } catch (error) {
                  context.ui.toast.show({ message: `ContextSift: ${error}`, variant: "error" })
                }
                return
              }

              const active = readMode() || fallback
              const summaries = active === "max" || options.compactSummaries === true
              const line = await readGainLine(currentSession)
              await context.ui.dialog.alert({
                title: "ContextSift",
                message: [
                  line,
                  `mode: ${active}${summaries ? " (summaries on)" : ""}`,
                  `conversation: ${currentSession || "(none in view)"}`,
                  "",
                  "Change: /sift off | /sift auto | /sift max",
                  "max also compacts OpenCode's summarizer input.",
                ].join("\n"),
              })
            },
          },
        ],
        bindings: ["context-sift.info"],
      }))
    } catch (error) {
      // Headless CLI runs have no Keymap provider; the footer still works.
      if (!keymapWarned) {
        keymapWarned = true
        console.warn(`[context-sift] keymap unavailable: ${error}`)
      }
    }

    ensure("")
    ensure(currentSession)
    const stop = context.data.on("session.execution.succeeded", () => {
      void refresh(currentSession)
      void refresh("")
    })

    return () => {
      stop()
    }
  },
})
