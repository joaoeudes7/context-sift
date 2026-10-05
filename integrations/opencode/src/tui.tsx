import { execFile } from "node:child_process"
import { promisify } from "node:util"

import { Plugin } from "@opencode/plugin/tui"

// CLI plugin: a persistent ContextSift savings line in the TUI footer.
// Reads `context-sift gain --oneline` (no model load — the gain command never
// imports torch), refreshed after each completed session execution and once on
// load. See the main README's *Savings report*.

const run = promisify(execFile)

async function readGainLine(): Promise<string> {
  try {
    const { stdout } = await run("context-sift", ["gain", "--oneline"], { timeout: 5000 })
    return stdout.trim() || "sift  no data"
  } catch {
    return "sift  unavailable"
  }
}

export default Plugin.define({
  id: "context-sift.cli",
  async setup(context) {
    const [state, update] = context.storage.memory("gain", {
      initial: { line: "sift  …" },
    })

    const refresh = async (): Promise<void> => {
      const line = await readGainLine()
      update((draft) => {
        draft.line = line
      })
    }

    const render = () => <text>{state.line}</text>
    context.ui.slot({ append: "home.footer.status", render })
    context.ui.slot({ append: "prompt.footer.status", render })

    void refresh()
    const stop = context.data.on("session.execution.succeeded", () => void refresh())

    return () => {
      stop()
    }
  },
})
