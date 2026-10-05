import { Plugin } from "@opencode/plugin"

import { compactEvent } from "./compact.ts"
import { Engine } from "./engine.ts"
import { DEFAULTS, type PolicyConfig } from "./policy.ts"

/**
 * OpenCode V2 plugin: compress the transcript that goes out to the model, using
 * the warm ContextSift daemon. Non-destructive by design — it edits only the
 * outgoing request (the `context` / `compaction` hooks), never persisted history.
 * Tool-call/result pairing and failure output are always preserved.
 */
export default Plugin.define({
  id: "context-sift",

  async setup(ctx) {
    const config = resolveConfig(ctx.options)
    const engine = new Engine({
      command: str(ctx.options.command, "context-sift"),
      args: stringArray(ctx.options.args, ["--serve"]),
      timeoutMs: num(ctx.options.timeoutMs, 30_000),
      onStderr: (line) => console.error(`[context-sift] ${line}`),
    })
    await engine.start()

    const contextLimit = await readContextLimit(ctx)

    const contextRegistration = await ctx.session.hook("context", async (event) => {
      await compactEvent(event, engine, config, contextLimit)
    })
    const compactionRegistration = await ctx.session.hook("compaction", async (event) => {
      await compactEvent(event, engine, config, contextLimit)
    })

    return async () => {
      await contextRegistration.dispose()
      await compactionRegistration.dispose()
      await engine.stop()
    }
  },
})

type ModelContext = {
  model: {
    list(): Promise<readonly { providerID: string; id: string; limit?: { context?: number } }[]>
    default(): Promise<{ providerID: string; modelID: string } | undefined>
  }
}

async function readContextLimit(ctx: ModelContext): Promise<number | undefined> {
  try {
    const selected = await ctx.model.default()
    if (!selected) return undefined
    const models = await ctx.model.list()
    const current = models.find(
      (model) => model.providerID === selected.providerID && model.id === selected.modelID,
    )
    return current?.limit?.context
  } catch {
    return undefined
  }
}

function resolveConfig(options: Record<string, unknown>): PolicyConfig {
  const config: PolicyConfig = { ...DEFAULTS }
  if (options.mode === "off" || options.mode === "auto" || options.mode === "aggressive") {
    config.mode = options.mode
  }
  config.minChars = num(options.minChars, config.minChars)
  config.keepRecent = num(options.keepRecent, config.keepRecent)
  config.budgetRatio = num(options.budgetRatio, config.budgetRatio)
  config.maxChars = num(options.maxChars, config.maxChars)
  if (typeof options.compactSystem === "boolean") config.compactSystem = options.compactSystem
  return config
}

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

function str(value: unknown, fallback: string): string {
  return typeof value === "string" && value.length > 0 ? value : fallback
}

function stringArray(value: unknown, fallback: string[]): string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string") ? value : fallback
}
