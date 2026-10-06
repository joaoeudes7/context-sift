import {
  budgetOpen,
  estimateTokens,
  shouldConsider,
  trimHeadTail,
  type PolicyConfig,
} from "./policy.ts"

/** Anything that can compact a string, e.g. `Engine`. */
export interface Compactor {
  compact(text: string, meta?: Record<string, unknown>): Promise<string>
}

/**
 * Shape of the OpenCode V2 request hook payload, narrowed to what we touch.
 * Confirmed against `@opencode/ai@2.0.22`:
 *   Message.content: ContentPart[] (never a string)
 *   ContentPart = text | media | tool-call | tool-result | reasoning | compaction | effort
 *   ToolResultValue = json | text | error | content (content = array of text|file)
 */
interface ContentPartLike {
  type?: string
  text?: string
  result?: { type?: string; value?: unknown }
}

interface MessageLike {
  content?: unknown
}

interface CompactibleEvent {
  system: readonly unknown[]
  messages: readonly unknown[]
}

export async function compactEvent(
  event: CompactibleEvent,
  engine: Compactor,
  config: PolicyConfig,
  contextLimit: number | undefined,
  meta: Record<string, unknown> = {},
): Promise<void> {
  if (config.mode === "off") return

  const totalTokens = estimateTokens(collectText(event.system, event.messages))
  if (!budgetOpen(totalTokens, contextLimit, config)) return

  if (config.compactSystem) {
    for (const part of event.system) await compactTextField(asPart(part), engine, config, meta)
  }

  // Keep the most recent messages verbatim: they carry the live task.
  const keep = Math.max(0, event.messages.length - config.keepRecent)
  for (let index = 0; index < keep; index++) {
    await compactMessage(event.messages[index], engine, config, meta)
  }
}

async function compactMessage(
  message: unknown,
  engine: Compactor,
  config: PolicyConfig,
  meta: Record<string, unknown>,
): Promise<void> {
  const content = (message as MessageLike | null)?.content
  if (!Array.isArray(content)) return
  for (const part of content) await compactContentPart(asPart(part), engine, config, meta)
}

async function compactContentPart(
  part: ContentPartLike | null,
  engine: Compactor,
  config: PolicyConfig,
  meta: Record<string, unknown>,
): Promise<void> {
  if (!part) return
  switch (part.type) {
    case "text":
      await compactTextField(part, engine, config, meta)
      return
    case "tool-result":
      await compactToolResult(part, engine, config, meta)
      return
    default:
      // media, tool-call (structured input), reasoning, compaction, effort: untouched.
      // Tool-call inputs are model-authored arguments; rewriting them risks tool semantics.
      return
  }
}

async function compactTextField(
  part: ContentPartLike | null,
  engine: Compactor,
  config: PolicyConfig,
  meta: Record<string, unknown>,
): Promise<void> {
  if (!part || part.type !== "text" || typeof part.text !== "string") return
  const compacted = await compactString(part.text, engine, config, meta)
  if (compacted !== undefined) part.text = compacted
}

async function compactToolResult(
  part: ContentPartLike,
  engine: Compactor,
  config: PolicyConfig,
  meta: Record<string, unknown>,
): Promise<void> {
  const result = part.result
  if (!result || typeof result !== "object") return
  if (result.type === "error") return // failures are never touched

  if (result.type === "text" || result.type === "json") {
    if (typeof result.value === "string") {
      const compacted = await compactString(result.value, engine, config, meta)
      if (compacted !== undefined) result.value = compacted
    }
    return
  }
  if (result.type === "content" && Array.isArray(result.value)) {
    for (const entry of result.value) {
      const item = entry as { type?: string; text?: string } | null
      if (item && item.type === "text" && typeof item.text === "string") {
        const compacted = await compactString(item.text, engine, config, meta)
        if (compacted !== undefined) item.text = compacted
      }
    }
  }
}

async function compactString(
  text: string,
  engine: Compactor,
  config: PolicyConfig,
  meta: Record<string, unknown>,
): Promise<string | undefined> {
  if (!shouldConsider(text, config)) return undefined
  try {
    // `in_chars` is the pre-trim size, so the ledger's saving is honest.
    return await engine.compact(trimHeadTail(text, config.maxChars), {
      ...meta,
      in_chars: text.length,
    })
  } catch (error) {
    console.error(`[context-sift] compaction failed, passing through: ${error}`)
    return undefined
  }
}

function collectText(system: readonly unknown[], messages: readonly unknown[]): string {
  const texts: string[] = []
  const push = (value: unknown): void => {
    if (typeof value === "string") texts.push(value)
  }
  for (const part of system) push(asPart(part)?.text)
  for (const message of messages) {
    const content = (message as MessageLike | null)?.content
    if (!Array.isArray(content)) continue
    for (const raw of content) {
      const part = asPart(raw)
      if (!part) continue
      push(part.text)
      const result = part.result
      if (part.type === "tool-result" && result && typeof result === "object") {
        push(result.value)
        if (Array.isArray(result.value)) {
          for (const entry of result.value) push((entry as { text?: unknown } | null)?.text)
        }
      }
    }
  }
  return texts.join(" ")
}

function asPart(value: unknown): ContentPartLike | null {
  return value && typeof value === "object" ? (value as ContentPartLike) : null
}
