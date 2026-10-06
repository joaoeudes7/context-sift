export type Mode = "off" | "auto" | "max"

export interface PolicyConfig {
  /** `off` = never touch; `auto` = gate by size + budget; `max` = compact everything eligible. */
  mode: Mode
  /** Minimum characters before a payload is worth compressing. */
  minChars: number
  /** Most recent messages kept verbatim, never compacted. */
  keepRecent: number
  /** Compact only when estimated context tokens exceed `budgetRatio * contextLimit`. `0` disables the gate. */
  budgetRatio: number
  /** Payloads larger than this are head/tail trimmed before hitting the engine. */
  maxChars: number
  /** Compact system instructions too. Off by default: they are usually load-bearing. */
  compactSystem: boolean
}

export const DEFAULTS: PolicyConfig = {
  mode: "auto",
  minChars: 2000,
  keepRecent: 6,
  budgetRatio: 0,
  maxChars: 100_000,
  compactSystem: false,
}

/** ponytail: chars/4 token heuristic; swap for a real tokenizer only if budget accuracy matters. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

const FAILURE_SIGNAL =
  /\b(?:error|failed|failure|fatal|panic|traceback|exception|warn(?:ing)?)\b|✗|FAILED|Segmentation fault/i

/** Failures and warnings are never safe to drop; ContextSift preserves them, and so must the gate. */
export function isFailure(text: string): boolean {
  return FAILURE_SIGNAL.test(text)
}

export function shouldConsider(text: string, config: PolicyConfig): boolean {
  if (config.mode === "off") return false
  if (text.length < config.minChars) return false
  return !isFailure(text)
}

/** The "when" gate: only intrude once the context is actually getting full. */
export function budgetOpen(
  totalTokens: number,
  contextLimit: number | undefined,
  config: PolicyConfig,
): boolean {
  if (config.mode === "max") return true
  if (config.budgetRatio <= 0) return true
  if (!contextLimit || contextLimit <= 0) return true
  return totalTokens > contextLimit * config.budgetRatio
}

/** Cap engine latency on very large payloads by keeping both ends, dropping the middle. */
export function trimHeadTail(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text
  const head = Math.floor(maxChars * 0.6)
  const tail = maxChars - head
  const removed = text.length - maxChars
  return `${text.slice(0, head)}\n…[context-sift: ${removed} chars trimmed]…\n${text.slice(-tail)}`
}
