# context-sift-opencode

OpenCode V2 plugin that compacts the transcript sent to the model, using a
shared [ContextSift](../..) daemon. It reduces tokens per request — and
therefore how often and how large context compaction has to be.

The plugin is the **decision layer** ("when/what to compress"). ContextSift is
the **compression engine** (extractive, 660K-param ML model).

## Install

1. Install the engine so `context-sift` is on `PATH`:

   ```sh
   pip install context-sift
   ```

2. Install the plugin as an auto-discovered local file:

   ```sh
   ./install.sh
   opencode reload
   ```

   (`opencode plugin add` only accepts npm/Git specs, so a local checkout goes
   through `~/.config/opencode/plugins/` auto-discovery.)

## Lifecycle — one warm daemon, shared

Every plugin runtime and every `opencode` CLI invocation talks to **one**
engine process over a Unix socket (`<tmpdir>/context-sift-<uid>.sock`):

- A client connects; if no daemon is listening it spawns one (detached) and waits.
- A `flock` start-lock plus a socket liveness probe guarantee exactly one process
  loads the model — concurrent spawns become waiters and exit once the socket is up.
- The daemon stays warm while any client is connected, then exits
  `idleTimeout` seconds after the last client leaves (default 60) and unlinks
  the socket. The next request re-spawns it.
- It does **not** die per request — that would forfeit model reuse.

## How it works

- Registers the `context` session hook (the agent-loop request). `compaction` is
  opt-in (`compactSummaries` or `mode: "max"`) because its summaries
  persist and pre-compressing the summarizer input can reduce fidelity.
- Both mutate **only the outgoing request** — persisted history is never touched.
- Keeps the most recent `keepRecent` messages verbatim.
- Never touches failures/warnings, `tool-call` inputs, or reasoning.
- Very large payloads are head/tail trimmed before hitting the engine.

```
messages → recency filter → failure guard → size gate → head/tail
         → context-sift (extractive) → dispatch
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `mode` | `"auto"` | `off` \| `auto` \| `max` (`max` also enables `compactSummaries`) |
| `minChars` | `2000` | Skip payloads smaller than this |
| `keepRecent` | `6` | Recent messages kept verbatim |
| `budgetRatio` | `0` | Only act past this fraction of the context window (`0` = always) |
| `maxChars` | `100000` | Head/tail trim threshold |
| `compactSystem` | `false` | Also compact system instructions |
| `compactSummaries` | `false` | Also compact the transcript sent to OpenCode's summarizer (can reduce checkpoint fidelity) |
| `command` | `"context-sift"` | Engine executable |
| `socketPath` | tmp dir | Shared daemon socket |
| `idleTimeout` | `60` | Seconds the daemon stays warm after the last client |

## Savings

Each compaction sends `source`, `cwd`, and the original payload size to the
engine, which logs sizes to a local ledger. Run `context-sift gain` for a
per-project/per-day reduction summary (see the main README's *Savings report*).

### TUI status

The plugin also ships a CLI (TUI) entrypoint (`src/tui.tsx`) that puts a live
savings line in the footer (`home.footer.status` / `prompt.footer.status`),
e.g. `sift  337.0K saved · 65.0% · 67 reqs · today 17.0K`. It reads
`context-sift gain --oneline` (no model load) once on load and after each
completed session execution. `install.sh` installs both entrypoints.

A `ContextSift: status & modes` palette command (slash `/sift`, alias `/gain`)
shows the current savings and mode, and switches it: `/sift off|auto|max`. The
mode is **machine-global** (`~/.local/share/context-sift/mode`, override with
`CONTEXT_SIFT_MODE_FILE`), not per-project.

## Tests

```sh
node --test test/*.test.ts
```

## Roadmap

Phase 2 adds an optional **Laya** decision layer (`laya-ts`, in-process ONNX) to
gate whole payloads by relevance to the current request before ContextSift runs
(filter-then-refine). Default `shadow` so gains are measured before it alters
requests.
