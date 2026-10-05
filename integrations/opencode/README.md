# context-sift-opencode

OpenCode V2 plugin that compacts the transcript sent to the model, using a warm
[ContextSift](../..) daemon. It reduces tokens per request — and therefore how
often and how large context compaction has to be.

The plugin is the **decision layer** ("when/what to compress"). ContextSift is
the **compression engine** (extractive, 660K-param ML model).

## Install

1. Install the engine and make it available on `PATH`:

   ```sh
   pip install context-sift
   context-sift --serve   # sanity check: prints {"ready": true}, then EOFs
   ```

2. Point OpenCode at the plugin directory (`opencode.jsonc`):

   ```jsonc
   {
     "plugins": [
       {
         "package": "/absolute/path/to/integrations/opencode",
         "options": { "mode": "auto" }
       }
     ]
   }
   ```

## How it works

- Registers the `context` and `compaction` session hooks. Both mutate **only the
  outgoing request** — persisted history is never touched.
- Keeps the most recent `keepRecent` messages verbatim.
- Never touches failures/warnings (ContextSift already preserves them).
- Very large payloads are head/tail trimmed before hitting the engine.

```
messages → recency filter → failure guard → size gate → head/tail
         → context-sift (extractive) → dispatch
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `mode` | `"auto"` | `off` \| `auto` \| `aggressive` |
| `minChars` | `2000` | Skip payloads smaller than this |
| `keepRecent` | `6` | Recent messages kept verbatim |
| `budgetRatio` | `0` | Only act past this fraction of the context window (`0` = always) |
| `maxChars` | `100000` | Head/tail trim threshold |
| `compactSystem` | `false` | Also compact system instructions |
| `command` | `"context-sift"` | Engine executable |
| `args` | `["--serve"]` | Engine arguments |
| `timeoutMs` | `30000` | Per-request engine timeout |

If `context-sift` is not on `PATH`, run it via Python:

```jsonc
{
  "options": {
    "command": "python3",
    "args": ["-m", "context_sift.runtime_cli", "--serve"]
  }
}
```

## Tests

```sh
node --test test/*.test.ts
```

## Roadmap

Phase 2 adds an optional **Laya** decision layer (`laya-ts`, in-process ONNX) to
gate whole payloads by relevance to the current request before ContextSift runs
(filter-then-refine). Default `shadow` so gains are measured before it alters
requests.
