# ContextSift

ContextSift compresses context before LLM prefill. Output stays extractive: fewer source clauses, same operational meaning.

Handles natural-language text, JSON API responses, log output, grep results, file listings, and embedded JSON — all in one call.

## Quickstart

```bash
pip install context-sift
```

```python
from context_sift import CompactorService

with CompactorService() as sift:
    reduced = sift("your long prompt here...")
```

Model ships inside the wheel. Zero config.

## Install

**PyPI** (works on any platform — CPU backend built-in):

```bash
pip install context-sift            # CPU (Linux, macOS, Windows)
pip install "context-sift[mlx]"     # Apple Silicon (MLX backend, faster)
```

**Git tag:**

```bash
pip install "context-sift @ git+https://github.com/joaoeudes7/context-sift.git@v1.0.0"
```

**Local dev:**

```bash
git clone https://github.com/joaoeudes7/context-sift.git
cd context-sift
pip install -e '.[mlx]'   # Apple Silicon
# or
pip install -e .           # CPU
```

**OpenCode plugin (V2):** installing provides the `context-sift` command; the plugin that compacts each model request lives in [`integrations/opencode`](integrations/opencode). See [OpenCode plugin](#opencode-plugin).

## Use as a library

```python
from context_sift import CompactorService

# Zero config — auto-selects MLX, CUDA, or CPU
sift = CompactorService()

first = sift(long_text)
second = sift(other_text)    # reuses loaded model

sift.stop()                  # application shutdown
```

**Context-manager form:**

```python
from context_sift import CompactorService

with CompactorService() as sift:
    reduced = sift(long_text)
```

### Parameters

```python
CompactorService(
    model_path=DEFAULT_MODEL_PATH,  # bundled model
    threshold=None,                  # None = read from model config (0.28)
    window_units=64,                 # token window for scoring
    backend="auto",                  # "auto" | "mlx" | "torch"
    device=None,                     # None = auto ("cuda" if available)
    always_compact=False,            # True = compact even short text
    autostart=True,                  # False = call start() manually
)
```

### Lifecycle

- One instance per worker process, not per request.
- `start()` and `stop()` are idempotent.
- A stopped service never reloads implicitly.
- Calls are serialized by a process-local lock.

### Backends

Auto-selection order: Apple MLX → NVIDIA CUDA → generic CPU.

Manual overrides for benchmarks or diagnostics:

```python
cpu  = CompactorService(backend="torch", device="cpu")
cuda = CompactorService(backend="torch", device="cuda")
mlx  = CompactorService(backend="mlx")
```

CUDA requires a PyTorch build compatible with your driver/CUDA stack. Requesting unavailable CUDA fails immediately — no silent fallback.

### Compression pipeline

ContextSift applies a multi-stage pipeline before the ML model scores clauses:

1. **Log compression** — ANSI stripping, error/warning preservation, repeated-line collapse, stack-trace truncation, warning deduplication
2. **JSON compression** — recursive routing of embedded JSON spans; arrays of objects get first/last items + error items preserved, remainder summarized; nested objects with large strings truncated
3. **Base64 replacement** — long base64 payloads replaced with size + SHA-256 identity
4. **Rule-based cleanup** — filler removal, exact-duplicate deduplication
5. **Whitespace normalization** — code-safe space collapsing (preserves indentation)
6. **ML scoring** — 660K-param BiGRU clause selector keeps high-scoring + protected spans

Each stage is a pure function, composable independently. See `context_sift.lossless`, `context_sift.json_compressor`, `context_sift.payloads`, `context_sift.rules`.

### Output trimming

Trim redundant model output before sending downstream:

```python
from context_sift import trim_output

trimmed = trim_output(model_output, context=user_prompt)
```

Removes ceremony preambles (`Sure! Let me...`, `Claro! Vou ajudar...`), trailing filler (`Let me know if...`, `Hope this helps!`), and lines that echo the context (high n-gram overlap). Supports EN, PT, ES, FR, DE.

### Whitespace normalization

Code-safe space collapsing — preserves indentation:

```python
from context_sift import collapse_spaces

code = "def foo():\n    x  =  1\n    if   True:\n        return   True"
print(collapse_spaces(code))
# def foo():
#     x = 1
#     if True:
#         return True
```

### Code compression (AST)

Strip comments, docstrings, and redundant syntax from source code:

```python
from context_sift import compress_code

result = compress_code(source, language="python")   # or "javascript"
print(result.ratio)      # 0.305
print(result.text)       # compressed source
```

Python: removes docstrings, comments, pass-through nodes, optional name minification.
JS/TS: removes comments, simplifies return, collapses whitespace.

### Input contract

**In scope:**

- System prompts and custom instructions
- Conversations and cross-model handoffs
- Plans, documentation, articles
- Multilingual prose with technical anchors
- JSON API responses and tool outputs (arrays of objects, nested structures)
- Log files and build output (pytest, npm, cargo, make, generic)
- Grep/ripgrep output (path:line:content rows)
- File path listings
- Mixed content (prose + embedded JSON)

**Rules:**

- Input below 200 characters returns unchanged (configurable via `always_compact`).
- Compression is adaptive. Dense text may stay mostly intact; repetitive text may shrink far beyond 30%.
- Goals, constraints, decisions, reasons, paths, commands, identifiers, numbers, URLs, and negations are always preserved.
- Error/fatal/critical items in JSON and logs are always preserved (100%).
- Stack trace heads (message + first 3 frames) are preserved.

### Compression ratios

Measured on real-world data:

| Input type | Compression | Method |
|---|---|---|
| HDFS logs (500 lines) | 94.5% | log compression + dedup |
| JSON log entries (100 items) | 94.4% | JSON dedup + lossless |
| App logs (ERROR/WARN/INFO) | 96.9% | level scoring + dedup |
| API response (50 users) | 52.6% | JSON structure + rules |
| Wikipedia prose (ML scoring) | 37.7% | clause selection (660K BiGRU) |
| Python source (AST) | 30.5% | docstrings + comments + pass |
| JavaScript source (AST) | 21.9% | comments + whitespace |

**Info retention**: 100% ERROR/WARN/CRITICAL preserved. 94-96% semantic similarity across all types.

## CLI

Pipe text or pass a file:

```bash
context-sift < prompt.txt > compact.txt
context-sift prompt.txt > compact.txt
```

CLI uses the same auto backend selection. Output is compacted text only.

## Proxy mode

Run an OpenAI-compatible API proxy that compacts requests before forwarding:

```bash
pip install 'context-sift[proxy]'
context-sift-proxy --upstream https://api.openai.com/v1 --key $OPENAI_API_KEY
```

The proxy compacts `messages[].content` (chat) and `input` (embeddings) before forwarding to the upstream provider. All other endpoints pass through untouched. Supports streaming (`stream: true`) SSE passthrough.

```bash
# Use like any OpenAI client — proxy sits in front
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Explain..."}]}'
```

| Option | Env | Default |
|---|---|---|
| `--host` | — | `127.0.0.1` |
| `--port` | — | `8000` |
| `--upstream` | `CONTEXT_SIFT_UPSTREAM_URL` | `https://api.openai.com/v1` |
| `--key` | `CONTEXT_SIFT_UPSTREAM_KEY` | relay client header |
| `--backend` | — | `auto` |
| `--threshold` | — | model config |

Key behavior:
- String `content` fields compacted; multimodal arrays (images) passed through untouched
- Auth: if `--key` set, used for all upstream requests; otherwise client's `Authorization` header relayed
- Streaming SSE chunks forwarded live (no buffering)
- `GET /health` returns `{"status":"ok","compactor":true}`

Use the proxy for clients without a plugin seam — any OpenAI-compatible tool. Inside OpenCode, use the [plugin](#opencode-plugin) instead: same engine, but policy-gated and non-destructive.

## OpenCode plugin

A V2 plugin that compacts the transcript before every model request, using one shared warm daemon. Fewer tokens per request means context compaction runs less often and smaller. It supersedes the [proxy](#proxy-mode) inside OpenCode; keep the proxy for other clients.

```bash
pip install context-sift        # provides the `context-sift` command
```

One engine process holds the model; every plugin runtime and `opencode` CLI invocation connects to it over a Unix socket. It spawns on demand and exits after an idle timeout — you never start it by hand.

Register the in-repo plugin in `opencode.jsonc`:

```jsonc
{
  "plugins": [
    {
      "package": "/absolute/path/to/context-sift/integrations/opencode",
      "options": { "mode": "auto" }
    }
  ]
}
```

- Non-destructive: edits only the outgoing request (`context` / `compaction` hooks), never persisted history.
- Keeps the most recent messages verbatim; never touches failures, warnings, tool-call inputs, or reasoning.
- Options: `mode` (`off` | `auto` | `aggressive`), `minChars`, `keepRecent`, `budgetRatio`, `maxChars`, `compactSystem`.

See [`integrations/opencode/README.md`](integrations/opencode/README.md) for details and tests (`node --test`).

## Savings report

Every compaction — plugin, proxy, or one-shot CLI — appends one line to a local
ledger (sizes and paths only, never the text). `context-sift gain` summarizes it
like `rtk gain`:

```bash
context-sift gain                 # totals, by source and project
context-sift gain --oneline       # one statusline-friendly line
context-sift gain --spark         # daily savings sparkline
context-sift gain --days 7        # only the last N days
context-sift gain --graph         # daily savings bars
context-sift gain -H              # recent requests
context-sift gain -p              # filter to the current directory
context-sift gain --json          # machine-readable
context-sift gain --reset --yes   # wipe the ledger
```

The ledger lives at `${XDG_DATA_HOME:-~/.local/share}/context-sift/gain.jsonl`
(override with `CONTEXT_SIFT_GAIN_FILE`). Token counts use a chars/4 estimate.

Inside OpenCode, the plugin's TUI entrypoint shows the same
`context-sift gain --oneline` summary live in the footer.

## Synthetic dataset

Generate training data with an OpenRouter teacher:

```bash
export OPENROUTER_API_KEY='...'
compact-dataset --count 100 --batch-size 4 --concurrency 4
```

Output defaults to `data/train.jsonl`. Reruns resume from valid rows. Generated rows require review before training. Never send private prompts to external teachers.

## Train ContextSift

Training requires Apple Silicon/MLX. Inference supports MLX, CPU, and CUDA.

```bash
PYTHONPATH=src python3 scripts/train_msc_rnn_mlx.py \
  --data data/msc/train.jsonl \
  --tokenizer models/context-sift/tokenizer.model \
  --output models/context-sift-next \
  --fast
```

Keep `models/context-sift` as the production model. Write experiments to another directory and promote only after validation.

## Validation

```bash
python -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/validate_prompt_scenarios.py --skip-judge
PYTHONPATH=src python3 scripts/validate_multilingual_handoffs.py
```

Multilingual fixtures cover EN, PT, ES, FR, DE, RU, AR, JA, and ZH. Technical anchors remain byte-exact.

## vs Headroom

| Dimension | ContextSift | Headroom |
|---|---|---|
| **Model** | 660K BiGRU (~2.5MB) | Kompress-v2-base (larger) |
| **Prose compression** | **37.7%** (ML clause selection) | ~15-20% |
| **Log compression** | **94.5%** (level scoring) | 60-95% |
| **AST languages** | Python, JS/TS | Python, JS/TS, Go, Rust, Java, C/C++, Perl |
| **Reversible** | No | Yes (CCR cache) |
| **Cache-aligned** | No | Yes (KV cache prefix) |
| **Deployment** | Library + CLI + Proxy + OpenCode plugin | Library + Proxy + MCP + Agent wrap |
| **Cross-agent** | No | Yes (SharedContext) |
| **Output trimming** | No | Yes (ceremony, effort routing) |
| **Cold start** | **280ms** | 15-200ms |
| **Dependencies** | torch, sentencepiece | torch + many extras |
| **License** | MIT | MIT |

**Choose ContextSift when**: you want a small, fast library with high compression and minimal dependencies. Library, CLI, OpenAI-compatible proxy, and an OpenCode plugin — `pip install` and go.

**Choose Headroom when**: you need reversible compression, cross-agent memory, proxy integration, or broad AST support.

## License

[MIT](LICENSE)
