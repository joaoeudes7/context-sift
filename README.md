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
pip install "context-sift @ git+https://github.com/joaoeudes7/compact_llm_summary.git@v1.0.0"
```

**Local dev:**

```bash
git clone https://github.com/joaoeudes7/compact_llm_summary.git
cd compact_llm_summary
pip install -e '.[mlx]'   # Apple Silicon
# or
pip install -e .           # CPU
```

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
5. **ML scoring** — 660K-param BiGRU clause selector keeps high-scoring + protected spans

Each stage is a pure function, composable independently. See `context_sift.lossless`, `context_sift.json_compressor`, `context_sift.payloads`, `context_sift.rules`.

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

Measured on realistic inputs:

| Input type | Saved |
|---|---|
| JSON API response (100 users) | 98.9% |
| Repeated log lines | 98.9% |
| Redundant prose | 95.1% |
| JSON log entries (50 items) | 89.1% |
| Embedded JSON in text | 76.1% |
| Long system prompt | 70.3% |
| Mixed prose + JSON | 64.7% |

## CLI

Pipe text or pass a file:

```bash
context-sift < prompt.txt > compact.txt
context-sift prompt.txt > compact.txt
```

CLI uses the same auto backend selection. Output is compacted text only.

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

## License

[MIT](LICENSE)
