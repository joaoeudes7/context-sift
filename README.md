# ContextSift

ContextSift removes redundant natural-language context before LLM prefill. Output stays extractive: fewer source clauses, same operational meaning.

Production goals:

- typical reduction of at least 30% when redundancy permits;
- preserve goals, constraints, decisions, reasons, paths, commands, identifiers, numbers, URLs, and negations;
- never force a fixed ratio;
- leave text below 200 characters unchanged;
- load one tiny model once and reuse it across requests.

Current model: `models/context-sift`, a 660,737-parameter SentencePiece + BiGRU clause selector.

## Runtime support

- generic CPU/Linux/macOS: PyTorch backend;
- NVIDIA: PyTorch CUDA backend;
- Apple Silicon/macOS: MLX backend, with automatic CPU fallback when Metal is unavailable.

Same bundled SafeTensors weights serve every backend. CPU is portable default because this 660K-parameter RNN is often too small to amortize CUDA kernel launch overhead.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[portable]'   # CPU or NVIDIA/CUDA
# or
pip install -e '.[mlx]'        # Apple Silicon
```

## Use as a reusable service

```python
from compact_dataset import CompactorService

sift = CompactorService()  # loads models/context-sift once

first = sift(long_text)
second = sift(other_text)   # same loaded model

sift.stop()                # application shutdown
```

CPU is automatic outside Apple Silicon. Explicit backend/device:

```python
cpu = CompactorService(backend="torch", device="cpu")
cuda = CompactorService(backend="torch", device="cuda")
mlx = CompactorService(backend="mlx")
```

CUDA requires a PyTorch build compatible with server NVIDIA driver/CUDA stack. Requesting unavailable CUDA fails immediately; it never falls back silently.

Create one instance per worker process, not per request. Calls are serialized by a process-local lock. `start()` and `stop()` are idempotent; a stopped service never reloads implicitly.

Context-manager form:

```python
from compact_dataset import CompactorService

with CompactorService() as sift:
    reduced_text = sift(long_text)
```

## Input contract

Input: decoded Unicode natural-language text. Caller must extract readable text first.

Primary use:

- long system prompts and custom instructions;
- conversations and cross-model handoffs;
- plans, documentation, and articles;
- multilingual prose containing technical anchors.

Outside primary scope: raw HTML/XML, source code, JSON/TOON, OCR repair, images, PDF binaries, and base64. Parse or bypass these formats before compaction. Git diff has a conservative compatibility path but is not part of main quality target.

No `source_type` is required. Compression is adaptive. Dense text may remain mostly intact; repetitive long context may shrink far beyond 30%.

For agent handoffs, retained context must identify current goal, governing rules, selected approach, rejected attempts, completed work, blockers, relevant paths/identifiers, and exact next action.

## Validation

```bash
python -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/validate_prompt_scenarios.py --skip-judge
PYTHONPATH=src python3 scripts/validate_multilingual_handoffs.py
```

Multilingual handoff fixtures cover EN, PT, ES, FR, DE, RU, AR, JA, and ZH. Technical anchors remain byte-exact. Semantic similarity alone is insufficient; operational details and protected spans are release gates.

## Synthetic dataset

Set key only in environment:

```bash
export OPENROUTER_API_KEY='...'
compact-dataset --count 100 --batch-size 4 --concurrency 4
```

Output defaults to `data/train.jsonl`; reruns resume from valid rows. `openrouter/free` selects an available free teacher. Pin one with `--model vendor/model:free`. Use `--plain-json` when model lacks OpenRouter `response_format` support.

Generated rows require review before training. Never send private prompts to external teachers.

## Train ContextSift

Training currently requires Apple Silicon/MLX. Inference supports MLX, generic CPU, and NVIDIA CUDA. Keep `models/context-sift` as production model; write experiments to another directory and promote only after validation.

```bash
PYTHONPATH=src python3 scripts/train_msc_rnn_mlx.py \
  --data data/msc/train.jsonl \
  --tokenizer models/context-sift/tokenizer.model \
  --output models/context-sift-next \
  --fast
```
