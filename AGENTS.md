# ContextSift Agent Rules

## Mission

ContextSift only reduces long natural-language context before LLM prefill. Prefer deletion over rewriting. Output stays extractive, ordered, grounded, and minimum-sufficient. Grammar/style unimportant; facts matter.

## Invariants

- Preserve central message, goals, constraints, decisions/reasons, negations, failures, pending actions, and security rules.
- Preserve exact paths, commands, symbols, identifiers, UUIDs, URLs, numbers, and technical anchors.
- Never invent or reorder information.
- Input below 200 characters stays unchanged.
- Ratio is adaptive; never force 30%. Dense text may stay large; redundant long text should normally reduce ≥30%.
- `models/context-sift` is only production model.
- Input is decoded Unicode natural-language text; no `source_type`.
- Raw HTML/XML, source code, JSON/TOON, OCR, images, binaries, and base64 require upstream parsing/bypass.

## Runtime

```python
from context_sift import CompactorService

sift = CompactorService()
compact = sift(long_text)
sift.stop()
```

One instance per worker; reuse across requests. Auto backend: Apple MLX → NVIDIA CUDA → CPU. CLI: `context-sift < prompt.txt > compact.txt`. Built wheel must contain model.

## Architecture

- `FastMinimumContextRNN`; SentencePiece 8K; 660,737 FP32 parameters; ~2.7 MB SafeTensors.
- MLX/PyTorch load same weights without conversion.
- Preserve trained pooling quirk: pad token `0`, mask token `3`. Changing requires retraining/parity validation.
- CPU may beat CUDA; never claim CUDA faster without real hardware benchmark.

Canonical paths: runtime `src/context_sift/msc.py`; portable backend `torch_backend.py`; MLX model `msc_model.py`; CLI `runtime_cli.py`; training `scripts/train_msc_rnn_mlx.py`; docs `README.md`; validation `reports/VALIDATION.md` and `reports/**/*.jsonl`.

## Preserve data

- `data/train.jsonl`
- `data/msc/train.jsonl`
- `data/wikipedia/`
- `data/validation/`

Do not restore retired RWKV, BarunLM, old checkpoints, duplicate tokenizer, empty JSONL, or stale SDD. Keep `word_pruner.py`: MSC training uses its label fallback. Never send private prompts, credentials, `.env`, or secrets to OpenRouter. Key comes only from `OPENROUTER_API_KEY`.

## Quality gates

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Runtime changes also prove stable repeated calls, explicit `stop()`, wheel execution outside repo, real production weights on CPU, real NVIDIA test when available, exact protected anchors, and multilingual critical recall.

Evidence: repetitive 50K handoff CPU ≈15.6 ms warm, 97.2% reduction, 100% fixture recall; cross-model handoff 91% reduction, 100% recall. Layered rules/tool permissions only reached 71–75% recall: known gap. Do not call universally production-safe until 100% or conservative protection. Similarity alone is insufficient; inspect output. Dense/scientific text favors preservation.

## Change discipline

- Think, simplify, edit surgically, verify; reuse before adding.
- Keep only necessary docs and validation evidence.
- Never delete training data/history as generic cleanup or modify unrelated files.
- Never commit secrets or experimental checkpoints.
- Do not commit, tag, push, publish, or release unless user explicitly requests that exact action in current turn.
- Current release: `v1.0.0`.

