# Structured compaction experiments

Purpose: test compact structures after extractive cleanup. Production model remains unchanged.

## Done

- Built multilingual extractive training and validation data.
- Trained tiny ContextSift RNN candidates; production artifact remains `models/context-sift`.
- Added CPU, NVIDIA/Torch, and Apple MLX execution paths.
- Added reusable `CompactorService`: load once, compact many, explicit `stop()`.
- Added `always_compact=False`: short prompts bypass compaction by default.
- Protected rules, permissions, prohibitions, user requests/questions, paths, commands, UUIDs, and code-sensitive content.
- Validated long coding-agent prompts, cross-model session handoff, tool-heavy context, Git diff, scientific text, code comments, and nine languages.
- Tested extractive KEEP/DROP/MERGE/REPLACE safety gates.
- Tested sequences, branches, dependencies, state records, tables, graph notation, abbreviations, and numeric precision.
- Added reproducible structured-format experiment. No structured rewrite enabled in production.

## Next flow

```text
freeze extractive baseline
-> run structured corpus across target tokenizers
-> expand long multilingual relation corpus
-> validate anchors + facts + edges + modality
-> measure downstream prefill latency
-> reject unsafe or low-gain rewrites
-> generate accepted training pairs
-> train candidate model
-> compare candidate vs frozen baseline
-> promote only with zero critical regression
```

Promotion requires:

```text
critical recall = 100%
relation precision/recall = 100%
modality + negation recall = 100%
no path/command/value corruption
median downstream token reduction improves
CPU latency and RAM stay within production budget
```

## Run

```bash
PYTHONPATH=src python3 scripts/experiment_structured_formats.py
```

Script uses ContextSift SentencePiece tokenizer and prints JSON only. It checks exact anchors, fixture-declared relations, modality markers, and token reduction.

`oracle_consistent` means candidate agrees with manually declared fixture oracle. It does not prove automatic semantic understanding. Relations are not extracted from arbitrary text yet.

Acceptance gate:

```text
oracle_consistent
and saved_tokens >= 8
and reduction >= 15%
```

Oracle consistency and economy are separate. Consistent rewrite below gate remains unused. Production safety additionally requires independent validation of actual model output.

## Current findings

Current corpus: 19 cases; 16 agree with manual oracle; 5 pass oracle and economy gates. Mean reduction among oracle-consistent candidates: 16.6%. These numbers measure handcrafted candidates, not current model rewrite ability.

Passing cases:

| Case | Tokens | Reduction |
|---|---:|---:|
| French service state | 37 -> 28 | 24.3% |
| German code-change description | 54 -> 42 | 22.2% |
| English agent workflow | 62 -> 45 | 27.4% |
| Portuguese repeated records | 64 -> 43 | 32.8% |
| French long repeated-state table | 177 -> 129 | 27.1% |

Consistent but short sequences, branches, dependencies, and effects were skipped because they saved fewer than eight tokens. Three deliberately invalid rewrites were rejected despite reducing tokens.

- `A -> B`: good for explicit sequence or flow. Never use it to guess causality.
- `A => B`: result or implication, not ordering.
- `A requires B`: safer than converting dependency into sequence.
- `A if C | B if D`: preserves branches when each condition remains explicit.
- `repeat(A)`: clearer cycle marker than an unexplained back edge.
- `key=value`: useful for object state.
- Compact table: useful for three or more records sharing fields.
- `quality up; bugs down`: cheaper than Unicode `quality↑; bugs↓` with current tokenizer.
- `+x/-x`: reserve for add/remove entities. It is ambiguous for quality trends.
- JSON and Mermaid are interchange formats, not default token-saving formats.
- Preserve source language. Translating labels during compaction adds semantic risk and requires separate validation.

Short examples often save 15–40% but fewer than eight tokens. Correct decision: do not rewrite. Long graphs and repeated records can amortize syntax and should provide larger absolute savings.

Long-format A/B changed this hypothesis:

| Long case | Reduction | Decision |
|---|---:|---|
| Portuguese agent workflow | -5.0% | reject: larger output |
| Spanish request branches | 1.9% | reject: negligible |
| German dependency graph | 14.1% | reject: below gate |
| English agent handoff | 4.1% | reject: negligible |
| French repeated-state table | 27.1% | retain experiment |

Graph syntax is not universally compact. Detailed rules and handoffs still need most semantic words; labels and operators add cost. Repeated records benefit because schema is written once. Prefer compact natural clauses for dense narrative handoffs.

Unsafe examples deliberately rejected:

- parallel work rewritten as serial chain;
- `requires` rewritten as chronological order;
- exact command changed;
- `only after`, `must`, `may`, `never`, or negation removed;
- relation inferred but absent from source.

## Pending

1. Freeze and record current production baseline.
2. Run same corpus with tokenizers of downstream LLMs. ContextSift token count alone does not prove prefill savings.
3. Expand to long PT, EN, ES, FR, DE, JA prompts with at least 50 relations each.
4. Add semantic judge comparing facts, edges, modality, quantities, paths, and commands.
5. Test automatic relation extraction. Current relation oracle is declared in fixtures.
6. Measure end-to-end prefill latency, not only token count.
7. Test abbreviation break-even per tokenizer and repetition count. Default remains no abbreviation.
8. Test tolerance-aware numeric rounding. Default remains exact precision.
9. Generate training pairs only from accepted rewrites.
10. Train and compare candidate against frozen baseline before any production promotion.
11. Validate full suite on host with accessible Metal device. Current headless macOS run reaches 110 tests but 16 MLX imports fail with `No Metal device available`.

## Production decision

Keep current extractive compactor as baseline. Structured rewrite becomes optional second stage only when all semantic gates pass and measured downstream token savings exceed gate.

Current RNN emits one KEEP/DROP score per unit and copies selected source clauses. Retraining it cannot generate arrows, tables, abbreviations, or rewritten values. Retrain current RNN only for demonstrated salience regressions. Learned structured output would require a separate generative decoder and `source -> structured` dataset; do not build it before deterministic validation succeeds.

Retain now:

- production extractive RNN;
- repeated-record table and simple `key=value` as offline candidates;
- explicit `requires`, `if`, `only`, and `repeat` grammar for evaluation fixtures;
- economic gate and negative regression cases;
- exact source language, precision, paths, commands, rules, and permissions.

Do not retain as production behavior:

- universal prose-to-graph conversion;
- Unicode trend symbols;
- automatic abbreviations or translation;
- free numeric rounding;
- fixture oracle presented as automatic semantic validation.
