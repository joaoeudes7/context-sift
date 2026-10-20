# MSC validation history

Production model: **ContextSift**, stored at `models/context-sift` (`production_ready: true`).

## Architecture experiments

- Hashed-word V2: 1.42M parameters; realistic agent fixture reduced 86.2%, critical recall 100%; language abstraction absent.
- Hierarchical MSC RNN V1: 1.53M parameters; validation BCE 0.6945 → 0.4031; 26.7% reduction at 96.4% oracle recall; position recall dropped 4.8 pp.
- Hierarchical MSC RNN V2: position augmentation removed position shortcut; only 15.8% reduction at 96.4% recall.
- Fast MSC RNN V2: 660,737 parameters; mean SentencePiece embeddings + document BiGRU; validation BCE 0.6930 → 0.3775.

## Fast MSC RNN V2

- Validation threshold 0.28: 30.4% reduction, 92.6% oracle unit recall.
- One 49,916-char Wikipedia document: 37.9% reduction, 95.5% target recall, 100% protected recall.
- Warm latency: 88 ms; measured MLX process RSS: 235 MB.
- Exact repeated fixture: 95.1% reduction after deterministic dedup integration.
- Full test suite before multilingual expansion: 99/99 passed.

## Multilingual validation

- Long EN, 10 documents: median reduction 29.1%; mean target recall 95.3%; protected recall 100%; p95 135 ms.
- Long PT, 10 documents: median reduction 36.4%; mean target recall 96.2%; protected recall 100%; p95 147 ms.
- Aligned Wikipedia tested in AR/DE/EN/ES/FR/JA/PT/RU/ZH. Protected recall 100% after the critical-span gate.
- AR/JA/RU/ZH currently fall back to little/no compression. Safe, but not useful yet.

## Real original-vs-compact audit

- The first 9-pair batch judge run at `2026-08-02T16:44:53Z` is invalid due cross-pair contamination and a schema typo.
- Corrected run at `2026-08-02T16:46:57Z`: one isolated judge request per language, strict schema.
- Central message preserved: 9/9. Contradictions: 0.
- Judge semantic recall: EN 0.92, PT 0.90, DE 0.95, ES/FR/AR/RU/ZH/JA 1.0.
- AR/RU/ZH/JA scores reflect no meaningful compression; they do not prove multilingual selection quality.

## Real domain audit

Run `2026-08-02T17:00:38Z`; isolated Ling 2.6 Flash judge requests; exact protected-item gate:

- Scientific paper, 42,208 chars: 19.1% token reduction; 100% protected recall; semantic recall 1.0.
- HDFS log slice, 50,000 chars: 0% reduction; 100% WARN/ERROR line recall; semantic recall 1.0. Safe fallback, currently ineffective.
- OpenCode git diff, 16,848 chars: 8.1% token reduction; 100% file/hunk/change-line recall; semantic recall 1.0.
- OpenCode agent prompt, 8,212 chars: 21.8% token reduction; 100% protected recall; semantic recall 1.0.
- Domain mean token reduction: 12.3%. Target 30% was not reached outside Wikipedia.
- Judge produced no contradictions or missing critical items, but perfect scores require future independent/downstream-task confirmation.

Original sources live in `data/validation/real`; compact outputs in `reports/domain_samples`.
Machine-readable record: `reports/domain_validation.jsonl`.

## Long agent-prompt and model-handoff audit

Run `2026-08-02T17:17:43Z`; synthetic Kilo-like prompt based on public behavior concepts,
synthetic model handoff/tool session, plus real `caveman` and `context-mode` skill text:

- Initial automatic gate reported 100% critical recall but used incomplete critical lists and substring
  matching. Manual output audit invalidated that result.
- Corrected exact-boundary audit: Kilo-like 91.3% reduction / 71.4% critical recall;
  cross-model handoff 89.5% / 85.7%; tool-heavy 92.8% / 75.0%; real skills + handoff
  29.8% / 90.9%.
- Missing content includes Kilo `pytest` and mode-switch condition, handoff active goal, and complete
  tool inventory/mutation prohibition. Current model fails operational-sufficiency gate.
- Runtime: 7–90 ms after model load.
- Rule gates now protect dotfiles, qualified symbols, test references, security/warning terms,
  and rejected/reverted/superseded/failed/pending/blocked state in English and Portuguese.
- Fixed corruption where cleanup changed Python test selector `path.py::test_name` into
  `path.py:test_name`.
- Fixed corruption where cleanup joined `read .env` into `read.env`; exact recall now rejects
  protected substrings embedded in another token.
- Handoff judge results are inconclusive: one evaluation requested details absent from ORIGINAL;
  another returned 0.70 with no missing critical item. Do not optimize against these scores.

Machine-readable record: `reports/prompt_validation.jsonl`.

## Multilingual model-handoff audit

Run `2026-08-02T17:49:29Z`; one aligned pure-text handoff in nine languages, each expanded with
80 redundant history reminders:

- Critical recall: 100% for EN, PT, ES, FR, DE, RU, AR, JA, ZH.
- Token reduction: EN 80.5%, PT 83.0%, ES 83.8%, FR 83.7%, DE 88.0%, RU 88.6%,
  AR 87.5%, JA 84.0%, ZH 84.0%.
- Warm latency: 4.3–10.6 ms.
- Anchors (paths, symbols, commands, issue IDs) are byte-exact. Natural phrases normalize only
  Unicode, whitespace, and spacing before punctuation.
- Initial FR and JA failures exposed two generic bugs: lost sentence boundaries in translated
  fixtures and Unicode `\b` around adjacent ASCII technical identifiers. Both fixed without
  language-specific model thresholds.
- Multilingual decision-state gates cover rejection/failure/pending/blocking vocabulary across
  evaluated languages. This is a deterministic safety gate, not proof of unseen-language quality.

Machine-readable record: `reports/multilingual_prompt_validation.jsonl`.

Machine-readable append-only records:

- `reports/validation_history.jsonl`
- `reports/semantic_validation.jsonl`

## Checkpoint projection audit

Run `2026-10-18T18:47:41Z`; checkpoint mode is the only path the OpenCode integration now
uses, and its output is folded into persisted memory, so critical recall is the gate that
matters. The projection is extractive and makes no model call; 12 items, p95 3.9 ms.

- Prompt scenarios (EN): mean token reduction 90.4%, min 89.2%; critical recall 100% (3/3).
- Aligned multilingual handoffs (EN/PT/ES/FR/DE/RU/AR/JA/ZH, 80 redundant reminders per item):
  mean token reduction 86.8%; critical recall 100% (9/9, exact anchors + natural phrases);
  missing items 0.
- Runtime gates: 5 repeated projections byte-identical, then `stop()` refuses further work.
- Two recall gaps were found by this audit and fixed before it recorded:
  - The 128-char cap on sentence expansion dropped the tail of a long condition
    (`... por lo que se revirtió` / `поэтому была отменена`), leaving the head clause
    unreadable. The checkpoint path now expands a meaning-changing anchor across its whole
    sentence; cost was multilingual mean reduction 87.1% → 86.8%.
  - The decision-state gate listed participles only, so ES `rechazó` (preterite) and RU
    `отменена` were unrecognised. It now uses stems for the reject/refuse/revert/cancel/fail
    families in the enumerated languages.
- Generic compaction was re-verified after that change: EN fixture mean ratio 0.086
  (unchanged), multilingual handoff critical recall 100% in all nine languages.

Machine-readable record: `reports/checkpoint_validation.jsonl`; samples in `reports/checkpoint_samples`.

## Next gates

1. Add multilingual selection labels for AR/JA/RU/ZH and repeat real-pair audit.
2. Train format-diverse selection labels; logs and diffs expose largest domain gap.
3. Add cross-window semantic dedup only after multilingual/domain recall remains stable.
4. Set `production_ready: true` only when critical recall stays 100% and semantic recall is non-inferior across all evaluated domains.
5. Checkpoint mode is a regex projection, not a model call: it reached 100% recall on the evaluated
   fixtures, but an unseen constraint phrasing with no status verb and no technical anchor can still
   be dropped. Extend the constraint gate from real session data before calling it conservative protection.
