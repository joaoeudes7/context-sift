"""Validate information retention, not just compression ratio.

Measures:
- Entity preservation (block IDs, IPs, node names, timestamps)
- Semantic similarity (TF-IDF cosine)
- Critical fact retention (ERROR/WARN preserved)
- Round-trip reversibility (lossless stages)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from context_sift.json_compressor import compact_json
from context_sift.lossless import (
    collapse_runs,
    compact_logs,
    expand_runs,
    fold_repeated_blocks,
    unfold_repeated_blocks,
)
from context_sift.payloads import compact_base64
from context_sift.rules import compress_rules


# ── Entity extraction ──────────────────────────────────────────────────

_ENTITY_PATTERNS: dict[str, re.Pattern] = {
    "block_id": re.compile(r"blk_-?\d+"),
    "ip": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "node": re.compile(r"dfs\.\w+(?:\$\w+)?"),
    "port": re.compile(r":\d{4,5}\b"),
    "timestamp": re.compile(r"\d{6}\s+\d{6}"),
    "url": re.compile(r"https?://\S+"),
    "path": re.compile(r"(?:/[\w.-]+){2,}"),
    "size": re.compile(r"\b\d+(?:\.\d+)?(?:KB|MB|GB|bytes?)\b", re.IGNORECASE),
}


def extract_entities(text: str) -> dict[str, set[str]]:
    return {name: set(pat.findall(text)) for name, pat in _ENTITY_PATTERNS.items()}


def entity_preservation(original: str, compressed: str) -> dict[str, dict]:
    orig_ents = extract_entities(original)
    comp_ents = extract_entities(compressed)
    result = {}
    for name in orig_ents:
        o, c = orig_ents[name], comp_ents.get(name, set())
        result[name] = {
            "original": len(o),
            "preserved": len(c & o),
            "lost": len(o - c),
            "ratio": round(len(c & o) / max(len(o), 1), 3),
        }
    return result


# ── Semantic similarity ────────────────────────────────────────────────

def semantic_similarity(original: str, compressed: str) -> float:
    """TF-IDF cosine similarity between original and compressed."""
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vec.fit_transform([original, compressed])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0])


# ── Critical fact retention ────────────────────────────────────────────

_ERROR_RE = re.compile(r"\b(FATAL|ERROR|CRITICAL|FAILED|Exception)\b", re.IGNORECASE)

def critical_facts(original: str, compressed: str) -> dict:
    orig_errors = [l for l in original.splitlines() if _ERROR_RE.search(l)]
    comp_errors = [l for l in compressed.splitlines() if _ERROR_RE.search(l)]
    # Normalize by removing numbers for dedup comparison
    orig_norm = {re.sub(r"\d+", "N", e.strip()) for e in orig_errors}
    comp_norm = {re.sub(r"\d+", "N", e.strip()) for e in comp_errors}
    return {
        "error_lines_original": len(orig_errors),
        "error_lines_compressed": len(comp_errors),
        "unique_error_patterns_preserved": len(orig_norm & comp_norm),
        "unique_error_patterns_total": len(orig_norm),
        "preservation_rate": round(len(orig_norm & comp_norm) / max(len(orig_norm), 1), 3),
    }


# ── Round-trip reversibility ───────────────────────────────────────────

def test_reversibility(text: str) -> dict:
    """Test that lossless stages can reverse."""
    # collapse_runs -> expand_runs
    collapsed = collapse_runs(text)
    expanded = expand_runs(collapsed)
    runs_ok = expanded == text

    # fold_repeated_blocks -> unfold_repeated_blocks
    folded = fold_repeated_blocks(text)
    unfolded = unfold_repeated_blocks(folded)
    blocks_ok = unfolded == text

    return {
        "collapse_expand_roundtrip": runs_ok,
        "fold_unfold_roundtrip": blocks_ok,
        "collapse_ratio": round(1 - len(collapsed) / max(len(text), 1), 3),
        "fold_ratio": round(1 - len(folded) / max(len(text), 1), 3),
    }


# ── Per-clause scoring (ML component) ──────────────────────────────────

def ml_keep_ratio(text: str) -> dict:
    """Run the ML scoring pipeline and report what it keeps."""
    from context_sift.clause_dataset import split_clauses
    import sentencepiece as spm
    from context_sift.msc import DEFAULT_MODEL_PATH

    path = Path(DEFAULT_MODEL_PATH)
    tokenizer = spm.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))

    clauses = split_clauses(text)
    if not clauses:
        return {"clause_count": 0}

    scores = []
    for clause in clauses:
        tokens = tokenizer.encode(clause.text, out_type=int, add_bos=True, add_eos=True)[:256]
        # Simple scoring: count how many tokens are "important" (not filler)
        important = sum(1 for t in tokens if t > 10)  # rough heuristic
        scores.append(important / max(len(tokens), 1))

    return {
        "clause_count": len(clauses),
        "avg_importance": round(np.mean(scores), 3) if scores else 0,
        "min_importance": round(float(np.min(scores)), 3) if scores else 0,
        "max_importance": round(float(np.max(scores)), 3) if scores else 0,
    }


# ── Main validation ────────────────────────────────────────────────────

def validate_sample(name: str, text: str) -> dict:
    result = {"name": name, "input_chars": len(text)}

    # Run full pipeline
    t = compact_base64(text)
    t = compact_json(t)
    t = compress_rules(t).text
    compressed = compact_logs(t)

    result["output_chars"] = len(compressed)
    result["compression_ratio"] = round(1 - len(compressed) / max(len(text), 1), 3)

    # Entity preservation
    result["entity_preservation"] = entity_preservation(text, compressed)

    # Semantic similarity
    result["semantic_similarity"] = round(semantic_similarity(text, compressed), 3)

    # Critical facts
    result["critical_facts"] = critical_facts(text, compressed)

    # Reversibility (on lossless stage only)
    result["reversibility"] = test_reversibility(text)

    return result


def main():
    samples = {}

    # 1. HDFS logs
    log_path = Path("data/validation/real/hdfs.log")
    if log_path.exists():
        lines = log_path.read_text().splitlines()
        samples["hdfs_logs_100"] = "\n".join(lines[:100])
        if len(lines) > 100:
            samples["hdfs_logs_500"] = "\n".join(lines[:500])

    # 2. Wikipedia article
    with open("data/wikipedia/long.jsonl") as f:
        rec = json.loads(f.readline())
        samples["wikipedia_article"] = rec["source"]

    # 3. JSON payloads
    samples["json_api_response"] = json.dumps({
        "status": "success",
        "data": {"items": [{"id": i, "name": f"Item {i}"} for i in range(50)]},
        "meta": {"total": 50, "page": 1}
    }, indent=2)

    samples["json_log_entries"] = json.dumps([
        {"ts": f"2026-01-01T00:{i:02d}:00Z", "level": "INFO", "msg": f"Batch {i} processed"}
        for i in range(100)
    ], indent=2)

    # 4. Application logs (mixed)
    app_log_lines = []
    for i in range(200):
        level = "ERROR" if i % 50 == 0 else "WARN" if i % 25 == 0 else "INFO"
        app_log_lines.append(
            f"2026-08-06 12:{i%60:02d}:00 [{level}] service=api "
            f"request_id=req-{i:04d} user_id=usr-{i%100:04d} "
            f"latency_ms={10 + i % 500} endpoint=/api/v1/items"
        )
    samples["app_logs_mixed"] = "\n".join(app_log_lines)

    # Run validation
    results = {}
    for name, text in samples.items():
        print(f"Validating: {name} ({len(text)} chars)...", file=sys.stderr)
        results[name] = validate_sample(name, text)

    # Output
    out_path = Path("reports/information_retention.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Sample':<25} {'Input':>8} {'Output':>8} {'Ratio':>7} {'Semantic':>9} {'Entities':>9} {'Critical':>9}")
    print("=" * 80)
    for name, r in results.items():
        ep = r["entity_preservation"]
        total_orig = sum(v["original"] for v in ep.values())
        total_pres = sum(v["preserved"] for v in ep.values())
        ent_ratio = total_pres / max(total_orig, 1)
        print(
            f"{name:<25} {r['input_chars']:>8} {r['output_chars']:>8} "
            f"{r['compression_ratio']:>6.1%} {r['semantic_similarity']:>8.3f} "
            f"{ent_ratio:>8.1%} {r['critical_facts']['preservation_rate']:>8.1%}"
        )
    print("=" * 80)

    # Detailed entity breakdown for HDFS
    print("\n--- HDFS Logs Entity Breakdown ---")
    hdfs = results["hdfs_logs_100"]["entity_preservation"]
    for ent_type, stats in hdfs.items():
        if stats["original"] > 0:
            print(f"  {ent_type:<12}: {stats['preserved']}/{stats['original']} ({stats['ratio']:.0%})")

    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
