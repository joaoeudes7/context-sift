"""Validate ContextSift compression pipeline on real-world data.

Loads Wikipedia articles, HDFS logs, and realistic JSON payloads.
Measures compression at each pipeline stage and overall.
Outputs results as JSON for reproducibility.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from context_sift.json_compressor import compact_json
from context_sift.lossless import compact_logs
from context_sift.payloads import compact_base64
from context_sift.rules import compress_rules


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_wikipedia_articles(n: int = 10) -> list[str]:
    """Load English Wikipedia article texts from long.jsonl."""
    path = Path("data/wikipedia/long.jsonl")
    texts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            texts.append(rec["source"])
    return texts


def load_aligned_articles(n: int = 5) -> list[str]:
    """Load multilingual article pairs (English text)."""
    path = Path("data/wikipedia/aligned.jsonl")
    texts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            en = rec.get("pages", {}).get("en", {})
            if "text" in en:
                texts.append(en["text"])
    return texts


def load_hdfs_logs(max_lines: int = 500) -> str:
    """Load HDFS log sample."""
    path = Path("data/validation/real/hdfs.log")
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()[:max_lines]
    return "\n".join(lines)


def load_app_diff() -> str:
    """Load the sample diff."""
    path = Path("data/validation/real/opencode.diff")
    if not path.exists():
        return ""
    return path.read_text()


def make_realistic_json_samples() -> list[str]:
    """Create realistic JSON payloads for testing."""
    samples = [
        # API response
        json.dumps({
            "status": "success",
            "data": {
                "users": [
                    {"id": i, "name": f"User {i}", "email": f"user{i}@example.com",
                     "roles": ["admin" if i % 5 == 0 else "viewer"],
                     "last_login": f"2026-01-{i:02d}T10:30:00Z"}
                    for i in range(1, 21)
                ],
                "total": 20, "page": 1, "per_page": 20
            },
            "meta": {"request_id": "abc-123-def-456", "latency_ms": 42}
        }, indent=2),
        # Log entries as JSON
        json.dumps([
            {"timestamp": f"2026-08-06T12:{i:02d}:00Z", "level": "INFO",
             "message": f"Processing batch {i}", "service": "worker",
             "trace_id": f"trace-{i:04d}", "duration_ms": 10 + i}
            for i in range(60)
        ], indent=2),
        # Config dump
        json.dumps({
            "database": {"host": "db.example.com", "port": 5432,
                         "name": "production", "pool_size": 20,
                         "ssl": True, "timeout_ms": 5000},
            "cache": {"backend": "redis", "host": "cache.example.com",
                      "port": 6379, "ttl_seconds": 3600, "max_memory_mb": 512},
            "features": {"dark_mode": True, "beta_features": False,
                         "max_upload_mb": 100, "rate_limit_rpm": 1000},
        }, indent=2),
    ]
    return samples


# ---------------------------------------------------------------------------
# Pipeline measurement helpers
# ---------------------------------------------------------------------------

def measure_stage(name: str, func: Any, text: str) -> tuple[str, dict]:
    """Run a pipeline stage, return (output, metrics)."""
    t0 = time.perf_counter()
    out = func(text)
    dt = time.perf_counter() - t0
    return out, {
        "stage": name,
        "input_chars": len(text),
        "output_chars": len(out),
        "ratio": round(1 - len(out) / max(len(text), 1), 3),
        "time_ms": round(dt * 1000, 2),
    }


def full_pipeline_stages(text: str) -> list[dict]:
    """Run each pipeline stage independently and measure."""
    stages = []
    text, m = measure_stage("base64", compact_base64, text)
    stages.append(m)
    text, m = measure_stage("json", compact_json, text)
    stages.append(m)
    text, m = measure_stage("rules", lambda t: compress_rules(t).text, text)
    stages.append(m)
    text, m = measure_stage("lossless", compact_logs, text)
    stages.append(m)
    return stages


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def run_validation() -> dict:
    results: dict[str, Any] = {"sections": {}}

    # --- 1. Wikipedia articles ---
    wiki_texts = load_wikipedia_articles(10)
    wiki_results = []
    for i, text in enumerate(wiki_texts):
        stages = full_pipeline_stages(text)
        final = text
        for stage_fn in [compact_base64, compact_json,
                         lambda t: compress_rules(t).text, compact_logs]:
            final = stage_fn(final)
        overall_ratio = round(1 - len(final) / len(text), 3)
        wiki_results.append({
            "article_index": i,
            "input_chars": len(text),
            "output_chars": len(final),
            "overall_ratio": overall_ratio,
            "stages": stages,
        })

    avg_wiki_ratio = round(sum(r["overall_ratio"] for r in wiki_results) / len(wiki_results), 3)
    results["sections"]["wikipedia_10_articles"] = {
        "count": len(wiki_results),
        "avg_ratio": avg_wiki_ratio,
        "min_ratio": round(min(r["overall_ratio"] for r in wiki_results), 3),
        "max_ratio": round(max(r["overall_ratio"] for r in wiki_results), 3),
        "details": wiki_results,
    }

    # --- 2. Aligned multilingual articles ---
    aligned_texts = load_aligned_articles(5)
    aligned_results = []
    for i, text in enumerate(aligned_texts):
        stages = full_pipeline_stages(text)
        final = text
        for stage_fn in [compact_base64, compact_json,
                         lambda t: compress_rules(t).text, compact_logs]:
            final = stage_fn(final)
        overall_ratio = round(1 - len(final) / len(text), 3)
        aligned_results.append({
            "article_index": i,
            "input_chars": len(text),
            "output_chars": len(final),
            "overall_ratio": overall_ratio,
            "stages": stages,
        })

    avg_aligned_ratio = round(sum(r["overall_ratio"] for r in aligned_results) / len(aligned_results), 3)
    results["sections"]["aligned_articles"] = {
        "count": len(aligned_results),
        "avg_ratio": avg_aligned_ratio,
        "min_ratio": round(min(r["overall_ratio"] for r in aligned_results), 3),
        "max_ratio": round(max(r["overall_ratio"] for r in aligned_results), 3),
        "details": aligned_results,
    }

    # --- 3. HDFS logs ---
    log_text = load_hdfs_logs(500)
    log_stages = full_pipeline_stages(log_text)
    final_log = log_text
    for stage_fn in [compact_base64, compact_json,
                     lambda t: compress_rules(t).text, compact_logs]:
        final_log = stage_fn(final_log)
    log_ratio = round(1 - len(final_log) / len(log_text), 3)
    results["sections"]["hdfs_logs"] = {
        "input_chars": len(log_text),
        "output_chars": len(final_log),
        "overall_ratio": log_ratio,
        "stages": log_stages,
    }

    # --- 4. JSON payloads ---
    json_samples = make_realistic_json_samples()
    json_results = []
    labels = ["api_response", "json_log_entries", "config_dump"]
    for i, (text, label) in enumerate(zip(json_samples, labels)):
        stages = full_pipeline_stages(text)
        final = text
        for stage_fn in [compact_base64, compact_json,
                         lambda t: compress_rules(t).text, compact_logs]:
            final = stage_fn(final)
        overall_ratio = round(1 - len(final) / len(text), 3)
        json_results.append({
            "label": label,
            "input_chars": len(text),
            "output_chars": len(final),
            "overall_ratio": overall_ratio,
            "stages": stages,
        })

    avg_json_ratio = round(sum(r["overall_ratio"] for r in json_results) / len(json_results), 3)
    results["sections"]["json_payloads"] = {
        "count": len(json_results),
        "avg_ratio": avg_json_ratio,
        "details": json_results,
    }

    # --- Summary ---
    all_ratios = (
        [r["overall_ratio"] for r in wiki_results]
        + [r["overall_ratio"] for r in aligned_results]
        + [log_ratio]
        + [r["overall_ratio"] for r in json_results]
    )
    results["summary"] = {
        "total_samples": len(all_ratios),
        "overall_avg_ratio": round(sum(all_ratios) / len(all_ratios), 3),
        "overall_min_ratio": round(min(all_ratios), 3),
        "overall_max_ratio": round(max(all_ratios), 3),
    }

    return results


if __name__ == "__main__":
    results = run_validation()
    out_path = Path("reports/validation_real_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results["summary"], indent=2))
    print(f"\nFull results: {out_path}")
