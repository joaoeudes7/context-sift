#!/usr/bin/env python3
"""Validate the checkpoint projection: exact anchors, multilingual recall, stability.

Checkpoint mode is the only path the OpenCode integration now uses, and its output is
folded into persisted memory, so recall is the gate that matters. The projection is
extractive and makes no model call; these metrics measure selection, not inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from context_sift import Compactor, CompactorService
from context_sift.prompt_scenarios import scenarios
from validate_multilingual_handoffs import contains_natural_phrase
from validate_prompt_scenarios import contains_exact


def stability(service: CompactorService, sample: str, calls: int) -> dict[str, object]:
    """Repeated projections must be byte-identical; stop() must then refuse work."""
    first = service.checkpoint_report(sample).text
    identical = all(service.checkpoint_report(sample).text == first for _ in range(calls - 1))
    service.stop()
    refused = False
    try:
        service.checkpoint_report(sample)
    except RuntimeError:
        refused = True
    return {
        "calls": calls,
        "repeated_calls_identical": identical,
        "stop_rejects_after_stop": refused,
    }


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/context-sift"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/validation/prompts/multilingual_handoffs.jsonl"),
    )
    parser.add_argument("--history", type=Path, default=Path("reports/checkpoint_validation.jsonl"))
    parser.add_argument("--outputs", type=Path, default=Path("reports/checkpoint_samples"))
    parser.add_argument("--repetitions", type=int, default=80)
    parser.add_argument("--stability-calls", type=int, default=5)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    items = scenarios()
    args.outputs.mkdir(parents=True, exist_ok=True)

    # Probe the service surface first and release it before loading the public
    # compactor, so peak memory stays at one model.
    service = CompactorService(args.model, backend=args.backend, device=args.device)
    runtime = stability(service, items[0].text, args.stability_calls)

    compactor = Compactor(args.model, backend=args.backend, device=args.device)
    scenario_results = []
    for scenario in items:
        start = time.perf_counter()
        report = compactor.checkpoint_report(scenario.text)
        elapsed = (time.perf_counter() - start) * 1_000
        source_tokens = len(compactor.tokenizer.encode(scenario.text))
        checkpoint_tokens = len(compactor.tokenizer.encode(report.text))
        missing = [value for value in scenario.critical if not contains_exact(report.text, value)]
        (args.outputs / f"{scenario.name}.checkpoint.txt").write_text(report.text, encoding="utf-8")
        scenario_results.append(
            {
                "id": scenario.name,
                "source_chars": len(scenario.text),
                "checkpoint_chars": len(report.text),
                "source_tokens": source_tokens,
                "checkpoint_tokens": checkpoint_tokens,
                "token_reduction": 1 - checkpoint_tokens / source_tokens,
                "units": report.units,
                "kept_units": report.kept,
                "protected_units": report.protected,
                "milliseconds": elapsed,
                "critical_recall": 1 - len(missing) / len(scenario.critical),
                "missing_critical": missing,
            }
        )

    multilingual = []
    for line in args.data.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source = row["source"] + "\n" + "\n".join([row["repetition"]] * args.repetitions)
        start = time.perf_counter()
        report = compactor.checkpoint_report(source)
        elapsed = (time.perf_counter() - start) * 1_000
        missing = [
            value for value in row["anchors"] if not contains_exact(report.text, value)
        ] + [
            value
            for value in row["critical_phrases"]
            if not contains_natural_phrase(report.text, value)
        ]
        required = len(row["anchors"]) + len(row["critical_phrases"])
        source_tokens = len(compactor.tokenizer.encode(source))
        checkpoint_tokens = len(compactor.tokenizer.encode(report.text))
        (args.outputs / f"{row['language']}.checkpoint.txt").write_text(
            report.text, encoding="utf-8"
        )
        multilingual.append(
            {
                "language": row["language"],
                "source_tokens": source_tokens,
                "checkpoint_tokens": checkpoint_tokens,
                "token_reduction": 1 - checkpoint_tokens / source_tokens,
                "critical_recall": 1 - len(missing) / required,
                "missing": missing,
                "milliseconds": elapsed,
            }
        )

    latencies = [item["milliseconds"] for item in [*scenario_results, *multilingual]]
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "mode": "checkpoint",
        "runtime": runtime,
        "scenarios": scenario_results,
        "multilingual": multilingual,
        "summary": {
            "scenario_mean_token_reduction": round(
                statistics.mean(item["token_reduction"] for item in scenario_results), 4
            ),
            "scenario_min_token_reduction": round(
                min(item["token_reduction"] for item in scenario_results), 4
            ),
            "multilingual_mean_token_reduction": round(
                statistics.mean(item["token_reduction"] for item in multilingual), 4
            ),
            "min_critical_recall": round(
                min(item["critical_recall"] for item in [*scenario_results, *multilingual]), 4
            ),
            "missing_items": sum(
                len(item["missing_critical"]) for item in scenario_results
            )
            + sum(len(item["missing"]) for item in multilingual),
            "p95_ms": round(p95(latencies), 2),
        },
    }
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
