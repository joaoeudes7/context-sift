"""Token-savings ledger and the ``context-sift gain`` report.

One JSON line per compaction is appended to a local store; the ``gain``
subcommand aggregates it into a summary like ``rtk gain``. The engine daemon is
the single writer for socket clients; the proxy and the one-shot CLI also record.

Only sizes and paths are stored — never the compacted text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ponytail: chars/4 matches the plugin policy heuristic; a real tokenizer buys
# accuracy the savings report does not need.
_TOKENS_PER_CHAR = 0.25


def store_path() -> Path:
    """Ledger location: ``$CONTEXT_SIFT_GAIN_FILE``, else XDG data dir."""
    override = os.environ.get("CONTEXT_SIFT_GAIN_FILE")
    if override:
        return Path(override)
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "context-sift" / "gain.jsonl"


def record(entry: dict[str, Any]) -> None:
    """Append one ledger entry. Best-effort: stats must never break compaction."""
    entry.setdefault("ts", time.time())
    try:
        path = store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        pass


def read_entries() -> list[dict[str, Any]]:
    path = store_path()
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


def _tokens(chars: float) -> int:
    return int(chars * _TOKENS_PER_CHAR)


def _human(count: float) -> str:
    value = float(count)
    for unit in ("", "K", "M", "B", "T"):
        if abs(value) < 1000:
            return f"{value:.1f}{unit}" if unit else f"{int(value)}"
        value /= 1000
    return f"{value:.1f}P"


def _summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total_in = sum(int(e.get("in", 0) or 0) for e in entries)
    total_out = sum(int(e.get("out", 0) or 0) for e in entries)
    total_ms = sum(float(e.get("ms", 0) or 0) for e in entries)
    saved = max(0, total_in - total_out)
    count = len(entries)
    return {
        "requests": count,
        "input_tokens": _tokens(total_in),
        "output_tokens": _tokens(total_out),
        "saved_tokens": _tokens(saved),
        "saved_pct": round(saved / total_in * 100, 1) if total_in else 0.0,
        "total_ms": round(total_ms, 1),
        "avg_ms": round(total_ms / count, 1) if count else 0.0,
    }


def _group(entries: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "in": 0, "out": 0})
    for entry in entries:
        name = str(entry.get(key) or "unknown")
        bucket = buckets[name]
        bucket["count"] += 1
        bucket["in"] += int(entry.get("in", 0) or 0)
        bucket["out"] += int(entry.get("out", 0) or 0)
    return dict(buckets)


def _by_day(entries: list[dict[str, Any]]) -> dict[str, int]:
    days: dict[str, int] = defaultdict(int)
    for entry in entries:
        ts = entry.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        days[day] += max(0, int(entry.get("in", 0) or 0) - int(entry.get("out", 0) or 0))
    return dict(days)


def _bar(value: int, peak: int, width: int = 20) -> str:
    filled = int(round(value / peak * width)) if peak > 0 else 0
    return "█" * filled + "░" * (width - filled)


_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[int]) -> str:
    peak = max(values) if values else 0
    if peak <= 0:
        return _SPARK[0] * len(values)
    return "".join(_SPARK[min(len(_SPARK) - 1, int(v / peak * (len(_SPARK) - 1)))] for v in values)


def _today_saved(entries: list[dict[str, Any]]) -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    return _by_day(entries).get(today, 0)


def _oneline(entries: list[dict[str, Any]]) -> str:
    summary = _summarize(entries)
    if not summary["requests"]:
        return "sift  no data"
    today = _human(_tokens(_today_saved(entries)))
    return (
        f"sift  {_human(summary['saved_tokens'])} saved · {summary['saved_pct']}% · "
        f"{summary['requests']} reqs · today {today}"
    )


def _render_text(
    entries: list[dict[str, Any]],
    project: bool,
    graph: bool,
    history: bool,
    spark: bool = False,
) -> str:
    scope = "Project Scope" if project else "Global Scope"
    summary = _summarize(entries)
    lines = [
        f"ContextSift Gain ({scope})",
        "════════════════════════════════════════════",
        f"Requests:        {summary['requests']}",
        f"Input tokens:    {_human(summary['input_tokens'])}",
        f"Output tokens:   {_human(summary['output_tokens'])}",
        f"Tokens saved:    {_human(summary['saved_tokens'])} ({summary['saved_pct']}%)",
        f"Engine time:     {summary['total_ms']}ms (avg {summary['avg_ms']}ms)",
    ]

    if spark:
        days = _by_day(entries)
        recent = sorted(days)[-14:]
        if recent:
            values = [days[d] for d in recent]
            peak = _human(_tokens(max(values)))
            lines.append(f"Last {len(recent)}d:      {_sparkline(values)} (peak {peak})")
    lines.append("")

    for key, title in (("source", "By Source"), ("cwd", "By Project")):
        groups = _group(entries, key)
        if not groups:
            continue
        lines.append(title)
        lines.append("────────────────────────────────────────────")
        for name, bucket in sorted(groups.items(), key=lambda kv: kv[1]["in"], reverse=True)[:10]:
            saved = max(0, bucket["in"] - bucket["out"])
            pct = round(saved / bucket["in"] * 100, 1) if bucket["in"] else 0.0
            lines.append(
                f"  {name[:28]:<28} {bucket['count']:>6}  {_human(_tokens(saved)):>7}  {pct:>5}%"
            )
        lines.append("")

    if graph:
        days = _by_day(entries)
        lines.append("Saved by day (tokens)")
        lines.append("────────────────────────────────────────────")
        if days:
            peak = max(days.values())
            for day in sorted(days)[-14:]:
                lines.append(f"  {day}  {_bar(days[day], peak)} {_human(_tokens(days[day]))}")
        else:
            lines.append("  (no data)")
        lines.append("")

    if history:
        lines.append("Recent requests")
        lines.append("────────────────────────────────────────────")
        for entry in entries[-20:]:
            ts = entry.get("ts", 0)
            when = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "?"
            saved = max(0, int(entry.get("in", 0) or 0) - int(entry.get("out", 0) or 0))
            lines.append(
                f"  {when}  {str(entry.get('source', '?'))[:8]:<8}  "
                f"{str(entry.get('cwd', ''))[-30:]:<30}  {_human(_tokens(saved)):>7} saved"
            )
        lines.append("")

    return "\n".join(lines)


def gain_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="context-sift gain", description="Token savings summary.")
    parser.add_argument("-p", "--project", action="store_true", help="filter to the current directory")
    parser.add_argument("-g", "--graph", action="store_true", help="show daily savings graph")
    parser.add_argument("--spark", action="store_true", help="show a daily savings sparkline")
    parser.add_argument("--days", type=int, default=0, help="only count the last N days (0 = all)")
    parser.add_argument("--oneline", action="store_true", help="one statusline-friendly line")
    parser.add_argument("-H", "--history", action="store_true", help="show recent requests")
    parser.add_argument("-j", "--json", action="store_true", dest="as_json", help="JSON output")
    parser.add_argument("--csv", action="store_true", help="CSV output")
    parser.add_argument("--reset", action="store_true", help="delete all recorded statistics")
    parser.add_argument("-y", "--yes", action="store_true", help="skip confirmation for --reset")
    args = parser.parse_args(argv)

    if args.reset:
        path = store_path()
        count = sum(1 for _ in read_entries())
        if not args.yes:
            print(f"Refusing to wipe {count} records without --yes.", file=sys.stderr)
            return 1
        path.unlink(missing_ok=True)
        print(f"Reset: removed {count} records from {path}.")
        return 0

    entries = read_entries()
    if args.project:
        entries = [e for e in entries if str(e.get("cwd") or "") == os.getcwd()]
    if args.days > 0:
        cutoff = time.time() - args.days * 86400
        entries = [e for e in entries if isinstance(e.get("ts"), (int, float)) and e["ts"] >= cutoff]

    if args.oneline:
        print(_oneline(entries))
        return 0

    if args.as_json:
        print(
            json.dumps(
                {
                    "summary": _summarize(entries),
                    "by_source": _group(entries, "source"),
                    "by_project": _group(entries, "cwd"),
                },
                indent=2,
            )
        )
        return 0

    if args.csv:
        print("ts,source,cwd,in,out,saved,ms")
        for entry in entries:
            saved = max(0, int(entry.get("in", 0) or 0) - int(entry.get("out", 0) or 0))
            print(
                f"{entry.get('ts', '')},{entry.get('source', '')},"
                f"{entry.get('cwd', '')},{entry.get('in', 0)},{entry.get('out', 0)},"
                f"{saved},{entry.get('ms', 0)}"
            )
        return 0

    print(
        _render_text(
            entries, project=args.project, graph=args.graph, history=args.history, spark=args.spark
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(gain_cli())
