"""Fetch long PT/EN Wikipedia rows through Hugging Face Dataset Viewer."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from compact_dataset.wikipedia_long import build_long_rows


API = "https://datasets-server.huggingface.co/rows"


async def fetch_config(config: str, wanted: int, start: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = start
    async with httpx.AsyncClient(timeout=60) as client:
        while len(rows) < wanted:
            response = await client.get(API, params={
                "dataset": "wikimedia/wikipedia",
                "config": config,
                "split": "train",
                "offset": offset,
                "length": 100,
            })
            response.raise_for_status()
            page = response.json().get("rows", [])
            if not page:
                break
            for item in page:
                article = item["row"]
                for row in build_long_rows(article["title"], article["url"], article["text"]):
                    row["language"] = config.rsplit(".", 1)[-1]
                    rows.append(row)
                    if len(rows) >= wanted:
                        break
                if len(rows) >= wanted:
                    break
            offset += len(page)
    return rows


async def run(output: Path, count_per_language: int, start: int) -> None:
    batches = await asyncio.gather(
        fetch_config("20231101.pt", count_per_language, start),
        fetch_config("20231101.en", count_per_language, start),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in [*batches[0], *batches[1]]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": sum(map(len, batches)), "pt": len(batches[0]), "en": len(batches[1])}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/wikipedia/long.jsonl"))
    parser.add_argument("--count-per-language", type=int, default=25)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args.output, args.count_per_language, args.start))


if __name__ == "__main__":
    main()
