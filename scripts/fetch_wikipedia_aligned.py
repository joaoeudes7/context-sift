"""Fetch same Wikipedia concepts across languages, grouped by Wikidata QID."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from context_sift.aligned_wikipedia import merge_page_queries, parse_page_extracts, parse_page_links_all


HEADERS = {"User-Agent": "compact-llm-summary/0.1 research dataset"}


async def api(client: httpx.AsyncClient, language: str, params: dict[str, str]) -> dict:
    for attempt in range(5):
        response = await client.get(f"https://{language}.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "formatversion": "1", **params,
        })
        if response.status_code == 429:
            await asyncio.sleep(float(response.headers.get("Retry-After", 2 ** attempt)))
            continue
        response.raise_for_status()
        return response.json()
    raise httpx.HTTPStatusError("rate limit retries exhausted", request=response.request, response=response)


async def fetch_groups(
    client: httpx.AsyncClient, titles: list[str], languages: set[str], min_chars: int
) -> list[dict]:
    params = {
        "prop": "langlinks|pageprops", "ppprop": "wikibase_item",
        "lllimit": "max", "redirects": "1", "titles": "|".join(titles),
    }
    body: dict = {}
    while True:
        page = await api(client, "en", params)
        merge_page_queries(body, page)
        continuation = page.get("continue")
        if not continuation:
            break
        params = {**params, **{key: str(value) for key, value in continuation.items()}}
    groups = parse_page_links_all(body, languages)
    requested = {
        language: [group["titles"][language] for group in groups if language in group["titles"]]
        for language in languages
    }
    responses = await asyncio.gather(*(
        api(client, language, {
            "prop": "extracts", "explaintext": "1", "exintro": "1", "exlimit": "max", "redirects": "1",
            "titles": "|".join(language_titles),
        }) if language_titles else asyncio.sleep(0, result={})
        for language, language_titles in requested.items()
    ))
    extracts = {
        language: parse_page_extracts(response, min_chars=min_chars)
        for language, response in zip(requested, responses, strict=True)
    }
    accepted = []
    for group in groups:
        pages = {}
        for language, title in group["titles"].items():
            text = extracts[language].get(title.casefold())
            if text:
                pages[language] = {"title": title, "text": text}
        if len(pages) >= 2:
            accepted.append({"group_id": group["group_id"], "pages": pages})
    return accepted


async def run(source: Path, output: Path, languages: set[str], count: int, min_chars: int) -> None:
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    titles = list(dict.fromkeys(
        row["metadata"]["title"] for row in rows if row.get("language") == "en"
    ))[:count]
    async with httpx.AsyncClient(headers=HEADERS, timeout=60) as client:
        accepted = []
        for start in range(0, len(titles), 20):
            accepted.extend(await fetch_groups(client, titles[start:start + 20], languages, min_chars))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for group in accepted:
            stream.write(json.dumps(group, ensure_ascii=False) + "\n")
    coverage = {language: sum(language in group["pages"] for group in accepted) for language in sorted(languages)}
    print(json.dumps({"groups": len(accepted), "coverage": coverage}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/wikipedia/long.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/wikipedia/aligned.jsonl"))
    parser.add_argument("--languages", default="en,pt,es,fr,de")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=1_000)
    args = parser.parse_args()
    asyncio.run(run(args.source, args.output, set(args.languages.split(",")), args.count, args.min_chars))


if __name__ == "__main__":
    main()
