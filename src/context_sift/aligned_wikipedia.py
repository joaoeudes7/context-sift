"""Helpers for Wikidata-QID grouped multilingual Wikipedia pages."""

from __future__ import annotations

from typing import Any


def merge_page_queries(target: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge paginated MediaWiki page properties without losing langlinks."""
    target_pages = target.setdefault("query", {}).setdefault("pages", {})
    for page_id, page in incoming.get("query", {}).get("pages", {}).items():
        current = target_pages.setdefault(page_id, {})
        for key, value in page.items():
            if key == "langlinks":
                current.setdefault(key, []).extend(value)
            else:
                current[key] = value
    return target


def parse_page_links(body: dict[str, Any], languages: set[str]) -> dict[str, Any] | None:
    pages = body.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    qid = page.get("pageprops", {}).get("wikibase_item")
    if not qid:
        return None
    titles = {"en": str(page["title"])} if "en" in languages else {}
    titles.update({
        link["lang"]: link["*"]
        for link in page.get("langlinks", [])
        if link.get("lang") in languages and link.get("*")
    })
    return {"group_id": qid, "titles": titles}


def parse_page_links_all(body: dict[str, Any], languages: set[str]) -> list[dict[str, Any]]:
    groups = []
    for page in body.get("query", {}).get("pages", {}).values():
        qid = page.get("pageprops", {}).get("wikibase_item")
        if not qid:
            continue
        titles = {"en": str(page["title"])} if "en" in languages else {}
        titles.update({
            link["lang"]: link["*"]
            for link in page.get("langlinks", [])
            if link.get("lang") in languages and link.get("*")
        })
        if len(titles) >= 2:
            groups.append({"group_id": qid, "titles": titles})
    return groups


def parse_page_extracts(body: dict[str, Any], min_chars: int = 1_000) -> dict[str, str]:
    """Return extracts indexed by requested and canonical title."""
    aliases: dict[str, str] = {}
    query = body.get("query", {})
    for key in ("normalized", "redirects"):
        for item in query.get(key, []):
            source, target = item.get("from"), item.get("to")
            if source and target:
                aliases[str(source).casefold()] = str(target).casefold()

    extracts: dict[str, str] = {}
    for page in query.get("pages", {}).values():
        title, text = page.get("title"), page.get("extract")
        if isinstance(title, str) and isinstance(text, str) and len(text) >= min_chars:
            extracts[title.casefold()] = text
    for source, target in aliases.items():
        if target in extracts:
            extracts[source] = extracts[target]
    return extracts
