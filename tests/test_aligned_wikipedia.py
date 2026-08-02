import unittest

from compact_dataset.aligned_wikipedia import (
    merge_page_queries,
    parse_page_extracts,
    parse_page_links,
    parse_page_links_all,
)


class AlignedWikipediaTests(unittest.TestCase):
    def test_extracts_qid_and_requested_language_titles(self) -> None:
        body = {"query": {"pages": {"1": {
            "title": "Dog",
            "pageprops": {"wikibase_item": "Q144"},
            "langlinks": [
                {"lang": "pt", "*": "Cão"},
                {"lang": "es", "*": "Canis familiaris"},
                {"lang": "ja", "*": "イヌ"},
            ],
        }}}}
        result = parse_page_links(body, {"en", "pt", "es"})
        self.assertEqual(result["group_id"], "Q144")
        self.assertEqual(result["titles"], {"en": "Dog", "pt": "Cão", "es": "Canis familiaris"})

    def test_rejects_page_without_qid(self) -> None:
        body = {"query": {"pages": {"1": {"title": "Unknown"}}}}
        self.assertIsNone(parse_page_links(body, {"en", "pt"}))

    def test_parses_multiple_pages_from_batched_response(self) -> None:
        body = {"query": {"pages": {
            "1": {"title": "Dog", "pageprops": {"wikibase_item": "Q144"},
                  "langlinks": [{"lang": "pt", "*": "Cão"}]},
            "2": {"title": "Cat", "pageprops": {"wikibase_item": "Q146"},
                  "langlinks": [{"lang": "pt", "*": "Gato"}]},
        }}}
        groups = parse_page_links_all(body, {"en", "pt"})
        self.assertEqual({group["group_id"] for group in groups}, {"Q144", "Q146"})

    def test_extracts_resolve_normalized_and_redirected_titles(self) -> None:
        text = "x" * 1_000
        body = {"query": {
            "normalized": [{"from": "dog", "to": "Dog"}],
            "redirects": [{"from": "Domestic dog", "to": "Dog"}],
            "pages": {"1": {"title": "Dog", "extract": text}},
        }}
        extracts = parse_page_extracts(body)
        self.assertEqual(extracts["dog"], text)
        self.assertEqual(extracts["domestic dog"], text)

    def test_merges_paginated_language_links(self) -> None:
        first = {"query": {"pages": {"1": {"title": "Dog", "langlinks": [{"lang": "pt", "*": "Cão"}]}}}}
        second = {"query": {"pages": {"1": {"title": "Dog", "langlinks": [{"lang": "es", "*": "Perro"}]}}}}
        merged = merge_page_queries(first, second)
        self.assertEqual([link["lang"] for link in merged["query"]["pages"]["1"]["langlinks"]], ["pt", "es"])


if __name__ == "__main__":
    unittest.main()
