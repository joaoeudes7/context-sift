"""Tests for collapse_spaces and trim_output."""

import unittest

from context_sift.lossless import collapse_spaces, trim_output


class CollapseSpacesTests(unittest.TestCase):
    def test_preserves_indentation(self) -> None:
        code = "def foo():\n    x  =  1\n    if   True:\n        return   True"
        result = collapse_spaces(code)
        self.assertEqual(result, "def foo():\n    x = 1\n    if True:\n        return True")

    def test_collapses_internal_spaces(self) -> None:
        text = "hello    world   foo"
        self.assertEqual(collapse_spaces(text), "hello world foo")

    def test_strips_trailing_whitespace(self) -> None:
        text = "hello   \nworld   \t  "
        self.assertEqual(collapse_spaces(text), "hello\nworld")

    def test_collapses_blank_lines(self) -> None:
        text = "a\n\n\n\n\nb"
        self.assertEqual(collapse_spaces(text), "a\n\nb")

    def test_preserves_single_newlines(self) -> None:
        text = "line1\nline2\nline3"
        self.assertEqual(collapse_spaces(text), "line1\nline2\nline3")

    def test_empty_input(self) -> None:
        self.assertEqual(collapse_spaces(""), "")


class TrimOutputTests(unittest.TestCase):
    def test_strips_ceremony_preamble(self) -> None:
        text = "Sure! Let me help you with that.\nThe answer is 42."
        result = trim_output(text)
        self.assertNotIn("Sure!", result)
        self.assertIn("The answer is 42.", result)

    def test_strips_trailing_filler(self) -> None:
        text = "The answer is 42.\n\nLet me know if you need anything else!"
        result = trim_output(text)
        self.assertIn("The answer is 42.", result)
        self.assertNotIn("Let me know", result)

    def test_removes_echoed_lines(self) -> None:
        context = "The quick brown fox jumps over the lazy dog. " * 3
        output = "The quick brown fox jumps over the lazy dog. " * 3 + "\nHere is the answer."
        result = trim_output(output, context=context)
        self.assertIn("Here is the answer.", result)

    def test_keeps_unique_content(self) -> None:
        text = "The answer is 42.\nThis is unique content."
        result = trim_output(text)
        self.assertIn("The answer is 42.", result)
        self.assertIn("This is unique content.", result)

    def test_returns_original_if_no_trim(self) -> None:
        text = "Short text."
        self.assertEqual(trim_output(text), text)

    def test_portuese_ceremony(self) -> None:
        text = "Claro! Vou ajudar com isso.\nA resposta é 42."
        result = trim_output(text)
        self.assertNotIn("Claro!", result)
        self.assertIn("A resposta é 42.", result)


if __name__ == "__main__":
    unittest.main()
