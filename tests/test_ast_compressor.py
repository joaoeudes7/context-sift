"""Tests for AST code compressor."""

import unittest

from context_sift.ast_compressor import (
    ASTCompressionResult,
    compress_code,
    compress_js_regex,
    compress_python_ast,
)


class PythonASTTests(unittest.TestCase):
    SAMPLE = '''
"""Module docstring."""

import os
from pathlib import Path


class MyClass:
    """Class docstring."""

    def __init__(self, x: int, y: str = "hello") -> None:
        """Init docstring."""
        self.x = x
        self.y = y
        pass

    def method(self) -> int:
        """Method docstring."""
        result = self.x + 1
        return result


def standalone() -> None:
    """Do something."""
    x = 42
    y = x * 2
    # This is a comment
    print(y)
    pass
'''

    def test_removes_docstrings(self) -> None:
        result = compress_python_ast(self.SAMPLE, remove_docstrings=True)
        self.assertNotIn('"""Module docstring."""', result)
        self.assertNotIn('"""Class docstring."""', result)
        self.assertNotIn('"""Init docstring."""', result)
        self.assertNotIn('"""Method docstring."""', result)
        self.assertNotIn('"""Do something."""', result)

    def test_keeps_docstrings(self) -> None:
        result = compress_python_ast(self.SAMPLE, remove_docstrings=False)
        self.assertIn('"""Module docstring."""', result)

    def test_removes_pass(self) -> None:
        result = compress_python_ast(self.SAMPLE)
        # __init__ has self.x, self.y, pass — pass should be removed
        # Method has result = ..., return — no pass to remove
        # standalone has x=..., y=..., print, pass — pass should be removed
        lines = result.splitlines()
        pass_lines = [l.strip() for l in lines if l.strip() == "pass"]
        self.assertEqual(len(pass_lines), 0)

    def test_preserves_structure(self) -> None:
        result = compress_python_ast(self.SAMPLE)
        self.assertIn("class MyClass", result)
        self.assertIn("def __init__", result)
        self.assertIn("def method", result)
        self.assertIn("def standalone", result)
        self.assertIn("import os", result)

    def test_removes_comments(self) -> None:
        result = compress_python_ast(self.SAMPLE)
        self.assertNotIn("# This is a comment", result)

    def test_minify_names(self) -> None:
        result = compress_python_ast(self.SAMPLE, minify_names=True)
        # Function/class names preserved, but args and locals are renamed
        self.assertIn("standalone", result)  # function name kept
        self.assertIn("MyClass", result)      # class name kept
        self.assertNotIn("greeting", result)   # local var renamed
        self.assertNotIn("def method(self, x: int, y: str", result)  # args renamed

    def test_roundtrip_valid(self) -> None:
        result = compress_python_ast(self.SAMPLE)
        # Should be valid Python
        compile(result, "<test>", "exec")

    def test_compression_ratio(self) -> None:
        result = compress_code(self.SAMPLE, language="python")
        self.assertIsInstance(result, ASTCompressionResult)
        self.assertGreater(result.ratio, 0.0)
        self.assertLess(result.compressed_chars, result.original_chars)


class JSRegexTests(unittest.TestCase):
    SAMPLE = '''
// This is a comment
function hello(name) {
    /* Block comment
       spanning multiple lines */
    const greeting = "Hello, " + name;
    console.log(greeting);
    return (greeting);
}

/**
 * JSDoc comment
 * @param {string} x
 */
const foo = (x) => {
    return x;
};

const bar = { };
'''

    def test_removes_line_comments(self) -> None:
        result = compress_js_regex(self.SAMPLE)
        self.assertNotIn("// This is a comment", result)

    def test_removes_block_comments(self) -> None:
        result = compress_js_regex(self.SAMPLE)
        self.assertNotIn("/* Block comment", result)
        self.assertNotIn("@param", result)

    def test_preserves_strings(self) -> None:
        result = compress_js_regex(self.SAMPLE)
        self.assertIn('"Hello, "', result)

    def test_simplifies_return(self) -> None:
        result = compress_js_regex(self.SAMPLE)
        self.assertIn("return greeting;", result)
        self.assertNotIn("return (greeting);", result)

    def test_removes_empty_blocks(self) -> None:
        result = compress_js_regex(self.SAMPLE)
        self.assertNotIn("{ }", result)

    def test_compression_ratio(self) -> None:
        result = compress_code(self.SAMPLE, language="javascript")
        self.assertIsInstance(result, ASTCompressionResult)
        self.assertGreater(result.ratio, 0.0)


class UnsupportedLanguageTests(unittest.TestCase):
    def test_raises_on_unsupported(self) -> None:
        with self.assertRaises(ValueError):
            compress_code("print('hi')", language="rust")


if __name__ == "__main__":
    unittest.main()
