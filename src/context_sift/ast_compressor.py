"""AST-based code compression.

Strips comments, docstrings, redundant whitespace, and optional name minification.
Supports Python (via stdlib ast) and JS/TS (regex-based).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


# ── Python AST compression ─────────────────────────────────────────────

@dataclass
class PythonASTCompressor(ast.NodeTransformer):
    """Strip docstrings, pass-through nodes, and redundant syntax."""

    remove_docstrings: bool = True
    remove_type_comments: bool = True

    def _strip_docstring(self, node: ast.AST) -> None:
        """Remove docstring from a module/class/function body."""
        if not self.remove_docstrings:
            return
        body = getattr(node, "body", None)
        if not body or len(body) < 1:
            return
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body.pop(0)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.expr | None:
        """Remove standalone string expressions (docstrings outside body)."""
        if self.remove_docstrings and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return node

    def visit_Pass(self, node: ast.Pass) -> ast.Pass | None:
        """Remove pass if body has other statements."""
        return None  # handled in parent




def compress_python_ast(
    source: str,
    *,
    remove_docstrings: bool = True,
    remove_type_comments: bool = True,
    minify_names: bool = False,
) -> str:
    """Compress Python source via AST round-trip.

    - Removes docstrings and type comments
    - Strips redundant pass statements
    - Optionally minifies local variable names
    """
    tree = ast.parse(source)

    # Strip docstrings and type comments
    compressor = PythonASTCompressor(
        remove_docstrings=remove_docstrings,
        remove_type_comments=remove_type_comments,
    )
    tree = ast.fix_missing_locations(compressor.visit(tree))

    # Remove orphaned pass statements (body with only pass → empty)
    _remove_orphaned_pass(tree)

    # Optionally minify local names
    if minify_names:
        _minify_local_names(tree)

    return ast.unparse(tree)


def _remove_orphaned_pass(tree: ast.Module) -> None:
    """Remove pass from bodies that have other statements."""
    for node in ast.walk(tree):
        for attr in ("body", "orelse", "finalbody", "handlers"):
            body = getattr(node, attr, None)
            if not isinstance(body, list):
                continue
            if len(body) > 1:
                body[:] = [n for n in body if not isinstance(n, ast.Pass)]


def _minify_local_names(tree: ast.Module) -> None:
    """Rename local variables to short names (a, b, c, ...)."""
    counter = [0]

    def _next_name() -> str:
        c = counter[0]
        counter[0] += 1
        if c < 26:
            return chr(ord("a") + c)
        return f"_v{c}"

    # Collect all names used in the module (builtins, globals)
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            used_names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            used_names.add(node.name)

    # Rename local variables in functions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local_names: dict[str, str] = {}
            # Rename args
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.arg in ("self", "cls"):
                    continue
                if arg.arg.startswith("_"):
                    continue
                new_name = _next_name()
                while new_name in used_names:
                    new_name = _next_name()
                local_names[arg.arg] = new_name
                used_names.add(new_name)

            # Rename local Assign targets
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in local_names:
                    child.id = local_names[child.id]
                elif isinstance(child, ast.arg) and child.arg in local_names:
                    child.arg = local_names[child.arg]


# ── JS/TS regex-based compression ──────────────────────────────────────

_JS_LINE_COMMENT = re.compile(r"//.*$", re.MULTILINE)
_JS_BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/", re.MULTILINE)
_JS_TEMPLATE_LITERAL = re.compile(r"`[^`]*`")
_JS_STRING = re.compile(r"(?:\"[^\"]*\"|'[^']*')")
_JS_TRAILING_WS = re.compile(r"[ \t]+(?=[\n;,\}\)\]])", re.MULTILINE)
_JS_BLANK_LINES = re.compile(r"\n\s*\n", re.MULTILINE)
_JS_RETURN_PAREN = re.compile(r"return\s*\(([^;]+)\)\s*;", re.MULTILINE)
_JS_EMPTY_BLOCK = re.compile(r"\{\s*\}", re.MULTILINE)


def compress_js_regex(source: str) -> str:
    """Strip JS/TS comments and redundant whitespace (regex-based, no parser).

    - Removes line/block comments (preserves strings/template literals)
    - Collapses whitespace
    - Removes empty blocks
    - Simplifies return (expr); → return expr;
    """
    # Protect strings and template literals
    protected: list[str] = []
    _place = [0]

    def _protect(m: re.Match) -> str:
        idx = _place[0]
        _place[0] += 1
        protected.append(m.group(0))
        return f"\x00PROT{idx}\x00"

    s = _JS_TEMPLATE_LITERAL.sub(_protect, source)
    s = _JS_STRING.sub(_protect, s)

    # Strip comments
    s = _JS_BLOCK_COMMENT.sub("", s)
    s = _JS_LINE_COMMENT.sub("", s)

    # Simplify return
    s = _JS_RETURN_PAREN.sub(r"return \1;", s)

    # Remove empty blocks
    s = _JS_EMPTY_BLOCK.sub("{}", s)

    # Collapse whitespace
    s = _JS_TRAILING_WS.sub("", s)
    s = _JS_BLANK_LINES.sub("\n", s)

    # Restore protected
    for i, orig in enumerate(protected):
        s = s.replace(f"\x00PROT{i}\x00", orig)

    return s.strip()


# ── Public API ──────────────────────────────────────────────────────────

@dataclass
class ASTCompressionResult:
    """Result of AST compression."""
    text: str
    language: str
    original_chars: int
    compressed_chars: int
    ratio: float = field(init=False)

    def __post_init__(self) -> None:
        self.ratio = round(1 - self.compressed_chars / max(self.original_chars, 1), 3)


def compress_code(
    source: str,
    *,
    language: str = "python",
    remove_docstrings: bool = True,
    remove_type_comments: bool = True,
    minify_names: bool = False,
) -> ASTCompressionResult:
    """Compress source code via AST or regex.

    Args:
        source: Source code string.
        language: 'python', 'javascript', 'typescript', 'jsx', 'tsx'.
        remove_docstrings: Remove docstrings (Python only).
        remove_type_comments: Remove type comments (Python only).
        minify_names: Minify local variable names (Python only).

    Returns:
        ASTCompressionResult with compressed text and metrics.
    """
    lang = language.lower()
    if lang == "python":
        compressed = compress_python_ast(
            source,
            remove_docstrings=remove_docstrings,
            remove_type_comments=remove_type_comments,
            minify_names=minify_names,
        )
    elif lang in ("javascript", "typescript", "jsx", "tsx", "js", "ts"):
        compressed = compress_js_regex(source)
    else:
        raise ValueError(f"unsupported language: {language}")

    return ASTCompressionResult(
        text=compressed,
        language=lang,
        original_chars=len(source),
        compressed_chars=len(compressed),
    )
