"""ContextSift — tiny multilingual context compactor for faster LLM prefill."""

from context_sift.ast_compressor import ASTCompressionResult, compress_code
from context_sift.lossless import collapse_spaces, trim_output
from context_sift.msc import DEFAULT_MODEL_PATH, Compactor, CompactorService

__all__ = [
    "Compactor",
    "CompactorService",
    "DEFAULT_MODEL_PATH",
    "compress_code",
    "ASTCompressionResult",
    "collapse_spaces",
    "trim_output",
]
