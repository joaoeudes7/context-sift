"""Synthetic dataset builder for prompt compaction."""

from compact_dataset.orchestrator import BuildConfig, DatasetBuilder
from compact_dataset.msc import DEFAULT_MODEL_PATH, Compactor, CompactorService

__all__ = ["BuildConfig", "Compactor", "CompactorService", "DEFAULT_MODEL_PATH", "DatasetBuilder"]
