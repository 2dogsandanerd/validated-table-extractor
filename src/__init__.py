"""
Validated Table Extractor - Audit-ready PDF table extraction.

The world's first table extractor with LLM-powered validation.
"""

from .extractor import TableExtractor
from .batch import BatchProcessor
from .config import Config
from .models import ValidationResult

__version__ = "0.1.0"
__all__ = ["TableExtractor", "BatchProcessor", "Config", "ValidationResult"]
