"""
Configuration for validated table extractor.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class Config(BaseModel):
    """Configuration for table extraction and validation."""

    # LLM Settings
    llm_provider: str = Field("ollama", description="LLM provider: ollama or openai")
    model: str = Field("llama3.2-vision:11b", description="Vision model for validation")
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama base URL")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")

    # Validation Settings
    confidence_threshold: float = Field(95.0, ge=0, le=100, description="Minimum confidence score")
    save_screenshots: bool = Field(True, description="Save table screenshots")
    output_dir: str = Field("validated_tables/", description="Output directory")

    # Extraction Settings
    docling_options: Dict[str, Any] = Field(
        default_factory=lambda: {
            "do_ocr": False,
            "do_table_structure": True
        },
        description="Docling pipeline options"
    )

    # Validation Rules
    validation_rules: Dict[str, Any] = Field(
        default_factory=lambda: {
            "require_headers": True,
            "min_columns": 2,
            "numeric_precision": 0.01,
            "detect_merged_cells": True,
            "verify_calculations": False
        },
        description="Validation rules"
    )

    # Performance
    timeout_seconds: int = Field(60, description="Timeout for LLM validation")
    max_retries: int = Field(3, description="Max retries on failure")

    class Config:
        extra = "allow"
