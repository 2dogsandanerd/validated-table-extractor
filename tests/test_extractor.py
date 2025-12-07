"""
Tests for TableExtractor.
"""
import pytest
from pathlib import Path
from src import TableExtractor, Config
from src.models import ValidationResult


class TestTableExtractor:
    """Test TableExtractor functionality."""

    def test_init_with_ollama(self):
        """Test initialization with Ollama provider."""
        extractor = TableExtractor(llm_provider="ollama")
        assert extractor.config.llm_provider == "ollama"
        assert extractor.llm is not None

    def test_init_with_config(self):
        """Test initialization with Config object."""
        config = Config(
            llm_provider="ollama",
            confidence_threshold=98.0,
            save_screenshots=False
        )
        extractor = TableExtractor(config)
        assert extractor.config.confidence_threshold == 98.0
        assert extractor.config.save_screenshots is False

    def test_init_invalid_provider(self):
        """Test initialization with invalid provider."""
        with pytest.raises(ValueError):
            TableExtractor(llm_provider="invalid")

    # Note: Full extraction tests require sample PDFs
    # Add integration tests with actual PDFs separately


class TestValidationResult:
    """Test ValidationResult model."""

    def test_create_result(self):
        """Test creating ValidationResult."""
        result = ValidationResult(
            source_file="test.pdf",
            source_page=1,
            table_index=0,
            raw_markdown_table="| A | B |\n|---|---|\n| 1 | 2 |",
            validation_summary="All correct",
            confidence_score=99.5,
            issues_found=[]
        )

        assert result.source_file == "test.pdf"
        assert result.confidence_score == 99.5
        assert result.is_high_confidence(95.0)
        assert not result.has_issues()

    def test_low_confidence(self):
        """Test low confidence detection."""
        result = ValidationResult(
            source_file="test.pdf",
            source_page=1,
            table_index=0,
            raw_markdown_table="| A |\n|---|\n| 1 |",
            validation_summary="Issues found",
            confidence_score=85.0,
            issues_found=["Missing column"]
        )

        assert not result.is_high_confidence(95.0)
        assert result.has_issues()
        assert len(result.issues_found) == 1

    def test_export_json(self, tmp_path):
        """Test JSON export."""
        result = ValidationResult(
            source_file="test.pdf",
            source_page=1,
            table_index=0,
            raw_markdown_table="| A | B |\n|---|---|\n| 1 | 2 |",
            validation_summary="OK",
            confidence_score=99.0,
            issues_found=[]
        )

        output_path = tmp_path / "result.json"
        result.export_json(str(output_path))

        assert output_path.exists()

        # Verify JSON content
        import json
        with open(output_path) as f:
            data = json.load(f)

        assert data['confidence_score'] == 99.0
        assert data['source_file'] == "test.pdf"


class TestConfig:
    """Test Config model."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.llm_provider == "ollama"
        assert config.confidence_threshold == 95.0
        assert config.save_screenshots is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = Config(
            llm_provider="openai",
            model="gpt-4o-mini",
            confidence_threshold=98.0,
            validation_rules={
                "require_headers": True,
                "min_columns": 3
            }
        )

        assert config.llm_provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.validation_rules["min_columns"] == 3
