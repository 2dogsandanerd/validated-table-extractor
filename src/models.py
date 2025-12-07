"""
Data models for validated table extraction.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json


class ValidationResult(BaseModel):
    """Result of table extraction and validation."""

    source_file: str = Field(..., description="Source PDF filename")
    source_page: int = Field(..., description="Page number (1-indexed)")
    table_index: int = Field(..., description="Table index on page (0-indexed)")

    raw_markdown_table: str = Field(..., description="Extracted table in Markdown format")

    validation_summary: str = Field(..., description="Human-readable validation summary")
    confidence_score: float = Field(..., ge=0, le=100, description="Validation confidence (0-100)")
    issues_found: List[str] = Field(default_factory=list, description="List of validation issues")

    screenshot_path: Optional[str] = Field(None, description="Path to table screenshot")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Extraction timestamp")

    # Detailed validation breakdown
    columns_validated: bool = Field(True, description="All columns present and correct")
    rows_validated: bool = Field(True, description="All rows present and correct")
    values_validated: bool = Field(True, description="Cell values match source")
    headers_validated: bool = Field(True, description="Headers correct")

    # Metadata
    extraction_time_ms: Optional[float] = Field(None, description="Extraction time in milliseconds")
    validation_time_ms: Optional[float] = Field(None, description="Validation time in milliseconds")
    llm_model_used: Optional[str] = Field(None, description="LLM model used for validation")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def export_json(self, path: str, pretty: bool = True) -> None:
        """Export result to JSON file."""
        with open(path, 'w') as f:
            if pretty:
                json.dump(self.model_dump(), f, indent=2, default=str)
            else:
                json.dump(self.model_dump(), f, default=str)

    def export_csv(self, path: str, include_metadata: bool = False) -> None:
        """Export table to CSV (with optional metadata header)."""
        import csv
        from io import StringIO

        # Parse markdown table
        lines = self.raw_markdown_table.strip().split('\n')
        rows = []
        for line in lines:
            if line.strip().startswith('|'):
                # Remove leading/trailing pipes and split
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                rows.append(cells)

        # Remove separator row (contains ---)
        rows = [row for row in rows if not all('---' in cell or '===' in cell for cell in row)]

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            if include_metadata:
                writer.writerow(['# Validated Table Extraction Metadata'])
                writer.writerow(['# Source:', self.source_file])
                writer.writerow(['# Page:', self.source_page])
                writer.writerow(['# Confidence:', f"{self.confidence_score}%"])
                writer.writerow(['# Timestamp:', self.timestamp.isoformat()])
                writer.writerow([])  # Empty line

            writer.writerows(rows)

    def export_excel(self, path: str, include_validation_sheet: bool = True) -> None:
        """Export to Excel with optional validation metadata sheet."""
        try:
            import pandas as pd
            from io import StringIO
        except ImportError:
            raise ImportError("pandas and openpyxl required for Excel export: pip install pandas openpyxl")

        # Parse markdown to DataFrame
        lines = self.raw_markdown_table.strip().split('\n')
        rows = []
        for line in lines:
            if line.strip().startswith('|'):
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                rows.append(cells)

        # Remove separator row
        rows = [row for row in rows if not all('---' in cell or '===' in cell for cell in row)]

        if not rows:
            raise ValueError("No table data to export")

        df = pd.DataFrame(rows[1:], columns=rows[0])

        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Table Data', index=False)

            if include_validation_sheet:
                metadata = {
                    'Field': [
                        'Source File',
                        'Page Number',
                        'Table Index',
                        'Confidence Score',
                        'Validation Summary',
                        'Issues Found',
                        'Columns Validated',
                        'Rows Validated',
                        'Values Validated',
                        'Timestamp',
                        'LLM Model'
                    ],
                    'Value': [
                        self.source_file,
                        self.source_page,
                        self.table_index,
                        f"{self.confidence_score}%",
                        self.validation_summary,
                        ', '.join(self.issues_found) if self.issues_found else 'None',
                        '✓' if self.columns_validated else '✗',
                        '✓' if self.rows_validated else '✗',
                        '✓' if self.values_validated else '✗',
                        self.timestamp.isoformat(),
                        self.llm_model_used or 'N/A'
                    ]
                }
                metadata_df = pd.DataFrame(metadata)
                metadata_df.to_excel(writer, sheet_name='Validation Metadata', index=False)

    def is_high_confidence(self, threshold: float = 95.0) -> bool:
        """Check if confidence score meets threshold."""
        return self.confidence_score >= threshold

    def has_issues(self) -> bool:
        """Check if any validation issues were found."""
        return len(self.issues_found) > 0


class BatchResult(BaseModel):
    """Result of batch processing."""

    total_pdfs: int = Field(..., description="Total PDFs processed")
    total_tables: int = Field(..., description="Total tables extracted")
    successful: int = Field(..., description="Successful extractions")
    failed: int = Field(..., description="Failed extractions")

    average_confidence: float = Field(..., description="Average confidence score")
    high_confidence_count: int = Field(..., description="Count of high-confidence extractions (>=95%)")
    low_confidence_count: int = Field(..., description="Count of low-confidence extractions (<95%)")

    results: List[ValidationResult] = Field(default_factory=list, description="Individual results")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Errors encountered")

    def get_low_confidence_results(self, threshold: float = 95.0) -> List[ValidationResult]:
        """Get all results below confidence threshold."""
        return [r for r in self.results if r.confidence_score < threshold]

    def get_failed_validations(self) -> List[ValidationResult]:
        """Get all results with validation issues."""
        return [r for r in self.results if r.has_issues()]

    def generate_report(self) -> str:
        """Generate human-readable report."""
        report = []
        report.append("=" * 60)
        report.append("VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Total PDFs processed: {self.total_pdfs}")
        report.append(f"Total tables extracted: {self.total_tables}")
        report.append(f"Successful extractions: {self.successful}")
        report.append(f"Failed extractions: {self.failed}")
        report.append(f"")
        report.append(f"Average confidence: {self.average_confidence:.1f}%")
        report.append(f"High confidence (>=95%): {self.high_confidence_count}")
        report.append(f"Low confidence (<95%): {self.low_confidence_count}")
        report.append("")

        if self.low_confidence_count > 0:
            report.append("⚠️  LOW CONFIDENCE EXTRACTIONS:")
            report.append("-" * 60)
            for result in self.get_low_confidence_results():
                report.append(f"  • {result.source_file} (page {result.source_page}): "
                            f"{result.confidence_score:.1f}%")
                if result.issues_found:
                    for issue in result.issues_found:
                        report.append(f"    - {issue}")
            report.append("")

        if self.errors:
            report.append("❌ ERRORS:")
            report.append("-" * 60)
            for error in self.errors:
                report.append(f"  • {error['file']}: {error['error']}")
            report.append("")

        report.append("=" * 60)
        return "\n".join(report)
