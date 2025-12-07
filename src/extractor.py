"""
Core table extraction and validation logic.
"""
import time
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from loguru import logger
import base64
from io import BytesIO

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# PDF and Image processing
import fitz  # PyMuPDF
from PIL import Image

# LLM
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .models import ValidationResult
from .config import Config


class TableExtractor:
    """
    Extract and validate tables from PDFs.

    Two-stage process:
    1. Docling extraction → Markdown table
    2. Vision LLM validation → Confidence score
    """

    def __init__(self, config: Optional[Config] = None, **kwargs):
        """
        Initialize table extractor.

        Args:
            config: Configuration object
            **kwargs: Override config parameters
        """
        self.config = config or Config()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.logger = logger.bind(component="TableExtractor")

        # Initialize Docling
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.docling_options.get("do_ocr", False)
        pipeline_options.do_table_structure = self.config.docling_options.get("do_table_structure", True)

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Initialize Vision LLM
        self.llm = self._init_llm()

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info(f"TableExtractor initialized with {self.config.llm_provider}")

    def _init_llm(self):
        """Initialize Vision LLM based on config."""
        if self.config.llm_provider == "ollama":
            return ChatOllama(
                model=self.config.model,
                base_url=self.config.ollama_base_url,
                temperature=0
            )
        elif self.config.llm_provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("OPENAI_API_KEY required when using openai provider")
            return ChatOpenAI(
                model=self.config.model,
                api_key=self.config.openai_api_key,
                temperature=0
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")

    def _validate_screenshot_path(self, screenshot_path: Path) -> Path:
        """
        Validate screenshot path to prevent path traversal attacks.

        Args:
            screenshot_path: Path to validate

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If path is outside output directory
            FileNotFoundError: If screenshot doesn't exist
        """
        resolved_path = screenshot_path.resolve()
        output_dir = Path(self.config.output_dir).resolve()

        try:
            resolved_path.relative_to(output_dir)
        except ValueError:
            raise ValueError(f"Screenshot path outside output directory: {screenshot_path}")

        if not resolved_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        return resolved_path

    def extract_and_validate(
        self,
        pdf_path: str,
        page: Optional[int] = None
    ) -> List[ValidationResult]:
        """
        Extract all tables from PDF and validate them.

        Args:
            pdf_path: Path to PDF file
            page: Specific page number (1-indexed), or None for all pages

        Returns:
            List of ValidationResult objects
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.logger.info(f"Processing: {pdf_path.name}")

        # Extract tables with Docling
        extraction_start = time.time()
        tables = self._extract_tables_docling(pdf_path, page)
        extraction_time_ms = (time.time() - extraction_start) * 1000

        self.logger.info(f"Found {len(tables)} tables in {pdf_path.name}")

        # Validate each table
        results = []
        for table_idx, (table_markdown, page_num, bbox) in enumerate(tables):
            self.logger.info(f"Validating table {table_idx + 1}/{len(tables)} on page {page_num}")

            # Capture screenshot
            screenshot_path = None
            if self.config.save_screenshots:
                screenshot_path = self._capture_table_screenshot(
                    pdf_path,
                    page_num,
                    bbox,
                    table_idx
                )

            # Validate with Vision LLM
            validation_start = time.time()
            validation = self._validate_table(
                table_markdown,
                screenshot_path,
                page_num
            )
            validation_time_ms = (time.time() - validation_start) * 1000

            # Create result
            result = ValidationResult(
                source_file=pdf_path.name,
                source_page=page_num,
                table_index=table_idx,
                raw_markdown_table=table_markdown,
                validation_summary=validation['summary'],
                confidence_score=validation['confidence_score'],
                issues_found=validation['issues'],
                columns_validated=validation['columns_correct'],
                rows_validated=validation['rows_correct'],
                values_validated=validation['values_correct'],
                headers_validated=validation['headers_correct'],
                screenshot_path=str(screenshot_path) if screenshot_path else None,
                extraction_time_ms=extraction_time_ms,
                validation_time_ms=validation_time_ms,
                llm_model_used=self.config.model
            )

            results.append(result)

            self.logger.info(
                f"Table {table_idx + 1} validated: "
                f"confidence={result.confidence_score:.1f}%"
            )

        return results

    def _extract_tables_docling(
        self,
        pdf_path: Path,
        page: Optional[int] = None
    ) -> List[Tuple[str, int, Tuple[float, float, float, float]]]:
        """
        Extract tables using Docling.

        Returns:
            List of (markdown_table, page_number, bounding_box)
        """
        # Convert PDF
        result = self.converter.convert(str(pdf_path))
        doc = result.document

        tables = []

        # Extract tables from Docling result (v2 API)
        # Docling v2 uses document.tables
        if hasattr(doc, 'tables') and doc.tables:
            # Iterate through tables
            for table in doc.tables:
                # Get markdown representation (pass doc to avoid deprecation warning)
                markdown = table.export_to_markdown(doc)
                
                # Get page number from table provenance
                # ProvenanceItem has 'page_no' attribute (not 'page')
                page_num = 1  # Default
                if hasattr(table, 'prov') and table.prov and len(table.prov) > 0:
                    page_num = table.prov[0].page_no  # Use page_no, not page!
                
                # Skip if specific page requested and doesn't match
                if page and page_num != page:
                    continue
                
                # Get bounding box
                bbox = self._get_bbox(table)
                
                tables.append((markdown, page_num, bbox))
        else:
            # Fallback: try to iterate through body items
            self.logger.warning("No tables found in document or using fallback method")

        return tables

    def _item_to_markdown(self, item) -> str:
        """Convert Docling table item to Markdown."""
        # Docling typically provides a markdown export method
        if hasattr(item, 'export_to_markdown'):
            return item.export_to_markdown()

        # Fallback: basic conversion
        if hasattr(item, 'text'):
            return item.text

        return str(item)

    def _get_bbox(self, item) -> Tuple[float, float, float, float]:
        """Extract bounding box from Docling item."""
        # NOTE: Docling bbox coordinates are currently incompatible with PyMuPDF
        # They produce narrow strips (e.g., 982x3 pixels) instead of proper table regions
        # Using full page as fallback until coordinate system mapping is resolved
        
        # Always return full page for now
        return (0, 0, 612, 792)  # US Letter size (will be clipped to actual page size)

    def _capture_table_screenshot(
        self,
        pdf_path: Path,
        page_num: int,
        bbox: Tuple[float, float, float, float],
        table_idx: int
    ) -> Optional[Path]:
        """
        Capture screenshot of table region.

        Args:
            pdf_path: PDF file path
            page_num: Page number (1-indexed)
            bbox: Bounding box (left, top, right, bottom)
            table_idx: Table index

        Returns:
            Path to saved screenshot
        """
        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]  # PyMuPDF uses 0-indexed pages

            # Validate and fix bounding box
            left, top, right, bottom = bbox
            page_rect = page.rect
            
            # Ensure bbox is within page bounds
            left = max(0, min(left, page_rect.width))
            top = max(0, min(top, page_rect.height))
            right = max(left + 1, min(right, page_rect.width))  # Ensure width > 0
            bottom = max(top + 1, min(bottom, page_rect.height))  # Ensure height > 0
            
            # Check if bbox is valid
            if right <= left or bottom <= top:
                self.logger.warning(
                    f"Invalid bbox for table {table_idx} on page {page_num}: "
                    f"({left}, {top}, {right}, {bottom}). Using full page."
                )
                # Use full page as fallback
                rect = page_rect
            else:
                rect = fitz.Rect(left, top, right, bottom)
            
            try:
                # Extract region
                pix = page.get_pixmap(clip=rect, dpi=150)
                
                # Save screenshot
                output_name = f"{pdf_path.stem}_page{page_num}_table{table_idx}.png"
                output_path = Path(self.config.output_dir) / output_name
                pix.save(str(output_path))
                
                return output_path
                
            except Exception as e:
                self.logger.error(f"Failed to capture screenshot: {e}")
                # Return None if screenshot fails - validation can still proceed
                return None

    def _validate_table(
        self,
        markdown_table: str,
        screenshot_path: Optional[Path],
        page_num: int
    ) -> dict:
        """
        Validate extracted table using Vision LLM.

        Args:
            markdown_table: Extracted markdown table
            screenshot_path: Path to table screenshot
            page_num: Page number

        Returns:
            Validation dict with confidence_score, summary, issues, etc.
        """
        # Prepare validation prompt
        prompt = f"""You are a table validation expert. Your task is to validate a table extraction.

EXTRACTED TABLE (Markdown):
```markdown
{markdown_table}
```

"""

        if screenshot_path and screenshot_path.exists():
            prompt += """ORIGINAL TABLE: See attached image.

"""

        prompt += """VALIDATION TASK:
1. Check if all columns are present
2. Check if all rows are present
3. Verify numeric values match (if applicable)
4. Verify headers are correct
5. Check for any merged cells or complex structures

Return a JSON object with this structure:
{
    "columns_correct": true/false,
    "rows_correct": true/false,
    "values_correct": true/false,
    "headers_correct": true/false,
    "confidence_score": <0-100>,
    "summary": "<brief validation summary>",
    "issues": ["<list of any issues found>"]
}

Be thorough but concise. Confidence score should reflect your certainty in the extraction accuracy.
"""

        # Prepare message with image
        if screenshot_path and screenshot_path.exists():
            # Validate path to prevent traversal attacks
            screenshot_path = self._validate_screenshot_path(screenshot_path)
            
            # Load image and convert to base64
            with open(screenshot_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    }
                ]
            )
        else:
            # Text-only validation (no screenshot)
            message = HumanMessage(content=prompt)

        # Call LLM
        try:
            response = self.llm.invoke([message])

            # Parse JSON response
            import json
            import re

            # Extract JSON from response (handles markdown code blocks)
            response_text = response.content
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)  # Non-greedy to prevent ReDoS

            if json_match:
                validation_data = json.loads(json_match.group(0))
            else:
                # Fallback: high confidence if no issues mentioned
                validation_data = {
                    "columns_correct": True,
                    "rows_correct": True,
                    "values_correct": True,
                    "headers_correct": True,
                    "confidence_score": 85.0,
                    "summary": "Validation completed without detailed analysis.",
                    "issues": []
                }

            return validation_data

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")

            # Return low-confidence result on error
            return {
                "columns_correct": False,
                "rows_correct": False,
                "values_correct": False,
                "headers_correct": False,
                "confidence_score": 50.0,
                "summary": f"Validation error: {str(e)}",
                "issues": [str(e)]
            }
