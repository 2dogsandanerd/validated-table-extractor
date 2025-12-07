"""
Batch processing for multiple PDFs.
"""
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger

from .extractor import TableExtractor
from .models import ValidationResult, BatchResult
from .config import Config


class BatchProcessor:
    """
    Process multiple PDFs in batch with progress tracking.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        max_workers: int = 4,
        **kwargs
    ):
        """
        Initialize batch processor.

        Args:
            config: Configuration object
            max_workers: Max parallel workers
            **kwargs: Config overrides
        """
        self.config = config or Config(**kwargs)
        self.max_workers = max_workers
        self.logger = logger.bind(component="BatchProcessor")

    def process_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        recursive: bool = False
    ) -> BatchResult:
        """
        Process all PDFs in a directory.

        Args:
            directory: Directory path
            pattern: File pattern (default: *.pdf)
            recursive: Search recursively

        Returns:
            BatchResult with aggregated results
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all PDFs
        if recursive:
            pdf_files = list(dir_path.rglob(pattern))
        else:
            pdf_files = list(dir_path.glob(pattern))

        self.logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        return self.process_files(pdf_files)

    def process_files(self, pdf_files: List[Path]) -> BatchResult:
        """
        Process list of PDF files.

        Args:
            pdf_files: List of PDF file paths

        Returns:
            BatchResult
        """
        all_results = []
        errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, pdf): pdf
                for pdf in pdf_files
            }

            # Process results with progress bar
            with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
                for future in as_completed(future_to_file):
                    pdf_file = future_to_file[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        self.logger.error(f"Failed to process {pdf_file.name}: {e}")
                        errors.append({
                            "file": str(pdf_file),
                            "error": str(e)
                        })
                    finally:
                        pbar.update(1)

        # Calculate statistics
        total_pdfs = len(pdf_files)
        total_tables = len(all_results)
        successful = len([r for r in all_results if r.confidence_score >= self.config.confidence_threshold])
        failed = total_tables - successful

        avg_confidence = sum(r.confidence_score for r in all_results) / total_tables if total_tables > 0 else 0
        high_confidence = len([r for r in all_results if r.confidence_score >= 95.0])
        low_confidence = total_tables - high_confidence

        return BatchResult(
            total_pdfs=total_pdfs,
            total_tables=total_tables,
            successful=successful,
            failed=failed,
            average_confidence=avg_confidence,
            high_confidence_count=high_confidence,
            low_confidence_count=low_confidence,
            results=all_results,
            errors=errors
        )

    def _process_single_file(self, pdf_path: Path) -> List[ValidationResult]:
        """Process single PDF file."""
        extractor = TableExtractor(self.config)
        return extractor.extract_and_validate(str(pdf_path))

    def generate_report(self, batch_result: BatchResult) -> str:
        """Generate human-readable batch report."""
        return batch_result.generate_report()
