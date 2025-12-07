"""
Quick start example for Validated Table Extractor.
"""
from src import TableExtractor, Config
from pathlib import Path


def main():
    """Run basic extraction example."""

    # Example 1: Simple extraction with Ollama (100% local, free)
    print("=" * 60)
    print("Example 1: Basic Table Extraction")
    print("=" * 60)

    extractor = TableExtractor(
        llm_provider="ollama",
        model="llama3.2-vision:11b",
        save_screenshots=True
    )

    # Extract from sample PDF (you need to provide your own)
    pdf_path = "sample_invoice.pdf"

    if not Path(pdf_path).exists():
        print(f"\n⚠️  Sample PDF not found: {pdf_path}")
        print("Please provide your own PDF file for testing.")
        return

    results = extractor.extract_and_validate(pdf_path)

    for i, result in enumerate(results):
        print(f"\n--- Table {i + 1} ---")
        print(f"Page: {result.source_page}")
        print(f"Confidence: {result.confidence_score:.1f}%")
        print(f"Validation: {result.validation_summary}")
        print(f"\nMarkdown Table:")
        print(result.raw_markdown_table)

        if result.issues_found:
            print(f"\n⚠️  Issues:")
            for issue in result.issues_found:
                print(f"  - {issue}")

        # Export to different formats
        if result.is_high_confidence():
            result.export_json(f"table_{i + 1}.json", pretty=True)
            result.export_csv(f"table_{i + 1}.csv", include_metadata=True)
            print(f"\n✅ Exported table_{i + 1}.json and table_{i + 1}.csv")

    # Example 2: Batch processing
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)

    from src import BatchProcessor

    processor = BatchProcessor(
        confidence_threshold=95.0,
        max_workers=4
    )

    # Process directory (create a 'pdfs/' directory with sample PDFs)
    if Path("pdfs/").exists():
        batch_result = processor.process_directory("pdfs/")

        print("\n" + batch_result.generate_report())

        # Save low-confidence results for review
        low_conf = batch_result.get_low_confidence_results()
        if low_conf:
            print(f"\n⚠️  {len(low_conf)} tables need manual review")
            for result in low_conf:
                print(f"  - {result.source_file} (page {result.source_page}): "
                      f"{result.confidence_score:.1f}%")
    else:
        print("\n⚠️  No 'pdfs/' directory found for batch processing example")

    # Example 3: Custom validation rules
    print("\n" + "=" * 60)
    print("Example 3: Custom Validation Rules")
    print("=" * 60)

    custom_config = Config(
        llm_provider="ollama",
        model="llama3.2-vision:11b",
        validation_rules={
            "require_headers": True,
            "min_columns": 3,
            "numeric_precision": 0.01,  # 1% variance allowed
            "detect_merged_cells": True,
            "verify_calculations": True  # Check if totals are correct
        }
    )

    extractor_custom = TableExtractor(custom_config)
    print("\n✅ Custom extractor initialized with strict validation rules")
    print(f"   - Minimum columns: 3")
    print(f"   - Numeric precision: ±1%")
    print(f"   - Verify calculations: Enabled")


if __name__ == "__main__":
    main()
