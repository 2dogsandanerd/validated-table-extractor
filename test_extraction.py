#!/usr/bin/env python3
"""
Test Script for Validated Table Extractor
Tests with a real PDF (Deutsche Bank Geschäftsbericht)
"""

import sys
from pathlib import Path
from src import TableExtractor

def main():
    # Check if PDF path is provided
    if len(sys.argv) < 2:
        print("❌ Usage: python test_extraction.py <path_to_pdf>")
        print("\nExample:")
        print("  python test_extraction.py deutsche_bank_bericht.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 Validated Table Extractor - Test")
    print("=" * 60)
    print(f"📄 PDF: {pdf_path}")
    print()
    
    # Initialize extractor
    print("🔧 Initializing TableExtractor...")
    print("   - LLM Provider: ollama")
    print("   - Model: llama3.2-vision:11b")
    print("   - Output Dir: validated_tables/")
    print()
    
    try:
        extractor = TableExtractor(
            llm_provider="ollama",
            model="llama3.2-vision:11b",
            save_screenshots=True
        )
        print("✅ TableExtractor initialized!")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Extract and validate tables
    print("📊 Extracting tables from PDF...")
    print("   (This may take a few minutes...)")
    print()
    
    try:
        results = extractor.extract_and_validate(pdf_path)
        
        print("=" * 60)
        print(f"✅ Extraction complete! Found {len(results)} table(s)")
        print("=" * 60)
        print()
        
        # Display results
        for idx, result in enumerate(results, 1):
            print(f"📋 Table {idx}:")
            print(f"   Page: {result.source_page}")
            print(f"   Confidence: {result.confidence_score:.1f}%")
            print(f"   Summary: {result.validation_summary}")
            
            if result.issues_found:
                print(f"   ⚠️  Issues: {', '.join(result.issues_found)}")
            
            print(f"   Screenshot: {result.screenshot_path}")
            print()
            
            # Show markdown preview (first 300 chars)
            markdown_preview = result.raw_markdown_table[:300]
            if len(result.raw_markdown_table) > 300:
                markdown_preview += "..."
            print(f"   Markdown Preview:")
            for line in markdown_preview.split('\n'):
                print(f"     {line}")
            print()
            print("-" * 60)
            print()
        
        # Summary statistics
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        high_confidence = len([r for r in results if r.confidence_score >= 95.0])
        
        print("📊 Summary:")
        print(f"   Total Tables: {len(results)}")
        print(f"   Average Confidence: {avg_confidence:.1f}%")
        print(f"   High Confidence (≥95%): {high_confidence}")
        print(f"   Output Directory: validated_tables/")
        print()
        
        # Export first result as example
        if results:
            print("💾 Exporting first table as example...")
            results[0].export_json("validated_tables/example_table.json", pretty=True)
            print("   ✅ Saved to: validated_tables/example_table.json")
            print()
        
        print("=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
