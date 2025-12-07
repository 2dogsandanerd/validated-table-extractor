# Quick Start Guide - Validated Table Extractor

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone or extract the repository
cd validated-table-extractor

# Run the setup script
./setup.sh

# Activate the virtual environment
source venv/bin/activate
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# OR install as package
pip install -e .
```

## Quick Test

### With Vision LLM (Full Validation)

```python
from src import TableExtractor

# Initialize with local Ollama (100% free)
extractor = TableExtractor(
    llm_provider="ollama",
    model="llama3.2-vision:11b"  # Vision model required
)

# Extract and validate
results = extractor.extract_and_validate("your_invoice.pdf")

for result in results:
    print(f"Confidence: {result.confidence_score}%")
    print(f"Summary: {result.validation_summary}")
```

### Without Vision LLM (Extraction Only)

**Don't have a Vision model? No problem!**

```python
from src import TableExtractor

# Use ANY text model (or skip LLM entirely)
extractor = TableExtractor(
    llm_provider="ollama",
    model="llama3.1:8b",  # Any model works
    save_screenshots=True  # Still captures audit screenshots
)

# Extract tables (validation will be skipped gracefully)
results = extractor.extract_and_validate("your_invoice.pdf")

for result in results:
    print(result.raw_markdown_table)  # Perfect extraction!
    # Note: confidence_score will be 50.0 (fallback)
```

**Extraction-Only Mode:**
- ✅ No Vision model needed
- ✅ Faster processing (no LLM calls)
- ✅ Perfect table extraction with Docling
- ✅ Screenshots still captured for audit
- ⚠️ No confidence scores (uses 50.0 fallback)

## Requirements

- **Python:** 3.10+
- **Ollama:** Install from https://ollama.ai (for local LLM)
- **Model (Optional):** 
  - Vision: `ollama pull llama3.2-vision:11b` (7GB)
  - OR smaller: `ollama pull llava:7b` (2GB)
  - OR any text model for extraction-only

## Dependencies

All dependencies are listed in:
- `requirements.txt` (for pip)
- `pyproject.toml` (for modern Python packaging)

Core dependencies:
- `docling>=2.0.0` - PDF table extraction
- `PyMuPDF>=1.23.0` - PDF rendering
- `pydantic>=2.0.0` - Data validation
- `langchain-ollama>=0.1.0` - Local LLM integration
- `loguru>=0.7.0` - Logging

## Troubleshooting

### Import Error
```bash
# Make sure venv is activated
source venv/bin/activate

# Verify installation
python -c "from src import TableExtractor; print('✅ OK')"
```

### Ollama Not Found
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the vision model
ollama pull llama3.2-vision:11b

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

## Next Steps

See [README.md](README.md) for full documentation and examples.
