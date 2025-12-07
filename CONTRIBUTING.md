# Contributing to Validated Table Extractor

Thank you for considering contributing! This project aims to provide audit-ready table extraction for compliance-first RAG systems.

## Development Setup

```bash
git clone https://github.com/yourusername/validated-table-extractor
cd validated-table-extractor
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use:
- **black** for formatting
- **ruff** for linting
- **mypy** for type checking

```bash
black src/
ruff check src/
mypy src/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Areas for Contribution

- **Vision Models:** Add support for more vision LLMs (Claude, Gemini, etc.)
- **Validation Rules:** Implement advanced validation logic
- **Export Formats:** Add more export options
- **Performance:** Optimize extraction speed
- **Documentation:** Improve examples and tutorials
- **Testing:** Add more test cases

## Questions?

Open an issue or start a discussion!
