#!/bin/bash
# Quick Setup Script for Validated Table Extractor
# Usage: ./setup.sh

set -e  # Exit on error

echo "🚀 Setting up Validated Table Extractor..."
echo ""

# Check Python version
echo "📌 Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  venv already exists, skipping creation"
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed from requirements.txt"

# Install package in editable mode
echo ""
echo "📦 Installing package in editable mode..."
pip install -e .
echo "✅ Package installed"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python -c 'from src import TableExtractor; print(\"✅ Import successful!\")'"
