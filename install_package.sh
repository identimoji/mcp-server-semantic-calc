#!/bin/bash
# Script to install the semantic calculator package

# Check if a virtual environment exists
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

echo "Installation complete!"
echo "To use the package, activate the virtual environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then you can import the package in Python:"
echo "  from semantic_calculator.core import SemanticCalculator"
echo ""
echo "Or run one of the example scripts:"
echo "  python examples/vector_operations.py"
