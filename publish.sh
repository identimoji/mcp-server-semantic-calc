#!/bin/bash
# Script to publish the semantic calculator package to PyPI

# Build the package
echo "Building the package..."
python -m build

# Upload to PyPI
echo "Uploading to PyPI..."
python -m twine upload dist/*

echo "Package published successfully!"
echo "To install with UV, run:"
echo "  uv tool install semantic-calculator"
