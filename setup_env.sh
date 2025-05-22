#!/bin/bash
# Script to set up a clean virtual environment for semantic calculator

# Location of UV
UV_PATH="/Users/rob/.local/bin/uv"

# Check if UV exists
if [ ! -f "$UV_PATH" ]; then
    echo "Error: UV not found at $UV_PATH"
    echo "Please install UV first or update the path in this script"
    exit 1
fi

echo "Setting up environment for M2 Mac..."

# Create and activate a new virtual environment
echo "Creating virtual environment..."
"$UV_PATH" venv -p 3.11

# Install dependencies with specific versions for arm64
echo "Installing dependencies for arm64 architecture..."
"$UV_PATH" pip install --force-reinstall --no-binary=numpy numpy==1.26.1
"$UV_PATH" pip install --upgrade "torch>=2.0.0"
"$UV_PATH" pip install scikit-learn==1.3.0
"$UV_PATH" pip install sentence-transformers==2.2.2
"$UV_PATH" pip install matplotlib==3.8.0
"$UV_PATH" pip install umap-learn==0.5.3
"$UV_PATH" pip install plotly==5.18.0
"$UV_PATH" pip install seaborn==0.13.0
"$UV_PATH" pip install pytest==7.4.0

# Install the package in development mode
echo "Installing semantic-calculator in development mode..."
"$UV_PATH" pip install -e .

# Verify installation
echo "Verifying installation..."
"$UV_PATH" python -c "import numpy; import torch; import sklearn; import sentence_transformers; print('All key libraries imported successfully!')"

echo "Setup complete! Run the MCP server with: .venv/bin/python -m semantic_calculator.mcp"