#!/bin/bash
# Make this script executable with: chmod +x install_with_uv.sh
# Script to install semantic-calculator package with uv

# Exit on error
set -e

echo "Installing semantic-calculator with uv..."

# Install package in editable mode
uvx pip install --global -e .

echo "Installation complete! Try running: semantic-calculator mcp --delay 3"
