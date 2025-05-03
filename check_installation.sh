#!/bin/bash
# Make this script executable with: chmod +x check_installation.sh
# Script to check the installation of the semantic calculator

# Activate the virtual environment
source .venv/bin/activate

# Check if the semantic_calculator package is installed
echo "Checking if semantic_calculator is installed..."
if python -c "import semantic_calculator" 2>/dev/null; then
    echo "✅ semantic_calculator is installed correctly"
else
    echo "❌ semantic_calculator is not installed"
    echo "Installing package in development mode..."
    pip install -e .
    
    # Check again
    if python -c "import semantic_calculator" 2>/dev/null; then
        echo "✅ semantic_calculator is now installed correctly"
    else
        echo "❌ Failed to install semantic_calculator. Please check for errors above."
        exit 1
    fi
fi

# Check Python path
echo "Current PYTHONPATH: $PYTHONPATH"
echo "Setting PYTHONPATH to include project directory..."
export PYTHONPATH=/Users/rob/repos/mcp-server-semantic-calc:$PYTHONPATH
echo "Updated PYTHONPATH: $PYTHONPATH"

# Check if the MCP module is accessible
echo "Checking if semantic_calculator.mcp is accessible..."
if python -c "from semantic_calculator import mcp" 2>/dev/null; then
    echo "✅ semantic_calculator.mcp is accessible"
else
    echo "❌ semantic_calculator.mcp is not accessible"
    echo "Please check the project structure and try again"
    exit 1
fi

echo "All checks passed! You can now use the semantic calculator MCP server."
echo ""
echo "To use with Claude Desktop:"
echo "1. Copy the claude_config.json content to your Claude Desktop config file"
echo "2. Restart Claude Desktop"
echo ""
echo "To test manually, run:"
echo "./test_mcp.sh"
