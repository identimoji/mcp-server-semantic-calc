#!/bin/bash
# Make this script executable with: chmod +x test_mcp.sh
# Simple script to test the MCP server

# Activate the virtual environment
source .venv/bin/activate

# Make sure the Python path includes the project directory
export PYTHONPATH=/Users/rob/repos/mcp-server-semantic-calc

# Run the server with a test initialize command
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}}' | python -m semantic_calculator mcp
