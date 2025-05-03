#!/bin/bash
# Make this script executable with: chmod +x setup_direct.sh
# This script sets up the direct MCP server (without needing to install the package)

echo "Setting up direct MCP server..."

# Make direct_mcp_server.py executable
chmod +x direct_mcp_server.py
echo "Made direct_mcp_server.py executable"

# Test if it can run directly
echo "Testing direct_mcp_server.py..."
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}}' | ./direct_mcp_server.py

echo ""
echo "To use this with Claude Desktop:"
echo "1. Copy the contents of ultra_simple_config.json to your Claude config file"
echo "2. Restart Claude Desktop"
echo ""
echo "Claude config file location: ~/Library/Application Support/Claude/claude_desktop_config.json"
