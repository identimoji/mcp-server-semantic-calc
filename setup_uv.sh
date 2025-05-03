#!/bin/bash
# Setup script for semantic calculator MCP using uv

echo "Setting up semantic calculator MCP with uv..."

# Make sure uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it with:"
    echo "pip install uv"
    exit 1
fi

# Create a fresh venv environment
echo "Creating a fresh venv environment..."
uv venv -p 3 .venv
source .venv/bin/activate

# Install the package in development mode
echo "Installing the package in development mode..."
uv pip install -e .

# Install MCP SDK
echo "Installing MCP SDK..."
uv pip install model-context-protocol

# Create simplified MCP module file
echo "Creating MCP module file..."
cat > semantic_calculator/mcp.py << 'EOL'
#!/usr/bin/env python3
"""
MCP server implementation for semantic calculator.
"""

import sys
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

try:
    # Import the MCP SDK
    from mcp.server.fastmcp import FastMCP
    import mcp.server.stdio
    import asyncio
    
    # Import our calculator
    from .core import SemanticCalculator
    
    # Create a FastMCP instance
    mcp = FastMCP("Semantic Calculator")
    
    # Initialize calculator
    calculator = SemanticCalculator()
    
    # Register tools
    @mcp.tool()
    def semantic_calculator_text_to_vector(text: str) -> Dict[str, Any]:
        """Convert text to a vector embedding"""
        logger.info(f"Converting text to vector: {text}")
        return calculator.text_to_vector(text)
    
    @mcp.tool()
    def semantic_calculator_emoji_to_vector(emoji: str) -> Dict[str, Any]:
        """Convert emoji to a vector embedding"""
        logger.info(f"Converting emoji to vector: {emoji}")
        return calculator.emoji_to_vector(emoji)
    
    @mcp.tool()
    def semantic_calculator_cosine_similarity(vector1: List[float], vector2: List[float]) -> Dict[str, Any]:
        """Calculate cosine similarity between two vectors"""
        logger.info("Calculating cosine similarity")
        return calculator.cosine_similarity(vector1, vector2)
    
    @mcp.tool()
    def semantic_calculator_euclidean_distance(vector1: List[float], vector2: List[float]) -> Dict[str, Any]:
        """Calculate Euclidean distance between two vectors"""
        logger.info("Calculating Euclidean distance")
        return calculator.euclidean_distance(vector1, vector2)
    
    @mcp.tool()
    def semantic_calculator_manhattan_distance(vector1: List[float], vector2: List[float]) -> Dict[str, Any]:
        """Calculate Manhattan distance between two vectors"""
        logger.info("Calculating Manhattan distance")
        return calculator.manhattan_distance(vector1, vector2)
    
    @mcp.tool()
    def semantic_calculator_dimension_distance(dimension1: Dict[str, Any], dimension2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate semantic similarity between two dimensions"""
        logger.info("Calculating dimension distance")
        return calculator.dimension_distance(dimension1, dimension2)
    
    @mcp.tool()
    def semantic_calculator_calculate_helical_components(
        magnitude: float, 
        phase_angle: float, 
        periods: List[int] = [2, 5, 10, 100]
    ) -> Dict[str, Any]:
        """Calculate helical components from magnitude and phase angle"""
        logger.info(f"Calculating helical components: mag={magnitude}, angle={phase_angle}")
        return calculator.calculate_helical_components(magnitude, phase_angle, periods)
    
    @mcp.tool()
    def semantic_calculator_parse_emojikey_string(emojikey: str) -> Dict[str, Any]:
        """Parse an emojikey string into a structured representation"""
        logger.info(f"Parsing emojikey: {emojikey}")
        return calculator.parse_emojikey_string(emojikey)
    
    # Main function for running the server
    def main():
        """Run the MCP server"""
        logger.info("Starting Semantic Calculator MCP server")
        asyncio.run(mcp.server.stdio.run_server(mcp))
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure you have installed the required dependencies:")
    logger.error("  - model-context-protocol")
    logger.error("  - sentence-transformers (optional, for full functionality)")
    
    # Create a simple placeholder that will exit with an error message
    def main():
        """Error placeholder"""
        logger.error("Cannot start MCP server due to missing dependencies")
        print(
            '{"jsonrpc": "2.0", "id": null, "error": {"code": -32000, "message": "Failed to initialize MCP server due to missing dependencies"}}'
        )
        sys.exit(1)

# Main entry point
if __name__ == "__main__":
    main()
EOL

# Create Claude config
echo "Creating Claude config..."
cat > claude_uv_config.json << EOL
{
  "mcpServers": {
    "semantic-calculator": {
      "command": "uvx",
      "args": [
        "-m", 
        "semantic_calculator.mcp"
      ]
    }
  }
}
EOL

# Make script executable
chmod +x semantic_calculator/mcp.py

echo ""
echo "Setup complete!"
echo ""
echo "To use the semantic calculator with Claude:"
echo "1. Copy the content of claude_uv_config.json to your Claude config file:"
echo "   ~/Library/Application Support/Claude/claude_desktop_config.json"
echo ""
echo "2. Restart Claude"
echo ""
echo "3. To test manually, run:"
echo "   uvx -m semantic_calculator.mcp"
echo ""
echo "Remember that you need to activate the virtual environment before running manually:"
echo "   source .venv/bin/activate"
