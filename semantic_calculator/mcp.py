#!/usr/bin/env python3
"""
MCP server implementation for semantic calculator using the official MCP SDK.
"""

import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Import the MCP SDK
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server import stdio
    logger.info("Successfully imported MCP SDK")
except ImportError as e:
    logger.error(f"Failed to import MCP SDK: {e}")
    sys.exit(1)

# Import the SemanticCalculator - require it to be available
try:
    from semantic_calculator.core import SemanticCalculator
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import sklearn
    
    logger.info("Successfully imported SemanticCalculator and dependencies")
    calculator = SemanticCalculator(model_name="all-mpnet-base-v2")
    logger.info("Initialized SemanticCalculator with model all-mpnet-base-v2")
except ImportError as e:
    logger.error(f"Failed to import required dependency: {e}")
    logger.error("The semantic calculator requires SentenceBERT, numpy, and scikit-learn")
    logger.error("Please install these dependencies with: uv pip install sentence-transformers numpy scikit-learn")
    sys.exit(1)

# Create a FastMCP instance
mcp = FastMCP("Semantic Calculator")

# Register tools
@mcp.tool()
def semantic_calc_text_to_vector(text: str) -> List[float]:
    """Convert text to a vector embedding"""
    logger.info(f"Converting text to vector: {text}")
    vector = calculator.semantic_calculator_text_to_vector(text)
    return vector.tolist()

@mcp.tool()
def semantic_calc_emoji_to_vector(emoji: str) -> List[float]:
    """Convert emoji to a vector embedding"""
    logger.info(f"Converting emoji to vector: {emoji}")
    vector = calculator.semantic_calculator_emoji_to_vector(emoji)
    return vector.tolist()

@mcp.tool()
def semantic_calc_cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    logger.info("Calculating cosine similarity")
    return calculator.semantic_calculator_cosine_similarity(
        np.array(vector1), np.array(vector2)
    )

@mcp.tool()
def semantic_calc_euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
    """Calculate Euclidean distance between two vectors"""
    logger.info("Calculating Euclidean distance")
    return calculator.semantic_calculator_euclidean_distance(
        np.array(vector1), np.array(vector2)
    )

@mcp.tool()
def semantic_calc_manhattan_distance(vector1: List[float], vector2: List[float]) -> float:
    """Calculate Manhattan distance between two vectors"""
    logger.info("Calculating Manhattan distance")
    return calculator.semantic_calculator_manhattan_distance(
        np.array(vector1), np.array(vector2)
    )

@mcp.tool()
def semantic_calc_dimension_distance(dimension1: Dict[str, str], dimension2: Dict[str, str]) -> float:
    """Calculate semantic similarity between two dimensions"""
    logger.info("Calculating dimension distance")
    return calculator.semantic_calculator_dimension_distance(dimension1, dimension2)

@mcp.tool()
def semantic_calc_calculate_helical_components(
    magnitude: float, 
    phase_angle: float, 
    periods: List[int] = [2, 5, 10, 100]
) -> Dict[str, float]:
    """Calculate helical components from magnitude and phase angle"""
    logger.info(f"Calculating helical components: mag={magnitude}, angle={phase_angle}")
    return calculator.semantic_calculator_calculate_helical_components(magnitude, phase_angle, periods)

@mcp.tool()
def semantic_calc_parse_emojikey_string(emojikey: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse an emojikey string into a structured representation"""
    logger.info(f"Parsing emojikey: {emojikey}")
    return calculator.semantic_calculator_parse_emojikey_string(emojikey)

# Main function to run the MCP server using the stdio protocol
async def run():
    """Run the MCP server"""
    logger.info("Starting Semantic Calculator MCP server")
    await mcp.run_stdio_async()

def main():
    """Entry point"""
    asyncio.run(run())

# Main entry point
if __name__ == "__main__":
    main()