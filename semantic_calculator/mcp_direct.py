#!/usr/bin/env python3
"""
FastMCP implementation for semantic calculator.
"""

import sys
import logging
import json
from typing import Dict, List, Any, Optional

# Configure logging with both file and stderr output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("semantic_calculator_mcp.log"),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

try:
    # Import the MCP SDK
    from mcp.server.fastmcp import FastMCP
    import mcp.server.stdio
    import asyncio
    
    # Import our calculator with compatibility
    from .compatibility import SemanticCalculator
    
    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder for numpy types."""
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.number):
                return float(obj)
            return json.JSONEncoder.default(self, obj)
    
    # Create a FastMCP instance
    mcp = FastMCP(
        "Semantic Calculator", 
        description="Tools for semantic embedding calculations",
        version="0.1.0"
    )
    
    # Initialize calculator
    calculator = SemanticCalculator()
    
    # Register text to vector tool
    @mcp.tool()
    def text_to_vector(text: str) -> Dict[str, Any]:
        """
        Convert text to a vector embedding
        
        Args:
            text: The text to convert to a vector embedding
            
        Returns:
            A vector embedding represented as a list of floats
        """
        logger.info(f"Converting text to vector: {text}")
        try:
            vector = calculator.text_to_vector(text)
            return {"vector": vector.tolist(), "success": True}
        except Exception as e:
            logger.error(f"Error converting text to vector: {e}")
            return {"error": str(e), "success": False}
    
    # Register emoji to vector tool
    @mcp.tool()
    def emoji_to_vector(emoji: str) -> Dict[str, Any]:
        """
        Convert emoji to a vector embedding
        
        Args:
            emoji: The emoji character to convert
            
        Returns:
            A vector embedding represented as a list of floats
        """
        logger.info(f"Converting emoji to vector: {emoji}")
        try:
            vector = calculator.emoji_to_vector(emoji)
            return {"vector": vector.tolist(), "success": True}
        except Exception as e:
            logger.error(f"Error converting emoji to vector: {e}")
            return {"error": str(e), "success": False}
    
    # Register cosine similarity tool
    @mcp.tool()
    def cosine_similarity(vector1: List[float], vector2: List[float]) -> Dict[str, Any]:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vector1: First vector as a list of floats
            vector2: Second vector as a list of floats
            
        Returns:
            The cosine similarity as a float between -1 and 1
        """
        logger.info("Calculating cosine similarity")
        try:
            import numpy as np
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            similarity = calculator.cosine_similarity(v1, v2)
            return {"similarity": similarity, "success": True}
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return {"error": str(e), "success": False}
    
    # Register euclidean distance tool
    @mcp.tool()
    def euclidean_distance(vector1: List[float], vector2: List[float]) -> Dict[str, Any]:
        """
        Calculate Euclidean distance between two vectors
        
        Args:
            vector1: First vector as a list of floats
            vector2: Second vector as a list of floats
            
        Returns:
            The Euclidean distance as a float
        """
        logger.info("Calculating Euclidean distance")
        try:
            import numpy as np
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            distance = calculator.euclidean_distance(v1, v2)
            return {"distance": distance, "success": True}
        except Exception as e:
            logger.error(f"Error calculating Euclidean distance: {e}")
            return {"error": str(e), "success": False}
    
    # Register manhattan distance tool
    @mcp.tool()
    def manhattan_distance(vector1: List[float], vector2: List[float]) -> Dict[str, Any]:
        """
        Calculate Manhattan distance between two vectors
        
        Args:
            vector1: First vector as a list of floats
            vector2: Second vector as a list of floats
            
        Returns:
            The Manhattan distance as a float
        """
        logger.info("Calculating Manhattan distance")
        try:
            import numpy as np
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            distance = calculator.manhattan_distance(v1, v2)
            return {"distance": distance, "success": True}
        except Exception as e:
            logger.error(f"Error calculating Manhattan distance: {e}")
            return {"error": str(e), "success": False}
    
    # Register dimension distance tool
    @mcp.tool()
    def dimension_distance(dimension1: Dict[str, str], dimension2: Dict[str, str]) -> Dict[str, Any]:
        """
        Calculate semantic similarity between two dimensions
        
        Args:
            dimension1: Dict with 'pole1' and 'pole2' keys for the first dimension
            dimension2: Dict with 'pole1' and 'pole2' keys for the second dimension
            
        Returns:
            A similarity score between 0 and 1
        """
        logger.info("Calculating dimension distance")
        try:
            distance = calculator.dimension_distance(dimension1, dimension2)
            return {"distance": distance, "success": True}
        except Exception as e:
            logger.error(f"Error calculating dimension distance: {e}")
            return {"error": str(e), "success": False}
    
    # Register helical components tool
    @mcp.tool()
    def calculate_helical_components(
        magnitude: float, 
        phase_angle: float, 
        periods: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Calculate helical components from magnitude and phase angle
        
        Args:
            magnitude: The magnitude (0-1)
            phase_angle: The phase angle in degrees (0-180)
            periods: List of periods for the helical components (optional)
            
        Returns:
            Dictionary with sine, cosine, and linear components
        """
        if periods is None:
            periods = [2, 5, 10, 100]
            
        logger.info(f"Calculating helical components: mag={magnitude}, angle={phase_angle}")
        try:
            components = calculator.calculate_helical_components(
                magnitude, phase_angle, periods
            )
            return {"components": components, "success": True}
        except Exception as e:
            logger.error(f"Error calculating helical components: {e}")
            return {"error": str(e), "success": False}
    
    # Register parse emojikey string tool
    @mcp.tool()
    def parse_emojikey_string(emojikey: str) -> Dict[str, Any]:
        """
        Parse an emojikey string into a structured representation
        
        Args:
            emojikey: The emojikey string to parse
            
        Returns:
            A structured representation of the emojikey
        """
        logger.info(f"Parsing emojikey: {emojikey}")
        try:
            parsed = calculator.parse_emojikey_string(emojikey)
            return {"parsed": parsed, "success": True}
        except Exception as e:
            logger.error(f"Error parsing emojikey: {e}")
            return {"error": str(e), "success": False}
    
    # Main function for running the server
    def main():
        """Run the MCP server"""
        logger.info("Starting Semantic Calculator MCP server")
        try:
            asyncio.run(mcp.server.stdio.run_server(mcp))
            logger.info("MCP server stopped")
        except Exception as e:
            logger.error(f"Error running MCP server: {e}", exc_info=True)
            sys.exit(1)
    
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
            json.dumps({
                "jsonrpc": "2.0", 
                "id": None, 
                "error": {
                    "code": -32000, 
                    "message": "Failed to initialize MCP server due to missing dependencies"
                }
            })
        )
        sys.exit(1)

# Main entry point
if __name__ == "__main__":
    main()