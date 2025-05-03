#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP wrapper for the semantic calculator.

This script provides a wrapper for using the semantic calculator
as an MCP server. It handles commands from stdin and outputs
results to stdout.
"""

import json
import sys
import logging
import numpy as np
from typing import Dict, Any

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="mcp_wrapper.log"
)
logger = logging.getLogger(__name__)

# Import semantic calculator
try:
    from semantic_calculator.core import SemanticCalculator
    semantic_calc = SemanticCalculator()
    logger.info("Initialized semantic calculator")
except ImportError as e:
    logger.error(f"Failed to import semantic calculator: {e}")
    sys.exit(1)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.number):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def handle_command(command_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle an MCP command.
    
    Args:
        command_data: The command data from stdin
        
    Returns:
        A dictionary with the result
    """
    try:
        command = command_data.get("command", "")
        params = command_data.get("params", {})
        
        # Log received command
        logger.info(f"Received command: {command}")
        
        # Dispatch command
        if command == "semantic_calculator_text_to_vector":
            text = params.get("text", "")
            result = semantic_calc.semantic_calculator_text_to_vector(text)
            
        elif command == "semantic_calculator_emoji_to_vector":
            emoji = params.get("emoji", "")
            result = semantic_calc.semantic_calculator_emoji_to_vector(emoji)
            
        elif command == "semantic_calculator_cosine_similarity":
            vector1 = np.array(params.get("vector1", []))
            vector2 = np.array(params.get("vector2", []))
            result = semantic_calc.semantic_calculator_cosine_similarity(vector1, vector2)
            
        elif command == "semantic_calculator_euclidean_distance":
            vector1 = np.array(params.get("vector1", []))
            vector2 = np.array(params.get("vector2", []))
            result = semantic_calc.semantic_calculator_euclidean_distance(vector1, vector2)
            
        elif command == "semantic_calculator_manhattan_distance":
            vector1 = np.array(params.get("vector1", []))
            vector2 = np.array(params.get("vector2", []))
            result = semantic_calc.semantic_calculator_manhattan_distance(vector1, vector2)
            
        elif command == "semantic_calculator_dimensionality_reduction":
            vectors = [np.array(v) for v in params.get("vectors", [])]
            dimensions = params.get("dimensions", 3)
            method = params.get("method", "t-SNE")
            result = semantic_calc.semantic_calculator_dimensionality_reduction(vectors, dimensions, method)
            
        elif command == "semantic_calculator_dimension_distance":
            dimension1 = params.get("dimension1", {})
            dimension2 = params.get("dimension2", {})
            result = semantic_calc.semantic_calculator_dimension_distance(dimension1, dimension2)
            
        elif command == "semantic_calculator_calculate_helical_components":
            magnitude = params.get("magnitude", 0.0)
            phase_angle = params.get("phase_angle", 0.0)
            periods = params.get("periods", [2, 5, 10, 100])
            result = semantic_calc.semantic_calculator_calculate_helical_components(magnitude, phase_angle, periods)
            
        elif command == "semantic_calculator_vectors_to_3d_coordinates":
            vectors = [np.array(v) for v in params.get("vectors", [])]
            method = params.get("method", "t-SNE")
            result = semantic_calc.semantic_calculator_vectors_to_3d_coordinates(vectors, method)
            
        elif command == "semantic_calculator_parse_emojikey_string":
            emojikey = params.get("emojikey", "")
            result = semantic_calc.semantic_calculator_parse_emojikey_string(emojikey)
            
        else:
            logger.error(f"Unknown command: {command}")
            result = {"error": f"Unknown command: {command}"}
        
        # Log successful result
        logger.info("Command processed successfully")
        return {"result": result}
    
    except Exception as e:
        # Log error
        logger.error(f"Error processing command: {e}", exc_info=True)
        return {"error": str(e)}

def main():
    """Main function for the MCP wrapper."""
    # Read command from stdin
    try:
        command_data = json.loads(sys.stdin.read())
        
        # Process command
        result = handle_command(command_data)
        
        # Output result to stdout
        sys.stdout.write(json.dumps(result, cls=NumpyEncoder))
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse command: {e}")
        sys.stdout.write(json.dumps({"error": f"Invalid JSON: {e}"}))
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.stdout.write(json.dumps({"error": f"Unexpected error: {str(e)}"}))

if __name__ == "__main__":
    main()
