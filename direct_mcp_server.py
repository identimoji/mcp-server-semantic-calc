#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained MCP server implementation for the semantic calculator.
This file includes all necessary code to run as a standalone server,
without needing to import from the semantic_calculator package.
"""

import sys
import json
import logging
import importlib.util
import os

# Configure logging to stderr only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

try:
    # Find the path to core.py
    project_dir = os.path.dirname(os.path.abspath(__file__))
    core_path = os.path.join(project_dir, 'semantic_calculator', 'core.py')
    
    # Dynamically import the core.py module
    logger.info(f"Attempting to import core from: {core_path}")
    if os.path.exists(core_path):
        spec = importlib.util.spec_from_file_location("core", core_path)
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        SemanticCalculator = core.SemanticCalculator
        logger.info("Successfully imported SemanticCalculator from core.py")
    else:
        logger.error(f"Could not find core.py at: {core_path}")
        raise ImportError(f"Could not find core.py at: {core_path}")
    
except ImportError as e:
    logger.error(f"Error importing SemanticCalculator: {e}")
    
    # Create a minimal fallback calculator for testing
    class SemanticCalculator:
        def __init__(self):
            logger.info("Using fallback SemanticCalculator")
            
        def text_to_vector(self, text):
            return {"text": text, "vector": [0.1, 0.2, 0.3], "note": "Fallback vector"}
            
        def emoji_to_vector(self, emoji):
            return {"emoji": emoji, "vector": [0.4, 0.5, 0.6], "note": "Fallback vector"}
            
        def cosine_similarity(self, vector1, vector2):
            return {"similarity": 0.5, "note": "Fallback similarity"}
            
        def euclidean_distance(self, vector1, vector2):
            return {"distance": 1.0, "note": "Fallback distance"}
            
        def manhattan_distance(self, vector1, vector2):
            return {"distance": 2.0, "note": "Fallback distance"}
            
        def dimensionality_reduction(self, vectors, dimensions, method):
            return {"reduced_vectors": [[0.1, 0.2], [0.3, 0.4]], "note": "Fallback reduction"}
            
        def dimension_distance(self, dimension1, dimension2):
            return {"distance": 0.5, "note": "Fallback dimension distance"}
            
        def calculate_helical_components(self, magnitude, phase_angle, periods):
            return {"components": [0.1, 0.2, 0.3], "note": "Fallback components"}
            
        def vectors_to_3d_coordinates(self, vectors, method):
            return {"coordinates": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "note": "Fallback coordinates"}
            
        def parse_emojikey_string(self, emojikey):
            return {"parsed": {"me": {}, "content": {}, "you": {}}, "note": "Fallback parsing"}


# Custom JSON encoder for any numpy types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.number):
                return float(obj)
        except ImportError:
            pass
        return json.JSONEncoder.default(self, obj)


class MCPServer:
    """MCP Server implementation for the semantic calculator."""
    
    def __init__(self):
        """Initialize the MCP server."""
        self.calculator = SemanticCalculator()
        logger.info("Initialized MCP server with semantic calculator")
    
    def handle_initialize(self, params, id_value):
        """Handle the initialize method."""
        logger.info(f"Handling initialize request with params: {params}")
        return {
            "jsonrpc": "2.0",
            "id": id_value,
            "result": {
                "capabilities": {},
                "serverInfo": {
                    "name": "semantic-calculator",
                    "version": "0.1.0"
                }
            }
        }
    
    def handle_method(self, method, params, id_value):
        """Handle semantic calculator methods."""
        logger.info(f"Handling method: {method} with params: {params}")
        
        try:
            # Dispatch to appropriate calculator method
            if method == "semantic_calculator_text_to_vector":
                text = params.get("text", "")
                result = self.calculator.text_to_vector(text)
                
            elif method == "semantic_calculator_emoji_to_vector":
                emoji = params.get("emoji", "")
                result = self.calculator.emoji_to_vector(emoji)
                
            elif method == "semantic_calculator_cosine_similarity":
                vector1 = params.get("vector1", [])
                vector2 = params.get("vector2", [])
                result = self.calculator.cosine_similarity(vector1, vector2)
                
            elif method == "semantic_calculator_euclidean_distance":
                vector1 = params.get("vector1", [])
                vector2 = params.get("vector2", [])
                result = self.calculator.euclidean_distance(vector1, vector2)
                
            elif method == "semantic_calculator_manhattan_distance":
                vector1 = params.get("vector1", [])
                vector2 = params.get("vector2", [])
                result = self.calculator.manhattan_distance(vector1, vector2)
                
            elif method == "semantic_calculator_dimensionality_reduction":
                vectors = params.get("vectors", [])
                dimensions = params.get("dimensions", 3)
                method_name = params.get("method", "t-SNE")
                result = self.calculator.dimensionality_reduction(vectors, dimensions, method_name)
                
            elif method == "semantic_calculator_dimension_distance":
                dimension1 = params.get("dimension1", {})
                dimension2 = params.get("dimension2", {})
                result = self.calculator.dimension_distance(dimension1, dimension2)
                
            elif method == "semantic_calculator_calculate_helical_components":
                magnitude = params.get("magnitude", 0.0)
                phase_angle = params.get("phase_angle", 0.0)
                periods = params.get("periods", [2, 5, 10, 100])
                result = self.calculator.calculate_helical_components(magnitude, phase_angle, periods)
                
            elif method == "semantic_calculator_vectors_to_3d_coordinates":
                vectors = params.get("vectors", [])
                method_name = params.get("method", "t-SNE")
                result = self.calculator.vectors_to_3d_coordinates(vectors, method_name)
                
            elif method == "semantic_calculator_parse_emojikey_string":
                emojikey = params.get("emojikey", "")
                result = self.calculator.parse_emojikey_string(emojikey)
                
            else:
                logger.error(f"Unknown method: {method}")
                return {
                    "jsonrpc": "2.0",
                    "id": id_value,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
            logger.info(f"Method {method} processed successfully")
            return {
                "jsonrpc": "2.0",
                "id": id_value,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error processing method {method}: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": id_value,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }
    
    def handle_request(self, request_data):
        """Handle an MCP request."""
        try:
            # Extract JSON-RPC parameters
            jsonrpc = request_data.get("jsonrpc", "2.0")
            method = request_data.get("method", "")
            params = request_data.get("params", {})
            id_value = request_data.get("id")
            
            # Log received method
            logger.info(f"Received method: {method} with id: {id_value}")
            
            # Handle MCP protocol methods
            if method == "initialize":
                return self.handle_initialize(params, id_value)
            else:
                return self.handle_method(method, params, id_value)
                
        except Exception as e:
            # Log error
            logger.error(f"Error processing request: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": id_value if "id_value" in locals() else None,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }
    
    def run(self):
        """Run the MCP server."""
        logger.info("Starting MCP server")
        
        try:
            # Process commands from stdin
            for line in sys.stdin:
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    # Parse request
                    request_data = json.loads(line)
                    
                    # Process request
                    response = self.handle_request(request_data)
                    
                    # Output response to stdout
                    print(json.dumps(response, cls=CustomJSONEncoder))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse request: {e}")
                    print(json.dumps({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: {e}"
                        }
                    }))
                    sys.stdout.flush()
                
                except Exception as e:
                    logger.error(f"Unexpected error: {e}", exc_info=True)
                    print(json.dumps({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32000,
                            "message": f"Unexpected error: {str(e)}"
                        }
                    }))
                    sys.stdout.flush()
        
        except KeyboardInterrupt:
            logger.info("MCP server stopped by user")
            sys.exit(0)


def main():
    """Main entry point."""
    server = MCPServer()
    server.run()


if __name__ == "__main__":
    main()
