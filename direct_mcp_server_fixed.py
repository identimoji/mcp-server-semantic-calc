#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct MCP server implementation using the official MCP SDK patterns.
This should be more aligned with the MCP protocol specification.
"""

import sys
import json
import logging
import os

# Configure logging to stderr only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Try to import the MCP library
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server import stdio
    logger.info("Successfully imported MCP SDK")
except ImportError:
    logger.warning("MCP SDK not found. Continuing with custom implementation.")
    FastMCP = None

# Try importing the core calculator
try:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)
    from semantic_calculator.core import SemanticCalculator
    calculator = SemanticCalculator()
    logger.info("Successfully imported SemanticCalculator")
except ImportError as e:
    logger.warning(f"Could not import SemanticCalculator: {e}")
    logger.warning("Using fallback implementation")
    
    # Define a minimal fallback calculator
    class FallbackCalculator:
        def __init__(self):
            logger.info("Using fallback calculator")
            
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
    
    calculator = FallbackCalculator()

# Define MCP server using the official FastMCP if available
if FastMCP:
    # Use the official SDK approach
    mcp = FastMCP("Semantic Calculator")
    
    @mcp.tool()
    def semantic_calculator_text_to_vector(text: str) -> dict:
        """Convert text to a vector embedding"""
        logger.info(f"Converting text to vector: {text}")
        return calculator.text_to_vector(text)
    
    @mcp.tool()
    def semantic_calculator_emoji_to_vector(emoji: str) -> dict:
        """Convert emoji to a vector embedding"""
        logger.info(f"Converting emoji to vector: {emoji}")
        return calculator.emoji_to_vector(emoji)
    
    @mcp.tool()
    def semantic_calculator_cosine_similarity(vector1: list, vector2: list) -> dict:
        """Calculate cosine similarity between two vectors"""
        logger.info("Calculating cosine similarity")
        return calculator.cosine_similarity(vector1, vector2)
    
    @mcp.tool()
    def semantic_calculator_euclidean_distance(vector1: list, vector2: list) -> dict:
        """Calculate Euclidean distance between two vectors"""
        logger.info("Calculating Euclidean distance")
        return calculator.euclidean_distance(vector1, vector2)
    
    @mcp.tool()
    def semantic_calculator_manhattan_distance(vector1: list, vector2: list) -> dict:
        """Calculate Manhattan distance between two vectors"""
        logger.info("Calculating Manhattan distance")
        return calculator.manhattan_distance(vector1, vector2)
    
    @mcp.tool()
    def semantic_calculator_dimension_distance(dimension1: dict, dimension2: dict) -> dict:
        """Calculate semantic similarity between two dimensions"""
        logger.info("Calculating dimension distance")
        return calculator.dimension_distance(dimension1, dimension2)
    
    @mcp.tool()
    def semantic_calculator_calculate_helical_components(magnitude: float, phase_angle: float, periods: list = [2, 5, 10, 100]) -> dict:
        """Calculate helical components from magnitude and phase angle"""
        logger.info(f"Calculating helical components: mag={magnitude}, angle={phase_angle}")
        return calculator.calculate_helical_components(magnitude, phase_angle, periods)
    
    @mcp.tool()
    def semantic_calculator_parse_emojikey_string(emojikey: str) -> dict:
        """Parse an emojikey string into a structured representation"""
        logger.info(f"Parsing emojikey: {emojikey}")
        return calculator.parse_emojikey_string(emojikey)
    
    # Run the server
    def main():
        logger.info("Starting FastMCP server")
        asyncio.run(stdio.run_server(mcp))

else:
    # Custom implementation if MCP SDK is not available
    import sys
    import json
    
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
    
    # Simple JSON-RPC message handler
    def handle_jsonrpc(message, calculator):
        try:
            jsonrpc = message.get("jsonrpc", "2.0")
            method = message.get("method", "")
            params = message.get("params", {})
            id_value = message.get("id")
            
            logger.info(f"Handling method: {method}")
            
            # Handle initialize method for MCP
            if method == "initialize":
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
            
            # Handle list_tools method for MCP
            elif method == "list_tools":
                return {
                    "jsonrpc": "2.0",
                    "id": id_value,
                    "result": {
                        "tools": [
                            {
                                "name": "semantic_calculator_text_to_vector",
                                "description": "Convert text to a vector embedding",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "The text to convert to a vector"
                                        }
                                    },
                                    "required": ["text"]
                                }
                            },
                            {
                                "name": "semantic_calculator_emoji_to_vector",
                                "description": "Convert emoji to a vector embedding",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "emoji": {
                                            "type": "string",
                                            "description": "The emoji to convert to a vector"
                                        }
                                    },
                                    "required": ["emoji"]
                                }
                            },
                            {
                                "name": "semantic_calculator_cosine_similarity",
                                "description": "Calculate cosine similarity between two vectors",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "vector1": {
                                            "type": "array",
                                            "description": "First vector"
                                        },
                                        "vector2": {
                                            "type": "array",
                                            "description": "Second vector"
                                        }
                                    },
                                    "required": ["vector1", "vector2"]
                                }
                            },
                            {
                                "name": "semantic_calculator_euclidean_distance",
                                "description": "Calculate Euclidean distance between two vectors",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "vector1": {
                                            "type": "array",
                                            "description": "First vector"
                                        },
                                        "vector2": {
                                            "type": "array",
                                            "description": "Second vector"
                                        }
                                    },
                                    "required": ["vector1", "vector2"]
                                }
                            },
                            {
                                "name": "semantic_calculator_dimension_distance",
                                "description": "Calculate semantic similarity between two dimensions",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "dimension1": {
                                            "type": "object",
                                            "description": "First dimension with pole1 and pole2 keys"
                                        },
                                        "dimension2": {
                                            "type": "object",
                                            "description": "Second dimension with pole1 and pole2 keys"
                                        }
                                    },
                                    "required": ["dimension1", "dimension2"]
                                }
                            },
                            {
                                "name": "semantic_calculator_calculate_helical_components",
                                "description": "Calculate helical components from magnitude and phase angle",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "magnitude": {
                                            "type": "number",
                                            "description": "Magnitude (0-1)"
                                        },
                                        "phase_angle": {
                                            "type": "number",
                                            "description": "Phase angle in degrees (0-180)"
                                        },
                                        "periods": {
                                            "type": "array",
                                            "description": "List of periods for helical components"
                                        }
                                    },
                                    "required": ["magnitude", "phase_angle"]
                                }
                            },
                            {
                                "name": "semantic_calculator_parse_emojikey_string",
                                "description": "Parse an emojikey string into a structured representation",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "emojikey": {
                                            "type": "string",
                                            "description": "Emojikey string to parse"
                                        }
                                    },
                                    "required": ["emojikey"]
                                }
                            }
                        ]
                    }
                }
            
            # Handle call_tool method for MCP
            elif method == "call_tool":
                tool_name = params.get("name", "")
                tool_params = params.get("parameters", {})
                
                logger.info(f"Calling tool: {tool_name} with params: {tool_params}")
                
                # Dispatch to appropriate tool method
                if tool_name == "semantic_calculator_text_to_vector":
                    text = tool_params.get("text", "")
                    result = calculator.text_to_vector(text)
                
                elif tool_name == "semantic_calculator_emoji_to_vector":
                    emoji = tool_params.get("emoji", "")
                    result = calculator.emoji_to_vector(emoji)
                
                elif tool_name == "semantic_calculator_cosine_similarity":
                    vector1 = tool_params.get("vector1", [])
                    vector2 = tool_params.get("vector2", [])
                    result = calculator.cosine_similarity(vector1, vector2)
                
                elif tool_name == "semantic_calculator_euclidean_distance":
                    vector1 = tool_params.get("vector1", [])
                    vector2 = tool_params.get("vector2", [])
                    result = calculator.euclidean_distance(vector1, vector2)
                
                elif tool_name == "semantic_calculator_dimension_distance":
                    dimension1 = tool_params.get("dimension1", {})
                    dimension2 = tool_params.get("dimension2", {})
                    result = calculator.dimension_distance(dimension1, dimension2)
                
                elif tool_name == "semantic_calculator_calculate_helical_components":
                    magnitude = tool_params.get("magnitude", 0.0)
                    phase_angle = tool_params.get("phase_angle", 0.0)
                    periods = tool_params.get("periods", [2, 5, 10, 100])
                    result = calculator.calculate_helical_components(magnitude, phase_angle, periods)
                
                elif tool_name == "semantic_calculator_parse_emojikey_string":
                    emojikey = tool_params.get("emojikey", "")
                    result = calculator.parse_emojikey_string(emojikey)
                
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": id_value,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {tool_name}"
                        }
                    }
                
                return {
                    "jsonrpc": "2.0",
                    "id": id_value,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, cls=CustomJSONEncoder)
                            }
                        ]
                    }
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": id_value,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": id_value if "id_value" in locals() else None,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }
    
    # Main server loop
    def main():
        logger.info("Starting custom JSON-RPC server")
        
        try:
            for line in sys.stdin:
                if not line.strip():
                    continue
                
                try:
                    # Parse the JSON-RPC message
                    message = json.loads(line)
                    
                    # Process the message
                    response = handle_jsonrpc(message, calculator)
                    
                    # Send the response
                    print(json.dumps(response, cls=CustomJSONEncoder))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
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
            logger.info("Server stopped by user")
            sys.exit(0)

# Main entry point
if __name__ == "__main__":
    # If using FastMCP, ensure asyncio is imported
    if FastMCP:
        import asyncio
    
    main()
