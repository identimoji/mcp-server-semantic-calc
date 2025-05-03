#!/usr/bin/env python3
"""
Simplified MCP server for semantic calculator that minimizes dependencies.
"""

import sys
import json
import logging
from typing import Dict, Any, List, Optional

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("semantic-calculator")

class SimpleJSONRPCServer:
    """A simple JSON-RPC server that reads from stdin and writes to stdout."""
    
    def __init__(self, name: str):
        self.name = name
        self.methods = {}
        
    def register_method(self, name: str, method):
        """Register a method with the server."""
        self.methods[name] = method
        
    def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC request."""
        request_id = request_data.get("id")
        method_name = request_data.get("method")
        params = request_data.get("params", {})
        
        logger.info(f"Received request: {method_name}")
        
        if method_name not in self.methods:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method_name}"
                },
                "id": request_id
            }
        
        try:
            result = self.methods[method_name](**params)
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
        except Exception as e:
            logger.error(f"Error executing method {method_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": f"Error executing method: {str(e)}"
                },
                "id": request_id
            }
    
    def run(self):
        """Run the server, reading from stdin and writing to stdout."""
        logger.info(f"Starting {self.name} MCP server")
        
        # Send initial ready message
        print(json.dumps({
            "jsonrpc": "2.0",
            "method": "mcp/ready",
            "params": {
                "name": self.name,
                "version": "0.1.0"
            }
        }))
        sys.stdout.flush()
        
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request_data = json.loads(line)
                    response = self.handle_request(request_data)
                    print(json.dumps(response))
                    sys.stdout.flush()
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {line}")
                    print(json.dumps({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        },
                        "id": None
                    }))
                    sys.stdout.flush()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)

# Dummy implementation of calculator functions
class DummyCalculator:
    """A dummy calculator that doesn't rely on external dependencies."""
    
    def text_to_vector(self, text: str) -> Dict[str, Any]:
        """Convert text to a dummy vector."""
        logger.info(f"Converting text to vector: {text}")
        # Generate a deterministic dummy vector based on the text
        import hashlib
        hash_val = hashlib.md5(text.encode()).hexdigest()
        vector = [int(hash_val[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
        return {"vector": vector}
    
    def emoji_to_vector(self, emoji: str) -> Dict[str, Any]:
        """Convert emoji to a dummy vector."""
        logger.info(f"Converting emoji to vector: {emoji}")
        # Similar approach as text_to_vector but with a different seed
        import hashlib
        hash_val = hashlib.md5((emoji + "emoji").encode()).hexdigest()
        vector = [int(hash_val[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
        return {"vector": vector}
    
    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> Dict[str, Any]:
        """Calculate dummy cosine similarity."""
        logger.info("Calculating cosine similarity")
        # Simple dot product for demonstration
        if len(vector1) != len(vector2):
            return {"error": "Vectors must have the same length"}
        
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = sum(a * a for a in vector1) ** 0.5
        magnitude2 = sum(b * b for b in vector2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            similarity = 0
        else:
            similarity = dot_product / (magnitude1 * magnitude2)
            
        return {"similarity": similarity}
    
    def euclidean_distance(self, vector1: List[float], vector2: List[float]) -> Dict[str, Any]:
        """Calculate dummy Euclidean distance."""
        logger.info("Calculating Euclidean distance")
        
        if len(vector1) != len(vector2):
            return {"error": "Vectors must have the same length"}
        
        distance = sum((a - b) ** 2 for a, b in zip(vector1, vector2)) ** 0.5
        return {"distance": distance}
    
    def manhattan_distance(self, vector1: List[float], vector2: List[float]) -> Dict[str, Any]:
        """Calculate dummy Manhattan distance."""
        logger.info("Calculating Manhattan distance")
        
        if len(vector1) != len(vector2):
            return {"error": "Vectors must have the same length"}
        
        distance = sum(abs(a - b) for a, b in zip(vector1, vector2))
        return {"distance": distance}
    
    def calculate_helical_components(
        self, 
        magnitude: float, 
        phase_angle: float, 
        periods: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Calculate helical components."""
        logger.info(f"Calculating helical components: mag={magnitude}, angle={phase_angle}")
        
        if periods is None:
            periods = [2, 5, 10, 100]
        
        import math
        phase_rad = math.radians(phase_angle)
        
        components = {
            "linear": magnitude,
        }
        
        for period in periods:
            scaled_angle = phase_rad * (360 / period)
            components[f"sin_{period}"] = magnitude * math.sin(scaled_angle)
            components[f"cos_{period}"] = magnitude * math.cos(scaled_angle)
        
        return {"components": components}
    
    def parse_emojikey_string(self, emojikey: str) -> Dict[str, Any]:
        """Parse an emojikey string."""
        logger.info(f"Parsing emojikey: {emojikey}")
        
        import re
        result = {
            "ME": [],
            "CONTENT": [],
            "YOU": []
        }
        
        component_pattern = r'\[(ME|CONTENT|YOU)\|(.*?)\]'
        component_matches = re.findall(component_pattern, emojikey)
        
        for component_type, pairs_str in component_matches:
            pairs = pairs_str.split('|')
            
            for pair in pairs:
                pair_match = re.match(r'([^0-9∠]+)([0-9])∠([0-9]+)', pair)
                if pair_match:
                    emojis, magnitude, angle = pair_match.groups()
                    
                    result[component_type].append({
                        "emojis": emojis,
                        "pole1": emojis[0],
                        "pole2": emojis[1],
                        "magnitude": float(magnitude),
                        "angle": float(angle)
                    })
        
        return {"parsed": result}

def main():
    """Main entry point for the MCP server."""
    # Create server
    server = SimpleJSONRPCServer("Semantic Calculator")
    
    # Create calculator
    calculator = DummyCalculator()
    
    # Register methods
    server.register_method("text_to_vector", calculator.text_to_vector)
    server.register_method("emoji_to_vector", calculator.emoji_to_vector)
    server.register_method("cosine_similarity", calculator.cosine_similarity)
    server.register_method("euclidean_distance", calculator.euclidean_distance)
    server.register_method("manhattan_distance", calculator.manhattan_distance)
    server.register_method("calculate_helical_components", calculator.calculate_helical_components)
    server.register_method("parse_emojikey_string", calculator.parse_emojikey_string)
    
    # Run server
    server.run()

if __name__ == "__main__":
    main()