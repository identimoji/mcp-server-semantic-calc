"""
Compatibility module to handle function name differences between core.py and mcp_direct.py.
This wrapper ensures that both naming conventions work.
"""

from .core import SemanticCalculator as BaseSemanticCalculator
import numpy as np
from typing import Dict, List, Any, Optional, Union

class SemanticCalculator(BaseSemanticCalculator):
    """
    Enhanced SemanticCalculator with compatibility methods.
    """
    
    def text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to a vector embedding."""
        return self.semantic_calculator_text_to_vector(text)
        
    def emoji_to_vector(self, emoji: str) -> np.ndarray:
        """Convert an emoji to a vector embedding."""
        return self.semantic_calculator_emoji_to_vector(emoji)
    
    def cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate the cosine similarity between two vectors."""
        return self.semantic_calculator_cosine_similarity(vector1, vector2)
    
    def euclidean_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate the Euclidean distance between two vectors."""
        return self.semantic_calculator_euclidean_distance(vector1, vector2)
    
    def manhattan_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate the Manhattan distance between two vectors."""
        return self.semantic_calculator_manhattan_distance(vector1, vector2)
    
    def dimension_distance(self, dimension1: Dict[str, str], dimension2: Dict[str, str]) -> float:
        """Calculate the semantic similarity between two dimensions."""
        return self.semantic_calculator_dimension_distance(dimension1, dimension2)
    
    def calculate_helical_components(
        self, 
        magnitude: float, 
        phase_angle: float,
        periods: List[float] = None
    ) -> Dict[str, float]:
        """Calculate helical components from magnitude and phase angle."""
        if periods is None:
            periods = [2, 5, 10, 100]
        return self.semantic_calculator_calculate_helical_components(magnitude, phase_angle, periods)
    
    def parse_emojikey_string(self, emojikey: str) -> Dict[str, Any]:
        """Parse an emojikey string into a structured representation."""
        return self.semantic_calculator_parse_emojikey_string(emojikey)