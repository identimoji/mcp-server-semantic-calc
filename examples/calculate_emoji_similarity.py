#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate semantic similarity between emoji pairs.

This script demonstrates the core functionality of calculating 
semantic relationships between emoji without visualization.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_calculator.core import SemanticCalculator

def main():
    """Main function for emoji similarity calculation."""
    print("Initializing semantic calculator...")
    calc = SemanticCalculator()
    
    # Define our emoji pairs
    pairs = [
        {"name": "Cognitive", "pole1": "🧠", "pole2": "🎨", "group": "ME"},
        {"name": "Structure", "pole1": "📏", "pole2": "🌊", "group": "ME"},
        {"name": "Detail", "pole1": "📚", "pole2": "📝", "group": "ME"},
        {"name": "Domain", "pole1": "💻", "pole2": "📊", "group": "CONTENT"},
        {"name": "Exploration", "pole1": "🔍", "pole2": "🔮", "group": "CONTENT"},
        {"name": "Complexity", "pole1": "🧩", "pole2": "🔄", "group": "CONTENT"},
        {"name": "Engagement", "pole1": "🔥", "pole2": "💤", "group": "YOU"},
        {"name": "Initiative", "pole1": "🧭", "pole2": "👣", "group": "YOU"},
        {"name": "Expertise", "pole1": "🎓", "pole2": "🌱", "group": "YOU"}
    ]
    
    print("\nCalculating within-pair pole distances:")
    print("=======================================")
    
    # Calculate distance between poles in the same pair
    for pair in pairs:
        pole1_vector = calc.semantic_calculator_emoji_to_vector(pair["pole1"])
        pole2_vector = calc.semantic_calculator_emoji_to_vector(pair["pole2"])
        
        # Calculate cosine similarity
        similarity = calc.semantic_calculator_cosine_similarity(pole1_vector, pole2_vector)
        
        # Calculate Euclidean distance
        distance = calc.semantic_calculator_euclidean_distance(pole1_vector, pole2_vector)
        
        print(f"{pair['name']} Pair: {pair['pole1']} ↔ {pair['pole2']}")
        print(f"  Cosine similarity: {similarity:.4f}")
        print(f"  Euclidean distance: {distance:.4f}")
        print()
    
    print("\nCalculating between-dimension similarities:")
    print("=========================================")
    
    # Calculate similarities between dimensions
    for i, pair1 in enumerate(pairs):
        for j, pair2 in enumerate(pairs[i+1:], i+1):
            # Calculate dimension distance
            dim_similarity = calc.semantic_calculator_dimension_distance(pair1, pair2)
            
            print(f"Dimensions {pair1['name']} and {pair2['name']}:")
            print(f"  Similarity: {dim_similarity:.4f}")
            print()
    
    print("\nCalculating helical components for phase angles:")
    print("=============================================")
    
    # Example phase angle calculations
    for angle in [0, 45, 90, 135, 180]:
        components = calc.semantic_calculator_calculate_helical_components(1.0, angle)
        
        print(f"Phase angle {angle}°:")
        for comp_name, comp_value in components.items():
            print(f"  {comp_name}: {comp_value:.4f}")
        print()

if __name__ == "__main__":
    main()
