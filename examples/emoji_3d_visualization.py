#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of 3D visualization of semantic emoji pairs.

This script demonstrates how to use the semantic calculator
to visualize emoji pairs in 3D space based on their semantic
relationships.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_calculator.core import SemanticCalculator

def main():
    """Main function for the example."""
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
    
    print("Calculating vector embeddings for emoji...")
    
    # Get vectors for each pole
    poles = []
    vectors = []
    for pair in pairs:
        # Add first pole
        poles.append({
            "name": pair["name"] + "-1",
            "emoji": pair["pole1"],
            "group": pair["group"]
        })
        vectors.append(calc.semantic_calculator_emoji_to_vector(pair["pole1"]))
        
        # Add second pole
        poles.append({
            "name": pair["name"] + "-2",
            "emoji": pair["pole2"],
            "group": pair["group"]
        })
        vectors.append(calc.semantic_calculator_emoji_to_vector(pair["pole2"]))
    
    print("Projecting to 3D space...")
    
    # Project to 3D space
    coords_3d = calc.semantic_calculator_vectors_to_3d_coordinates(vectors)
    
    # Define connections (pairs that should be connected)
    connections = [(i*2, i*2+1) for i in range(len(pairs))]
    
    print("Creating visualization...")
    
    # Create the visualization
    fig = calc.semantic_calculator_plot_3d_poles(poles, coords_3d, connections)
    
    # Check if running in Jupyter notebook
    try:
        # If in notebook, display the figure
        from IPython import get_ipython
        if get_ipython() is not None:
            # Show figure inline
            fig.show()
        else:
            # Save to HTML file if not in notebook
            import plotly.io as pio
            pio.write_html(fig, "emoji_3d_visualization.html")
            print("Visualization saved to 'emoji_3d_visualization.html'")
    except (ImportError, NameError):
        # Not in Jupyter, save to file
        import plotly.io as pio
        pio.write_html(fig, "emoji_3d_visualization.html")
        print("Visualization saved to 'emoji_3d_visualization.html'")

if __name__ == "__main__":
    main()
