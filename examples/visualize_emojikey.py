#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of visualizing an Emojikey V3 string in 3D space.

This script demonstrates how to use the semantic calculator
to visualize the dimensions in an Emojikey V3 string.
"""

import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_calculator.core import SemanticCalculator

def main():
    """Main function for the example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize an Emojikey V3 string in 3D space")
    parser.add_argument("--emojikey", type=str, required=False,
                        help="The Emojikey V3 string to visualize")
    parser.add_argument("--output", type=str, default="emojikey_visualization.html",
                        help="Output HTML file (default: emojikey_visualization.html)")
    args = parser.parse_args()
    
    # Use default example if no emojikey provided
    emojikey = args.emojikey
    if not emojikey:
        emojikey = "[ME|🧠🎨8∠40|📏🌊7∠60|📚📝8∠45]|[CONTENT|💻📊9∠20|🔍🔮8∠65|🧩🔄8∠35]|[YOU|🔥💤8∠30|🤔✅8∠40|🎓🌱7∠65]"
        print(f"No emojikey provided, using example:\n{emojikey}")
    
    print("Initializing semantic calculator...")
    calc = SemanticCalculator()
    
    print("Parsing emojikey...")
    parsed = calc.semantic_calculator_parse_emojikey_string(emojikey)
    
    # Print parsed components
    for component in ["ME", "CONTENT", "YOU"]:
        if component in parsed and parsed[component]:
            print(f"\n{component} Component:")
            for pair in parsed[component]:
                print(f"  {pair['emojis']} - Magnitude: {pair['magnitude']}, Angle: {pair['angle']}°")
    
    print("\nCreating 3D visualization...")
    fig = calc.semantic_calculator_emojikey_to_3d_visualization(emojikey)
    
    # Save or display the visualization
    try:
        # If in notebook, display the figure
        from IPython import get_ipython
        if get_ipython() is not None:
            # Show figure inline
            fig.show()
        else:
            # Save to HTML file if not in notebook
            import plotly.io as pio
            pio.write_html(fig, args.output)
            print(f"Visualization saved to '{args.output}'")
    except (ImportError, NameError):
        # Not in Jupyter, save to file
        import plotly.io as pio
        pio.write_html(fig, args.output)
        print(f"Visualization saved to '{args.output}'")

if __name__ == "__main__":
    main()
