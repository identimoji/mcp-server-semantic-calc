#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse and analyze an Emojikey V3 string.

This script demonstrates how to use the semantic calculator
to analyze the components of an Emojikey V3 string.
"""

import argparse
from ..core import SemanticCalculator

def main():
    """Main function for the example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parse and analyze an Emojikey V3 string")
    parser.add_argument("--emojikey", type=str, required=False,
                        help="The Emojikey V3 string to analyze")
    args = parser.parse_args()
    
    # Use default example if no emojikey provided
    emojikey = args.emojikey
    if not emojikey:
        emojikey = "[ME|🧠🎨8∠40|📏🌊7∠60|📚📝8∠45]|[CONTENT|💻📊9∠20|🔍🔮8∠65|🧩🔄8∠35]|[YOU|🔥💤8∠30|🤔✅8∠40|🎓🌱7∠65]"
        print(f"No emojikey provided, using example:\n{emojikey}")
    
    print("\nInitializing semantic calculator...")
    calc = SemanticCalculator()
    
    print("\nParsing emojikey components:")
    print("==========================")
    parsed = calc.semantic_calculator_parse_emojikey_string(emojikey)
    
    # Print parsed components
    for component in ["ME", "CONTENT", "YOU"]:
        if component in parsed and parsed[component]:
            print(f"\n{component} Component:")
            for pair in parsed[component]:
                print(f"  {pair['emojis']} - Magnitude: {pair['magnitude']}, Angle: {pair['angle']}°")
    
    print("\nCalculating helical components for each dimension:")
    print("=============================================")
    for component in ["ME", "CONTENT", "YOU"]:
        if component in parsed and parsed[component]:
            print(f"\n{component} Component Helical Representation:")
            for pair in parsed[component]:
                components = calc.semantic_calculator_calculate_helical_components(
                    pair['magnitude'], 
                    pair['angle']
                )
                print(f"  {pair['emojis']}:")
                for comp_name, comp_value in components.items():
                    print(f"    {comp_name}: {comp_value:.4f}")
    
    print("\nCalculating semantic similarity between dimensions:")
    print("==============================================")
    
    # Collect all dimensions
    all_dimensions = []
    for component in ["ME", "CONTENT", "YOU"]:
        if component in parsed and parsed[component]:
            for pair in parsed[component]:
                all_dimensions.append({
                    "name": f"{component}-{pair['emojis']}",
                    "pole1": pair["pole1"],
                    "pole2": pair["pole2"],
                    "group": component
                })
    
    # Calculate and print similarities
    for i, dim1 in enumerate(all_dimensions):
        for j, dim2 in enumerate(all_dimensions[i+1:], i+1):
            sim = calc.semantic_calculator_dimension_distance(
                {"pole1": dim1["pole1"], "pole2": dim1["pole2"]},
                {"pole1": dim2["pole1"], "pole2": dim2["pole2"]}
            )
            print(f"  {dim1['name']} ↔ {dim2['name']}: {sim:.4f}")

if __name__ == "__main__":
    main()
