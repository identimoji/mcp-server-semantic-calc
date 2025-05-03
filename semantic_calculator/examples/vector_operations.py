#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic vector operations for semantic calculations.

This script demonstrates the core vector operations
for working with semantic vectors.
"""

import numpy as np
from ..core import SemanticCalculator

def main():
    """Main function for vector operations demo."""
    print("Initializing semantic calculator...")
    calc = SemanticCalculator()
    
    # Define a set of emoji to work with
    emojis = [
        "🧠", "🎨", "📏", "🌊", "📚", "📝", 
        "💻", "📊", "🔍", "🔮", "🧩", "🔄",
        "🔥", "💤", "🤔", "✅", "🎓", "🌱"
    ]
    
    print("\nConverting emoji to vectors:")
    print("==========================")
    
    # Convert emoji to vectors
    vectors = {}
    for emoji in emojis:
        vector = calc.semantic_calculator_emoji_to_vector(emoji)
        vectors[emoji] = vector
        print(f"Vector for {emoji}: shape={vector.shape}, first 5 values={vector[:5]}")
    
    print("\nCalculating cosine similarities:")
    print("=============================")
    
    # Calculate cosine similarities between emoji
    similarity_pairs = [
        ("🧠", "🎨"),  # Analytical vs. Creative
        ("🧠", "📏"),  # Analytical vs. Structured
        ("🧠", "💻"),  # Analytical vs. Tech
        ("🎨", "🔮"),  # Creative vs. Exploratory
        ("📚", "📝"),  # Detailed vs. Concise
        ("🔥", "💤"),  # Engaged vs. Disengaged
        ("🎓", "🌱")   # Expert vs. Novice
    ]
    
    for emoji1, emoji2 in similarity_pairs:
        sim = calc.semantic_calculator_cosine_similarity(vectors[emoji1], vectors[emoji2])
        print(f"Similarity between {emoji1} and {emoji2}: {sim:.4f}")
    
    print("\nCalculating Euclidean distances:")
    print("==============================")
    
    for emoji1, emoji2 in similarity_pairs[:5]:  # Just use first 5 pairs
        dist = calc.semantic_calculator_euclidean_distance(vectors[emoji1], vectors[emoji2])
        print(f"Distance between {emoji1} and {emoji2}: {dist:.4f}")
    
    print("\nSimulating 3D coordinates (manually calculated, not using dimensionality reduction):")
    print("=================================================================================")
    
    # Create a manual 3D projection without using dimensionality reduction
    # This avoids the t-SNE error and lets us see the approximate spatial relationships
    
    # Use PCA for simple dimensionality reduction (this will work with small samples)
    from sklearn.decomposition import PCA
    
    # Stack vectors into a matrix
    matrix = np.stack([vectors[emoji] for emoji in emojis])
    
    # Reduce to 3D using PCA (much simpler than t-SNE and works with small samples)
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(matrix)
    
    # Print the 3D coordinates
    for i, emoji in enumerate(emojis):
        print(f"3D coordinates for {emoji}: ({coords_3d[i, 0]:.4f}, {coords_3d[i, 1]:.4f}, {coords_3d[i, 2]:.4f})")
    
    # Calculate distances between pairs in 3D space
    print("\n3D distances between pairs:")
    for emoji1, emoji2 in similarity_pairs[:5]:
        i1 = emojis.index(emoji1)
        i2 = emojis.index(emoji2)
        coord1 = coords_3d[i1]
        coord2 = coords_3d[i2]
        dist_3d = np.linalg.norm(coord1 - coord2)
        print(f"3D distance between {emoji1} and {emoji2}: {dist_3d:.4f}")

if __name__ == "__main__":
    main()
