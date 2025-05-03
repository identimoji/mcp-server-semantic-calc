#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of analyzing semantic relationships between dimensions.

This script demonstrates how to calculate and visualize
similarities between different oppositional pairs.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_calculator.core import SemanticCalculator

def main():
    """Main function for the example."""
    print("Initializing semantic calculator...")
    calc = SemanticCalculator()
    
    # Define dimensions to analyze
    dimensions = [
        {"name": "Cognitive", "pole1": "🧠", "pole2": "🎨", "group": "ME"},
        {"name": "Structure", "pole1": "📏", "pole2": "🌊", "group": "ME"},
        {"name": "Detail", "pole1": "📚", "pole2": "📝", "group": "ME"},
        {"name": "Trust", "pole1": "🔒", "pole2": "🔓", "group": "ME"},
        {"name": "Complexity", "pole1": "🧩", "pole2": "🔄", "group": "ME"},
        {"name": "Emotion", "pole1": "😊", "pole2": "😔", "group": "ME"},
        {"name": "Agency", "pole1": "👑", "pole2": "🤝", "group": "ME"},
        
        {"name": "Domain", "pole1": "💻", "pole2": "📊", "group": "CONTENT"},
        {"name": "Exploration", "pole1": "🔍", "pole2": "🔮", "group": "CONTENT"},
        {"name": "Progress", "pole1": "🚧", "pole2": "🏁", "group": "CONTENT"},
        {"name": "Flow", "pole1": "⬆️", "pole2": "⬇️", "group": "CONTENT"},
        {"name": "Velocity", "pole1": "🐢", "pole2": "🚀", "group": "CONTENT"},
        
        {"name": "Engagement", "pole1": "🔥", "pole2": "💤", "group": "YOU"},
        {"name": "Initiative", "pole1": "🧭", "pole2": "👣", "group": "YOU"},
        {"name": "Expertise", "pole1": "🎓", "pole2": "🌱", "group": "YOU"},
        {"name": "Receptivity", "pole1": "📖", "pole2": "🔒", "group": "YOU"},
        {"name": "Curiosity", "pole1": "🤔", "pole2": "✅", "group": "YOU"}
    ]
    
    # Calculate similarity matrix
    n = len(dimensions)
    similarity_matrix = np.zeros((n, n))
    
    print("Calculating dimension similarities...")
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity is 1.0
            else:
                similarity_matrix[i, j] = calc.semantic_calculator_dimension_distance(
                    dimensions[i], dimensions[j]
                )
    
    print("Creating heatmap visualization...")
    
    # Create labels for the heatmap
    labels = [f"{d['pole1']}{d['pole2']} ({d['name']})" for d in dimensions]
    
    # Create a mask for the upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
    
    # Set up the figure
    plt.figure(figsize=(16, 14))
    
    # Draw the heatmap
    ax = sns.heatmap(
        similarity_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"label": "Semantic Similarity"}
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add title
    plt.title("Semantic Similarity Between Oppositional Pairs", fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("dimension_similarity.png", dpi=300, bbox_inches="tight")
    print("Heatmap saved to 'dimension_similarity.png'")
    
    # Find clusters of similar dimensions
    from sklearn.cluster import AgglomerativeClustering
    
    # Convert similarity to distance for clustering
    distance_matrix = 1 - similarity_matrix
    
    # Determine optimal number of clusters
    from sklearn.metrics import silhouette_score
    
    # Try different numbers of clusters
    silhouette_scores = []
    max_clusters = min(10, n-1)  # Don't try more clusters than dimensions-1
    
    for num_clusters in range(2, max_clusters + 1):
        clustering = AgglomerativeClustering(
            n_clusters=num_clusters, 
            affinity='precomputed', 
            linkage='average'
        ).fit(distance_matrix)
        
        # Only calculate score if we have more than one sample per cluster
        if len(set(clustering.labels_)) > 1:
            score = silhouette_score(distance_matrix, clustering.labels_, metric='precomputed')
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)  # Invalid score
    
    # Find optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because we started at 2
    
    # Perform clustering with optimal number of clusters
    clustering = AgglomerativeClustering(
        n_clusters=optimal_clusters, 
        affinity='precomputed', 
        linkage='average'
    ).fit(distance_matrix)
    
    # Print clusters
    print(f"\nDimension Clusters (optimal {optimal_clusters} clusters):")
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(dimensions[i])
    
    for cluster_id, cluster_dims in clusters.items():
        print(f"\nCluster {cluster_id + 1}:")
        for dim in cluster_dims:
            print(f"  {dim['pole1']}{dim['pole2']} ({dim['name']}) - {dim['group']}")
    
    # Perform dimensionality reduction to visualize in 2D
    from sklearn.manifold import MDS
    
    # Use MDS to project to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos = mds.fit_transform(distance_matrix)
    
    # Create scatter plot
    plt.figure(figsize=(14, 10))
    
    # Create colors by group
    group_colors = {
        "ME": "blue",
        "CONTENT": "green",
        "YOU": "red"
    }
    
    # Create colors by cluster
    cluster_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot each point
    for i in range(n):
        plt.scatter(
            pos[i, 0], 
            pos[i, 1], 
            color=group_colors[dimensions[i]['group']],
            marker=cluster_markers[clustering.labels_[i] % len(cluster_markers)],
            s=100,
            alpha=0.7,
            edgecolors='black'
        )
        
        # Add label
        plt.annotate(
            f"{dimensions[i]['pole1']}{dimensions[i]['pole2']}",
            (pos[i, 0], pos[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=12
        )
    
    # Add legend for groups
    for group, color in group_colors.items():
        plt.scatter([], [], color=color, label=group, s=100, alpha=0.7, edgecolors='black')
    
    # Add legend for clusters
    for i in range(optimal_clusters):
        plt.scatter(
            [], [], 
            color='gray', 
            marker=cluster_markers[i % len(cluster_markers)],
            label=f'Cluster {i+1}',
            s=100,
            alpha=0.7,
            edgecolors='black'
        )
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.title("2D Projection of Dimension Semantic Space", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.savefig("dimension_clusters.png", dpi=300, bbox_inches="tight")
    print("Cluster visualization saved to 'dimension_clusters.png'")

if __name__ == "__main__":
    main()
