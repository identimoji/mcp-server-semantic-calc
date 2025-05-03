"""
Core implementation of the Semantic Calculator.

This module provides functions for operating on semantic vectors,
with special support for emoji vectors and the Emojikey V3 system.
"""

import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Any
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticCalculator:
    """
    A calculator for semantic operations on vectors, text, and emoji.
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the semantic calculator.
        
        Args:
            model_name: The name of the sentence transformer model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized SemanticCalculator with model {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Some functions will be unavailable.")
            self.model = None
            
        # Cache for emoji embeddings to improve performance
        self.emoji_cache = {}

    def semantic_calculator_text_to_vector(self, text: str) -> np.ndarray:
        """
        Convert text to a vector embedding.
        
        Args:
            text: The text to convert
            
        Returns:
            A vector embedding of the text
        """
        if self.model is None:
            raise ImportError("sentence-transformers not installed. Cannot convert text to vector.")
            
        return self.model.encode(text)
    
    def semantic_calculator_emoji_to_vector(self, emoji: str) -> np.ndarray:
        """
        Convert an emoji to a vector embedding.
        
        Args:
            emoji: The emoji character to convert
            
        Returns:
            A vector embedding of the emoji
        """
        if emoji in self.emoji_cache:
            return self.emoji_cache[emoji]
        
        if self.model is None:
            raise ImportError("sentence-transformers not installed. Cannot convert emoji to vector.")
        
        # First, try to get emoji Unicode name for better semantic representation
        import unicodedata
        try:
            emoji_name = unicodedata.name(emoji)
            # Add "emoji" to the name for better context
            text_representation = f"emoji {emoji_name.lower()}"
        except (ValueError, TypeError):
            # If we can't get the name, just use the emoji itself
            text_representation = emoji
            
        vector = self.model.encode(text_representation)
        
        # Cache the result
        self.emoji_cache[emoji] = vector
        return vector
    
    def semantic_calculator_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            The cosine similarity as a float between -1 and 1
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Reshape for sklearn
        v1 = vector1.reshape(1, -1)
        v2 = vector2.reshape(1, -1)
        
        return float(cosine_similarity(v1, v2)[0][0])
    
    def semantic_calculator_euclidean_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            The Euclidean distance
        """
        return float(np.linalg.norm(vector1 - vector2))
    
    def semantic_calculator_manhattan_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate the Manhattan distance between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            The Manhattan distance
        """
        return float(np.sum(np.abs(vector1 - vector2)))
    
    def semantic_calculator_dimensionality_reduction(
        self, 
        vectors: List[np.ndarray], 
        dimensions: int = 3, 
        method: str = "t-SNE"
    ) -> np.ndarray:
        """
        Project a set of vectors to a lower-dimensional space.
        
        Args:
            vectors: List of vectors to project
            dimensions: Number of dimensions in the output (2 or 3)
            method: Reduction method: "t-SNE" or "UMAP"
            
        Returns:
            Array of reduced vectors
        """
        vectors_array = np.array(vectors)
        
        if method.lower() == "t-sne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=dimensions, random_state=42)
        elif method.lower() == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=dimensions, random_state=42)
            except ImportError:
                logger.warning("umap-learn not installed. Falling back to t-SNE.")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=dimensions, random_state=42)
        else:
            raise ValueError(f"Method {method} not supported. Use 't-SNE' or 'UMAP'.")
            
        return reducer.fit_transform(vectors_array)
    
    def semantic_calculator_dimension_distance(
        self, 
        dimension1: Dict[str, str], 
        dimension2: Dict[str, str]
    ) -> float:
        """
        Calculate the semantic similarity between two dimensions.
        
        Each dimension is defined by its two poles (e.g., Analytical↔Creative).
        
        Args:
            dimension1: Dict with 'pole1' and 'pole2' keys for the first dimension
            dimension2: Dict with 'pole1' and 'pole2' keys for the second dimension
            
        Returns:
            A similarity score between 0 and 1
        """
        # Get vector embeddings for each pole
        dim1_pole1 = self.semantic_calculator_emoji_to_vector(dimension1['pole1'])
        dim1_pole2 = self.semantic_calculator_emoji_to_vector(dimension1['pole2'])
        dim2_pole1 = self.semantic_calculator_emoji_to_vector(dimension2['pole1'])
        dim2_pole2 = self.semantic_calculator_emoji_to_vector(dimension2['pole2'])
        
        # Calculate cross-dimension similarities
        sim1 = self.semantic_calculator_cosine_similarity(dim1_pole1, dim2_pole1)
        sim2 = self.semantic_calculator_cosine_similarity(dim1_pole2, dim2_pole2)
        sim3 = self.semantic_calculator_cosine_similarity(dim1_pole1, dim2_pole2)
        sim4 = self.semantic_calculator_cosine_similarity(dim1_pole2, dim2_pole1)
        
        # More sophisticated approach:
        # Higher similarity if the mapping of poles is consistent
        # (i.e. pole1→pole1, pole2→pole2 has higher similarity than pole1→pole2, pole2→pole1)
        direct_mapping = (sim1 + sim2) / 2
        cross_mapping = (sim3 + sim4) / 2
        
        # If direct mapping is stronger, use that; otherwise use cross mapping
        if direct_mapping > cross_mapping:
            return float(direct_mapping)
        else:
            return float(cross_mapping)
    
    def semantic_calculator_calculate_helical_components(
        self, 
        magnitude: float, 
        phase_angle: float,
        periods: List[float] = [2, 5, 10, 100]
    ) -> Dict[str, float]:
        """
        Calculate helical components from magnitude and phase angle.
        
        This implements the "Clock algorithm" concept from the MIT paper
        on helical representations in LLMs.
        
        Args:
            magnitude: The magnitude (0-1)
            phase_angle: The phase angle in degrees (0-180)
            periods: List of periods for the helical components
            
        Returns:
            Dictionary with sine, cosine, and linear components
        """
        # Convert phase angle from degrees to radians
        phase_rad = np.radians(phase_angle)
        
        # Initialize components dictionary
        components = {
            "linear": magnitude,
        }
        
        # Calculate sine and cosine components for each period
        for period in periods:
            # Scale the phase angle by the period
            scaled_angle = phase_rad * (360 / period)
            
            # Calculate components
            components[f"sin_{period}"] = magnitude * np.sin(scaled_angle)
            components[f"cos_{period}"] = magnitude * np.cos(scaled_angle)
        
        return components
    
    def semantic_calculator_vectors_to_3d_coordinates(
        self, 
        vectors: List[np.ndarray],
        method: str = "t-SNE"
    ) -> np.ndarray:
        """
        Convert vectors to 3D coordinates preserving relative distances.
        
        Args:
            vectors: List of vectors to convert
            method: Dimensionality reduction method
            
        Returns:
            Array of 3D coordinates
        """
        return self.semantic_calculator_dimensionality_reduction(vectors, dimensions=3, method=method)
    
    def semantic_calculator_plot_3d_poles(
        self, 
        poles: List[Dict[str, str]], 
        coordinates: np.ndarray, 
        connections: Optional[List[Tuple[int, int]]] = None,
        colors: Optional[List[str]] = None,
        interactive: bool = True
    ) -> Any:
        """
        Create a 3D visualization of semantic poles.
        
        Args:
            poles: List of pole information (name, emoji, group, etc.)
            coordinates: 3D coordinates for each pole
            connections: List of pairs to connect with lines
            colors: List of colors for each group
            interactive: Whether to create an interactive plot
            
        Returns:
            The plot object
        """
        if interactive:
            try:
                import plotly.graph_objects as go
                
                # Create groups for coloring
                groups = []
                for pole in poles:
                    if 'group' in pole and pole['group'] not in groups:
                        groups.append(pole['group'])
                
                # Default colors if not provided
                if colors is None:
                    import matplotlib.pyplot as plt
                    cmap = plt.cm.get_cmap('tab10', len(groups))
                    colors = [f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" 
                             for r, g, b, _ in [cmap(i) for i in range(len(groups))]]
                
                # Map groups to colors
                group_colors = {group: colors[i % len(colors)] for i, group in enumerate(groups)}
                
                # Create color list for points
                point_colors = [group_colors.get(pole.get('group', 'default'), 'rgb(100,100,100)') 
                               for pole in poles]
                
                # Create figure
                fig = go.Figure()
                
                # Add points
                fig.add_trace(go.Scatter3d(
                    x=coordinates[:, 0],
                    y=coordinates[:, 1],
                    z=coordinates[:, 2],
                    mode='markers+text',
                    marker=dict(
                        size=10,
                        color=point_colors,
                    ),
                    text=[f"{p.get('emoji', '')} {p.get('name', '')}" for p in poles],
                    hoverinfo='text'
                ))
                
                # Add lines for connected poles
                if connections:
                    for i, j in connections:
                        # Determine color based on endpoints
                        color1 = point_colors[i]
                        color2 = point_colors[j]
                        # Use the same color if they're from the same group
                        line_color = color1 if color1 == color2 else 'rgba(100, 100, 100, 0.8)'
                        
                        fig.add_trace(go.Scatter3d(
                            x=[coordinates[i, 0], coordinates[j, 0]],
                            y=[coordinates[i, 1], coordinates[j, 1]],
                            z=[coordinates[i, 2], coordinates[j, 2]],
                            mode='lines',
                            line=dict(width=4, color=line_color),
                            hoverinfo='none'
                        ))
                
                # Update layout
                fig.update_layout(
                    title="Semantic Pole Visualization",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z"
                    ),
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
                return fig
            
            except ImportError:
                logger.warning("Plotly not installed. Falling back to matplotlib.")
                interactive = False
        
        if not interactive:
            # Fallback to matplotlib for non-interactive plot
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points
            ax.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
                c=range(len(poles)),
                cmap='viridis',
                s=50
            )
            
            # Add labels
            for i, pole in enumerate(poles):
                ax.text(
                    coordinates[i, 0],
                    coordinates[i, 1],
                    coordinates[i, 2],
                    f"{pole.get('emoji', '')} {pole.get('name', '')}",
                    size=8
                )
            
            # Add lines
            if connections:
                for i, j in connections:
                    ax.plot(
                        [coordinates[i, 0], coordinates[j, 0]],
                        [coordinates[i, 1], coordinates[j, 1]],
                        [coordinates[i, 2], coordinates[j, 2]],
                        'gray', alpha=0.7
                    )
            
            ax.set_title("Semantic Pole Visualization")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            
            return fig
    
    def semantic_calculator_parse_emojikey_string(self, emojikey: str) -> Dict[str, Any]:
        """
        Parse an emojikey string into a structured representation.
        
        Args:
            emojikey: The emojikey string to parse
            
        Returns:
            A dictionary representing the parsed emojikey
        """
        import re
        
        # Initialize result structure
        result = {
            "ME": [],
            "CONTENT": [],
            "YOU": []
        }
        
        # Pattern to extract components
        component_pattern = r'\[(ME|CONTENT|YOU)\|(.*?)\]'
        component_matches = re.findall(component_pattern, emojikey)
        
        for component_type, pairs_str in component_matches:
            # Split pairs by pipe
            pairs = pairs_str.split('|')
            
            for pair in pairs:
                # Extract emoji pair, magnitude, and angle
                pair_match = re.match(r'([^0-9∠]+)([0-9])∠([0-9]+)', pair)
                if pair_match:
                    emojis, magnitude, angle = pair_match.groups()
                    
                    # Add to result
                    result[component_type].append({
                        "emojis": emojis,
                        "pole1": emojis[0],  # First emoji
                        "pole2": emojis[1],  # Second emoji
                        "magnitude": float(magnitude),
                        "angle": float(angle)
                    })
        
        return result
    
    def semantic_calculator_emojikey_to_3d_visualization(
        self, 
        emojikey: str,
        pairs_to_visualize: Optional[List[str]] = None
    ) -> Any:
        """
        Create a 3D visualization from an emojikey string.
        
        Args:
            emojikey: The emojikey string to visualize
            pairs_to_visualize: List of emoji pairs to include, or None for all
            
        Returns:
            A 3D plot of the emojikey dimensions
        """
        # Parse the emojikey
        parsed = self.semantic_calculator_parse_emojikey_string(emojikey)
        
        # Collect all pairs from all components
        all_pairs = []
        for component in ["ME", "CONTENT", "YOU"]:
            for pair_data in parsed[component]:
                all_pairs.append({
                    "emojis": pair_data["emojis"],
                    "component": component
                })
        
        # Filter if requested
        if pairs_to_visualize:
            all_pairs = [p for p in all_pairs if p["emojis"] in pairs_to_visualize]
        
        # Extract all unique poles
        poles = []
        pole_indices = {}
        idx = 0
        
        for pair in all_pairs:
            emojis = pair["emojis"]
            component = pair["component"]
            
            # Add first pole
            pole1_name = f"{component}-{emojis[0]}"
            poles.append({
                "name": pole1_name,
                "emoji": emojis[0],
                "group": component
            })
            pole_indices[pole1_name] = idx
            idx += 1
            
            # Add second pole
            pole2_name = f"{component}-{emojis[1]}"
            poles.append({
                "name": pole2_name,
                "emoji": emojis[1],
                "group": component
            })
            pole_indices[pole2_name] = idx
            idx += 1
        
        # Get vectors for all poles
        vectors = []
        for pole in poles:
            vectors.append(self.semantic_calculator_emoji_to_vector(pole["emoji"]))
        
        # Create connections list
        connections = []
        for pair in all_pairs:
            emojis = pair["emojis"]
            component = pair["component"]
            
            idx1 = pole_indices[f"{component}-{emojis[0]}"]
            idx2 = pole_indices[f"{component}-{emojis[1]}"]
            connections.append((idx1, idx2))
        
        # Project to 3D
        coords_3d = self.semantic_calculator_vectors_to_3d_coordinates(vectors)
        
        # Create colors for components
        colors = {
            "ME": "rgb(59, 130, 246)",      # Blue
            "CONTENT": "rgb(16, 185, 129)", # Green
            "YOU": "rgb(239, 68, 68)"       # Red
        }
        
        # Plot
        return self.semantic_calculator_plot_3d_poles(
            poles=poles, 
            coordinates=coords_3d, 
            connections=connections,
            colors=list(colors.values())
        )
