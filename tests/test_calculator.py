"""
Tests for the semantic calculator.

Run with: pytest -v
"""

import pytest
import numpy as np

from semantic_calculator.core import SemanticCalculator

@pytest.fixture
def calculator():
    """Initialize a semantic calculator for testing."""
    return SemanticCalculator()

def test_emoji_to_vector(calculator):
    """Test the emoji to vector conversion."""
    # Get vector for an emoji
    vector = calculator.semantic_calculator_emoji_to_vector("🧠")
    
    # Check that it's a numpy array
    assert isinstance(vector, np.ndarray)
    
    # Check it has reasonable size
    assert vector.shape[0] > 10
    
    # Test cache functionality - should return same vector
    vector2 = calculator.semantic_calculator_emoji_to_vector("🧠")
    assert np.array_equal(vector, vector2)

def test_cosine_similarity(calculator):
    """Test cosine similarity calculation."""
    # Create some test vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([1, 1, 0])
    
    # Test orthogonal vectors
    assert calculator.semantic_calculator_cosine_similarity(v1, v2) == 0.0
    
    # Test identical vectors
    assert calculator.semantic_calculator_cosine_similarity(v1, v1) == 1.0
    
    # Test 45 degree angle
    sim = calculator.semantic_calculator_cosine_similarity(v1, v3)
    assert abs(sim - 1/np.sqrt(2)) < 1e-10

def test_helical_components(calculator):
    """Test helical component calculation."""
    # Test with magnitude 1.0 and angle 0
    components = calculator.semantic_calculator_calculate_helical_components(1.0, 0.0)
    
    # Linear component should equal magnitude
    assert components["linear"] == 1.0
    
    # Sine should be zero for angle 0
    assert abs(components["sin_2"]) < 1e-10
    
    # Cosine should be 1.0 for angle 0
    assert abs(components["cos_2"] - 1.0) < 1e-10
    
    # Test with magnitude 1.0 and angle 90
    components = calculator.semantic_calculator_calculate_helical_components(1.0, 90.0)
    
    # Sine should be 1.0 for angle 90
    assert abs(components["sin_2"] - 1.0) < 1e-10
    
    # Cosine should be 0.0 for angle 90
    assert abs(components["cos_2"]) < 1e-10

def test_parse_emojikey_string(calculator):
    """Test parsing emojikey strings."""
    # Test a simple emojikey
    emojikey = "[ME|🧠🎨8∠40]|[CONTENT|💻📊9∠20]|[YOU|🔥💤8∠30]"
    
    parsed = calculator.semantic_calculator_parse_emojikey_string(emojikey)
    
    # Check structure
    assert "ME" in parsed
    assert "CONTENT" in parsed
    assert "YOU" in parsed
    
    # Check ME component
    assert len(parsed["ME"]) == 1
    assert parsed["ME"][0]["emojis"] == "🧠🎨"
    assert parsed["ME"][0]["magnitude"] == 8.0
    assert parsed["ME"][0]["angle"] == 40.0
    
    # Check CONTENT component
    assert len(parsed["CONTENT"]) == 1
    assert parsed["CONTENT"][0]["emojis"] == "💻📊"
    assert parsed["CONTENT"][0]["magnitude"] == 9.0
    assert parsed["CONTENT"][0]["angle"] == 20.0
    
    # Check YOU component
    assert len(parsed["YOU"]) == 1
    assert parsed["YOU"][0]["emojis"] == "🔥💤"
    assert parsed["YOU"][0]["magnitude"] == 8.0
    assert parsed["YOU"][0]["angle"] == 30.0

def test_dimension_distance(calculator):
    """Test dimension distance calculation."""
    # Define some test dimensions
    dim1 = {"pole1": "🧠", "pole2": "🎨"}
    dim2 = {"pole1": "📏", "pole2": "🌊"}
    
    # Calculate distance
    distance = calculator.semantic_calculator_dimension_distance(dim1, dim2)
    
    # Should return a float between 0 and 1
    assert isinstance(distance, float)
    assert 0.0 <= distance <= 1.0
    
    # Distance to self should be high
    self_distance = calculator.semantic_calculator_dimension_distance(dim1, dim1)
    assert self_distance > 0.8
