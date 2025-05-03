# Semantic Calculator MCP

A Python-based MCP tool for semantic operations on vectors, text, and emoji, with specialized support for the Emojikey V3 system.

## Features

- Calculate semantic similarities between vectors
- Convert text and emoji to vector embeddings
- Calculate helical components for phase angle representations
- Parse and analyze Emojikey V3 strings
- Calculate semantic field distance between dimensions

## Installation

### Installation for Development (Apple Silicon Macs)

If you're on an Apple Silicon Mac (M1/M2/M3), use the provided installation script that ensures the correct architecture is used:

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-server-semantic-calc.git
cd mcp-server-semantic-calc

# Run the installation script (uses native arm64 architecture)
./install_native.sh
```

The script will:
1. Switch to arm64 architecture if needed
2. Install the package in editable mode
3. Install all required dependencies
4. Register the MCP server with Claude Desktop

### Running the Server Manually

```bash
# Run in native arm64 mode (Apple Silicon Macs)
./run_native.sh

# Or run directly with Python in the correct architecture
arch -arm64 python3 -m semantic_calculator.mcp

# Or run directly using the MCP CLI (if architecture issues are resolved)
arch -arm64 ~/Library/Python/3.11/bin/mcp run semantic_calculator/mcp.py
```

### Development Workflow

For development or customization:

1. Make changes to the code
2. No need to reinstall - editable mode (`-e`) allows changes to be detected automatically
3. Restart Claude Desktop to pick up the changes

## Usage

### Direct Python Usage

```python
from semantic_calculator.core import SemanticCalculator

# Initialize the calculator
calc = SemanticCalculator()

# Calculate similarity between two emoji
similarity = calc.semantic_calculator_cosine_similarity(
    calc.semantic_calculator_emoji_to_vector("🧠"),
    calc.semantic_calculator_emoji_to_vector("🎨")
)
print(f"Similarity: {similarity}")
```

### MCP Usage (in Claude)

Once configured in Claude Desktop, you can use it with:

```javascript
// Convert emoji to vector
const brainVector = semantic_calc_emoji_to_vector({
  emoji: "🧠"
});

const artVector = semantic_calc_emoji_to_vector({
  emoji: "🎨"
});

// Calculate similarity
const similarity = semantic_calc_cosine_similarity({
  vector1: brainVector,
  vector2: artVector
});

console.log(`Similarity: ${similarity}`);
```

Note: All tool functions have been prefixed with `semantic_calc_` to avoid naming conflicts with other tools.

### Example Scripts

```bash
# After installation, run examples
python -m semantic_calculator.examples.vector_operations
python -m semantic_calculator.examples.calculate_emoji_similarity
python -m semantic_calculator.examples.analyze_emojikey
```

## Core Functions

- `semantic_calc_text_to_vector`: Convert text to a vector embedding
- `semantic_calc_emoji_to_vector`: Convert emoji to a vector embedding
- `semantic_calc_cosine_similarity`: Calculate cosine similarity between vectors
- `semantic_calc_euclidean_distance`: Calculate Euclidean distance between vectors
- `semantic_calc_manhattan_distance`: Calculate Manhattan distance between vectors
- `semantic_calc_dimension_distance`: Calculate similarity between dimensions
- `semantic_calc_calculate_helical_components`: Calculate helical components from magnitude/phase
- `semantic_calc_parse_emojikey_string`: Parse emojikey strings

## Dependencies

The semantic calculator strictly requires the following dependencies to work:

- sentence-transformers (SentenceBERT) - Required for semantic vector embeddings
- numpy - Required for vector operations
- scikit-learn - Required for distance calculations
- torch - Required by SentenceBERT

For visualization features, these additional dependencies are needed:
- matplotlib - For basic visualization
- plotly - For interactive 3D visualization
- umap-learn - For dimensionality reduction
- seaborn - For enhanced visualizations

Note: The semantic calculator will refuse to start if any of the core dependencies (SentenceBERT, numpy, scikit-learn) are not installed.

## License

MIT