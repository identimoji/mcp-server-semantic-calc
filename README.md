# Semantic Calculator MCP

A Python-based MCP tool for semantic operations on vectors, text, and emoji, with specialized support for the Emojikey V3 system.

## Features

- Calculate semantic similarities between vectors
- Convert text and emoji to vector embeddings
- Calculate helical components for phase angle representations
- Parse and analyze Emojikey V3 strings
- Calculate semantic field distance between dimensions

## Installation

### Install with UV (Recommended)

```bash
# Install with uv
uv tool install semantic-calculator
```

### Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "semantic-calculator": {
      "command": "uvx",
      "args": [
        "semantic-calculator",
        "mcp"
      ]
    }
  }
}
```

### Manual Installation (Development)

For development or customization:

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-server-semantic-calc.git
cd mcp-server-semantic-calc

# Install in development mode
pip install -e .
```

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
const brainVector = semantic_calculator_emoji_to_vector({
  emoji: "🧠"
});

const artVector = semantic_calculator_emoji_to_vector({
  emoji: "🎨"
});

// Calculate similarity
const similarity = semantic_calculator_cosine_similarity({
  vector1: brainVector,
  vector2: artVector
});

console.log(`Similarity: ${similarity}`);
```

### Example Scripts

```bash
# After installation, run examples
python -m semantic_calculator.examples.vector_operations
python -m semantic_calculator.examples.calculate_emoji_similarity
python -m semantic_calculator.examples.analyze_emojikey
```

## Core Functions

- `semantic_calculator_text_to_vector`: Convert text to a vector embedding
- `semantic_calculator_emoji_to_vector`: Convert emoji to a vector embedding
- `semantic_calculator_cosine_similarity`: Calculate cosine similarity between vectors
- `semantic_calculator_euclidean_distance`: Calculate Euclidean distance between vectors
- `semantic_calculator_dimension_distance`: Calculate similarity between dimensions
- `semantic_calculator_calculate_helical_components`: Calculate helical components from magnitude/phase
- `semantic_calculator_parse_emojikey_string`: Parse emojikey strings

## License

MIT
