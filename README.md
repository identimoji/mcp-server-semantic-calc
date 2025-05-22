# Semantic Calculator MCP

A Python-based MCP tool for semantic operations on vectors, text, and emoji, with specialized support for the Emojikey V3 system.

## Features

- Calculate semantic similarities between vectors
- Convert text and emoji to vector embeddings
- Calculate helical components for phase angle representations
- Parse and analyze Emojikey V3 strings
- Calculate semantic field distance between dimensions

## Requirements

- **Python 3.10+** (required by MCP SDK)
- Apple Silicon Mac (M1/M2/M3) or Intel Mac
- Claude Desktop application

## Installation

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-server-semantic-calc.git
cd mcp-server-semantic-calc

# Install dependencies using uv (recommended)
uv sync

# Or install in development mode
uv pip install -e .
```

### 2. Configure Claude Desktop

Add this configuration to your Claude Desktop settings (typically `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "Semantic Calculator": {
      "command": "arch",
      "args": [
        "-arm64",
        "/Users/rob/.local/bin/uv",
        "--directory",
        "/Users/rob/repos/mcp-server-semantic-calc",
        "run",
        "-m",
        "semantic_calculator",
        "mcp"
      ]
    }
  }
}
```

**Important Notes:**
- Replace `/Users/rob/` with your actual home directory path
- The `-arm64` flag ensures native Apple Silicon execution
- Make sure `/Users/rob/.local/bin/uv` exists (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 3. Restart Claude Desktop

Restart Claude Desktop to load the new MCP server.

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

### Core Dependencies (Required)

These dependencies are automatically installed via `pyproject.toml`:

- **Python 3.10+** - Required by MCP SDK
- **mcp>=0.9.0** - Model Context Protocol SDK
- **sentence-transformers>=2.2.0** - For semantic vector embeddings
- **numpy>=1.20.0** - For vector operations
- **scikit-learn>=1.0.0** - For distance calculations
- **torch>=1.10.0** - Required by SentenceBERT

### Visualization Dependencies (Optional)

- **matplotlib>=3.4.0** - For basic visualization
- **plotly>=5.5.0** - For interactive 3D visualization
- **umap-learn>=0.5.2** - For dimensionality reduction
- **seaborn>=0.11.2** - For enhanced visualizations

### Development Dependencies

- **pytest>=7.0.0** - For testing (install with `uv sync --dev`)

**Note:** The semantic calculator will refuse to start if any of the core dependencies are not installed. All dependencies are managed through `pyproject.toml` and installed automatically with `uv sync`.

## License

MIT