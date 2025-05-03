#!/bin/bash
# Setup script for the semantic calculator MCP on macOS/Linux

echo "Setting up the semantic calculator MCP..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    pip install uv || pip3 install uv
    
    if ! command -v uv &> /dev/null; then
        echo "Failed to install UV. Please install it manually with: pip install uv"
        exit 1
    fi
fi

echo "Creating virtual environment..."
uv venv

if [ ! -d ".venv" ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install numpy scikit-learn sentence-transformers torch matplotlib umap-learn plotly pytest seaborn

echo "Setup complete!"
echo ""
echo "To use the semantic calculator, first activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run one of the example scripts:"
echo "  python examples/emoji_3d_visualization.py"
echo "  python examples/visualize_emojikey.py"
echo "  python examples/dimension_analysis.py"
echo ""
echo "To deactivate the virtual environment when you're done, run:"
echo "  deactivate"
