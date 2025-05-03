#!/bin/bash
# Make all scripts executable

# Make installation scripts executable
chmod +x install_package.sh
chmod +x mcp_wrapper.py
chmod +x setup.py

# Make example scripts executable
chmod +x examples/calculate_emoji_similarity.py
chmod +x examples/analyze_emojikey.py
chmod +x examples/vector_operations.py
chmod +x examples/emoji_3d_visualization.py
chmod +x examples/visualize_emojikey.py
chmod +x examples/dimension_analysis.py

echo "All scripts are now executable!"
