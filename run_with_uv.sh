#!/bin/bash
# Run script using UV at your specific location

# Your UV location
LOCAL_UV="/Users/rob/.local/bin/uv"

# Check if UV exists
if [ -f "$LOCAL_UV" ]; then
    echo "Using UV at $LOCAL_UV"
    
    # Use UV run to execute the module, with the current project installed
    "$LOCAL_UV" run -m semantic_calculator.mcp --with-editable .
else
    echo "UV not found at $LOCAL_UV"
    echo "Trying with system Python instead..."
    
    # Fall back to system Python
    python3 -m semantic_calculator.mcp
fi