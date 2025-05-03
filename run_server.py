#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple script to run the MCP server directly for testing.
"""

import os
import sys
import logging

# Configure logging to stderr only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

try:
    # Add the project directory to the Python path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)
    logger.info(f"Added {project_dir} to Python path")
    
    # Print current sys.path for debugging
    logger.info(f"Current sys.path: {sys.path}")
    
    # Try importing the module
    logger.info("Attempting to import semantic_calculator.mcp...")
    from semantic_calculator.mcp import main
    logger.info("Successfully imported semantic_calculator.mcp")
    
    # Run the MCP server
    logger.info("Starting MCP server...")
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("This might be because the semantic_calculator package is not installed or not in the Python path.")
    logger.error("Try installing the package with: pip install -e .")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    sys.exit(1)
