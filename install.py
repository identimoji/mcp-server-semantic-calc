#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Installation script for the semantic calculator MCP.
Uses UV for dependency management.
"""

import subprocess
import os
import sys
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dependencies
DEPENDENCIES = [
    "numpy",
    "scikit-learn",
    "sentence-transformers",
    "torch",
    "matplotlib",
    "umap-learn",
    "plotly",
    "pytest",  # For running tests
]

def check_uv_installation():
    """Check if UV is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def create_venv():
    """Create a virtual environment using UV."""
    logger.info("Creating virtual environment...")
    
    # Check if UV is installed
    if not check_uv_installation():
        logger.error("UV not found. Please install UV first.")
        logger.info("You can install UV with: pip install uv")
        return False
    
    # Create venv
    try:
        cmd = ["uv", "venv"]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Virtual environment created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install dependencies using UV."""
    logger.info("Installing dependencies with UV...")
    
    # Check if UV is installed
    if not check_uv_installation():
        logger.error("UV not found. Please install UV first.")
        logger.info("You can install UV with: pip install uv")
        return False
    
    # Install dependencies
    try:
        cmd = ["uv", "pip", "install"] + DEPENDENCIES
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def main():
    """Main installation function."""
    logger.info("Starting installation of semantic calculator MCP...")
    
    # Create virtual environment
    if not create_venv():
        logger.error("Failed to create virtual environment.")
        logger.info("You can try manually creating a venv with: python -m venv env")
        logger.info("Then activate it with: source env/bin/activate (Unix) or env\\Scripts\\activate (Windows)")
        return
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Installation failed.")
        return
    
    # Test the installation
    logger.info("Testing installation...")
    try:
        from semantic_calculator.core import SemanticCalculator
        calc = SemanticCalculator()
        logger.info("Installation successful!")
    except ImportError as e:
        logger.error(f"Installation test failed: {e}")
    
    logger.info("Installation complete!")
    logger.info("To use this package, activate the virtual environment with:")
    logger.info("  source .venv/bin/activate  (on Unix/Mac)")
    logger.info("  or")
    logger.info("  .venv\\Scripts\\activate  (on Windows)")
    logger.info("Then run the examples with: python examples/emoji_3d_visualization.py")
    

if __name__ == "__main__":
    main()
