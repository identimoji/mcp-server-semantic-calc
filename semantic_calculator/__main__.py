#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for the semantic calculator when called as a module.
"""

import sys
import argparse
import json
import logging
from .mcp_server import run_mcp_server

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point when run as a module."""
    parser = argparse.ArgumentParser(description="Semantic Calculator MCP")
    parser.add_argument("command", choices=["mcp"], help="Command to run (currently only 'mcp' is supported)")
    
    args = parser.parse_args()
    
    if args.command == "mcp":
        logger.info("Starting semantic calculator MCP server")
        run_mcp_server()
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
