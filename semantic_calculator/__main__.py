#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for the semantic calculator when called as a module.
"""

import sys
import argparse
import logging
import time
import threading
from .mcp import main as mcp_main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

def delayed_startup(delay_seconds=2):
    """Start the MCP server after a delay to allow for terminal output."""
    logger.info(f"Will start the semantic calculator MCP server in {delay_seconds} seconds...")
    time.sleep(delay_seconds)
    logger.info("Starting semantic calculator MCP server now")
    mcp_main()

def main():
    """Main entry point when run as a module."""
    parser = argparse.ArgumentParser(description="Semantic Calculator MCP")
    parser.add_argument("command", choices=["mcp"], help="Command to run (currently only 'mcp' is supported)")
    parser.add_argument("--delay", type=int, default=2, help="Startup delay in seconds (default: 2)")
    
    args = parser.parse_args()
    
    if args.command == "mcp":
        logger.info("Initializing semantic calculator")
        print("Semantic Calculator is starting up. This might take a moment as models are loaded...")
        # Start in a separate thread to not block terminal output
        thread = threading.Thread(target=delayed_startup, args=(args.delay,))
        thread.daemon = True
        thread.start()
        thread.join()  # Wait for the thread to finish
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
