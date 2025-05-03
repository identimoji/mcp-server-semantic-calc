#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for running examples as a module.
"""

import sys
import argparse

def main():
    """Main entry point for running examples."""
    parser = argparse.ArgumentParser(description="Semantic Calculator Examples")
    parser.add_argument("example", choices=["vector_operations", "calculate_emoji_similarity", "analyze_emojikey"],
                        help="The example to run")
    parser.add_argument("--emojikey", type=str, required=False,
                        help="Emojikey string (for analyze_emojikey example)")
    
    args = parser.parse_args()
    
    if args.example == "vector_operations":
        from .vector_operations import main
        main()
    elif args.example == "calculate_emoji_similarity":
        from .calculate_emoji_similarity import main
        main()
    elif args.example == "analyze_emojikey":
        from .analyze_emojikey import main
        # If emojikey is provided, pass it as a command line argument
        if args.emojikey:
            sys.argv = [sys.argv[0], "--emojikey", args.emojikey]
        main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
