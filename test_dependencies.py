#!/usr/bin/env python3
"""
Test script to check all dependencies and identify issues.
"""

import sys
import os
import importlib
import traceback

def check_dependency(module_name, version_attr=None):
    try:
        module = importlib.import_module(module_name)
        if version_attr:
            version = getattr(module, version_attr, None)
            if version:
                print(f"✅ {module_name} (version: {version})")
            else:
                print(f"✅ {module_name} (version attribute {version_attr} not found)")
        else:
            print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name} - Unexpected error: {e}")
        traceback.print_exc()
        return False

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("\nChecking dependencies...\n")

# Check core dependencies
dependencies = [
    ("numpy", "__version__"),
    ("scikit-learn", None),
    ("torch", "__version__"),
    ("sentence_transformers", "__version__"),
    ("matplotlib", "__version__"),
    ("umap", None),
    ("plotly", "__version__"),
    ("mcp.server.fastmcp", None),
]

for dep, version_attr in dependencies:
    check_dependency(dep, version_attr)

# Try to import our core module
print("\nTrying to import semantic calculator core...")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from semantic_calculator.core import SemanticCalculator
    print("✅ semantic_calculator.core imported successfully")
    
    # Create calculator instance
    calc = SemanticCalculator()
    print("✅ SemanticCalculator initialized")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    try:
        # Test parsing emojikey
        test_key = "[ME|🧠🎨8∠40]"
        parsed = calc.semantic_calculator_parse_emojikey_string(test_key)
        print(f"✅ Parsed emojikey: {parsed}")
        
        # Test helical components calculation
        comp = calc.semantic_calculator_calculate_helical_components(5, 40)
        print(f"✅ Helical components: {comp}")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        traceback.print_exc()
    
except Exception as e:
    print(f"❌ Failed to import semantic_calculator.core: {e}")
    traceback.print_exc()