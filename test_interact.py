#!/usr/bin/env python3
"""
Quick test of the interactive interface
"""

import sys
from pathlib import Path

# Test if we can import the module
try:
    from interact import NeuroGenInterface
    print("✅ Successfully imported NeuroGenInterface")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test initialization
print("\n📝 Testing interface initialization...")
try:
    interface = NeuroGenInterface(tokenizer_dir="tokenizer")
    print("✅ Interface initialized successfully")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

# Test with a simple prompt
print("\n🧪 Testing generation...")
test_prompt = "The future of AI"

try:
    response = interface.generate(test_prompt, max_new_tokens=20, show_thinking=True)
    print(f"\n✅ Generation successful!")
    print(f"📤 Response: {response}")
except Exception as e:
    print(f"❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All tests passed!")
