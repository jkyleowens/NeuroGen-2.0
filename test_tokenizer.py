#!/usr/bin/env python3
"""
Test script to verify SentencePiece tokenizer is working correctly
"""

import json
from pathlib import Path

try:
    import sentencepiece as spm
    print("✅ SentencePiece module imported successfully")
except ImportError:
    print("❌ SentencePiece not installed. Run: pip install sentencepiece")
    exit(1)

# Load tokenizer configuration
tokenizer_dir = Path(__file__).parent / "tokenizer"
tokenizer_state_path = tokenizer_dir / "tokenizer_state.json"

print(f"\n📂 Loading tokenizer config from: {tokenizer_state_path}")

try:
    with open(tokenizer_state_path, 'r') as f:
        tokenizer_config = json.load(f)
    print("✅ Config loaded successfully")
    print(f"   Config: {tokenizer_config}")
except Exception as e:
    print(f"❌ Failed to load config: {e}")
    exit(1)

# Load tokenizer model
vocab_size = tokenizer_config.get('vocab_size', 32000)
model_path = tokenizer_dir / tokenizer_config['model_path'].replace('./', '')

print(f"\n📝 Loading SentencePiece model from: {model_path}")

try:
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(model_path))
    print("✅ Tokenizer loaded successfully")
    print(f"   Vocab size (from config): {vocab_size}")
    print(f"   Vocab size (from model): {tokenizer.GetPieceSize()}")
except Exception as e:
    print(f"❌ Failed to load tokenizer: {e}")
    exit(1)

# Test tokenization
print("\n🧪 Testing tokenization:")
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by biological neurons.",
    "Hello, world!",
    "This is a test sentence with multiple tokens."
]

for text in test_texts:
    # Encode
    token_ids = tokenizer.EncodeAsIds(text)
    
    # Decode
    decoded_text = tokenizer.DecodeIds(token_ids)
    
    # Get pieces (subwords)
    pieces = tokenizer.EncodeAsPieces(text)
    
    print(f"\n  Original: {text}")
    print(f"  Tokens:   {len(token_ids)} tokens")
    print(f"  IDs:      {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
    print(f"  Pieces:   {pieces[:10]}{'...' if len(pieces) > 10 else ''}")
    print(f"  Decoded:  {decoded_text}")
    print(f"  Match:    {'✅' if decoded_text.strip() == text.strip() else '❌'}")

print("\n" + "="*60)
print("🎉 Tokenizer test complete!")
print("="*60)
print("\nThe tokenizer is ready for use in training scripts:")
print("  - train_simple.py")
print("  - train_advanced.py")
print("  - train_slimpajama.py")
