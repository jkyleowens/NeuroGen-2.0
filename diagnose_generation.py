#!/usr/bin/env python3
"""
Diagnostic script to identify generation issues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "bin"))

import json
import sentencepiece as spm
import libneurogen

print("=" * 80)
print("🔍 NeuroGen Generation Diagnostics")
print("=" * 80 + "\n")

# Load tokenizer
print("📝 Loading tokenizer...")
tokenizer_dir = Path("tokenizer")
with open(tokenizer_dir / "tokenizer_state.json", 'r') as f:
    config = json.load(f)

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(str(tokenizer_dir / config['model_path'].replace('./', '')))
vocab_size = config['vocab_size']
print(f"✅ Tokenizer loaded: vocab_size={vocab_size}\n")

# Initialize model
print("🚀 Initializing model...")
model = libneurogen.NeuroGenModel(vocab_size=vocab_size, embedding_dim=512, gpu_device=0)
print("✅ Model initialized\n")

# Load latest checkpoint
checkpoint_dir = Path("checkpoints")
checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.bin"))
if checkpoints:
    checkpoint = checkpoints[-1]
    print(f"📂 Loading checkpoint: {checkpoint}")
    model.load_checkpoint(str(checkpoint))
    print("✅ Checkpoint loaded\n")
else:
    print("⚠️  No checkpoint found - using random weights\n")

# Test 1: Check if forward pass gives reasonable logits
print("=" * 80)
print("TEST 1: Forward Pass Distribution")
print("=" * 80)

test_text = "Hello world"
test_ids = tokenizer.EncodeAsIds(test_text)
print(f"Input: '{test_text}'")
print(f"Token IDs: {test_ids}\n")

# Generate just 1 token to see what happens
print("Generating 1 token (max_length=1)...")
try:
    output_ids = model.generate(test_ids, 1)
    generated_id = output_ids[-1] if len(output_ids) > len(test_ids) else None
    
    if generated_id:
        generated_text = tokenizer.DecodeIds([generated_id])
        print(f"Generated token ID: {generated_id}")
        print(f"Generated text: '{generated_text}'")
    else:
        print("❌ No token generated!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Generate multiple times with same input
print("=" * 80)
print("TEST 2: Sampling Diversity (5 generations)")
print("=" * 80)
print("Testing if sampling is deterministic or varied...\n")

generations = []
for i in range(5):
    try:
        output_ids = model.generate(test_ids, 10)
        gen_ids = output_ids[len(test_ids):]
        gen_text = tokenizer.DecodeIds(gen_ids)
        generations.append(gen_text)
        print(f"Generation {i+1}: {gen_text[:50]}...")
    except Exception as e:
        print(f"Generation {i+1}: ERROR - {e}")

# Check if all generations are identical
if len(set(generations)) == 1:
    print("\n❌ WARNING: All generations are IDENTICAL!")
    print("   This indicates deterministic sampling or stuck sampler")
elif len(set(generations)) < 3:
    print("\n⚠️  WARNING: Very low diversity (only {} unique outputs)".format(len(set(generations))))
else:
    print(f"\n✅ Good diversity: {len(set(generations))} unique outputs")

print()

# Test 3: Check for repetition in single generation
print("=" * 80)
print("TEST 3: Repetition Detection")
print("=" * 80)

print("Generating 50 tokens...")
try:
    output_ids = model.generate(test_ids, 50)
    gen_ids = output_ids[len(test_ids):]
    gen_text = tokenizer.DecodeIds(gen_ids)
    
    print(f"Generated: {gen_text}\n")
    
    # Check for repeated tokens
    unique_tokens = len(set(gen_ids))
    total_tokens = len(gen_ids)
    diversity_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    print(f"Unique tokens: {unique_tokens}/{total_tokens}")
    print(f"Diversity ratio: {diversity_ratio:.2%}")
    
    if diversity_ratio < 0.3:
        print("❌ CRITICAL: Very low diversity - model is stuck in loops!")
    elif diversity_ratio < 0.5:
        print("⚠️  WARNING: Low diversity - possible repetition issues")
    else:
        print("✅ Good diversity")
    
    # Check for exact repeating patterns
    token_counts = {}
    for tid in gen_ids:
        token_counts[tid] = token_counts.get(tid, 0) + 1
    
    most_common = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nMost common tokens:")
    for tid, count in most_common:
        token_text = tokenizer.DecodeIds([tid])
        print(f"  '{token_text}' (ID {tid}): {count} times ({count/total_tokens*100:.1f}%)")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("📊 DIAGNOSIS COMPLETE")
print("=" * 80)
print("\nPossible Issues:")
print("1. If all generations identical → Sampling bug in C++ code")
print("2. If low diversity → Temperature too low or argmax instead of sampling")
print("3. If repetitive tokens → Model not trained or weights not loaded")
print("4. If random gibberish → Checkpoint not loaded correctly")
print("\nNext steps: Check the specific issue identified above")
print("=" * 80 + "\n")
