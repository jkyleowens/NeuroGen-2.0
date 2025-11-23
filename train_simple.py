#!/usr/bin/env python3
"""
NeuroGen 2.0 - Simple Training Test Script

Quick training test without large dataset downloads.
Uses synthetic or small built-in datasets for rapid iteration.
"""

import sys
import time
import random
import json
from pathlib import Path
from typing import List, Tuple

# Add bin to path
sys.path.insert(0, str(Path(__file__).parent / "bin"))

try:
    import sentencepiece as spm
    import libneurogen
    HAS_DEPS = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install: pip install sentencepiece")
    HAS_DEPS = False
    sys.exit(1)

# Sample training texts (no download needed)
TRAINING_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by biological neurons in the brain.",
    "Deep learning has revolutionized computer vision and natural language processing.",
    "The Transformer architecture uses self-attention mechanisms.",
    "GPT models are trained on massive amounts of text data.",
    "Recurrent neural networks process sequential data.",
    "Convolutional neural networks excel at image recognition.",
    "Reinforcement learning agents learn through trial and error.",
    "Natural language understanding requires contextual reasoning.",
    "The brain contains approximately 86 billion neurons.",
    "Synaptic plasticity enables learning and memory formation.",
    "Dopamine plays a crucial role in reward-based learning.",
    "The cerebral cortex is responsible for higher-order thinking.",
    "Neural oscillations coordinate activity across brain regions.",
    "Spiking neural networks model biological neurons more closely.",
    "Sparse representations are computationally efficient.",
    "Working memory maintains temporary information in the prefrontal cortex.",
    "The hippocampus is essential for forming new memories.",
    "Attention mechanisms allow models to focus on relevant information.",
]

def train_simple():
    """Simple training loop with synthetic data"""
    print("=" * 80)
    print("🧠 NeuroGen 2.0 - Simple Training Test")
    print("=" * 80 + "\n")
    
    # Initialize
    print("🚀 Initializing model...")
    
    # Load tokenizer configuration
    tokenizer_dir = Path(__file__).parent / "tokenizer"
    tokenizer_state_path = tokenizer_dir / "tokenizer_state.json"
    
    with open(tokenizer_state_path, 'r') as f:
        tokenizer_config = json.load(f)
    
    vocab_size = tokenizer_config.get('vocab_size', 32000)
    model_path = tokenizer_dir / tokenizer_config['model_path'].replace('./', '')
    
    model = libneurogen.NeuroGenModel(
        vocab_size=vocab_size,
        embedding_dim=2048,  # Quadrupled from 512 for maximum capacity
        gpu_device=0
    )
    print("✅ Model initialized\n")
    
    print("📝 Loading SentencePiece tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(model_path))
    print(f"✅ Tokenizer loaded from {model_path}")
    print(f"   Vocab size: {vocab_size}\n")
    
    # Training parameters
    num_epochs = 10
    steps_per_epoch = len(TRAINING_TEXTS)
    total_steps = num_epochs * steps_per_epoch
    
    print(f"🎯 Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Samples per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    print(f"   Dataset: {len(TRAINING_TEXTS)} synthetic sentences\n")
    
    print("🏃 Starting training...\n")
    
    start_time = time.time()
    total_loss = 0.0
    total_accuracy = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_start = time.time()
        
        # Shuffle data each epoch
        texts = TRAINING_TEXTS.copy()
        random.shuffle(texts)
        
        for step, text in enumerate(texts):
            # Tokenize with SentencePiece
            tokens = tokenizer.EncodeAsIds(text)
            # Truncate if needed
            if len(tokens) > 128:
                tokens = tokens[:128]
            if len(tokens) < 2:
                continue
            
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
            
            # Training step
            step_start = time.time()
            loss, accuracy = model.train_step(input_ids, target_ids)
            step_time = time.time() - step_start
            
            epoch_loss += loss
            epoch_acc += accuracy
            total_loss += loss
            total_accuracy += accuracy
            
            # Print progress
            global_step = epoch * steps_per_epoch + step + 1
            if (step + 1) % 5 == 0 or step == 0:
                tokens_per_sec = len(input_ids) / step_time if step_time > 0 else 0
                print(f"Step {global_step:3d}/{total_steps} | "
                      f"Epoch {epoch+1}/{num_epochs} | "
                      f"Loss: {loss:.4f} | "
                      f"Acc: {accuracy*100:5.1f}% | "
                      f"Speed: {tokens_per_sec:5.1f} tok/s | "
                      f"Sample: \"{text[:40]}...\"")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / steps_per_epoch
        avg_epoch_acc = epoch_acc / steps_per_epoch
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Avg Loss: {avg_epoch_loss:.4f}")
        print(f"   Avg Accuracy: {avg_epoch_acc*100:.1f}%")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   Samples/sec: {steps_per_epoch/epoch_time:.2f}\n")
    
    # Final summary
    total_time = time.time() - start_time
    avg_loss = total_loss / total_steps
    avg_acc = total_accuracy / total_steps
    
    print("=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"📊 Total Steps: {total_steps}")
    print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"📉 Avg Loss: {avg_loss:.4f}")
    print(f"📈 Avg Accuracy: {avg_acc*100:.1f}%")
    print(f"🚀 Throughput: {total_steps/total_time:.2f} samples/sec")
    print("=" * 80 + "\n")
    
    # Test generation
    print("🧪 Testing text generation...\n")
    test_prompts = [
        "The brain",
        "Neural networks",
        "Machine learning"
    ]
    
    for prompt in test_prompts:
        prompt_ids = tokenizer.EncodeAsIds(prompt)
        generated_ids = model.generate(prompt_ids, max_length=20)
        generated_text = tokenizer.DecodeIds(generated_ids)
        print(f"Prompt: \"{prompt}\"")
        print(f"Generated: \"{generated_text}\"\n")
    
    print("✅ All tests complete!\n")

if __name__ == "__main__":
    if not HAS_DEPS:
        print("❌ Cannot run without dependencies")
        sys.exit(1)
    
    try:
        train_simple()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

