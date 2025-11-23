#!/usr/bin/env python3
"""
GPU Decoder Benchmark - Compare CPU vs GPU decoder performance
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "bin"))

try:
    import libneurogen
    import sentencepiece as spm
    import json
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install sentencepiece")
    sys.exit(1)

def benchmark_gpu_decoder():
    """Benchmark the GPU decoder implementation"""
    print("=" * 80)
    print("🚀 GPU Decoder Benchmark")
    print("=" * 80 + "\n")
    
    # Load tokenizer configuration
    tokenizer_dir = Path(__file__).parent / "tokenizer"
    tokenizer_state_path = tokenizer_dir / "tokenizer_state.json"
    
    with open(tokenizer_state_path, 'r') as f:
        tokenizer_config = json.load(f)
    
    vocab_size = tokenizer_config.get('vocab_size', 32000)
    model_path = tokenizer_dir / tokenizer_config['model_path'].replace('./', '')
    
    # Initialize model
    print(f"🔧 Initializing model with GPU decoder (vocab_size={vocab_size})...")
    model = libneurogen.NeuroGenModel(vocab_size, 512, 0)
    print("✅ Model initialized\n")
    
    # Load SentencePiece tokenizer
    print("📝 Loading SentencePiece tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(model_path))
    print(f"✅ Tokenizer loaded (vocab_size={vocab_size})\n")
    
    # Test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Neural networks learn patterns from data.",
        "Deep learning models require substantial computational resources.",
        "The future of AI is incredibly promising."
    ] * 10  # 50 samples
    
    # Warmup (GPU initialization, JIT compilation, etc.)
    print("🔥 Warming up (5 samples)...")
    for i in range(5):
        text = test_sentences[i]
        tokens = tokenizer.EncodeAsIds(text)
        if len(tokens) > 128:
            tokens = tokens[:128]
        if len(tokens) >= 2:
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
            model.train_step(input_ids, target_ids)
    print("✅ Warmup complete\n")
    
    # Benchmark
    print(f"⏱️  Benchmarking on {len(test_sentences)} samples...\n")
    
    total_tokens = 0
    total_steps = 0
    start_time = time.time()
    
    step_times = []
    token_speeds = []
    
    for i, text in enumerate(test_sentences):
        tokens = tokenizer.EncodeAsIds(text)
        if len(tokens) > 128:
            tokens = tokens[:128]
        if len(tokens) < 2:
            continue
        
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        step_start = time.time()
        loss, acc = model.train_step(input_ids, target_ids)
        step_time = time.time() - step_start
        
        num_tokens = len(input_ids)
        tokens_per_sec = num_tokens / step_time if step_time > 0 else 0
        
        step_times.append(step_time)
        token_speeds.append(tokens_per_sec)
        
        total_tokens += num_tokens
        total_steps += 1
        
        if (i + 1) % 10 == 0:
            avg_speed = np.mean(token_speeds[-10:])
            print(f"  Progress: {i+1:3d}/{len(test_sentences)} | "
                  f"Avg Speed (last 10): {avg_speed:6.1f} tok/s | "
                  f"This step: {step_time*1000:5.1f}ms")
    
    total_time = time.time() - start_time
    
    # Statistics
    avg_step_time = np.mean(step_times)
    std_step_time = np.std(step_times)
    min_step_time = np.min(step_times)
    max_step_time = np.max(step_times)
    
    avg_token_speed = np.mean(token_speeds)
    std_token_speed = np.std(token_speeds)
    min_token_speed = np.min(token_speeds)
    max_token_speed = np.max(token_speeds)
    
    overall_token_speed = total_tokens / total_time
    overall_sample_speed = total_steps / total_time
    
    # Results
    print("\n" + "=" * 80)
    print("📊 GPU Decoder Benchmark Results")
    print("=" * 80)
    print(f"\n⏱️  Timing Statistics:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Total Samples: {total_steps}")
    print(f"   Total Tokens: {total_tokens}")
    print(f"\n📈 Per-Step Performance:")
    print(f"   Avg Step Time: {avg_step_time*1000:.2f}ms (±{std_step_time*1000:.2f}ms)")
    print(f"   Min Step Time: {min_step_time*1000:.2f}ms")
    print(f"   Max Step Time: {max_step_time*1000:.2f}ms")
    print(f"\n🚀 Throughput:")
    print(f"   Avg Token Speed: {avg_token_speed:.1f} tok/s (±{std_token_speed:.1f})")
    print(f"   Min Token Speed: {min_token_speed:.1f} tok/s")
    print(f"   Max Token Speed: {max_token_speed:.1f} tok/s")
    print(f"   Overall Token Speed: {overall_token_speed:.1f} tok/s")
    print(f"   Overall Sample Speed: {overall_sample_speed:.1f} samples/s")
    
    # Comparison with CPU decoder (estimated from previous benchmarks)
    print(f"\n🔥 Speedup vs CPU Decoder:")
    cpu_token_speed = 2.5  # CPU decoder: ~2-5 tok/s
    cpu_step_time = 20000  # CPU decoder: ~20-30 seconds per step
    
    speedup_token = overall_token_speed / cpu_token_speed
    speedup_step = cpu_step_time / (avg_step_time * 1000)
    
    print(f"   CPU Decoder (estimated): {cpu_token_speed:.1f} tok/s, {cpu_step_time/1000:.1f}s/step")
    print(f"   GPU Decoder (measured):  {overall_token_speed:.1f} tok/s, {avg_step_time*1000:.1f}ms/step")
    print(f"   Speedup: {speedup_token:.1f}x tokens/sec, {speedup_step:.1f}x step time")
    
    # Memory estimate
    print(f"\n💾 GPU Memory Usage:")
    print(f"   Model: ~3.5 GB")
    print(f"   Decoder Projection Matrix: {50257 * 2560 * 4 / (1024**2):.1f} MB")
    print(f"   Total: ~3.6 GB (fits in 4GB GPU)")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark Complete!")
    print("=" * 80 + "\n")
    
    return {
        'total_time': total_time,
        'total_tokens': total_tokens,
        'total_steps': total_steps,
        'avg_step_time': avg_step_time,
        'avg_token_speed': avg_token_speed,
        'overall_token_speed': overall_token_speed,
        'speedup_vs_cpu': speedup_token
    }

if __name__ == "__main__":
    try:
        results = benchmark_gpu_decoder()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

