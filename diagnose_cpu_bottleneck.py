#!/usr/bin/env python3
"""
CPU Bottleneck Diagnostic Tool

Identifies performance issues in train_slimpajama.py by profiling:
1. GPU-CPU synchronization overhead
2. Token-by-token processing inefficiency
3. Python-C++ boundary crossing costs
4. Memory transfer bottlenecks
"""

import sys
import time
import cProfile
import pstats
from pathlib import Path
import psutil
import os

sys.path.insert(0, str(Path(__file__).parent / "bin"))

try:
    import libneurogen
    HAS_NEUROGEN = True
except ImportError:
    print("❌ libneurogen not found - cannot profile")
    sys.exit(1)

def profile_train_step():
    """Profile a single training step to identify bottlenecks"""
    
    print("="*80)
    print("CPU BOTTLENECK DIAGNOSTIC")
    print("="*80)
    
    # Initialize model
    print("\n1️⃣  Initializing model...")
    model = libneurogen.NeuroGenModel(
        vocab_size=32000,
        embedding_dim=1536,
        gpu_device=0
    )
    
    # Create test data (simulate typical sequence)
    print("\n2️⃣  Creating test data...")
    sequence_lengths = [10, 50, 100, 500]
    
    for seq_len in sequence_lengths:
        input_ids = list(range(1, seq_len + 1))  # Dummy tokens
        target_ids = list(range(2, seq_len + 2))
        
        print(f"\n{'='*80}")
        print(f"Testing sequence length: {seq_len} tokens")
        print(f"{'='*80}")
        
        # Get CPU usage before
        process = psutil.Process(os.getpid())
        cpu_before = process.cpu_percent(interval=0.1)
        
        # Time the training step
        start = time.time()
        loss, accuracy = model.train_step(input_ids, target_ids)
        elapsed = time.time() - start
        
        # Get CPU usage after
        cpu_after = process.cpu_percent(interval=0.1)
        
        # Calculate metrics
        tokens_per_sec = seq_len / elapsed if elapsed > 0 else 0
        ms_per_token = (elapsed * 1000) / seq_len
        
        print(f"\n📊 Results:")
        print(f"   Total Time:        {elapsed:.3f}s")
        print(f"   Time per Token:    {ms_per_token:.2f}ms")
        print(f"   Throughput:        {tokens_per_sec:.1f} tokens/sec")
        print(f"   Loss:              {loss:.4f}")
        print(f"   Accuracy:          {accuracy*100:.1f}%")
        print(f"   CPU Usage:         {cpu_after:.1f}%")
        
        # Identify bottlenecks
        print(f"\n🔍 Bottleneck Analysis:")
        if ms_per_token > 50:
            print(f"   ⚠️  SEVERE: {ms_per_token:.1f}ms/token is extremely slow")
            print(f"       Expected: <1ms/token for GPU processing")
            print(f"       Issue: Likely GPU→CPU synchronization on every token")
        elif ms_per_token > 10:
            print(f"   ⚠️  HIGH: {ms_per_token:.1f}ms/token is slow")
            print(f"       Expected: <1ms/token for GPU processing")
            print(f"       Issue: Token-by-token processing overhead")
        elif ms_per_token > 1:
            print(f"   ⚙️  MODERATE: {ms_per_token:.1f}ms/token")
            print(f"       Could be improved with batching")
        else:
            print(f"   ✅ GOOD: {ms_per_token:.2f}ms/token is acceptable")
        
        if cpu_after > 80:
            print(f"   🔥 CPU BOTTLENECK: {cpu_after:.1f}% usage")
            print(f"       GPU should be doing the work, not CPU!")
        elif cpu_after > 50:
            print(f"   ⚠️  High CPU usage: {cpu_after:.1f}%")
        else:
            print(f"   ✅ Normal CPU usage: {cpu_after:.1f}%")

def profile_detailed():
    """Run cProfile to identify specific function bottlenecks"""
    
    print("\n" + "="*80)
    print("DETAILED FUNCTION PROFILING")
    print("="*80)
    
    # Initialize model
    model = libneurogen.NeuroGenModel(
        vocab_size=32000,
        embedding_dim=1536,
        gpu_device=0
    )
    
    # Create test data
    input_ids = list(range(1, 101))  # 100 tokens
    target_ids = list(range(2, 102))
    
    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    loss, accuracy = model.train_step(input_ids, target_ids)
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    print("\nTop 20 functions by cumulative time:")
    stats.print_stats(20)

def main():
    print("\n🔬 Starting CPU Bottleneck Diagnostic...\n")
    
    # Run basic profiling
    profile_train_step()
    
    # Detailed profiling
    try:
        profile_detailed()
    except Exception as e:
        print(f"\n⚠️  Detailed profiling failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF IDENTIFIED BOTTLENECKS")
    print("="*80)
    print("""
The main CPU bottlenecks in train_slimpajama.py are:

1. 🔴 GPU→CPU SYNCHRONIZATION (CRITICAL)
   - decodeAndSample() forces CPU to wait for GPU result
   - Happens EVERY token (3000+ times per chunk)
   - Causes massive pipeline stall
   
2. 🟠 TOKEN-BY-TOKEN PROCESSING (HIGH)
   - No batching - processes one token at a time
   - Sequential loop with 3000+ iterations
   - Python overhead on each iteration
   
3. 🟡 MEMORY COPIES (MODERATE)
   - std::vector copies between Python and C++
   - Embedding lookups copy data
   - Brain output copies

RECOMMENDED FIXES:

✅ 1. BATCH PROCESSING (High Impact)
   - Process multiple tokens in parallel
   - Reduce synchronization points
   - Use GPU streams for overlap

✅ 2. ASYNC GPU OPERATIONS (High Impact)
   - Use cudaStreamCreate() for async execution
   - Don't wait for GPU results until needed
   - Pipeline CPU and GPU work

✅ 3. REDUCE SYNCHRONIZATION (Critical)
   - Only sync at end of sequence, not per token
   - Use cudaEventRecord() to track completion
   - Buffer predictions on GPU

✅ 4. OPTIMIZE PYTHON BINDINGS (Moderate)
   - Pass numpy arrays instead of lists
   - Use py::buffer_info for zero-copy
   - Batch multiple sequences together
    """)
    print("="*80)

if __name__ == "__main__":
    main()
