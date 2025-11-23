#!/usr/bin/env python3
"""
CPU Memory Bottleneck Diagnostic Tool

Profiles memory allocations and CPU usage in the training loop to identify:
1. Large memory allocations causing CPU bottleneck
2. Excessive vector copies (std::vector<float> returns)
3. Python↔C++ boundary memory transfers
4. Memory fragmentation issues
"""

import sys
import time
import tracemalloc
from pathlib import Path
import psutil
import os
import gc

sys.path.insert(0, str(Path(__file__).parent / "bin"))

try:
    import libneurogen
    HAS_NEUROGEN = True
except ImportError:
    print("❌ libneurogen not found - cannot profile")
    sys.exit(1)

def format_bytes(bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def profile_memory_usage():
    """Profile memory usage during training"""
    
    print("="*80)
    print("CPU MEMORY BOTTLENECK DIAGNOSTIC")
    print("="*80)
    
    # Get process info
    process = psutil.Process(os.getpid())
    
    # Start memory tracking
    tracemalloc.start()
    
    print("\n1️⃣  Initial Memory State:")
    mem_info = process.memory_info()
    print(f"   RSS (Resident Set Size):  {format_bytes(mem_info.rss)}")
    print(f"   VMS (Virtual Memory):     {format_bytes(mem_info.vms)}")
    
    # Initialize model
    print("\n2️⃣  Initializing NeuroGen model...")
    snapshot_before = tracemalloc.take_snapshot()
    
    model = libneurogen.NeuroGenModel(
        vocab_size=32000,
        embedding_dim=1536,
        gpu_device=0
    )
    
    snapshot_after = tracemalloc.take_snapshot()
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    
    mem_info = process.memory_info()
    print(f"   Memory after init:")
    print(f"   RSS:  {format_bytes(mem_info.rss)}")
    print(f"   VMS:  {format_bytes(mem_info.vms)}")
    
    print(f"\n   Top 5 memory allocations during init:")
    for stat in top_stats[:5]:
        print(f"   {stat}")
    
    # Test different sequence lengths
    test_cases = [
        (10, "Small sequence"),
        (100, "Medium sequence"),
        (500, "Large sequence"),
        (1000, "Very large sequence")
    ]
    
    for seq_len, description in test_cases:
        print(f"\n{'='*80}")
        print(f"3️⃣  Testing: {description} ({seq_len} tokens)")
        print(f"{'='*80}")
        
        # Create test data
        input_ids = list(range(1, seq_len + 1))
        target_ids = list(range(2, seq_len + 2))
        
        # Get baseline memory
        gc.collect()
        mem_before = process.memory_info()
        snapshot_before = tracemalloc.take_snapshot()
        cpu_percent_before = process.cpu_percent(interval=0.1)
        
        # Run training step
        start = time.time()
        loss, accuracy = model.train_step(input_ids, target_ids)
        elapsed = time.time() - start
        
        # Get memory after
        snapshot_after = tracemalloc.take_snapshot()
        mem_after = process.memory_info()
        cpu_percent_after = process.cpu_percent(interval=0.1)
        
        # Calculate deltas
        rss_delta = mem_after.rss - mem_before.rss
        vms_delta = mem_after.vms - mem_before.vms
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Time:              {elapsed:.3f}s ({elapsed*1000/seq_len:.2f}ms per token)")
        print(f"   Throughput:        {seq_len/elapsed:.1f} tokens/sec")
        print(f"   Loss:              {loss:.4f}")
        print(f"   Accuracy:          {accuracy*100:.1f}%")
        print(f"   CPU Usage:         {cpu_percent_after:.1f}%")
        
        print(f"\n💾 Memory Usage:")
        print(f"   RSS Before:        {format_bytes(mem_before.rss)}")
        print(f"   RSS After:         {format_bytes(mem_after.rss)}")
        print(f"   RSS Delta:         {format_bytes(abs(rss_delta))} {'↑' if rss_delta > 0 else '↓'}")
        print(f"   VMS Delta:         {format_bytes(abs(vms_delta))} {'↑' if vms_delta > 0 else '↓'}")
        
        # Analyze memory allocations
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        
        if top_stats:
            print(f"\n🔍 Top Memory Allocations:")
            for i, stat in enumerate(top_stats[:10], 1):
                size_diff = stat.size_diff
                count_diff = stat.count_diff
                print(f"   {i}. {format_bytes(size_diff)} ({count_diff:+d} blocks)")
                print(f"      {stat.traceback.format()[0]}")
        
        # Identify bottlenecks
        print(f"\n⚠️  Bottleneck Analysis:")
        
        # Check if memory grew significantly
        if abs(rss_delta) > 50 * 1024 * 1024:  # 50MB
            print(f"   🔥 MEMORY LEAK: RSS grew by {format_bytes(abs(rss_delta))}")
            print(f"      This indicates memory is not being freed properly")
        
        # Check for excessive allocations
        total_allocs = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        if total_allocs > 100 * 1024 * 1024:  # 100MB
            print(f"   ⚠️  HIGH ALLOCATION: {format_bytes(total_allocs)} allocated")
            print(f"      Likely cause: Large std::vector copies on every token")
        
        # Check CPU usage
        if cpu_percent_after > 80:
            print(f"   🔥 CPU BOTTLENECK: {cpu_percent_after:.1f}% usage")
            print(f"      Expected: <20% for GPU-accelerated training")
        
        # Check throughput
        tokens_per_sec = seq_len / elapsed
        if tokens_per_sec < 100:
            print(f"   🐌 SLOW THROUGHPUT: {tokens_per_sec:.1f} tokens/sec")
            print(f"      Expected: >1000 tokens/sec for GPU processing")
            print(f"      Root cause likely: GPU→CPU sync on every token")
    
    print(f"\n{'='*80}")
    print("4️⃣  DIAGNOSIS SUMMARY")
    print(f"{'='*80}")
    
    print("\n🔍 Known Issues in train_step():")
    print("   1. Token-by-token loop (lines 113-141 in neurogen_bindings.cpp)")
    print("      - Creates std::vector<float> for EVERY token")
    print("      - embedding_->encodeById() returns vector by VALUE (copy)")
    print("      - brain_->getBrocaOutput() returns vector by VALUE (copy)")
    print("      - With embedding_dim=1536: 1536 floats × 4 bytes = 6KB per token")
    print("      - For 500 tokens: 500 × 6KB × 2 = ~6MB copied!")
    
    print("\n   2. GPU→CPU synchronization on every token")
    print("      - gpu_decoder_->decodeAndSample() forces GPU sync")
    print("      - CPU waits for GPU on EVERY token")
    print("      - This is the PRIMARY bottleneck (40-50% of time)")
    
    print("\n   3. No memory reuse")
    print("      - New vectors allocated/deallocated on every token")
    print("      - Causes memory fragmentation")
    print("      - Increases CPU memory pressure")
    
    print("\n💡 SOLUTIONS:")
    print("   A. Use references/pointers instead of returning by value:")
    print("      const float* embedded = embedding_->encodeByIdPtr(input_ids[i]);")
    print("      const float* brain_output = brain_->getBrocaOutputPtr();")
    
    print("\n   B. Batch processing (process all tokens at once):")
    print("      std::vector<int> predicted = model.train_step_batch(input_ids, target_ids);")
    print("      This eliminates N-1 GPU syncs and reduces memory copies")
    
    print("\n   C. Pre-allocate buffers:")
    print("      Keep reusable buffers for embedding/output vectors")
    print("      Reuse across tokens instead of allocate/free")
    
    print("\n📝 See CPU_BOTTLENECK_FIX.h for complete implementation")
    print("="*80)
    
    tracemalloc.stop()

if __name__ == "__main__":
    profile_memory_usage()
