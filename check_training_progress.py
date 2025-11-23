#!/usr/bin/env python3
"""
Quick script to check training progress and visualization status
"""

import json
from pathlib import Path

viz_dir = Path("training_viz")
checkpoints_dir = Path("checkpoints")

print("="*80)
print("🔍 TRAINING PROGRESS CHECK")
print("="*80)

# Check if visualization directory exists
if not viz_dir.exists():
    print("❌ Visualization directory not found: training_viz/")
    print("   Run training first to generate visualizations")
else:
    print(f"✅ Visualization directory exists: {viz_dir}")
    
    # Check metrics file
    metrics_file = viz_dir / "metrics_history.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        num_steps = len(metrics.get('steps', []))
        print(f"\n📊 METRICS:")
        print(f"   Total steps recorded: {num_steps}")
        
        if num_steps > 0:
            losses = metrics.get('loss', [])
            accs = metrics.get('accuracy', [])
            
            print(f"   Loss:     min={min(losses):.4f}, max={max(losses):.4f}, latest={losses[-1]:.4f}")
            print(f"   Accuracy: min={min(accs)*100:.1f}%, max={max(accs)*100:.1f}%, latest={accs[-1]*100:.1f}%")
            print(f"   Throughput: {metrics.get('throughput', [0])[-1]:.1f} tokens/sec")
    else:
        print("\n⚠️  No metrics file found yet")
    
    # Check text samples
    samples_file = viz_dir / "text_samples.json"
    if samples_file.exists():
        with open(samples_file) as f:
            samples = json.load(f)
        print(f"\n📝 TEXT SAMPLES:")
        print(f"   Total samples captured: {len(samples)}")
        if samples:
            print(f"   Latest sample (step {samples[-1]['step']}):")
            print(f"      Input:  {samples[-1]['input'][:60]}...")
            print(f"      Output: {samples[-1]['output'][:60]}...")
    else:
        print("\n⚠️  No text samples file found yet")
    
    # Check PNG files
    png_files = list(viz_dir.glob("training_step_*.png"))
    print(f"\n🖼️  VISUALIZATION IMAGES:")
    print(f"   PNG files generated: {len(png_files)}")
    if png_files:
        latest_png = max(png_files, key=lambda p: p.stat().st_mtime)
        print(f"   Latest: {latest_png.name}")
        print(f"   Size: {latest_png.stat().st_size / (1024*1024):.2f} MB")

# Check checkpoints
if checkpoints_dir.exists():
    checkpoint_files = list(checkpoints_dir.glob("checkpoint_step_*.bin"))
    print(f"\n💾 CHECKPOINTS:")
    print(f"   Checkpoint files: {len(checkpoint_files)}")
    if checkpoint_files:
        for ckpt in sorted(checkpoint_files)[-3:]:  # Show last 3
            size = ckpt.stat().st_size / (1024*1024*1024)
            print(f"   - {ckpt.name}: {size:.2f} GB")

print("\n" + "="*80)
print("💡 TIPS:")
print("   - Visualizations generate every 50 steps (default)")
print("   - Use --viz-interval 5 for more frequent charts")
print("   - Use --test mode to see visualization quickly (5 steps)")
print("   - Check training_viz/training_latest.png for latest dashboard")
print("="*80)
