#!/usr/bin/env python3
"""
Quick verification that train_slimpajama.py progress updates work correctly.
Tests the fix for the "hangup after dataset load" issue.
"""

import subprocess
import time
import sys
from pathlib import Path

def test_progress_updates():
    """Test that train_slimpajama.py shows progress updates"""
    
    print("="*80)
    print("Testing train_slimpajama.py progress updates")
    print("="*80)
    print()
    print("This test verifies that the script:")
    print("  1. Loads the dataset")
    print("  2. Shows progress updates during accumulation")
    print("  3. Completes training steps with visible output")
    print()
    print("Expected behavior: Progress updates every 2-3 seconds")
    print("Test duration: ~1-2 minutes")
    print()
    print("Starting test in 3 seconds...")
    time.sleep(3)
    
    # Run train_slimpajama.py in test mode
    script_path = Path(__file__).parent / "train_slimpajama.py"
    
    try:
        print("\n" + "="*80)
        print("Running: python3 train_slimpajama.py --test")
        print("="*80 + "\n")
        
        # Run the script and capture output in real-time
        process = subprocess.Popen(
            ["python3", str(script_path), "--test"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        output_lines = []
        progress_updates = 0
        last_line_time = time.time()
        silence_duration = 0
        max_silence = 0
        
        # Track output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='')
                output_lines.append(line)
                
                # Check for progress update indicators
                if "Accumulating data:" in line or "Step" in line or "complete" in line:
                    current_time = time.time()
                    silence_duration = current_time - last_line_time
                    max_silence = max(max_silence, silence_duration)
                    last_line_time = current_time
                    
                    if "Accumulating data:" in line:
                        progress_updates += 1
        
        process.wait()
        
        # Analyze results
        print("\n" + "="*80)
        print("Test Results")
        print("="*80)
        
        success = True
        
        # Check 1: Progress updates appeared
        if progress_updates > 0:
            print(f"✅ Progress updates detected: {progress_updates} updates")
        else:
            print(f"❌ No progress updates detected (expected several)")
            success = False
        
        # Check 2: Max silence time reasonable
        if max_silence < 30:
            print(f"✅ Max silence between outputs: {max_silence:.1f}s (acceptable)")
        else:
            print(f"⚠️  Max silence between outputs: {max_silence:.1f}s (longer than expected)")
            success = False
        
        # Check 3: Process completed
        if process.returncode == 0:
            print(f"✅ Process completed successfully")
        else:
            print(f"❌ Process failed with code {process.returncode}")
            success = False
        
        # Check 4: Look for key output
        output_text = ''.join(output_lines)
        
        if "Dataset loaded successfully" in output_text:
            print(f"✅ Dataset loaded successfully")
        else:
            print(f"⚠️  Dataset load message not found")
        
        if "Training on chunk" in output_text:
            print(f"✅ Training steps executed")
        else:
            print(f"⚠️  No training steps found in output")
        
        print()
        if success:
            print("🎉 Test PASSED: Progress updates working correctly!")
            print("   The hangup issue has been fixed.")
        else:
            print("⚠️  Test completed with warnings")
            print("   Check output above for details")
        
        return success
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        if process:
            process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print()
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "SlimPajama Progress Test" + " "*34 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    try:
        success = test_progress_updates()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")
        sys.exit(1)
