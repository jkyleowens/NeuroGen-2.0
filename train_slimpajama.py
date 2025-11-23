#!/usr/bin/env python3
"""
NeuroGen Modular Brain - SlimPajama Training Script
UPDATED: GTX 1650 Optimized with Lightweight Checkpointing

This version uses an adaptive checkpoint strategy that:
- Saves training state frequently (no OOM risk)
- Attempts full model saves rarely (every 100 steps)
- Disables full saves if OOM persists
- Saves final model at end of training

Requirements:
    pip install datasets transformers torch numpy tqdm zstandard matplotlib
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
import signal
import random
import datetime
import subprocess
import gc

try:
    from datasets import load_dataset
    import sentencepiece as spm
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install datasets sentencepiece torch numpy tqdm zstandard matplotlib")
    sys.exit(1)

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


@dataclass
class TrainingConfig:
    """Training configuration"""
    vocab_size: int = 32000
    embedding_dim: int = 768
    gpu_device: int = 0
    temperature: float = 1.0
    dataset_name: str = "cerebras/SlimPajama-627B"
    dataset_split: str = "train"
    streaming: bool = True
    max_seq_length: int = 256
    batch_size: int = 1
    learning_rate: float = 0.001
    num_epochs: int = 1
    tokens_per_chunk: int = 4096
    max_chunks: Optional[int] = None
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    verbose_logging: bool = True
    viz_dir: str = "training_viz"
    viz_interval_chunks: int = 5
    tokenizer_dir: str = "tokenizer"
    tokenizer_model: str = "nlp_agent_tokenizer.model"


# ==============================================================================
# LIGHTWEIGHT CHECKPOINT MANAGER FOR GTX 1650
# ==============================================================================

class TrainingStateCheckpoint:
    """Saves only training metrics (no model weights) - NO OOM RISK"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.state_file = self.checkpoint_dir / "training_state.json"
    
    def save(self, global_step: int, total_loss: float, 
             tokens_processed: int, start_time: float) -> bool:
        """Save training metrics only (< 1KB, instant)"""
        try:
            state = {
                'global_step': global_step,
                'total_loss': total_loss,
                'tokens_processed': tokens_processed,
                'start_time': start_time,
                'timestamp': time.time(),
                'avg_loss': total_loss / max(global_step, 1)
            }
            
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            temp_file.replace(self.state_file)
            
            return True
        except Exception as e:
            print(f"   ⚠️  State save failed: {e}")
            return False
    
    def load(self) -> Optional[Dict]:
        """Load training state"""
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None


class GTX1650CheckpointManager:
    """
    Optimized for GTX 1650 (4GB VRAM) - Adaptive checkpoint strategy.
    
    Strategy:
    - Always save training state (lightweight, no OOM)
    - Attempt full model save every 100 steps initially
    - After OOM failures, increase interval or disable
    - Save final model at end
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training state manager (lightweight)
        self.state_mgr = TrainingStateCheckpoint(checkpoint_dir)
        
        # Full checkpoint settings
        self.full_checkpoint_path = self.checkpoint_dir / "full_model.ckpt"
        self.oom_failures = 0
        self.full_saves_enabled = True
        self.last_full_save_step = 0
        
        print("\n" + "="*70)
        print("🎮 GTX 1650 Checkpoint Manager Initialized")
        print("="*70)
        print("Strategy:")
        print("  • Training state: Saved every 5 steps (< 1KB, instant)")
        print("  • Full model: Attempted every 100 steps (1-2 GB, 10-30s)")
        print("  • Adapts if OOM occurs (increases interval or disables)")
        print("  • Final model: Saved at end of training")
        print("="*70 + "\n")
    
    def _get_full_save_interval(self) -> int:
        """Adaptive interval based on OOM history"""
        if self.oom_failures == 0:
            return 100  # Initial: try every 100 steps
        elif self.oom_failures == 1:
            return 200  # After 1st OOM: double interval
        elif self.oom_failures == 2:
            return 500  # After 2nd OOM: save very rarely
        else:
            return 999999  # After 3+ OOMs: effectively disabled
    
    def save(self, model, global_step: int, total_loss: float,
             tokens_processed: int, start_time: float) -> bool:
        """
        Adaptive save strategy:
        1. Always save training state (no OOM risk)
        2. Attempt full model save based on adaptive interval
        """
        
        # STEP 1: Always save training state (lightweight, no OOM)
        self.state_mgr.save(global_step, total_loss, tokens_processed, start_time)
        
        # STEP 2: Decide if we should attempt full model save
        full_interval = self._get_full_save_interval()
        should_save_full = (global_step % full_interval == 0) and self.full_saves_enabled
        
        if not should_save_full:
            return True
        
        # Disable full saves after 3 failures
        if self.oom_failures >= 3 and self.full_saves_enabled:
            print("\n" + "!"*70)
            print("⚠️  FULL CHECKPOINT SAVING HAS BEEN DISABLED")
            print("!"*70)
            print("Reason: Multiple OOM failures during checkpoint save")
            print("Impact:")
            print("  • Training state continues to be saved (progress tracked)")
            print("  • Model will be saved ONLY at the end of training")
            print("  • This prevents OOM kills during training")
            print("!"*70 + "\n")
            self.full_saves_enabled = False
            return True
        
        # STEP 3: Attempt full model save with maximum memory management
        if not model:
            return True
        
        print(f"\n{'='*70}")
        print(f"💾 FULL MODEL CHECKPOINT ATTEMPT (Step {global_step})")
        print(f"{'='*70}")
        print(f"⚠️  This will use significant memory and may take 10-60 seconds")
        print(f"⚠️  If OOM occurs, full saves will be reduced/disabled")
        
        # Delete old checkpoint
        if self.full_checkpoint_path.exists():
            try:
                old_size_gb = self.full_checkpoint_path.stat().st_size / (1024**3)
                self.full_checkpoint_path.unlink()
                print(f"   🗑️  Deleted old checkpoint ({old_size_gb:.2f} GB)")
            except Exception as e:
                print(f"   ⚠️  Could not delete old: {e}")
        
        # Maximum memory cleanup
        print("   🧹 Maximum memory cleanup...")
        for i in range(5):
            collected = gc.collect()
            if i == 0:
                print(f"      Pass 1: {collected} objects collected")
        time.sleep(1.0)  # Give system time to reclaim memory
        
        # Attempt save
        try:
            save_start = time.time()
            print("   💾 Saving...")
            
            model.save_checkpoint(str(self.full_checkpoint_path))
            
            save_duration = time.time() - save_start
            
            # Verify save succeeded
            if self.full_checkpoint_path.exists():
                size_gb = self.full_checkpoint_path.stat().st_size / (1024**3)
                print(f"   ✅ CHECKPOINT SAVED SUCCESSFULLY!")
                print(f"   📊 Size: {size_gb:.2f} GB")
                print(f"   ⏱️  Time: {save_duration:.1f}s")
                print(f"{'='*70}\n")
                
                self.last_full_save_step = global_step
                
                # Post-save cleanup
                for _ in range(3):
                    gc.collect()
                
                return True
            else:
                print(f"   ❌ Checkpoint file not found after save")
                self.oom_failures += 1
                return False
        
        except MemoryError:
            print(f"\n   ❌ OUT OF MEMORY during checkpoint save!")
            self.oom_failures += 1
            print(f"   📊 OOM failure count: {self.oom_failures}")
            print(f"   💡 Next full save attempt: step {global_step + self._get_full_save_interval()}")
            print(f"{'='*70}\n")
            
            # Cleanup after failed save
            for _ in range(5):
                gc.collect()
            
            return False
        
        except Exception as e:
            print(f"\n   ❌ Checkpoint save failed: {e}")
            self.oom_failures += 1
            print(f"{'='*70}\n")
            return False
    
    def load(self, model):
        """Load checkpoint if available"""
        # Load training state
        state = self.state_mgr.load()
        if state:
            print(f"📥 Loaded training state:")
            print(f"   Step: {state['global_step']}")
            print(f"   Tokens: {state['tokens_processed']:,}")
            print(f"   Avg Loss: {state['avg_loss']:.4f}")
        
        # Load full model if available
        if self.full_checkpoint_path.exists() and model:
            try:
                size_gb = self.full_checkpoint_path.stat().st_size / (1024**3)
                print(f"\n📥 Loading full model checkpoint ({size_gb:.2f} GB)...")
                model.load_checkpoint(str(self.full_checkpoint_path))
                print(f"   ✅ Model loaded successfully!")
            except Exception as e:
                print(f"   ⚠️  Could not load model: {e}")
        
        return state
    
    def save_final_model(self, model):
        """Save final model at end of training (one last attempt)"""
        if not model:
            print("⚠️  No model to save")
            return False
        
        final_path = self.checkpoint_dir / "final_model.ckpt"
        
        print("\n" + "="*70)
        print("💾 SAVING FINAL MODEL")
        print("="*70)
        print(f"Path: {final_path}")
        print("This is the final checkpoint save and may take several minutes")
        print("⚠️  If this fails due to OOM, your training state is still saved")
        
        # Maximum possible cleanup
        print("\n🧹 Maximum memory cleanup before final save...")
        for i in range(10):
            collected = gc.collect()
            if i == 0:
                print(f"   Collected {collected} objects")
        time.sleep(2.0)
        
        try:
            save_start = time.time()
            model.save_checkpoint(str(final_path))
            save_duration = time.time() - save_start
            
            if final_path.exists():
                size_gb = final_path.stat().st_size / (1024**3)
                print(f"\n✅ FINAL MODEL SAVED SUCCESSFULLY!")
                print(f"   Size: {size_gb:.2f} GB")
                print(f"   Time: {save_duration:.1f}s")
                print(f"   Location: {final_path}")
                print("="*70 + "\n")
                return True
        except Exception as e:
            print(f"\n❌ Final model save failed: {e}")
            print("💡 Training state is saved in training_state.json")
            print("="*70 + "\n")
            return False


# ==============================================================================
# VISUALIZATION MANAGER
# ==============================================================================

class VisualizationManager:
    """Manages training visualization"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.viz_dir = Path(config.viz_dir)
        self.viz_dir.mkdir(exist_ok=True)
        self.history: Dict[str, List] = {
            'chunks': [], 'loss': [], 'accuracy': [],
            'throughput': [], 'tokens_processed': []
        }
        self.text_samples: List[Dict] = []
    
    def update(self, chunk: int, loss: float, accuracy: float, throughput: float,
               tokens_processed: int, input_text: str = "", output_text: str = "",
               expected_text: str = "", predicted_text: str = "",
               step: Optional[int] = None, sample_index: Optional[int] = None):
        """Update metrics"""
        self.history['chunks'].append(chunk)
        self.history['loss'].append(loss)
        self.history['accuracy'].append(accuracy)
        self.history['throughput'].append(throughput)
        self.history['tokens_processed'].append(tokens_processed)
        
        if input_text and predicted_text:
            self.text_samples.append({
                'chunk': chunk, 'step': step or chunk,
                'input': input_text, 'output': predicted_text,
                'expected_next': expected_text,
                'predicted_next': predicted_text,
                'accuracy': accuracy,
            })
    
    def save_text_samples(self):
        """Save text samples to JSON"""
        samples_path = self.viz_dir / 'text_samples.json'
        with open(samples_path, 'w') as f:
            json.dump(self.text_samples, f, indent=2)
        print(f"💾 Updated {samples_path}")
    
    def generate_charts(self, chunk: int):
        """Generate training charts"""
        if len(self.history['chunks']) < 2:
            return
        
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Loss
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(self.history['chunks'], self.history['loss'], 'b-', linewidth=2)
            ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Chunk')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            
            # Accuracy
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(self.history['chunks'], [a*100 for a in self.history['accuracy']], 'g-', linewidth=2)
            ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Chunk')
            ax2.set_ylabel('Accuracy (%)')
            ax2.grid(True, alpha=0.3)
            
            # Throughput
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(self.history['chunks'], self.history['throughput'], 'purple', linewidth=2)
            ax3.set_title('Token Throughput', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Chunk')
            ax3.set_ylabel('Tokens/Second')
            ax3.grid(True, alpha=0.3)
            
            # Cumulative tokens
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(self.history['chunks'], self.history['tokens_processed'], 'orange', linewidth=2)
            ax4.set_title('Cumulative Tokens', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Chunk')
            ax4.set_ylabel('Total Tokens')
            ax4.grid(True, alpha=0.3)
            
            fig.suptitle(f'NeuroGen Training - Chunk {chunk}', fontsize=16, fontweight='bold')
            
            output_path = self.viz_dir / f'training_chunk_{chunk:06d}.png'
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Saved visualization: {output_path.name}")
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")


# ==============================================================================
# TRAINER
# ==============================================================================

class SlimPajamaTrainer:
    """Trainer optimized for GTX 1650"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Load tokenizer
        print(f"📝 Loading tokenizer from {config.tokenizer_dir}")
        tokenizer_path = Path(config.tokenizer_dir) / config.tokenizer_model
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(tokenizer_path))
        
        tokenizer_state_path = Path(config.tokenizer_dir) / "tokenizer_state.json"
        with open(tokenizer_state_path, 'r') as f:
            tokenizer_config = json.load(f)
        actual_vocab_size = tokenizer_config.get('vocab_size', 32000)
        print(f"✅ Tokenizer loaded: vocab_size={actual_vocab_size}")
        
        if actual_vocab_size != config.vocab_size:
            config.vocab_size = actual_vocab_size
        
        # Training state
        self.global_step = 0
        self.total_loss = 0.0
        self.tokens_processed = 0
        self.start_time = time.time()
        self.sample_step = 0
        
        # GTX 1650 optimized checkpoint manager
        self.checkpoint_mgr = GTX1650CheckpointManager(str(self.checkpoint_dir))
        
        # Visualization
        self.viz = VisualizationManager(config)
        
        # Load model
        sys.path.insert(0, str(Path(__file__).parent / "bin"))
        try:
            import libneurogen
            self.model = libneurogen.NeuroGenModel(
                vocab_size=config.vocab_size,
                embedding_dim=config.embedding_dim,
                gpu_device=config.gpu_device
            )
            print("✅ Loaded libneurogen bindings")
        except ImportError:
            lib_path = Path(__file__).parent / "bin" / "libneurogen.so"
            if lib_path.exists():
                print(f"✓ Found shared library at {lib_path}")
                print("   (Running in simulation mode)")
            else:
                print("⚠️  libneurogen.so not found")
            self.model = None
    
    def tokenize_text(self, text: str) -> Tuple[List[int], int]:
        """Tokenize text"""
        tokens = self.tokenizer.EncodeAsIds(text)
        if len(tokens) < 2:
            return [], -1
        if len(tokens) > self.config.max_seq_length:
            tokens = tokens[:self.config.max_seq_length]
        visible_len = min(32, len(tokens) - 1)
        return tokens[:visible_len], tokens[visible_len]
    
    def train_on_chunk(self, texts: List[str]) -> Tuple[float, float, str, str, str, int, int]:
        """Train on chunk"""
        chunk_loss = chunk_accuracy = chunk_tokens = 0.0
        processed = skipped = 0
        sample_input = expected_next = predicted_next = ""
        first_idx = first_step = -1
        
        for idx, text in enumerate(texts):
            context_ids, target_id = self.tokenize_text(text)
            if not context_ids or target_id < 0:
                skipped += 1
                continue
            
            processed += 1
            self.sample_step += 1
            
            if self.model:
                loss, accuracy, predicted_id = self.model.train_step(context_ids, [target_id])
            else:
                time.sleep(0.001)
                loss, accuracy, predicted_id = 2.5, 0.1, target_id
            
            chunk_loss += loss * len(context_ids)
            chunk_accuracy += accuracy * len(context_ids)
            
            if not sample_input:
                sample_input = self.tokenizer.DecodeIds(context_ids)
                expected_next = self.tokenizer.DecodeIds([target_id])
                predicted_next = self.tokenizer.DecodeIds([predicted_id]) if predicted_id >= 0 else ""
                first_idx, first_step = idx, self.sample_step
            
            chunk_tokens += len(context_ids)
            self.tokens_processed += len(context_ids)
        
        if self.config.verbose_logging:
            print(f"      Processed {processed} samples ({skipped} skipped), {int(chunk_tokens)} tokens")
        
        avg_loss = chunk_loss / max(chunk_tokens, 1)
        avg_accuracy = chunk_accuracy / max(chunk_tokens, 1)
        return avg_loss, avg_accuracy, sample_input, expected_next, predicted_next, first_idx, first_step
    
    def train(self):
        """Main training loop"""
        print(f"🚀 Starting training")
        
        # Load checkpoint
        if self.model:
            print("\n🔍 Checking for checkpoint...")
            state = self.checkpoint_mgr.load(self.model)
            if state:
                # Restore training state
                self.global_step = state.get('global_step', 0)
                self.total_loss = state.get('total_loss', 0.0)
                self.tokens_processed = state.get('tokens_processed', 0)
                print("✅ Resuming from checkpoint\n")
            else:
                print("ℹ️  Starting fresh\n")
        
        # Load dataset
        try:
            print("⏳ Loading dataset...")
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split,
                streaming=self.config.streaming
            )
            print("✅ Dataset loaded\n")
        except Exception as e:
            print(f"❌ Dataset load failed: {e}")
            return
        
        # Training loop
        chunk_texts = []
        chunk_tokens = samples_in_chunk = 0
        last_update = time.time()
        
        print("📊 Starting training...")
        print(f"   Tokens per chunk: {self.config.tokens_per_chunk}\n")
        
        for i, example in enumerate(dataset):
            text = example.get("text", "")
            if not text:
                continue
            
            # Progress update
            if time.time() - last_update >= 2.0:
                print(f"📥 Accumulating: {samples_in_chunk} samples, ~{chunk_tokens} tokens")
                last_update = time.time()
            
            chunk_texts.append(text)
            samples_in_chunk += 1
            chunk_tokens += len(text) // 4
            
            if chunk_tokens >= self.config.tokens_per_chunk:
                print(f"\n✓ Chunk complete: {samples_in_chunk} samples, ~{chunk_tokens} tokens")
                print(f"🏋️  Training on chunk {self.global_step + 1}...")
                
                chunk_start = time.time()
                loss, accuracy, sample_input, expected, predicted, idx, step = self.train_on_chunk(chunk_texts)
                chunk_time = time.time() - chunk_start
                
                self.global_step += 1
                self.total_loss += loss
                avg_loss = self.total_loss / self.global_step
                elapsed = time.time() - self.start_time
                tps = self.tokens_processed / elapsed if elapsed > 0 else 0
                
                print(f"✅ Step {self.global_step} complete in {chunk_time:.2f}s")
                print(f"   Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}% | Avg Loss: {avg_loss:.4f}")
                print(f"   Throughput: {tps:.1f} tokens/sec | Total: {self.tokens_processed:,}")
                
                # Update viz
                self.viz.update(self.global_step, loss, accuracy, tps, self.tokens_processed,
                               sample_input, predicted, expected, predicted, step, idx)
                self.viz.save_text_samples()
                
                # Save checkpoint (lightweight state every time, full model rarely)
                if self.global_step % 5 == 0:
                    self.checkpoint_mgr.save(
                        self.model, self.global_step, self.total_loss,
                        self.tokens_processed, self.start_time
                    )
                
                # Generate charts
                if self.global_step % self.config.viz_interval_chunks == 0:
                    print("📊 Generating charts...")
                    self.viz.generate_charts(self.global_step)
                
                print()
                
                # Reset for next chunk
                chunk_texts = []
                chunk_tokens = samples_in_chunk = 0
                last_update = time.time()
                
                if self.config.max_chunks and self.global_step >= self.config.max_chunks:
                    print(f"✅ Reached max_chunks ({self.config.max_chunks})")
                    break
        
        # Final saves
        print("\n📊 Final visualizations...")
        self.viz.save_text_samples()
        if self.global_step > 0:
            self.viz.generate_charts(self.global_step)
        
        # Save final model
        if self.model:
            self.checkpoint_mgr.save_final_model(self.model)
        
        print("\n" + "="*70)
        print("🎉 Training complete!")
        print("="*70)
        print(f"Steps: {self.global_step}")
        print(f"Tokens: {self.tokens_processed:,}")
        print(f"Avg loss: {self.total_loss / max(self.global_step, 1):.4f}")
        print(f"Time: {time.time() - self.start_time:.1f}s")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Train NeuroGen on SlimPajama")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--tokens-per-chunk", type=int, default=4096)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    if args.test:
        print("🧪 TEST MODE")
        config = TrainingConfig(
            gpu_device=args.gpu, tokens_per_chunk=1024,
            max_chunks=5, max_seq_length=256
        )
    else:
        config = TrainingConfig(
            gpu_device=args.gpu, tokens_per_chunk=args.tokens_per_chunk,
            max_chunks=args.max_chunks, max_seq_length=args.max_seq_length
        )
    
    print("\n" + "="*70)
    print("🧠 NeuroGen 2.0 - SlimPajama Training (GTX 1650 Optimized)")
    print("="*70)
    print(f"GPU: {config.gpu_device}")
    print(f"Tokens/chunk: {config.tokens_per_chunk}")
    print(f"Max chunks: {config.max_chunks or 'unlimited'}")
    print("="*70 + "\n")
    
    try:
        trainer = SlimPajamaTrainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()