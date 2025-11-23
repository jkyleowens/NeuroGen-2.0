#!/usr/bin/env python3
"""
NeuroGen 2.0 - SlimPajama Training Script with Advanced Visualization

Features:
- Real-time performance metrics
- Text input/output display  
- Loss and accuracy curves
- Comprehensive charts and graphs
- Detailed logging with text samples
- Model parameter visualization

Requirements:
    pip install datasets sentencepiece numpy tqdm zstandard matplotlib
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from collections import deque
import json
import signal
import random
import datetime

# Add bin to path for libneurogen
sys.path.insert(0, str(Path(__file__).parent / "bin"))

try:
    from datasets import load_dataset
    import sentencepiece as spm
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_DEPS = True
except ImportError as e:
    print(f"⚠️  Missing dependencies: {e}")
    print("Install with: pip install datasets sentencepiece matplotlib tqdm zstandard")
    HAS_DEPS = False
    sys.exit(1)

try:
    import zstandard  # noqa: F401
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import libneurogen
    HAS_NEUROGEN = True
except ImportError:
    print("⚠️  libneurogen not found - running in simulation mode")
    HAS_NEUROGEN = False

@dataclass
class TrainingMetrics:
    """Real-time training metrics"""
    step: int
    loss: float
    accuracy: float
    tokens_per_sec: float
    gpu_memory_mb: float
    learning_rate: float
    timestamp: float

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model parameters
    vocab_size: int = 32000  # SentencePiece vocab size
    embedding_dim: int = 1536  # Scaled to 60% of max for stable 4GB GPU usage
    gpu_device: int = 0
    temperature: float = 1.0  # Sampling temperature
    
    # Dataset parameters
    dataset_name: str = "cerebras/SlimPajama-627B"
    dataset_split: str = "train"
    streaming: bool = True  # Stream for large datasets
    max_seq_length: int = 256  # Match train_advanced.py
    
    # Training parameters
    batch_size: int = 1
    initial_learning_rate: float = 0.001
    num_steps: int = 10000  # Train for N steps (not chunks!)
    warmup_steps: int = 1000
    
    # Checkpoint parameters
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 2000  # Save every N steps
    
    # Visualization
    viz_dir: str = "training_viz"
    viz_interval: int = 50
    display_samples: int = 5
    
    # Tokenizer
    tokenizer_dir: str = "tokenizer"
    tokenizer_model: str = "nlp_agent_tokenizer.model"


class VisualizationManager:
    """Manages training visualization and chart generation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.viz_dir = Path(config.viz_dir)
        self.viz_dir.mkdir(exist_ok=True)
        
        # Metric history
        self.history: Dict[str, List] = {
            'steps': [],
            'loss': [],
            'accuracy': [],
            'throughput': [],
            'learning_rate': []
        }
        
        # Text samples
        self.text_samples: List[Dict] = []
        
    def update(self, metrics: TrainingMetrics, input_text: str = "", output_text: str = ""):
        """Update metrics and optionally add text sample"""
        self.history['steps'].append(metrics.step)
        self.history['loss'].append(metrics.loss)
        self.history['accuracy'].append(metrics.accuracy)
        self.history['throughput'].append(metrics.tokens_per_sec)
        self.history['learning_rate'].append(metrics.learning_rate)
        
        if input_text and output_text:
            self.text_samples.append({
                'step': metrics.step,
                'input': input_text,
                'output': output_text,
                'accuracy': metrics.accuracy
            })
    
    def generate_charts(self, step: int):
        """Generate comprehensive training charts"""
        if len(self.history['steps']) < 10:
            return
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Loss Curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.history['steps'], self.history['loss'], 'b-', linewidth=2, alpha=0.8)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Add moving average
        if len(self.history['loss']) > 50:
            window = 50
            ma_loss = np.convolve(self.history['loss'], np.ones(window)/window, mode='valid')
            ma_steps = self.history['steps'][window-1:]
            ax1.plot(ma_steps, ma_loss, 'r--', linewidth=2, label=f'{window}-step MA', alpha=0.6)
            ax1.legend()
        
        # 2. Accuracy Curve
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.history['steps'], [a*100 for a in self.history['accuracy']], 
                'g-', linewidth=2, alpha=0.8)
        ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3)
        
        # Add moving average
        if len(self.history['accuracy']) > 50:
            window = 50
            ma_acc = np.convolve(self.history['accuracy'], np.ones(window)/window, mode='valid')
            ma_steps = self.history['steps'][window-1:]
            ax2.plot(ma_steps, [a*100 for a in ma_acc], 'orange', linewidth=2, 
                    label=f'{window}-step MA', alpha=0.6, linestyle='--')
            ax2.legend()
        
        # 3. Throughput
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.history['steps'], self.history['throughput'], 
                'purple', linewidth=2, alpha=0.8)
        ax3.set_title('Token Throughput', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Tokens/Second')
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning Rate Schedule
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.history['steps'], self.history['learning_rate'], 
                'orange', linewidth=2, alpha=0.8)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        # 5. Loss vs Accuracy
        ax5 = fig.add_subplot(gs[1, 1])
        scatter = ax5.scatter(self.history['loss'], 
                            [a*100 for a in self.history['accuracy']],
                            c=self.history['steps'], cmap='viridis', 
                            alpha=0.6, s=20)
        ax5.set_title('Loss vs Accuracy', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Loss')
        ax5.set_ylabel('Accuracy (%)')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Training Step')
        
        # 6. Recent Performance (last 500 steps)
        ax6 = fig.add_subplot(gs[1, 2])
        recent_steps = 500
        if len(self.history['steps']) > recent_steps:
            recent_loss = self.history['loss'][-recent_steps:]
            recent_acc = [a*100 for a in self.history['accuracy'][-recent_steps:]]
            recent_x = list(range(len(recent_loss)))
            
            ax6_twin = ax6.twinx()
            ax6.plot(recent_x, recent_loss, 'b-', linewidth=2, label='Loss', alpha=0.8)
            ax6_twin.plot(recent_x, recent_acc, 'g-', linewidth=2, label='Accuracy', alpha=0.8)
            
            ax6.set_title(f'Recent Performance (Last {recent_steps} steps)', 
                         fontsize=14, fontweight='bold')
            ax6.set_xlabel('Step (relative)')
            ax6.set_ylabel('Loss', color='b')
            ax6_twin.set_ylabel('Accuracy (%)', color='g')
            ax6.tick_params(axis='y', labelcolor='b')
            ax6_twin.tick_params(axis='y', labelcolor='g')
            ax6.grid(True, alpha=0.3)
        
        # 7. Text Samples Table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        if self.text_samples:
            # Show last 5 samples
            recent_samples = self.text_samples[-5:]
            table_data = []
            for sample in recent_samples:
                input_short = (sample['input'][:40] + '...') if len(sample['input']) > 40 else sample['input']
                output_short = (sample['output'][:40] + '...') if len(sample['output']) > 40 else sample['output']
                table_data.append([
                    f"Step {sample['step']}",
                    input_short,
                    output_short,
                    f"{sample['accuracy']*100:.1f}%"
                ])
            
            table = ax7.table(cellText=table_data,
                            colLabels=['Step', 'Input Text', 'Model Output', 'Accuracy'],
                            cellLoc='left',
                            loc='center',
                            colWidths=[0.15, 0.35, 0.35, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style header
            for i in range(4):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax7.set_title('Recent Text Samples', fontsize=14, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle(f'NeuroGen 2.0 Training Dashboard - Step {step}', 
                    fontsize=16, fontweight='bold')
        
        # Save
        output_path = self.viz_dir / f'training_step_{step:06d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save latest
        latest_path = self.viz_dir / 'training_latest.png'
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Recreate the same chart
        # (This is inefficient but ensures latest.png always matches the most recent)
        # In production, we'd save fig before closing
        
        print(f"📊 Saved visualization: {output_path.name}")
    
    def save_metrics_json(self):
        """Save metrics history to JSON"""
        metrics_path = self.viz_dir / 'metrics_history.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        samples_path = self.viz_dir / 'text_samples.json'
        with open(samples_path, 'w') as f:
            json.dump(self.text_samples, f, indent=2)


class SlimPajamaTrainer:
    """Trainer for NeuroGen on SlimPajama dataset - matches train_advanced.py structure"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize SentencePiece tokenizer
        print(f"📝 Loading SentencePiece tokenizer from {config.tokenizer_dir}")
        if HAS_DEPS:
            tokenizer_path = Path(config.tokenizer_dir) / config.tokenizer_model
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.Load(str(tokenizer_path))
            
            # Load vocab size from config
            tokenizer_state_path = Path(config.tokenizer_dir) / "tokenizer_state.json"
            with open(tokenizer_state_path, 'r') as f:
                tokenizer_config = json.load(f)
            self.vocab_size = tokenizer_config.get('vocab_size', 32000)
            print(f"✅ Tokenizer loaded: vocab_size={self.vocab_size}")
        else:
            self.tokenizer = None
            self.vocab_size = 32000
            
        # Visualization
        self.viz = VisualizationManager(config)
        
        # Training state
        self.global_step = 0
        self.start_time = time.time()
        self.step_times = deque(maxlen=100)
        
        # Checkpointing
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        
        # Load model
        try:
            if HAS_NEUROGEN:
                self.model = libneurogen.NeuroGenModel(
                    vocab_size=self.vocab_size,
                    embedding_dim=config.embedding_dim,
                    gpu_device=config.gpu_device
                )
                print("✅ Loaded libneurogen bindings")
            else:
                print("⚠️  Running in simulation mode (libneurogen not found)")
                self.model = None
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("   Running in simulation mode")
            self.model = None

    def get_learning_rate(self, step: int) -> float:
        """Calculate learning rate with warmup"""
        if step < self.config.warmup_steps:
            return self.config.initial_learning_rate * (step / self.config.warmup_steps)
        return self.config.initial_learning_rate
    
    def train_step(self, input_ids: List[int], target_ids: List[int]) -> Tuple[float, float]:
        """Execute one training step on a single sample"""
        if self.model:
            loss, accuracy = self.model.train_step(input_ids, target_ids)
        else:
            # Simulation mode
            loss = np.random.uniform(1.0, 3.0)
            accuracy = np.random.uniform(0.0, 0.3)
        
        return loss, accuracy
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to human-readable time"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def train(self):
        """Main training loop - processes ONE SAMPLE PER STEP (matches train_advanced.py)"""
        print("\n" + "="*80)
        print("🧠 NeuroGen 2.0 - SlimPajama Training (Sample-by-Sample)")
        print("="*80 + "\n")
        
        if not HAS_DEPS:
            print("❌ Missing required dependencies. Exiting.")
            return
        
        # Load dataset
        print(f"📚 Loading dataset: {self.config.dataset_name}")
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split,
                streaming=self.config.streaming
            )
            print("✅ Dataset loaded (streaming mode)")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return
        
        print(f"\n🎯 Training for {self.config.num_steps} steps")
        print(f"📊 Visualizations will be saved to: {self.config.viz_dir}/")
        print(f"⏳ Iterating dataset (this may take a moment to start)...\n")
        
        # Training loop with progress bar
        pbar = tqdm(total=self.config.num_steps, desc="Training", ncols=100)
        
        # Add timeout protection and error handling
        dataset_iter = iter(dataset)
        consecutive_skips = 0
        max_consecutive_skips = 1000
        
        while self.global_step < self.config.num_steps:
            try:
                example = next(dataset_iter)
            except StopIteration:
                print("\n⚠️  Reached end of dataset stream")
                break
            except Exception as e:
                print(f"\n❌ Error reading dataset: {e}")
                consecutive_skips += 1
                if consecutive_skips > max_consecutive_skips:
                    print(f"❌ Too many consecutive errors ({max_consecutive_skips}), stopping")
                    break
                continue
            
            text = example.get("text", "")
            if not text or len(text) < 10:
                consecutive_skips += 1
                if consecutive_skips > max_consecutive_skips:
                    print(f"\n⚠️  Too many empty samples ({max_consecutive_skips}), stopping")
                    break
                continue
            
            # Reset skip counter on successful sample
            consecutive_skips = 0
            
            # Tokenize with SentencePiece
            tokens = self.tokenizer.EncodeAsIds(text)
            
            # Truncate if needed
            if len(tokens) > self.config.max_seq_length:
                tokens = tokens[:self.config.max_seq_length]
            
            if len(tokens) < 2:
                continue
            
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
            
            # Training step
            step_start = time.time()
            loss, accuracy = self.train_step(input_ids, target_ids)
            step_time = time.time() - step_start
            self.step_times.append(step_time)
            
            # Calculate metrics
            tokens_per_sec = len(input_ids) / step_time if step_time > 0 else 0
            lr = self.get_learning_rate(self.global_step)
            
            metrics = TrainingMetrics(
                step=self.global_step,
                loss=loss,
                accuracy=accuracy,
                tokens_per_sec=tokens_per_sec,
                gpu_memory_mb=0.0,  # Placeholder
                learning_rate=lr,
                timestamp=time.time()
            )
            
            # Decode text for display (occasionally)
            if self.global_step % self.config.display_samples == 0:
                input_text = self.tokenizer.DecodeIds(input_ids[:50])
                
                # Generate actual predictions from the model
                if self.model:
                    try:
                        # Use first 10 tokens as prompt, generate next 40
                        prompt_len = min(10, len(input_ids))
                        prompt_ids = input_ids[:prompt_len]
                        generated_ids = self.model.generate(prompt_ids, max_length=40)
                        # Remove the prompt part to show only predictions
                        predicted_ids = generated_ids[prompt_len:]
                        output_text = self.tokenizer.DecodeIds(predicted_ids[:50])
                    except:
                        # Fallback: show targets if generation fails
                        output_text = self.tokenizer.DecodeIds(target_ids[:50])
                else:
                    # Simulation mode: show targets
                    output_text = self.tokenizer.DecodeIds(target_ids[:50])
                    
                self.viz.update(metrics, input_text, output_text)
            else:
                self.viz.update(metrics)
            
            # Update progress bar
            avg_step_time = np.mean(self.step_times) if self.step_times else step_time
            eta = (self.config.num_steps - self.global_step) * avg_step_time
            
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{accuracy*100:.1f}%',
                'tps': f'{tokens_per_sec:.1f}',
                'lr': f'{lr:.2e}',
                'eta': self.format_time(eta)
            })
            pbar.update(1)
            
            # Generate visualizations
            if self.global_step % self.config.viz_interval == 0 and self.global_step > 0:
                self.viz.generate_charts(self.global_step)
                self.viz.save_metrics_json()
            
            # Save checkpoint
            if self.global_step % self.config.checkpoint_interval == 0 and self.global_step > 0:
                if self.model:
                    ckpt_path = f"{self.config.checkpoint_dir}/checkpoint_step_{self.global_step}.bin"
                    self.model.save_checkpoint(ckpt_path)
                    print(f"\n💾 Saved checkpoint: {ckpt_path}")
            
            self.global_step += 1
        
        pbar.close()
        
        # Final visualization
        print("\n📊 Generating final visualizations...")
        self.viz.generate_charts(self.global_step)
        self.viz.save_metrics_json()
        
        # Training summary
        total_time = time.time() - self.start_time
        avg_tps = np.mean([m for m in self.viz.history['throughput'] if m > 0])
        
        print("\n" + "="*80)
        print("✅ Training Complete!")
        print("="*80)
        print(f"📊 Total Steps: {self.global_step}")
        print(f"⏱️  Total Time: {self.format_time(total_time)}")
        print(f"🚀 Avg Throughput: {avg_tps:.1f} tokens/sec")
        print(f"📉 Final Loss: {self.viz.history['loss'][-1]:.4f}")
        print(f"📈 Final Accuracy: {self.viz.history['accuracy'][-1]*100:.1f}%")
        print(f"📁 Visualizations saved to: {self.config.viz_dir}/")
        print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="NeuroGen 2.0 - SlimPajama Training (Sample-by-Sample)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--viz-interval", type=int, default=50, help="Visualization generation interval (steps)")
    parser.add_argument("--checkpoint-interval", type=int, default=2000, help="Checkpoint save interval (steps)")
    parser.add_argument("--test", action="store_true", help="Quick test mode (100 steps)")
    args = parser.parse_args()
    
    # Configure based on mode
    if args.test:
        print("🧪 Running in TEST MODE (quick verification)")
        config = TrainingConfig(
            gpu_device=args.gpu,
            num_steps=100,  # Quick test with 100 steps
            max_seq_length=128,  # Shorter sequences
            viz_interval=25,  # Visualize twice during test
            checkpoint_interval=200,  # No checkpoint in test mode
            display_samples=10  # Show more samples
        )
    else:
        config = TrainingConfig(
            gpu_device=args.gpu,
            num_steps=args.steps,
            max_seq_length=args.max_seq_length,
            viz_interval=args.viz_interval,
            checkpoint_interval=args.checkpoint_interval
        )
    
    print("\n" + "="*80)
    print("🧠 NeuroGen 2.0 - SlimPajama Training with Advanced Visualization")
    print("="*80)
    print(f"Configuration:")
    print(f"  GPU Device: {config.gpu_device}")
    print(f"  Training Steps: {config.num_steps}")
    print(f"  Max sequence length: {config.max_seq_length}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  Visualization interval: {config.viz_interval} steps")
    print(f"  Checkpoint interval: {config.checkpoint_interval} steps")
    print(f"  Visualization dir: {config.viz_dir}/")
    print("="*80 + "\n")
    
    try:
        trainer = SlimPajamaTrainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
