#!/usr/bin/env python3
"""
NeuroGen 2.0 - Advanced Training Script with Visualization

Features:
- Real-time performance metrics
- Text input/output display
- Loss and accuracy curves
- Learning rate scheduling
- Comprehensive charts and graphs
- Model parameter visualization
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from collections import deque
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
    print("Install with: pip install datasets sentencepiece matplotlib tqdm")
    HAS_DEPS = False

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
    vocab_size: int = 32000  # SentencePiece vocab size
    embedding_dim: int = 1536  # Scaled to 60% of max for stable 4GB GPU usage
    gpu_device: int = 0
    
    # Dataset
    dataset_name: str = "cerebras/SlimPajama-627B"
    dataset_split: str = "train"
    streaming: bool = True
    max_seq_length: int = 256
    
    # Training
    batch_size: int = 1
    initial_learning_rate: float = 0.001
    num_steps: int = 10000
    warmup_steps: int = 1000
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 2000  # Save every 2000 steps (reduced from 500)
    
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
        plt.savefig(latest_path, dpi=150, bbox_inches='tight')
        
        print(f"📊 Saved visualization: {output_path.name}")
    
    def save_metrics_json(self):
        """Save metrics history to JSON"""
        metrics_path = self.viz_dir / 'metrics_history.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        samples_path = self.viz_dir / 'text_samples.json'
        with open(samples_path, 'w') as f:
            json.dump(self.text_samples, f, indent=2)

class AdvancedTrainer:
    """Advanced trainer with comprehensive logging and visualization"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize model
        print("🚀 Initializing NeuroGen 2.0 (SCALED UP LLM)")
        if HAS_NEUROGEN:
            self.model = libneurogen.NeuroGenModel(
                vocab_size=config.vocab_size,
                embedding_dim=config.embedding_dim,
                gpu_device=config.gpu_device
            )
            print("✅ Model loaded successfully")
        else:
            self.model = None
            print("⚠️  Running in simulation mode")
        
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
    
    def get_learning_rate(self, step: int) -> float:
        """Calculate learning rate with warmup"""
        if step < self.config.warmup_steps:
            return self.config.initial_learning_rate * (step / self.config.warmup_steps)
        return self.config.initial_learning_rate
    
    def train_step(self, input_ids: List[int], target_ids: List[int]) -> Tuple[float, float]:
        """Execute one training step"""
        if self.model:
            loss, accuracy = self.model.train_step(input_ids, target_ids)
        else:
            # Simulation
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
        """Main training loop"""
        print("\n" + "="*80)
        print("🧠 NeuroGen 2.0 - Advanced Training Loop")
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
        
        # Training loop
        pbar = tqdm(total=self.config.num_steps, desc="Training", ncols=100)
        
        # Add timeout protection and better error handling
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
                target_text = self.tokenizer.DecodeIds(target_ids[:50])
                self.viz.update(metrics, input_text, target_text)
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
    parser = argparse.ArgumentParser(description="NeuroGen 2.0 Advanced Training")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--viz-interval", type=int, default=50, help="Visualization interval")
    parser.add_argument("--checkpoint-interval", type=int, default=2000, help="Checkpoint save interval (steps)")
    args = parser.parse_args()
    
    config = TrainingConfig(
        gpu_device=args.gpu,
        num_steps=args.steps,
        viz_interval=args.viz_interval,
        checkpoint_interval=args.checkpoint_interval
    )
    
    trainer = AdvancedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()

