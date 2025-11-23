#!/usr/bin/env python3
"""
NeuroGen 2.0 - Interactive Model Interface

Provides a command-line interface to interact with the trained NeuroGen model.
The model generates multi-token responses and decides when to stop generation.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Optional

# Add bin to path for libneurogen
sys.path.insert(0, str(Path(__file__).parent / "bin"))

try:
    import sentencepiece as spm
    HAS_SPM = True
except ImportError:
    print("❌ Error: sentencepiece not installed")
    print("Install with: pip install sentencepiece")
    HAS_SPM = False
    sys.exit(1)

try:
    import libneurogen
    HAS_NEUROGEN = True
except ImportError:
    print("⚠️  Warning: libneurogen not found")
    print("Make sure you've compiled the library: make -j$(nproc)")
    HAS_NEUROGEN = False


class NeuroGenInterface:
    """Interactive interface for NeuroGen model"""
    
    def __init__(self, tokenizer_dir: str = "tokenizer", checkpoint_path: Optional[str] = None):
        """
        Initialize the model interface
        
        Args:
            tokenizer_dir: Path to tokenizer directory
            checkpoint_path: Optional path to model checkpoint
        """
        print("\n" + "="*80)
        print("🧠 NeuroGen 2.0 - Interactive Model Interface")
        print("="*80 + "\n")
        
        # Load tokenizer
        print("📝 Loading SentencePiece tokenizer...")
        tokenizer_path = Path(tokenizer_dir) / "nlp_agent_tokenizer.model"
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(tokenizer_path))
        
        # Load vocab size from config
        tokenizer_state_path = Path(tokenizer_dir) / "tokenizer_state.json"
        with open(tokenizer_state_path, 'r') as f:
            tokenizer_config = json.load(f)
        self.vocab_size = tokenizer_config.get('vocab_size', 32000)
        print(f"✅ Tokenizer loaded: vocab_size={self.vocab_size}\n")
        
        # Initialize model
        print("🚀 Initializing NeuroGen model...")
        if HAS_NEUROGEN:
            self.model = libneurogen.NeuroGenModel(
                vocab_size=self.vocab_size,
                embedding_dim=512,
                gpu_device=0
            )
            
            # Load checkpoint if provided
            if checkpoint_path and Path(checkpoint_path).exists():
                print(f"📂 Loading checkpoint: {checkpoint_path}")
                self.model.load_checkpoint(checkpoint_path)
                print("✅ Checkpoint loaded")
            
            print("✅ Model initialized\n")
        else:
            self.model = None
            print("⚠️  Running in simulation mode (no model)\n")
        
        # Generation parameters
        self.max_length = 100
        self.temperature = 0.8
        self.top_k = 40
        self.top_p = 0.9
        self.repetition_penalty = 1.2
        
        # Special tokens
        self.eos_token_id = self.tokenizer.eos_id()
        self.bos_token_id = self.tokenizer.bos_id()
        self.pad_token_id = self.tokenizer.pad_id()
        
        print("⚙️  Generation Parameters:")
        print(f"   Max Length: {self.max_length}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Top-K: {self.top_k}")
        print(f"   Top-P: {self.top_p}")
        print(f"   Repetition Penalty: {self.repetition_penalty}\n")
    
    def generate(self, prompt: str, max_new_tokens: int = 50, 
                 show_thinking: bool = False) -> str:
        """
        Generate a response from the model
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            show_thinking: Whether to show token-by-token generation
            
        Returns:
            Generated text
        """
        if not self.model:
            return "[Model not available - running in simulation mode]"
        
        # Tokenize input
        input_ids = self.tokenizer.EncodeAsIds(prompt)
        
        if show_thinking:
            print(f"\n🔍 Input tokens ({len(input_ids)}): {input_ids[:10]}...")
            print(f"📝 Generating response...\n")
        
        generated_ids = []
        generation_start = time.time()
        
        # Track token frequencies for repetition penalty
        token_freq = {}
        
        # Generate tokens one by one
        for step in range(max_new_tokens):
            # Prepare context (input + generated so far)
            context_ids = input_ids + generated_ids
            
            # Get model predictions
            step_start = time.time()
            logits = self.model.forward(context_ids)
            step_time = time.time() - step_start
            
            # Get logits for next token (last position)
            next_token_logits = logits[-1]  # Shape: [vocab_size]
            
            # Apply temperature
            next_token_logits = [l / self.temperature for l in next_token_logits]
            
            # Apply repetition penalty
            for token_id, freq in token_freq.items():
                if token_id < len(next_token_logits):
                    next_token_logits[token_id] /= (self.repetition_penalty ** freq)
            
            # Sample next token (using top-k, top-p)
            next_token_id = self._sample_token(next_token_logits)
            
            # Update frequency tracking
            token_freq[next_token_id] = token_freq.get(next_token_id, 0) + 1
            
            # Check for end-of-sequence
            if next_token_id == self.eos_token_id:
                if show_thinking:
                    print("\n🛑 [EOS token generated]")
                break
            
            # Add to generated sequence
            generated_ids.append(next_token_id)
            
            # Decode and display token
            if show_thinking:
                token_text = self.tokenizer.DecodeIds([next_token_id])
                print(f"Token {step+1:3d} ({step_time*1000:.1f}ms): '{token_text}' (id={next_token_id})")
            
            # Check for natural stopping points
            if self._should_stop(generated_ids):
                if show_thinking:
                    print("\n✋ [Natural stopping point detected]")
                break
        
        generation_time = time.time() - generation_start
        
        # Decode generated tokens
        generated_text = self.tokenizer.DecodeIds(generated_ids)
        
        if show_thinking:
            tokens_per_sec = len(generated_ids) / generation_time if generation_time > 0 else 0
            print(f"\n✅ Generated {len(generated_ids)} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        return generated_text
    
    def _sample_token(self, logits: List[float]) -> int:
        """
        Sample next token using top-k and top-p (nucleus) sampling
        
        Args:
            logits: Logit scores for each token in vocabulary
            
        Returns:
            Sampled token ID
        """
        import math
        import random
        
        # Convert logits to probabilities
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]
        
        # Create list of (token_id, prob) pairs
        token_probs = [(i, p) for i, p in enumerate(probs)]
        
        # Sort by probability (descending)
        token_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top-k filtering
        if self.top_k > 0:
            token_probs = token_probs[:self.top_k]
        
        # Apply top-p (nucleus) filtering
        if self.top_p < 1.0:
            cumulative_prob = 0.0
            filtered_probs = []
            for token_id, prob in token_probs:
                cumulative_prob += prob
                filtered_probs.append((token_id, prob))
                if cumulative_prob >= self.top_p:
                    break
            token_probs = filtered_probs
        
        # Renormalize probabilities
        total_prob = sum(p for _, p in token_probs)
        token_probs = [(tid, p / total_prob) for tid, p in token_probs]
        
        # Sample from distribution
        rand_val = random.random()
        cumulative = 0.0
        for token_id, prob in token_probs:
            cumulative += prob
            if rand_val <= cumulative:
                return token_id
        
        # Fallback to most likely token
        return token_probs[0][0]
    
    def _should_stop(self, generated_ids: List[int]) -> bool:
        """
        Determine if generation should stop based on generated tokens
        
        Args:
            generated_ids: List of generated token IDs
            
        Returns:
            True if generation should stop
        """
        if len(generated_ids) < 3:
            return False
        
        # Decode last few tokens
        last_text = self.tokenizer.DecodeIds(generated_ids[-5:])
        
        # Check for sentence-ending punctuation followed by space
        ending_patterns = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        for pattern in ending_patterns:
            if pattern in last_text:
                return True
        
        # Check for paragraph breaks
        if '\n\n' in last_text:
            return True
        
        # Check for repeated tokens (potential loop)
        if len(generated_ids) >= 6:
            last_three = generated_ids[-3:]
            prev_three = generated_ids[-6:-3]
            if last_three == prev_three:
                return True
        
        return False
    
    def chat_loop(self):
        """Run interactive chat loop"""
        print("\n" + "="*80)
        print("💬 Interactive Chat Mode")
        print("="*80)
        print("\nCommands:")
        print("  - Type your message and press Enter")
        print("  - '/debug' - Toggle debug mode (show token generation)")
        print("  - '/temp <value>' - Set temperature (0.1-2.0)")
        print("  - '/length <value>' - Set max generation length")
        print("  - '/clear' - Clear conversation history")
        print("  - '/quit' or 'exit' - Exit the program")
        print("\n" + "="*80 + "\n")
        
        conversation_history = []
        debug_mode = False
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/quit' or user_input == 'exit':
                        print("\n👋 Goodbye!\n")
                        break
                    elif user_input == '/debug':
                        debug_mode = not debug_mode
                        print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}\n")
                        continue
                    elif user_input.startswith('/temp '):
                        try:
                            temp = float(user_input.split()[1])
                            if 0.1 <= temp <= 2.0:
                                self.temperature = temp
                                print(f"🌡️  Temperature set to {temp}\n")
                            else:
                                print("⚠️  Temperature must be between 0.1 and 2.0\n")
                        except (ValueError, IndexError):
                            print("⚠️  Usage: /temp <value>\n")
                        continue
                    elif user_input.startswith('/length '):
                        try:
                            length = int(user_input.split()[1])
                            if 1 <= length <= 500:
                                self.max_length = length
                                print(f"📏 Max length set to {length}\n")
                            else:
                                print("⚠️  Length must be between 1 and 500\n")
                        except (ValueError, IndexError):
                            print("⚠️  Usage: /length <value>\n")
                        continue
                    elif user_input == '/clear':
                        conversation_history = []
                        print("🗑️  Conversation history cleared\n")
                        continue
                    else:
                        print("⚠️  Unknown command\n")
                        continue
                
                # Add to conversation history
                conversation_history.append(f"User: {user_input}")
                
                # Build prompt from conversation history
                if len(conversation_history) > 6:  # Keep last 3 exchanges
                    context = conversation_history[-6:]
                else:
                    context = conversation_history
                
                prompt = "\n".join(context) + "\nAssistant:"
                
                # Generate response
                print("\n🧠 Generating...\n")
                response = self.generate(
                    prompt, 
                    max_new_tokens=self.max_length,
                    show_thinking=debug_mode
                )
                
                # Display response
                if not debug_mode:
                    print(f"NeuroGen: {response}\n")
                else:
                    print(f"\n📤 Final output: {response}\n")
                
                # Add to history
                conversation_history.append(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()
    
    def single_prompt(self, prompt: str, show_thinking: bool = True):
        """Generate a single response without chat loop"""
        print("\n" + "="*80)
        print("📝 Single Prompt Mode")
        print("="*80)
        print(f"\nPrompt: {prompt}\n")
        print("="*80 + "\n")
        
        response = self.generate(prompt, max_new_tokens=self.max_length, show_thinking=show_thinking)
        
        print("\n" + "="*80)
        print("📤 Generated Response:")
        print("="*80)
        print(f"\n{response}\n")
        print("="*80 + "\n")
        
        return response


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroGen 2.0 Interactive Interface")
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='tokenizer', help='Path to tokenizer directory')
    parser.add_argument('--prompt', type=str, help='Single prompt mode (no chat loop)')
    parser.add_argument('--max-length', type=int, default=50, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=40, help='Top-K sampling parameter')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-P (nucleus) sampling parameter')
    parser.add_argument('--debug', action='store_true', help='Show token-by-token generation')
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = NeuroGenInterface(
        tokenizer_dir=args.tokenizer,
        checkpoint_path=args.checkpoint
    )
    
    # Set parameters
    interface.max_length = args.max_length
    interface.temperature = args.temperature
    interface.top_k = args.top_k
    interface.top_p = args.top_p
    
    # Run in appropriate mode
    if args.prompt:
        # Single prompt mode
        interface.single_prompt(args.prompt, show_thinking=args.debug)
    else:
        # Interactive chat mode
        interface.chat_loop()


if __name__ == "__main__":
    if not HAS_SPM:
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
