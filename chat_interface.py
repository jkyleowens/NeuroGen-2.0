#!/usr/bin/env python3
"""
NeuroGen 2.0 - Interactive Chat Interface

Real-time text generation interface with the NeuroGen brain-inspired model.
Supports multi-token generation with intelligent stopping.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple
import os

# Add bin to path for libneurogen
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


class NeuroGenChat:
    """Interactive chat interface for NeuroGen model"""
    
    def __init__(self, checkpoint_path: Optional[str] = None, auto_load_latest: bool = True):
        """Initialize the model and tokenizer"""
        print("=" * 80)
        print("🧠 NeuroGen 2.0 - Interactive Chat Interface")
        print("=" * 80 + "\n")
        
        # Auto-load latest checkpoint if not specified
        if checkpoint_path is None and auto_load_latest:
            checkpoint_dir = Path("checkpoints")
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.bin"))
                if checkpoints:
                    checkpoint_path = str(checkpoints[-1])
                    print(f"🔍 Auto-loading latest checkpoint: {checkpoint_path}\n")
        
        # Load tokenizer configuration
        print("📝 Loading SentencePiece tokenizer...")
        tokenizer_dir = Path(__file__).parent / "tokenizer"
        tokenizer_state_path = tokenizer_dir / "tokenizer_state.json"
        
        with open(tokenizer_state_path, 'r') as f:
            tokenizer_config = json.load(f)
        
        self.vocab_size = tokenizer_config.get('vocab_size', 32000)
        model_path = tokenizer_dir / tokenizer_config['model_path'].replace('./', '')
        
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(model_path))
        print(f"✅ Tokenizer loaded: vocab_size={self.vocab_size}\n")
        
        # Initialize model
        print("🚀 Initializing NeuroGen model...")
        self.model = libneurogen.NeuroGenModel(
            vocab_size=self.vocab_size,
            embedding_dim=1536,  # Scaled to 60% of max for stable 4GB GPU usage
            gpu_device=0
        )
        print("✅ Model initialized\n")
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"📂 Loading checkpoint from {checkpoint_path}...")
            self.model.load_checkpoint(checkpoint_path)
            print("✅ Checkpoint loaded\n")
        elif checkpoint_path:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print("   Starting with fresh model weights\n")
        
        # Generation parameters
        self.max_tokens = 100
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9
        
        # Stop tokens (end of sentence markers)
        self.stop_tokens = set([
            self.tokenizer.eos_id(),  # End of sequence
            self.tokenizer.piece_to_id('.'),
            self.tokenizer.piece_to_id('!'),
            self.tokenizer.piece_to_id('?'),
            self.tokenizer.piece_to_id('\n'),
        ])
        
        print("🎯 Generation Parameters:")
        print(f"   Max tokens: {self.max_tokens}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Top-K: {self.top_k}")
        print(f"   Top-P (nucleus): {self.top_p}")
        print()
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        return self.tokenizer.EncodeAsIds(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.DecodeIds(token_ids)
    
    def should_stop(self, token_id: int, generated_count: int, recent_tokens: List[int]) -> bool:
        """
        Determine if generation should stop.
        
        Stopping conditions:
        1. Hit max_tokens limit
        2. Generated an end-of-sentence token
        3. Generated multiple newlines (end of paragraph)
        4. Model outputs EOS token
        """
        # Max length check
        if generated_count >= self.max_tokens:
            return True
        
        # End of sequence token
        if token_id == self.tokenizer.eos_id():
            return True
        
        # End of sentence punctuation (stop after punctuation + space)
        if len(recent_tokens) >= 2:
            # Check if we just completed a sentence (punctuation followed by space/newline)
            prev_token = recent_tokens[-2] if len(recent_tokens) >= 2 else None
            if prev_token in self.stop_tokens:
                # Allow one more token after punctuation (usually a space)
                if generated_count > 10:  # Don't stop too early
                    return True
        
        # Multiple consecutive newlines (end of paragraph)
        if len(recent_tokens) >= 3:
            newline_id = self.tokenizer.piece_to_id('\n')
            if all(t == newline_id for t in recent_tokens[-3:]):
                return True
        
        return False
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                 temperature: Optional[float] = None, stream: bool = True) -> str:
        """
        Generate text continuation from prompt.
        
        Args:
            prompt: Input text to continue
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            stream: If True, print tokens as they're generated
        
        Returns:
            Generated text
        """
        # Use provided parameters or defaults
        max_tokens = max_tokens or self.max_tokens
        
        # Encode prompt
        input_ids = self.encode(prompt)
        
        if not input_ids:
            return ""
        
        if stream:
            print(f"\n🤖 NeuroGen: ", end='', flush=True)
        
        start_time = time.time()
        
        try:
            # Use the C++ generate method
            # The C++ implementation handles token-by-token generation,
            # sampling, and stopping conditions internally
            all_generated_ids = self.model.generate(input_ids, max_tokens)
            
            # Extract only the newly generated tokens (exclude prompt)
            generated_ids = all_generated_ids[len(input_ids):]
            
            # Decode and display
            if stream:
                # Stream output token by token for better UX
                for token_id in generated_ids:
                    token_text = self.decode([token_id])
                    print(token_text, end='', flush=True)
            
            elapsed = time.time() - start_time
            
            if stream:
                print(f"\n\n⏱️  Generated {len(generated_ids)} tokens in {elapsed:.2f}s ({len(generated_ids)/elapsed:.1f} tok/s)\n")
            
            # Decode full generation
            generated_text = self.decode(generated_ids)
            return generated_text
            
        except Exception as e:
            print(f"\n⚠️  Generation error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def chat_loop(self):
        """Main interactive chat loop"""
        print("=" * 80)
        print("💬 Chat Mode")
        print("=" * 80)
        print("\nCommands:")
        print("  /help     - Show this help message")
        print("  /params   - Show/change generation parameters")
        print("  /clear    - Clear conversation history")
        print("  /save     - Save checkpoint")
        print("  /quit     - Exit chat")
        print("\nJust type your message to chat with NeuroGen!")
        print("=" * 80 + "\n")
        
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/quit' or user_input == '/exit':
                        print("\n👋 Goodbye!\n")
                        break
                    
                    elif user_input == '/help':
                        print("\n📖 Available Commands:")
                        print("  /help     - Show this help message")
                        print("  /params   - Show/change generation parameters")
                        print("  /clear    - Clear conversation history")
                        print("  /save     - Save model checkpoint")
                        print("  /quit     - Exit chat")
                        print()
                        continue
                    
                    elif user_input == '/params':
                        print("\n⚙️  Current Parameters:")
                        print(f"  Max tokens: {self.max_tokens}")
                        print(f"  Temperature: {self.temperature}")
                        print(f"  Top-K: {self.top_k}")
                        print(f"  Top-P: {self.top_p}")
                        print("\nTo change: /params <param>=<value>")
                        print("Example: /params temperature=0.5")
                        print()
                        continue
                    
                    elif user_input.startswith('/params '):
                        # Parse parameter changes
                        try:
                            param_str = user_input[8:].strip()
                            param, value = param_str.split('=')
                            param = param.strip()
                            value = value.strip()
                            
                            if param == 'max_tokens':
                                self.max_tokens = int(value)
                                print(f"✅ Set max_tokens = {self.max_tokens}\n")
                            elif param == 'temperature':
                                self.temperature = float(value)
                                print(f"✅ Set temperature = {self.temperature}\n")
                            elif param == 'top_k':
                                self.top_k = int(value)
                                print(f"✅ Set top_k = {self.top_k}\n")
                            elif param == 'top_p':
                                self.top_p = float(value)
                                print(f"✅ Set top_p = {self.top_p}\n")
                            else:
                                print(f"⚠️  Unknown parameter: {param}\n")
                        except Exception as e:
                            print(f"⚠️  Invalid parameter format: {e}")
                            print("   Use: /params <param>=<value>\n")
                        continue
                    
                    elif user_input == '/clear':
                        conversation_history = []
                        print("\n🗑️  Conversation history cleared\n")
                        continue
                    
                    elif user_input == '/save':
                        checkpoint_dir = Path("checkpoints")
                        checkpoint_dir.mkdir(exist_ok=True)
                        checkpoint_path = checkpoint_dir / f"chat_checkpoint_{int(time.time())}.bin"
                        try:
                            self.model.save_checkpoint(str(checkpoint_path))
                            print(f"\n💾 Checkpoint saved to {checkpoint_path}\n")
                        except Exception as e:
                            print(f"\n⚠️  Error saving checkpoint: {e}\n")
                        continue
                    
                    else:
                        print(f"⚠️  Unknown command: {user_input}")
                        print("   Type /help for available commands\n")
                        continue
                
                # Add user message to history
                conversation_history.append(f"User: {user_input}")
                
                # Build prompt with conversation history (last N turns)
                context_turns = 5  # Keep last 5 turns
                recent_history = conversation_history[-context_turns*2:] if len(conversation_history) > context_turns*2 else conversation_history
                
                # Format prompt
                prompt = "\n".join(recent_history) + f"\nUser: {user_input}\nAssistant:"
                
                # Generate response
                response = self.generate(prompt, stream=True)
                
                # Add to history
                conversation_history.append(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type /quit to exit or continue chatting.\n")
                continue
            except EOFError:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()
    
    def single_generation(self, prompt: str):
        """Generate a single response and exit"""
        print(f"\n📝 Prompt: {prompt}\n")
        response = self.generate(prompt, stream=True)
        print(f"\n📄 Full Response:\n{response}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroGen 2.0 Interactive Chat Interface")
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to model checkpoint to load')
    parser.add_argument('--prompt', '-p', type=str, default=None,
                        help='Single prompt (non-interactive mode)')
    parser.add_argument('--max-tokens', '-m', type=int, default=100,
                        help='Maximum tokens to generate (default: 100)')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    parser.add_argument('--no-auto-load', action='store_true',
                        help='Disable automatic loading of latest checkpoint')
    
    args = parser.parse_args()
    
    # Initialize chat interface with auto-load enabled by default
    # (unless --no-auto-load is specified or --checkpoint is provided)
    auto_load = not args.no_auto_load and args.checkpoint is None
    chat = NeuroGenChat(checkpoint_path=args.checkpoint, auto_load_latest=auto_load)
    
    # Override parameters if provided
    if args.max_tokens:
        chat.max_tokens = args.max_tokens
    if args.temperature:
        chat.temperature = args.temperature
    
    # Single prompt mode or interactive chat
    if args.prompt:
        chat.single_generation(args.prompt)
    else:
        chat.chat_loop()


if __name__ == "__main__":
    if not HAS_DEPS:
        print("❌ Cannot run without dependencies")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
