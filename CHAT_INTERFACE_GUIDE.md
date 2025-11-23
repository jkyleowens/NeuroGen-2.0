# NeuroGen 2.0 - Interactive Chat Interface Guide

## Overview

The `chat_interface.py` provides an interactive way to chat with your trained NeuroGen model. It supports:
- 💬 Multi-turn conversations with context memory
- 🎯 Intelligent stopping (ends at sentence completion)
- ⚙️ Adjustable generation parameters
- 💾 Checkpoint loading/saving
- 🚀 Streaming output (see tokens as they generate)

## Quick Start

### Interactive Chat Mode

```bash
# Start chat with fresh model
python3 chat_interface.py

# Load a trained checkpoint
python3 chat_interface.py --checkpoint checkpoints/checkpoint_step_8000.bin

# Auto-load the latest checkpoint
python3 chat_interface.py --auto-load-latest
```

### Single Prompt Mode

```bash
# Generate a single response (non-interactive)
python3 chat_interface.py --prompt "The future of AI is"

# With custom parameters
python3 chat_interface.py --prompt "Once upon a time" --max-tokens 200 --temperature 0.9
```

## Usage Examples

### Example 1: Basic Chat

```
👤 You: What is machine learning?
🤖 NeuroGen: a subset of artificial intelligence that focuses on...
```

### Example 2: Story Generation

```
👤 You: Tell me a story about a robot
🤖 NeuroGen: Once upon a time, there was a robot named Atlas who...
```

### Example 3: Code Completion

```
👤 You: def fibonacci(n):
🤖 NeuroGen:     if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## Interactive Commands

While in chat mode, you can use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/params` | Show current generation parameters |
| `/params <param>=<value>` | Change a parameter |
| `/clear` | Clear conversation history |
| `/save` | Save current model checkpoint |
| `/quit` or `/exit` | Exit the chat |

## Generation Parameters

### Adjusting Parameters

```bash
# In chat mode:
/params temperature=0.5      # Make output more focused
/params max_tokens=200       # Generate longer responses
/params                      # View current settings
```

### Parameter Descriptions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 100 | Maximum tokens to generate per response |
| `temperature` | 0.8 | Sampling temperature (0.0=deterministic, 1.0=creative) |
| `top_k` | 50 | Consider top-K most likely tokens (future feature) |
| `top_p` | 0.9 | Nucleus sampling threshold (future feature) |

### Temperature Guide

- **0.1-0.3**: Very focused, repetitive, deterministic
- **0.5-0.7**: Balanced, good for factual content
- **0.8-1.0**: Creative, diverse, good for stories
- **1.2+**: Very random, may be incoherent

## Stopping Conditions

The model automatically stops generating when:

1. **Max tokens reached**: Hit the `max_tokens` limit
2. **End of sentence**: Generated `.`, `!`, or `?` followed by space
3. **End of paragraph**: Multiple consecutive newlines
4. **EOS token**: Model explicitly signals end-of-sequence
5. **Natural completion**: Model finishes a coherent thought

You can adjust stopping behavior by changing `max_tokens` or implementing custom logic in `should_stop()`.

## Command Line Options

```bash
python3 chat_interface.py [OPTIONS]

Options:
  -c, --checkpoint PATH        Load model checkpoint from PATH
  -p, --prompt TEXT           Single prompt mode (non-interactive)
  -m, --max-tokens N          Maximum tokens to generate (default: 100)
  -t, --temperature FLOAT     Sampling temperature (default: 0.8)
  -a, --auto-load-latest      Auto-load the latest checkpoint
  -h, --help                  Show help message
```

## Advanced Usage

### Loading Specific Checkpoints

```bash
# Load checkpoint from step 5000
python3 chat_interface.py -c checkpoints/checkpoint_step_5000.bin

# Compare different checkpoints
python3 chat_interface.py -c checkpoints/checkpoint_step_1000.bin -p "Hello world"
python3 chat_interface.py -c checkpoints/checkpoint_step_8000.bin -p "Hello world"
```

### Batch Generation

```bash
# Generate multiple completions
for prompt in "The sky is" "Once upon" "In the future"; do
    echo "=== Prompt: $prompt ==="
    python3 chat_interface.py -p "$prompt" -m 50
done
```

### Conversation Context

The chat interface maintains conversation history and includes the last 5 turns (10 messages) as context for generation. This allows for:

- Follow-up questions
- Pronouns referring to previous messages
- Coherent multi-turn conversations

Example:
```
👤 You: What is Python?
🤖 NeuroGen: Python is a high-level programming language...

👤 You: What is it used for?
🤖 NeuroGen: It is commonly used for web development, data science...
       ↑ "It" refers to Python from previous context
```

## Performance Metrics

The interface displays generation performance:

```
⏱️  Generated 45 tokens in 0.23s (195.7 tok/s)
```

This shows:
- Number of tokens generated
- Time taken
- Throughput (tokens per second)

Expected performance:
- **Without checkpoint**: ~50-100 tok/s (random initialization)
- **With trained model**: ~150-250 tok/s (GPU decoder enabled)

## Troubleshooting

### Issue: "ImportError: libneurogen"

**Solution**: Make sure the model is compiled:
```bash
make clean && make -j$(nproc)
ls -lh bin/libneurogen.so  # Should exist
```

### Issue: "Checkpoint not found"

**Solution**: Check available checkpoints:
```bash
ls -lh checkpoints/
# Use one that exists or omit --checkpoint to start fresh
```

### Issue: Generation is slow

**Causes**:
1. No checkpoint loaded (model not trained)
2. GPU not available
3. Long context window

**Solutions**:
```bash
# Load trained checkpoint
python3 chat_interface.py --auto-load-latest

# Check GPU
nvidia-smi

# Reduce max_tokens for faster responses
python3 chat_interface.py -m 50
```

### Issue: Output is gibberish

**Cause**: Model not trained yet

**Solution**: Train the model first:
```bash
# Quick training test
python3 train_simple.py

# Or full training
python3 train_advanced.py --steps 5000
```

### Issue: Output is repetitive

**Cause**: Temperature too low

**Solution**: Increase temperature:
```bash
python3 chat_interface.py -t 0.9
# Or in chat mode:
/params temperature=0.9
```

## Integration with Training

### Workflow

1. **Train the model**:
   ```bash
   python3 train_advanced.py --steps 10000
   ```

2. **Test during training**:
   ```bash
   # In another terminal
   python3 chat_interface.py --auto-load-latest -p "Test prompt"
   ```

3. **Evaluate quality**:
   ```bash
   # Try various prompts to assess performance
   python3 chat_interface.py --auto-load-latest
   ```

4. **Continue training if needed**:
   ```bash
   python3 train_advanced.py --steps 20000 --checkpoint checkpoints/checkpoint_step_10000.bin
   ```

## API Usage (Python Integration)

You can also use the `NeuroGenChat` class in your own Python scripts:

```python
from chat_interface import NeuroGenChat

# Initialize
chat = NeuroGenChat(checkpoint_path="checkpoints/checkpoint_step_8000.bin")

# Generate single response
response = chat.generate("The future of AI", stream=False)
print(response)

# Adjust parameters
chat.max_tokens = 150
chat.temperature = 0.7

# Generate with custom settings
response = chat.generate("Once upon a time", max_tokens=200, temperature=0.9)
```

## Future Enhancements

Planned features:
- [ ] Top-K and Top-P sampling in C++ backend
- [ ] Beam search for better quality
- [ ] Multi-token batching for faster generation
- [ ] Conversation saving/loading
- [ ] Model comparison mode
- [ ] Perplexity calculation
- [ ] Token probability display
- [ ] Interactive parameter tuning UI

## Tips for Best Results

1. **Use trained checkpoints**: The model needs training to generate coherent text
2. **Adjust temperature**: Lower for factual, higher for creative
3. **Provide context**: Give detailed prompts for better responses
4. **Experiment with max_tokens**: Longer doesn't always mean better
5. **Clear history**: Use `/clear` if context becomes confused
6. **Save good checkpoints**: Use `/save` when you get good results

## Examples of Good Prompts

### Factual
```
"Explain quantum computing in simple terms"
"What are the benefits of exercise?"
"How does photosynthesis work?"
```

### Creative
```
"Write a short poem about the ocean"
"Tell me a story about a time traveler"
"Imagine a world where humans can fly"
```

### Code
```
"Write a Python function to sort a list"
"Implement a binary search algorithm"
"Create a class for a stack data structure"
```

### Conversation
```
"Hello! How are you today?"
"What's your favorite color?"
"Can you tell me a joke?"
```

## Comparison with Other Models

To benchmark against other models:

```bash
# NeuroGen (your model)
python3 chat_interface.py -p "The capital of France" -c checkpoints/checkpoint_step_8000.bin

# Compare perplexity, coherence, speed, etc.
```

---

**Last Updated**: November 22, 2025  
**Version**: 1.0  
**Status**: ✅ Production Ready  

For more information, see:
- `TRAINING_GUIDE.md` - How to train the model
- `SETUP_AND_USAGE.md` - Installation and setup
- `README.md` - Project overview
