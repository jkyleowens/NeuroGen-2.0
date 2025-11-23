# NeuroGen 2.0 - Interactive Interface Guide

## Overview

The `interact.py` script provides an interactive command-line interface to chat with your trained NeuroGen model. The model generates multi-token responses and intelligently decides when to stop generation.

## Quick Start

### Basic Chat Mode

```bash
# Start interactive chat (uses latest checkpoint if available)
python3 interact.py

# Start chat with specific checkpoint
python3 interact.py --checkpoint checkpoints/checkpoint_step_8000.bin

# Start with custom generation parameters
python3 interact.py --temperature 0.9 --max-length 100 --top-k 50
```

### Single Prompt Mode

```bash
# Generate response to a single prompt
python3 interact.py --prompt "The future of artificial intelligence is"

# With debug mode (shows token-by-token generation)
python3 interact.py --prompt "Once upon a time" --debug

# Generate longer response
python3 interact.py --prompt "Explain how neural networks work" --max-length 150
```

## Interactive Commands

Once in chat mode, you can use these commands:

| Command | Description |
|---------|-------------|
| `Type your message` | Generate a response from the model |
| `/debug` | Toggle debug mode (show token generation details) |
| `/temp <value>` | Set temperature (0.1-2.0, default: 0.8) |
| `/length <value>` | Set max generation length (1-500, default: 50) |
| `/clear` | Clear conversation history |
| `/quit` or `exit` | Exit the program |

## Generation Parameters

### Temperature
Controls randomness of generation:
- **0.1-0.5**: More focused, deterministic outputs
- **0.7-0.9**: Balanced creativity and coherence
- **1.0-2.0**: More creative, diverse outputs

```bash
python3 interact.py --temperature 0.5  # More deterministic
python3 interact.py --temperature 1.2  # More creative
```

### Max Length
Maximum number of tokens to generate:

```bash
python3 interact.py --max-length 30   # Short responses
python3 interact.py --max-length 200  # Long responses
```

### Top-K Sampling
Only sample from the K most likely tokens:

```bash
python3 interact.py --top-k 10   # Very focused
python3 interact.py --top-k 100  # More diverse
```

### Top-P (Nucleus) Sampling
Sample from smallest set of tokens whose cumulative probability exceeds P:

```bash
python3 interact.py --top-p 0.7  # More focused
python3 interact.py --top-p 0.95 # More diverse
```

## Features

### 1. Multi-Token Generation
The model generates complete thoughts, not just single tokens:
```
You: Tell me about neural networks
NeuroGen: Neural networks are computational models inspired by the human brain...
```

### 2. Intelligent Stopping
The model automatically stops when it detects:
- Sentence-ending punctuation (`.`, `!`, `?`)
- Paragraph breaks (`\n\n`)
- Repeated token patterns (prevents loops)
- EOS (End-of-Sequence) token

### 3. Conversation History
The interface maintains context across the conversation:
```
You: What's the capital of France?
NeuroGen: Paris.

You: What's the population?
NeuroGen: Approximately 2.2 million in the city proper...
```

### 4. Debug Mode
See detailed token generation:
```bash
You: Hello
🧠 Generating...

🔍 Input tokens (5): [259, 884, ...]
📝 Generating response...

Token   1 (35.2ms): ' Hello' (id=259)
Token   2 (33.8ms): '!' (id=0)
Token   3 (34.1ms): ' How' (id=1829)
Token   4 (33.5ms): ' can' (id=476)
Token   5 (34.2ms): ' I' (id=306)
Token   6 (33.9ms): ' help' (id=1037)
Token   7 (34.0ms): ' you' (id=345)
Token   8 (33.7ms): '?' (id=32)

✋ [Natural stopping point detected]

✅ Generated 8 tokens in 0.27s (29.6 tok/s)

📤 Final output: Hello! How can I help you?
```

## Example Sessions

### Example 1: Story Generation

```bash
$ python3 interact.py --checkpoint checkpoints/checkpoint_step_8000.bin --temperature 0.9

You: Write a short story about a robot learning to paint

NeuroGen: In a dusty workshop, Unit-7 discovered an old paintbrush among the spare 
parts. Its sensors analyzed the bristles, confused by their lack of utility. But 
something unexpected happened when it dipped them in color...
```

### Example 2: Question Answering

```bash
$ python3 interact.py --checkpoint checkpoints/checkpoint_step_8000.bin --temperature 0.5

You: What is machine learning?

NeuroGen: Machine learning is a subset of artificial intelligence that enables 
computers to learn and improve from experience without being explicitly programmed. 
It uses algorithms to identify patterns in data and make predictions or decisions 
based on those patterns.
```

### Example 3: Code Generation

```bash
$ python3 interact.py --checkpoint checkpoints/checkpoint_step_8000.bin --temperature 0.7

You: Write a Python function to calculate factorial

NeuroGen: def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

## Advanced Usage

### Batch Processing

Process multiple prompts from a file:

```bash
# Create prompts file
cat > prompts.txt << EOF
What is artificial intelligence?
Explain quantum computing.
Write a haiku about technology.
EOF

# Process each prompt
while IFS= read -r prompt; do
    echo "=== Prompt: $prompt ===" 
    python3 interact.py --prompt "$prompt" --max-length 80
    echo ""
done < prompts.txt
```

### Python API Usage

Use the interface programmatically:

```python
from interact import NeuroGenInterface

# Initialize
interface = NeuroGenInterface(
    checkpoint_path="checkpoints/checkpoint_step_8000.bin"
)

# Configure
interface.temperature = 0.8
interface.max_length = 100

# Generate
response = interface.generate("Tell me a joke", max_new_tokens=50)
print(response)
```

### Integration with Other Tools

```python
import sys
sys.path.insert(0, 'bin')
from interact import NeuroGenInterface

# Create interface
model = NeuroGenInterface()

# Use in your application
def get_ai_response(user_message):
    return model.generate(user_message, max_new_tokens=50)

# Example: Simple chatbot server
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']
    response = get_ai_response(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Troubleshooting

### Issue: "libneurogen not found"

**Solution**: Make sure the library is compiled:
```bash
make clean && make -j$(nproc)
ls -lh bin/libneurogen.so  # Should exist
```

### Issue: "sentencepiece not installed"

**Solution**: Install dependencies:
```bash
pip install sentencepiece
```

### Issue: Generation is too slow

**Causes**:
- CPU-only mode (GPU not detected)
- Large checkpoint loading
- High max_length setting

**Solutions**:
```bash
# Use shorter max_length
python3 interact.py --max-length 30

# Check GPU is detected
nvidia-smi

# Rebuild with GPU support
make clean && make -j$(nproc)
```

### Issue: Repetitive outputs

**Solution**: Adjust sampling parameters:
```bash
# Increase temperature and repetition penalty
python3 interact.py --temperature 1.0 --top-k 50
```

In the Python code, you can also adjust:
```python
interface.repetition_penalty = 1.5  # Default: 1.2
```

### Issue: Incoherent outputs

**Solution**: Lower temperature for more focused generation:
```bash
python3 interact.py --temperature 0.5 --top-p 0.85
```

## Performance Tips

1. **Use the latest checkpoint** for best results:
   ```bash
   ls -lt checkpoints/*.bin | head -1  # Find latest
   python3 interact.py --checkpoint checkpoints/checkpoint_step_9500.bin
   ```

2. **Optimize parameters** for your use case:
   - Creative writing: `--temperature 0.9 --top-p 0.95`
   - Factual responses: `--temperature 0.5 --top-k 20`
   - Balanced: `--temperature 0.7 --top-k 40 --top-p 0.9`

3. **Monitor performance** in debug mode:
   ```bash
   python3 interact.py --debug
   # Look for tokens/sec in output
   ```

4. **Clear conversation history** periodically to avoid context overflow:
   ```
   You: /clear
   ```

## Next Steps

- **Train longer**: More training steps generally improve quality
- **Experiment with parameters**: Find the sweet spot for your use case
- **Fine-tune on specific domains**: Train on domain-specific data
- **Build applications**: Use the Python API to integrate into your projects

## Technical Details

### Stopping Criteria

The model stops generation when:

1. **EOS Token**: Special end-of-sequence token is generated
2. **Punctuation**: Sentence-ending punctuation followed by space (`. `, `! `, `? `)
3. **Paragraph breaks**: Double newline (`\n\n`)
4. **Repetition**: Same 3-token sequence appears twice in a row
5. **Max length**: Reaches `max_new_tokens` limit

### Sampling Algorithm

1. **Temperature scaling**: Divide logits by temperature
2. **Repetition penalty**: Penalize recently generated tokens
3. **Top-K filtering**: Keep only K most likely tokens
4. **Top-P filtering**: Keep smallest set with cumulative probability ≥ P
5. **Sampling**: Sample from filtered distribution

### Context Window

- The interface keeps the last 3 user-assistant exchanges (6 messages)
- Older messages are dropped to stay within context limits
- Use `/clear` to reset conversation history

---

**Last Updated**: November 22, 2024  
**Status**: Production ready  
**Performance**: ~30-50 tokens/sec on GTX 1650
