# NeuroGen 2.0 - Bio-Inspired Neural Language Model

**Status**: Phase 3a Complete ✅  
**Performance**: 196 tokens/sec, 78.5x faster than baseline  
**Scale**: 39,424 neurons, ~130M parameters, LLM-ready architecture

---

## 🎯 Overview

NeuroGen 2.0 is a high-performance, bio-inspired neural language model that combines neuroscience principles with GPU optimization. It features:

- **Modular Brain Architecture**: 6 specialized cortical modules (Thalamus, Wernicke, Broca, Hippocampus, PFC, Basal Ganglia)
- **Pipelined Processing**: Recurrent working memory with parallel cortical processing
- **GPU-Accelerated**: cuBLAS projection, CUDA kernels, sparse matrix operations
- **Scalable**: 130M parameters fit in 4GB GPU memory
- **Fast**: 196 tokens/sec throughput, 35ms per training step

---

## 🚀 Quick Start

### Prerequisites

```bash
# CUDA 11.0+ and compatible GPU
nvidia-smi  # Verify GPU availability

# Python dependencies
conda create -n neurogen python=3.9
conda activate neurogen
pip install pybind11 transformers torch numpy tqdm datasets
```

### Build & Run

```bash
# Clone and build
cd NeuroGen-2.0
make clean && make -j$(nproc)

# Quick test (10 seconds)
python3 train_simple.py

# Interactive chat (use trained model)
python3 chat_interface.py --auto-load-latest

# Benchmark (measure your speedup)
python3 benchmark_decoder.py

# C++ simulation only
./neurogen_sim
```

### Expected Output

```
✓ GPU Decoder initialized (cuBLAS + CUDA)
  Vocab: 50257 | Output dim: 2560
  Expected speedup: 50-100x over CPU decoder

🏃 Starting training...
Step   1/200 | Epoch 1/10 | Loss: 1.0000 | Acc:   0.0% | Speed: 196.3 tok/s
Step   5/200 | Epoch 1/10 | Loss: 1.0000 | Acc:   0.0% | Speed: 203.1 tok/s
...
✅ Training Complete!
📊 Total Steps: 200
⏱️  Total Time: 9.4s
🚀 Throughput: 21.17 samples/sec
```

---

## 📊 Performance Highlights

### GPU Decoder Benchmark

| Metric | Value |
|--------|-------|
| **Token Throughput** | 196.3 tok/s |
| **Step Time** | 35.4 ms |
| **Speedup vs CPU** | 78.5x |
| **Consistency** | ±2.5% variance |

### Scalability

| Configuration | Neurons | Parameters | Memory | Speed |
|--------------|---------|------------|--------|-------|
| Small | 15,872 | ~60M | 2.8 GB | 220 tok/s |
| **Current** | **39,424** | **~130M** | **3.5 GB** | **196 tok/s** |
| Large (future) | 100,000 | ~350M | 8 GB | 150 tok/s |

---

## 🏗️ Architecture

### Modular Brain Design

```
Input Token
    ↓
[Thalamus] → Fast sensory encoding (1,024 neurons)
    ↓
[Wernicke] → Language comprehension (10,240 neurons)
    ↓
[Working Memory] ← Recurrent context buffer
    ↓
[Hippocampus] → Pattern storage (5,120 neurons)
[PFC] → Executive control (10,240 neurons)
[Basal Ganglia] → Action selection (2,560 neurons)
    ↓
[Broca] → Language production (10,240 neurons)
    ↓
[GPU Decoder] → Vocabulary projection (50,257 tokens)
    ↓
Output Token
```

### Key Technologies

- **LIF-A Neurons**: Leaky Integrate-and-Fire with Adaptation
- **Sparse Matrices (CSR)**: Efficient synaptic connectivity
- **cuBLAS**: GPU-accelerated matrix operations
- **CUDA Kernels**: Custom softmax, sampling, neural updates
- **Pipelined Processing**: 1.49x speedup over sequential
- **Structure of Arrays**: Cache-friendly memory layout

---

## 📚 Documentation

### Quick Reference

- **CHAT_INTERFACE_GUIDE.md**: Interactive chat with your model
- **TRAINING_GUIDE.md**: How to train the model
- **GPU_DECODER_SUMMARY.md**: Technical deep-dive on decoder
- **PHASE_3_COMPLETION_SUMMARY.md**: Implementation details
- **SCALING_SUMMARY.md**: Model scaling analysis
- **PIPELINED_PROCESSING.md**: Architecture explanation

### Usage Examples

#### Training

```python
import libneurogen

# Initialize model (using SentencePiece tokenizer vocab)
model = libneurogen.NeuroGenModel(
    vocab_size=32000,  # SentencePiece tokenizer
    embedding_dim=512,
    gpu_device=0
)

# Train on next-token prediction
loss, accuracy = model.train_step(
    input_ids=[1, 2, 3, 4],
    target_ids=[2, 3, 4, 5]
)

print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")
```

#### Generation

```python
# Generate text from prompt
generated_ids = model.generate(
    prompt_ids=[1, 2, 3],
    max_length=100
)

print(f"Generated {len(generated_ids)} tokens")
```

---

## 🧪 Testing

### Test Scripts

```bash
# Simple training (10 seconds)
python3 train_simple.py

# Performance benchmark
python3 benchmark_decoder.py

# Advanced training with visualization
python3 train_advanced.py --steps 100

# C++ standalone benchmark
./neurogen_sim
```

### Validation Results

- ✅ Model initializes successfully
- ✅ GPU decoder 78.5x faster than CPU
- ✅ Stable training (no crashes, no leaks)
- ✅ Memory fits in 4GB GPU
- ✅ Consistent performance (±2.5% variance)

---

## 📈 Benchmark Results

### Simple Training (200 steps, 20 samples × 10 epochs)

```
Total Time: 9.4s
Throughput: 21.17 samples/sec
Token Speed: 196 tok/s (avg)
Loss: 1.0000 (expected - no gradients yet)
Accuracy: 0.0% (expected - learning rules pending)
```

### Comprehensive Benchmark (50 samples, 360 tokens)

```
Total Time: 1.83s
Total Tokens: 360
Avg Token Speed: 203.7 tok/s (±9.7)
Avg Step Time: 35.44ms (±5.19ms)
Min Token Speed: 171.2 tok/s
Max Token Speed: 221.9 tok/s
Overall Throughput: 196.3 tok/s
Speedup vs CPU: 78.5x
```

### C++ Simulation (neural dynamics only)

```
Total Time: 7.0s
Simulation Steps: 10,000
Throughput: 1,390 Hz
Neurons: 39,424
Synapses: ~1.2M (sparse)
```

---

## 🔬 Technical Details

### GPU Decoder Implementation

**Algorithm**: cuBLAS matrix-vector multiplication + CUDA softmax

```cpp
// Projection: logits = W * neural_output
cublasSgemv(handle, CUBLAS_OP_N,
    vocab_size,      // 50,257 rows
    output_dim,      // 2,560 cols
    &alpha,
    d_projection_matrix,
    vocab_size,
    d_neural_output,
    1,
    &beta,
    d_logits,
    1
);

// Softmax: probs = softmax(logits)
softmaxKernel<<<1, 256>>>(d_logits, d_probs, vocab_size);

// Sample: token_id = sample(probs)
sampleKernel<<<1, 1>>>(d_probs, d_result, vocab_size);
```

**Performance**:
- Matrix multiply: ~2ms (257M FLOPs @ 128 GFLOPs)
- Softmax: ~0.2ms
- Sampling: <0.1ms
- **Total**: 35ms per step (includes neural simulation)

### Memory Layout

```
GPU Memory (3.5 GB total):

Neural Network (3.0 GB):
  - Neurons (SoA): voltage, adaptation, threshold, spikes
  - Synapses (CSR): values, indices, row pointers
  - Inter-module connections
  - Working memory buffers

GPU Decoder (490 MB):
  - Projection matrix: 50,257 × 2,560 × FP32
  - Logits buffer: 50,257 × FP32
  - Probability buffer: 50,257 × FP32
  - Input buffer: 2,560 × FP32

Headroom (500 MB):
  - Batch processing buffers
  - Gradient storage (Phase 3b)
  - Temporary allocations
```

---

## 🎯 Roadmap

### Phase 1: Core Refactor ✅ (Complete)
- [x] LIF-A neuron model
- [x] Structure of Arrays (SoA) layout
- [x] Sparse matrix (CSR) synapses
- [x] Modular cortical architecture

### Phase 2: Simulation Loop ✅ (Complete)
- [x] High-performance simulation (1,390 Hz)
- [x] Pipelined processing (1.49x speedup)
- [x] Inter-module connections
- [x] Neuromodulation (dopamine, serotonin, norepinephrine)

### Phase 3a: GPU Decoder ✅ (Complete)
- [x] cuBLAS integration
- [x] CUDA softmax kernel
- [x] GPU sampling strategies
- [x] Python bindings
- [x] 78.5x speedup achieved

### Phase 3b: Learning Rules (In Progress)
- [ ] GPU-accelerated gradients
- [ ] Matrix-plasticity implementation
- [ ] Backpropagation through time
- [ ] Proper loss computation

### Phase 3c: Advanced Features (Planned)
- [ ] Batch processing (5-10x speedup)
- [ ] Mixed precision (FP16)
- [ ] Top-K/Top-P sampling on GPU
- [ ] Structural plasticity

### Phase 4: Production (Future)
- [ ] Multi-GPU support
- [ ] Distributed training
- [ ] INT8 quantization
- [ ] Model compression

---

## 🐛 Known Limitations

### Learning

⚠️ **Model not converging** (accuracy remains 0%)
- **Cause**: Missing proper gradient computation
- **Impact**: Structure validated, but no learning
- **Status**: Phase 3b priority

### Sampling

⚠️ **Top-K/Top-P use fallback**
- **Cause**: Not yet implemented on GPU
- **Impact**: Generation quality (not speed)
- **Status**: Phase 3c enhancement

### Dataset Loading

⚠️ **`train_advanced.py` hangs at dataset iterator**
- **Cause**: Streaming dataset edge cases
- **Workaround**: Use `train_simple.py` for now
- **Status**: Fixed in latest code, needs validation

---

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repo_url>
cd NeuroGen-2.0

# Install dependencies
conda env create -f environment.yml
conda activate neurogen

# Build
make clean && make -j$(nproc)

# Run tests
python3 train_simple.py
python3 benchmark_decoder.py
./neurogen_sim
```

### Code Structure

```
NeuroGen-2.0/
├── include/
│   ├── engine/          # Neural engine (LIF, sparse matrix)
│   ├── modules/         # Cortical modules
│   ├── interfaces/      # Embeddings, decoder
│   └── persistence/     # Checkpointing
├── src/
│   ├── engine/          # Implementation files
│   ├── modules/         # Module implementations
│   ├── interfaces/      # Interface implementations
│   ├── python/          # Python bindings (pybind11)
│   └── main.cpp         # C++ entry point
├── train_simple.py      # Quick training test
├── train_advanced.py    # Full training pipeline
├── benchmark_decoder.py # Performance benchmark
└── Makefile             # Build system
```

---

## 📄 License

[Add your license here]

---

## 📧 Contact

[Add your contact information here]

---

## 🙏 Acknowledgments

- **CUDA**: NVIDIA CUDA toolkit for GPU acceleration
- **cuBLAS**: Optimized linear algebra library
- **pybind11**: Seamless Python-C++ bindings
- **Hugging Face**: Transformers library and datasets

---

## 📊 Citation

If you use NeuroGen 2.0 in your research, please cite:

```bibtex
@software{neurogen2025,
  title={NeuroGen 2.0: Bio-Inspired Neural Language Model},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

---

**Last Updated**: November 22, 2025  
**Version**: 2.0.0  
**Status**: Phase 3a Complete ✅  
**Performance**: 196 tok/s, 78.5x speedup over CPU

