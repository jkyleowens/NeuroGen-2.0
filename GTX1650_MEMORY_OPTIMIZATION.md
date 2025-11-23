# GTX 1650 Memory Optimization

## Problem
CUDA is running out of memory on GTX 1650 (4GB VRAM). The current model is too large.

## Current Memory Usage Analysis

### Current Configuration
- **Embedding Dim**: 2048 (quadrupled)
- **Vocab Size**: 32000
- **Broca Output Dim**: 32768 (quadrupled)
- **Max Sequence Length**: 512

### Memory Breakdown

1. **Token Embeddings**: 
   - Size: 32000 vocab × 2048 dim × 4 bytes = **256 MB**

2. **GPU Decoder Projection Matrix**:
   - Size: 32768 (Broca) × 32000 (vocab) × 4 bytes = **4,194 MB** (4.1 GB!)
   - This alone exceeds GTX 1650 capacity!

3. **Brain Module Activations**:
   - Approximately **500-800 MB** for all neural modules

4. **Working Memory** (activations, gradients):
   - Approximately **200-400 MB**

**Total Estimated**: ~5+ GB (exceeds 4GB limit!)

## Root Cause

The `output_dim = 32768` in GPUDecoder is causing massive memory usage:
- Projection matrix: 32768 × 32000 × 4 bytes = **4.1 GB**
- This single matrix uses more than the entire GPU memory!

## Solution: Scale Down for GTX 1650

### Target Memory Budget (4GB VRAM)
- Token Embeddings: **128 MB** (25%)
- Decoder Matrix: **1.5 GB** (37.5%)
- Brain Modules: **600 MB** (15%)
- Working Memory: **400 MB** (10%)
- Safety Buffer: **500 MB** (12.5%)
- **Total**: ~3.1 GB (leaves ~900MB headroom)

### New Configuration

```cpp
// For GTX 1650 (4GB VRAM) - Conservative Scaling
vocab_size = 32000
embedding_dim = 768      // Reduced from 2048 (÷2.67)
broca_output_dim = 8192  // Reduced from 32768 (÷4)
max_seq_length = 256     // Reduced from 512 (÷2)
```

### Memory Calculation with New Config

1. **Token Embeddings**: 
   - 32000 × 768 × 4 bytes = **96 MB** ✅

2. **GPU Decoder Matrix**:
   - 8192 × 32000 × 4 bytes = **1,024 MB** (1 GB) ✅

3. **Brain Modules**: 
   - ~600 MB ✅

4. **Working Memory**: 
   - ~400 MB ✅

**Total**: ~2.1 GB (48% safe margin!)

## Implementation

### 1. Update `train_slimpajama.py`
```python
embedding_dim: int = 768  # Optimized for GTX 1650 (4GB)
max_seq_length: int = 256
```

### 2. Update `neurogen_bindings.cpp`
```cpp
gpu_decoder_config.output_dim = 8192; // Optimized for GTX 1650
```

### 3. Update `train_advanced.py` (if needed)
```python
embedding_dim: int = 768
max_seq_length: int = 256
```

## Performance Impact

### Compared to 2048 embedding / 32768 output:
- **Memory**: 5+ GB → 2.1 GB (58% reduction)
- **Speed**: Slightly faster (less data to move)
- **Capacity**: Reduced but still functional for language modeling
- **Fit**: ✅ Comfortably fits on GTX 1650

### Compared to Original 512 embedding / 12288 output:
- **Memory**: Similar or slightly better
- **Capacity**: 50% more than original (768 vs 512)
- **Performance**: Good balance for 4GB GPU

## Alternative Configurations

### Ultra-Conservative (for stability)
```python
embedding_dim: int = 512   # Original size
broca_output_dim: int = 6144  # Half of original 12288
max_seq_length: int = 256
# Total: ~1.5 GB
```

### Balanced (recommended)
```python
embedding_dim: int = 768   # 50% increase over original
broca_output_dim: int = 8192  # 33% decrease from original
max_seq_length: int = 256
# Total: ~2.1 GB
```

### Aggressive (maximum capacity)
```python
embedding_dim: int = 1024  # Doubled from original
broca_output_dim: int = 12288  # Original size
max_seq_length: int = 256
# Total: ~2.8 GB
```

## Recommendation

Use the **Balanced** configuration:
- embedding_dim = 768
- output_dim = 8192
- max_seq_length = 256

This provides:
- ✅ Safe 48% memory margin
- ✅ 50% more capacity than original
- ✅ Stable training
- ✅ Room for future growth

## Verification

After applying changes, verify with:
```bash
nvidia-smi  # Check GPU memory usage
python train_slimpajama.py --test
```

Expected GPU usage: 2.1-2.5 GB (safe for 4GB GPU)
