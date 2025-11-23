# Voltage-Based Readout Fix

## Problem: Token Repetition Due to Sparse Spike-Based Output

### Root Cause

The neural network was using **spike-based readout** in `NeuralEngine::getNeuronOutputs()`, which read binary spike events (0 or 1) from the `d_spikes` buffer. This resulted in:

1. **Sparse output vectors** - Mostly zeros since neurons don't spike every timestep
2. **Bias-dominated decoding** - When neural_output is mostly 0, the decoder's logit calculation becomes:
   ```cpp
   logit ≈ projection_bias[i]  // Matrix multiplication term ≈ 0
   ```
3. **Repetitive token generation** - The decoder outputs the same token (highest bias) regardless of neural state
4. **Pattern: AAAAAAA → BBBBBBB** - Eventually noise causes a switch, then locks into new token

### Solution: Voltage-Based Readout

Changed from binary spikes to **continuous membrane potential** (voltage), providing a rich signal that varies with neural activity.

## Implementation

### File: `src/engine/NeuralEngine.cu`

**Before (Lines 162-173):**
```cpp
std::vector<float> NeuralEngine::getNeuronOutputs() {
    std::vector<float> outputs(num_outputs_);

    std::vector<uint8_t> host_spikes(num_outputs_);
    cudaMemcpy(host_spikes.data(), network_state_.d_spikes,
               num_outputs_ * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    for(size_t i=0; i<num_outputs_; ++i) {
        outputs[i] = (float)host_spikes[i];  // ❌ Binary: 0 or 1
    }

    return outputs;
}
```

**After:**
```cpp
std::vector<float> NeuralEngine::getNeuronOutputs() {
    std::vector<float> outputs(num_outputs_);

    // ✅ VOLTAGE-BASED READOUT (continuous signal)
    // Read membrane potentials instead of binary spikes
    std::vector<float> host_voltages(num_outputs_);
    cudaMemcpy(host_voltages.data(), network_state_.d_voltage,
               num_outputs_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert voltage to activity signal
    // Voltage range: -65mV (rest) to ~+30mV (spike peak)
    // Map to [0, 1] range with sigmoid-like function
    for(size_t i=0; i<num_outputs_; ++i) {
        float v = host_voltages[i];

        // Shift so resting potential (-65) maps to 0
        float shifted = v + 65.0f;  // Now: 0 (rest) to 95 (peak)

        // Rectify to remove sub-threshold activity
        float rectified = std::max(0.0f, shifted);

        // Normalize to [0, 1] with sigmoid activation
        // Center around 10mV above rest (typical active neurons)
        float normalized = 1.0f / (1.0f + std::exp(-(rectified - 10.0f) * 0.1f));

        outputs[i] = normalized;
    }

    return outputs;
}
```

### File: `include/engine/NeuralEngine.h`

Updated documentation to reflect voltage-based readout:
```cpp
/**
 * @brief Retrieve voltage-based activity of output neurons
 * Returns continuous signal (0-1) based on membrane potential
 */
std::vector<float> getNeuronOutputs();
```

## Signal Processing Pipeline

The voltage-based readout applies biologically-inspired transformations:

1. **Voltage Input**: Raw membrane potential (-65 to +30 mV)
2. **Shift**: Move resting potential to 0 (add 65)
3. **Rectification**: Remove sub-threshold activity (max with 0)
4. **Sigmoid**: Smooth activation centered around 10mV above rest
5. **Output**: Continuous signal in [0, 1] range

This provides:
- **Continuous variability** based on neural state
- **Biological realism** - activity proportional to depolarization
- **Rich decoder input** - matrix projection has meaningful signal
- **Token diversity** - different neural states → different tokens

## Expected Impact

### Before Fix:
- Output vector: `[0, 1, 0, 0, 0, 1, 0, 0, ...]` (sparse, binary)
- Decoder logits: Dominated by bias
- Token distribution: Peaked at highest bias token
- Generation: Repetitive (AAAA → BBBB → CCCC)

### After Fix:
- Output vector: `[0.23, 0.67, 0.12, 0.89, 0.45, ...]` (dense, continuous)
- Decoder logits: Vary with neural state
- Token distribution: Spread across vocabulary
- Generation: Diverse, context-dependent

## Related Changes

- Added `#include <algorithm>` for `std::max()`
- Existing `#include <cmath>` provides `std::exp()`

## Testing Recommendations

1. **Verify decoder input diversity**: Log output vector statistics
2. **Check token entropy**: Measure distribution spread during generation
3. **Validate generation quality**: Test with various prompts
4. **Monitor convergence**: Ensure training loss decreases properly

## Biological Justification

Real neurons encode information in **rate codes** (firing frequency) and **temporal codes** (spike timing), not just binary spike events. Voltage-based readout captures:

- **Sub-threshold dynamics** - Neurons near threshold but not spiking
- **Population encoding** - Average depolarization of neuron groups
- **Graded responses** - Proportional to input strength

This is analogous to how calcium imaging and voltage-sensitive dyes are used in neuroscience to measure population activity.

## Alternative Approaches (Not Implemented)

1. **Firing rate estimation**: Count spikes over time window (requires temporal buffer)
2. **Synaptic current**: Read `d_input_current` (less informative post-spike)
3. **Population average**: Group neurons and average voltages (implemented via num_outputs < num_neurons)

The current voltage-based approach provides the best balance of biological realism, computational efficiency, and signal richness.
