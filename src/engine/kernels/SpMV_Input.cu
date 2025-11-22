#include "engine/kernels/SpMV_Input.cuh"
#include <cuda_runtime.h>

namespace neurogen {
namespace kernels {

// Conversion kernel (reused from NeuralEngine but centralized here)
__global__ void castSpikesToFloatKernel(const uint8_t* spikes, float* float_spikes, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float_spikes[idx] = (float)spikes[idx];
    }
}

void computeSynapticInputs(
    neurogen::SparseMatrix& matrix,
    const uint8_t* d_spikes,
    float* d_input_current,
    float* d_temp_float_spikes,
    int num_neurons,
    cudaStream_t stream)
{
    // 1. Convert uint8 spikes to float for cuSPARSE compatibility
    int threads = 256;
    int blocks = (num_neurons + threads - 1) / threads;
    
    castSpikesToFloatKernel<<<blocks, threads, 0, stream>>>(
        d_spikes, d_temp_float_spikes, num_neurons
    );

    // 2. Perform SpMV
    // Input_Current += Weights * Float_Spikes
    // alpha=1.0 (add to existing input), beta=1.0 (preserve existing input accumulator)
    // Note: The matrix class handles the CuSparse complexity
    
    // Ideally we should expose the stream to the matrix class.
    // For now, assuming matrix uses default stream or we need to update it.
    // Phase 2 enhancement: ensure stream is passed down if possible, 
    // but SparseMatrix currently doesn't take stream in vectorMultiply.
    // We'll rely on default stream synchronization or update SparseMatrix later.
    // Wait, checking SparseMatrix.h... vectorMultiply doesn't take stream.
    // TODO: Update SparseMatrix to accept stream for full async.
    
    matrix.vectorMultiply(d_temp_float_spikes, d_input_current, 1.0f, 1.0f);
}

} // namespace kernels
} // namespace neurogen

