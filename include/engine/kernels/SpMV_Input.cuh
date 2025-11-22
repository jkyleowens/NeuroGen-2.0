#pragma once
#include "engine/SparseMatrix.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace neurogen {
namespace kernels {

/**
 * @brief Computes synaptic inputs using Sparse Matrix-Vector Multiplication.
 * 
 * Optimized for neural network connectivity:
 * Input Current += WeightMatrix * Spikes
 * 
 * @param matrix The sparse weight matrix (BSR/CSR)
 * @param d_spikes Spike array (uint8_t) from neurons
 * @param d_input_current Output current accumulator
 * @param d_temp_float_spikes Temporary buffer for float conversion
 * @param num_neurons Total neurons
 * @param stream CUDA stream
 */
void computeSynapticInputs(
    neurogen::SparseMatrix& matrix,
    const uint8_t* d_spikes,
    float* d_input_current,
    float* d_temp_float_spikes,
    int num_neurons,
    cudaStream_t stream
);

} // namespace kernels
} // namespace neurogen

