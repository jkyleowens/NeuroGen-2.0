/**
 * @file ALIFKernels.cuh
 * @brief CUDA Kernels for Adaptive LIF Neuron Updates
 * 
 * High-performance CUDA kernels for ALIF neuron dynamics.
 * Optimized for:
 * - Coalesced memory access (SoA layout)
 * - Minimal FLOPs per neuron (5 vs 40 for Izhikevich)
 * - Warp-level efficiency
 * 
 * @version 2.0-Cortical
 * @date November 29, 2025
 */

#ifndef ALIF_KERNELS_CUH
#define ALIF_KERNELS_CUH

#include "ALIFNeuron.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace neurogen {
namespace cortical {
namespace kernels {

// ============================================================================
// KERNEL LAUNCH CONFIGURATION
// ============================================================================

constexpr int ALIF_BLOCK_SIZE = 256;  // Threads per block
constexpr int ALIF_WARPS_PER_BLOCK = ALIF_BLOCK_SIZE / 32;

// Calculate optimal grid size
inline int getALIFGridSize(int num_neurons) {
    return (num_neurons + ALIF_BLOCK_SIZE - 1) / ALIF_BLOCK_SIZE;
}

// ============================================================================
// BASIC ALIF UPDATE KERNEL
// ============================================================================

/**
 * @brief Main ALIF neuron update kernel
 * 
 * Discrete-time ALIF dynamics:
 *   V[t+1] = α * V[t] + (1-α) * V_rest + I[t] * R
 *   if V > θ: spike, V = V_reset, A += Δ_adapt
 *   A[t+1] = β * A[t]
 *   θ[t+1] = θ_base + A[t+1]
 * 
 * Total: ~5 FLOPs per neuron (vs 40+ for Izhikevich)
 * 
 * @param voltage       Membrane potentials [num_neurons]
 * @param adaptation    Adaptation values [num_neurons]
 * @param threshold     Current thresholds [num_neurons]
 * @param current       Input currents [num_neurons]
 * @param last_spike    Last spike times [num_neurons]
 * @param refractory    Remaining refractory time [num_neurons]
 * @param spiked        Spike output flags [num_neurons]
 * @param spike_count   Cumulative spike counts [num_neurons]
 * @param firing_rate   Exponential average firing rate [num_neurons]
 * @param activity_trace STDP activity trace [num_neurons]
 * @param params        ALIF parameters (in constant memory)
 * @param dt            Timestep (ms)
 * @param current_time  Current simulation time (ms)
 * @param num_neurons   Number of neurons
 */
__attribute__((unused)) static __global__ void alif_update_kernel(
    float* __restrict__ voltage,
    float* __restrict__ adaptation,
    float* __restrict__ threshold,
    const float* __restrict__ current,
    float* __restrict__ last_spike,
    float* __restrict__ refractory,
    uint8_t* __restrict__ spiked,
    uint32_t* __restrict__ spike_count,
    float* __restrict__ firing_rate,
    float* __restrict__ activity_trace,
    const ALIFParameters params,
    float dt,
    float current_time,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // Load state (coalesced reads)
    float V = voltage[idx];
    float A = adaptation[idx];
    float I = current[idx];
    float ref = refractory[idx];
    
    // Precompute decay factors
    float alpha = __expf(-dt / params.tau_mem);
    float beta = __expf(-dt / params.tau_adaptation);
    
    // Check refractory period
    bool in_refractory = (ref > 0.0f);
    
    if (!in_refractory) {
        // Membrane potential update (leaky integration)
        // V = α*V + (1-α)*V_rest + I*R*dt/τ
        float V_inf = params.v_rest + I * params.input_resistance;
        V = alpha * V + (1.0f - alpha) * V_inf;
    }
    
    // Compute current threshold
    float theta = params.v_threshold_base + A;
    
    // Spike detection
    bool spike = (!in_refractory) && (V >= theta);
    
    if (spike) {
        // Reset and update adaptation
        V = params.v_reset;
        A += params.adaptation_increment;
        ref = params.refractory_period;
        
        // Update spike count
        atomicAdd(&spike_count[idx], 1);
        
        // Record spike time
        last_spike[idx] = current_time;
    }
    
    // Decay adaptation
    A *= beta;
    
    // Update refractory timer
    ref = fmaxf(0.0f, ref - dt);
    
    // Update firing rate (exponential moving average)
    float rate_alpha = __expf(-dt / 100.0f);  // 100ms time constant
    float fr = firing_rate[idx];
    fr = rate_alpha * fr + (spike ? (1.0f - rate_alpha) * (1000.0f / dt) : 0.0f);
    
    // Update activity trace (for STDP)
    float trace_alpha = __expf(-dt / 20.0f);  // 20ms STDP window
    float trace = activity_trace[idx];
    trace = trace_alpha * trace + (spike ? 1.0f : 0.0f);
    
    // Store state (coalesced writes)
    voltage[idx] = V;
    adaptation[idx] = A;
    threshold[idx] = theta;
    refractory[idx] = ref;
    spiked[idx] = spike ? 1 : 0;
    firing_rate[idx] = fr;
    activity_trace[idx] = trace;
}

// ============================================================================
// COMPARTMENTAL ALIF UPDATE KERNEL
// ============================================================================

/**
 * @brief Compartmental ALIF update for pyramidal cells
 * 
 * Implements dendritic computation with 3 compartments:
 * - Soma: Integration and spike generation
 * - Basal: Feedforward input (bottom-up)
 * - Apical: Feedback input (top-down context)
 * 
 * Coupling: V_soma receives current from basal and apical via conductances
 */
__attribute__((unused)) static __global__ void compartmental_alif_update_kernel(
    // Somatic state
    float* __restrict__ v_soma,
    float* __restrict__ adaptation,
    float* __restrict__ threshold,
    float* __restrict__ last_spike,
    float* __restrict__ refractory,
    uint8_t* __restrict__ spiked,
    uint32_t* __restrict__ spike_count,
    float* __restrict__ firing_rate,
    float* __restrict__ activity_trace,
    // Dendritic state
    float* __restrict__ v_basal,
    float* __restrict__ v_apical,
    // Input currents
    const float* __restrict__ i_soma,
    const float* __restrict__ i_basal,
    const float* __restrict__ i_apical,
    // Calcium (for learning)
    float* __restrict__ ca_soma,
    float* __restrict__ ca_basal,
    float* __restrict__ ca_apical,
    // Parameters
    const ALIFParameters params,
    float g_basal,      // Basal-soma coupling conductance
    float g_apical,     // Apical-soma coupling conductance
    float dt,
    float current_time,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // Load state
    float Vs = v_soma[idx];
    float Vb = v_basal[idx];
    float Va = v_apical[idx];
    float A = adaptation[idx];
    float Is = i_soma[idx];
    float Ib = i_basal[idx];
    float Ia = i_apical[idx];
    float ref = refractory[idx];
    
    // Decay factors
    float alpha = __expf(-dt / params.tau_mem);
    float alpha_dend = __expf(-dt / (params.tau_mem * 2.0f));  // Slower for dendrites
    float beta = __expf(-dt / params.tau_adaptation);
    float alpha_ca = __expf(-dt / 50.0f);  // 50ms calcium decay
    
    bool in_refractory = (ref > 0.0f);
    
    // === DENDRITIC COMPARTMENT UPDATES ===
    // V_dend = α * V_dend + (1-α) * V_rest + I_dend * R
    Vb = alpha_dend * Vb + (1.0f - alpha_dend) * params.v_rest + Ib * params.input_resistance * 0.5f;
    Va = alpha_dend * Va + (1.0f - alpha_dend) * params.v_rest + Ia * params.input_resistance * 0.3f;
    
    // === SOMATIC COMPARTMENT UPDATE ===
    if (!in_refractory) {
        // Current from dendrites
        float I_from_basal = g_basal * (Vb - Vs);
        float I_from_apical = g_apical * (Va - Vs);
        
        // Total current
        float I_total = Is + I_from_basal + I_from_apical;
        
        // Leaky integration
        float V_inf = params.v_rest + I_total * params.input_resistance;
        Vs = alpha * Vs + (1.0f - alpha) * V_inf;
    }
    
    // Threshold
    float theta = params.v_threshold_base + A;
    
    // Spike detection
    bool spike = (!in_refractory) && (Vs >= theta);
    
    if (spike) {
        Vs = params.v_reset;
        A += params.adaptation_increment;
        ref = params.refractory_period;
        atomicAdd(&spike_count[idx], 1);
        last_spike[idx] = current_time;
    }
    
    // Decay adaptation
    A *= beta;
    
    // Update refractory
    ref = fmaxf(0.0f, ref - dt);
    
    // === CALCIUM DYNAMICS (for STDP) ===
    float Ca_s = ca_soma[idx];
    float Ca_b = ca_basal[idx];
    float Ca_a = ca_apical[idx];
    
    // Calcium decays, spikes add to somatic calcium
    Ca_s = alpha_ca * Ca_s + (spike ? 1.0f : 0.0f);
    Ca_b = alpha_ca * Ca_b;
    Ca_a = alpha_ca * Ca_a;
    
    // === FIRING RATE AND TRACE ===
    float rate_alpha = __expf(-dt / 100.0f);
    float fr = firing_rate[idx];
    fr = rate_alpha * fr + (spike ? (1.0f - rate_alpha) * (1000.0f / dt) : 0.0f);
    
    float trace_alpha = __expf(-dt / 20.0f);
    float trace = activity_trace[idx];
    trace = trace_alpha * trace + (spike ? 1.0f : 0.0f);
    
    // Store state
    v_soma[idx] = Vs;
    v_basal[idx] = Vb;
    v_apical[idx] = Va;
    adaptation[idx] = A;
    threshold[idx] = theta;
    refractory[idx] = ref;
    spiked[idx] = spike ? 1 : 0;
    firing_rate[idx] = fr;
    activity_trace[idx] = trace;
    ca_soma[idx] = Ca_s;
    ca_basal[idx] = Ca_b;
    ca_apical[idx] = Ca_a;
}

// ============================================================================
// INITIALIZATION KERNELS
// ============================================================================

/**
 * @brief Initialize ALIF neurons to resting state
 */
__attribute__((unused)) static __global__ void alif_init_kernel(
    float* __restrict__ voltage,
    float* __restrict__ adaptation,
    float* __restrict__ threshold,
    float* __restrict__ current,
    float* __restrict__ last_spike,
    float* __restrict__ refractory,
    uint8_t* __restrict__ spiked,
    uint32_t* __restrict__ spike_count,
    float* __restrict__ firing_rate,
    float* __restrict__ activity_trace,
    const ALIFParameters params,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    voltage[idx] = params.v_rest;
    adaptation[idx] = 0.0f;
    threshold[idx] = params.v_threshold_base;
    current[idx] = 0.0f;
    last_spike[idx] = -1000.0f;  // Long ago
    refractory[idx] = 0.0f;
    spiked[idx] = 0;
    spike_count[idx] = 0;
    firing_rate[idx] = 0.0f;
    activity_trace[idx] = 0.0f;
}

/**
 * @brief Reset input currents to zero
 */
__attribute__((unused)) static __global__ void reset_currents_kernel(
    float* __restrict__ current,
    int num_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    current[idx] = 0.0f;
}

// ============================================================================
// UTILITY KERNELS
// ============================================================================

/**
 * @brief Count spikes (reduction kernel)
 */
__attribute__((unused)) static __global__ void count_spikes_kernel(
    const uint8_t* __restrict__ spiked,
    int* __restrict__ spike_count,
    int num_neurons
) {
    // Shared memory for block-level reduction
    __shared__ int block_sum;
    if (threadIdx.x == 0) block_sum = 0;
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;
    
    if (idx < num_neurons) {
        local_count = spiked[idx];
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    
    // First thread in warp adds to block sum
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&block_sum, local_count);
    }
    __syncthreads();
    
    // First thread adds block sum to global
    if (threadIdx.x == 0) {
        atomicAdd(spike_count, block_sum);
    }
}

/**
 * @brief Compute mean firing rate (reduction)
 */
__attribute__((unused)) static __global__ void mean_firing_rate_kernel(
    const float* __restrict__ firing_rate,
    float* __restrict__ mean_rate,
    int num_neurons
) {
    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    
    if (idx < num_neurons) {
        local_sum = firing_rate[idx];
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&block_sum, local_sum);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(mean_rate, block_sum / num_neurons);
    }
}

// ============================================================================
// HOST-SIDE WRAPPER FUNCTIONS
// ============================================================================

/**
 * @brief Update all ALIF neurons
 */
inline cudaError_t launchALIFUpdate(
    ALIFNeuronArrays& neurons,
    float dt,
    float current_time,
    cudaStream_t stream = 0
) {
    int grid_size = getALIFGridSize(neurons.num_neurons);
    
    alif_update_kernel<<<grid_size, ALIF_BLOCK_SIZE, 0, stream>>>(
        neurons.d_voltage,
        neurons.d_adaptation,
        neurons.d_threshold,
        neurons.d_current,
        neurons.d_last_spike_time,
        neurons.d_refractory_remaining,
        neurons.d_spiked,
        neurons.d_spike_count,
        neurons.d_firing_rate,
        neurons.d_activity_trace,
        neurons.params,
        dt,
        current_time,
        neurons.num_neurons
    );
    
    return cudaGetLastError();
}

/**
 * @brief Initialize ALIF neurons
 */
inline cudaError_t launchALIFInit(
    ALIFNeuronArrays& neurons,
    cudaStream_t stream = 0
) {
    int grid_size = getALIFGridSize(neurons.num_neurons);
    
    alif_init_kernel<<<grid_size, ALIF_BLOCK_SIZE, 0, stream>>>(
        neurons.d_voltage,
        neurons.d_adaptation,
        neurons.d_threshold,
        neurons.d_current,
        neurons.d_last_spike_time,
        neurons.d_refractory_remaining,
        neurons.d_spiked,
        neurons.d_spike_count,
        neurons.d_firing_rate,
        neurons.d_activity_trace,
        neurons.params,
        neurons.num_neurons
    );
    
    return cudaGetLastError();
}

/**
 * @brief Reset input currents
 */
inline cudaError_t launchResetCurrents(
    float* d_currents,
    int num_neurons,
    cudaStream_t stream = 0
) {
    int grid_size = getALIFGridSize(num_neurons);
    reset_currents_kernel<<<grid_size, ALIF_BLOCK_SIZE, 0, stream>>>(
        d_currents, num_neurons
    );
    return cudaGetLastError();
}

} // namespace kernels
} // namespace cortical
} // namespace neurogen

#endif // ALIF_KERNELS_CUH
