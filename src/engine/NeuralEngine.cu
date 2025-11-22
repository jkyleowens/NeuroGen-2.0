#include "engine/NeuralEngine.h"
#include "engine/kernels/LIF_Update.cuh"
#include "engine/kernels/SpMV_Input.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>

namespace neurogen {

// ============================================================================
// NEURAL ENGINE IMPLEMENTATION
// ============================================================================

NeuralEngine::NeuralEngine(int device_id) 
    : device_id_(device_id), is_initialized_(false),
      num_neurons_(0), num_synapses_(0),
      d_input_buffer_(nullptr), d_output_buffer_(nullptr) {
    
    cudaSetDevice(device_id_);
    cudaStreamCreate(&compute_stream_);
}

NeuralEngine::~NeuralEngine() {
    freeMemory();
    cudaStreamDestroy(compute_stream_);
}

bool NeuralEngine::initialize(const NetworkConfig& config) {
    cudaSetDevice(device_id_);
    
    num_neurons_ = config.num_neurons;
    num_synapses_ = config.num_synapses;
    num_inputs_ = config.num_inputs;
    num_outputs_ = config.num_outputs;

    allocateMemory();
    initializeState();
    
    // Initialize Synaptic Matrix (Sparse)
    synaptic_matrix_.initialize(num_neurons_, num_neurons_, num_synapses_);
    
    // Populate with random connections (Phase 1 Baseline)
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> values;
    rows.reserve(num_synapses_);
    cols.reserve(num_synapses_);
    values.reserve(num_synapses_);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_neuron(0, num_neurons_ - 1);
    std::normal_distribution<float> dist_weight(0.1f, 0.05f);

    for (size_t i = 0; i < num_synapses_; ++i) {
        rows.push_back(dist_neuron(rng)); // Post-synaptic
        cols.push_back(dist_neuron(rng)); // Pre-synaptic
        values.push_back(dist_weight(rng));
    }

    synaptic_matrix_.setFromTriplets(rows, cols, values);
    
    is_initialized_ = true;
    return true;
}

void NeuralEngine::allocateMemory() {
    // TensorNetwork
    network_state_.num_neurons = num_neurons_;
    cudaMalloc(&network_state_.d_voltage, num_neurons_ * sizeof(float));
    cudaMalloc(&network_state_.d_adaptation, num_neurons_ * sizeof(float));
    cudaMalloc(&network_state_.d_threshold, num_neurons_ * sizeof(float));
    cudaMalloc(&network_state_.d_spikes, num_neurons_ * sizeof(uint8_t));
    cudaMalloc(&network_state_.d_input_current, num_neurons_ * sizeof(float));
    cudaMalloc(&network_state_.d_last_spike_time, num_neurons_ * sizeof(float));
    
    // Buffers
    cudaMalloc(&d_input_buffer_, num_neurons_ * sizeof(float));
    cudaMalloc(&d_output_buffer_, num_neurons_ * sizeof(float));
}

void NeuralEngine::freeMemory() {
    if (network_state_.d_voltage) cudaFree(network_state_.d_voltage);
    if (network_state_.d_adaptation) cudaFree(network_state_.d_adaptation);
    if (network_state_.d_threshold) cudaFree(network_state_.d_threshold);
    if (network_state_.d_spikes) cudaFree(network_state_.d_spikes);
    if (network_state_.d_input_current) cudaFree(network_state_.d_input_current);
    if (network_state_.d_last_spike_time) cudaFree(network_state_.d_last_spike_time);
    
    if (d_input_buffer_) cudaFree(d_input_buffer_);
    if (d_output_buffer_) cudaFree(d_output_buffer_);
}

void NeuralEngine::initializeState() {
    // Initialize to resting potentials
    std::vector<float> initial_v(num_neurons_, -65.0f);
    cudaMemcpy(network_state_.d_voltage, initial_v.data(), num_neurons_ * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemset(network_state_.d_adaptation, 0, num_neurons_ * sizeof(float));
    cudaMemset(network_state_.d_spikes, 0, num_neurons_ * sizeof(uint8_t));
    cudaMemset(network_state_.d_input_current, 0, num_neurons_ * sizeof(float));
}

void NeuralEngine::update(float dt, float reward_signal) {
    if (!is_initialized_) return;

    static float current_time = 0.0f;
    current_time += dt;

    // Phase 2 Compute Pipeline
    
    // 1. Compute Synaptic Inputs (SpMV)
    // Uses optimized kernel from kernels/SpMV_Input.cu
    // Re-uses d_output_buffer_ as temporary float spike buffer
    neurogen::kernels::computeSynapticInputs(
        synaptic_matrix_,
        network_state_.d_spikes,
        network_state_.d_input_current,
        d_output_buffer_, // Temp float spikes
        num_neurons_,
        compute_stream_
    );

    // 2. Update Neurons (LIF + kWTA)
    // Uses optimized fused kernel from kernels/LIF_Update.cu
    neurogen::kernels::LIFParams lif_params;
    lif_params.alpha = 0.95f;
    lif_params.beta = 0.90f;
    lif_params.delta = 1.0f;
    lif_params.v_thresh = -50.0f;
    lif_params.v_reset = -65.0f;
    lif_params.k_winners = 25; // Top 10% of 256 block
    
    neurogen::kernels::launchLIFUpdate(
        network_state_.d_voltage,
        network_state_.d_adaptation,
        network_state_.d_spikes,
        network_state_.d_input_current,
        network_state_.d_last_spike_time,
        current_time,
        num_neurons_,
        dt,
        lif_params,
        compute_stream_
    );
    
    // 3. Reset Input Current for next step
    // Essential for "Emulation" vs "Simulation" - we clear accumulator
    cudaMemsetAsync(network_state_.d_input_current, 0, num_neurons_ * sizeof(float), compute_stream_);
    
    cudaStreamSynchronize(compute_stream_);
}

void NeuralEngine::processInput(const std::vector<float>& inputs) {
    if (!is_initialized_) return;
    
    size_t count = std::min(inputs.size(), num_neurons_);
    // Copy directly to input current accumulator
    cudaMemcpyAsync(network_state_.d_input_current, inputs.data(), count * sizeof(float), cudaMemcpyHostToDevice, compute_stream_);
}

std::vector<float> NeuralEngine::getNeuronOutputs() {
    std::vector<float> outputs(num_outputs_);
    
    std::vector<uint8_t> host_spikes(num_outputs_);
    cudaMemcpy(host_spikes.data(), network_state_.d_spikes, num_outputs_ * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    for(size_t i=0; i<num_outputs_; ++i) {
        outputs[i] = (float)host_spikes[i];
    }
    
    return outputs;
}

// Backward compatibility stubs
std::vector<GPUNeuronState> NeuralEngine::getNeuronStates() const {
    std::vector<float> v(num_neurons_);
    cudaMemcpy(v.data(), network_state_.d_voltage, num_neurons_ * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<GPUNeuronState> states(num_neurons_);
    for(size_t i=0; i<num_neurons_; ++i) {
        states[i].V = v[i];
    }
    return states;
}

std::vector<GPUSynapse> NeuralEngine::getSynapseStates() const {
    return std::vector<GPUSynapse>(num_synapses_);
}

bool NeuralEngine::setNeuronStates(const std::vector<GPUNeuronState>& states) {
    if (states.size() != num_neurons_) return false;

    std::vector<float> v(num_neurons_);
    std::vector<float> a(num_neurons_);

    for(size_t i=0; i<num_neurons_; ++i) {
        v[i] = states[i].V;
        a[i] = states[i].u;
    }

    cudaMemcpy(network_state_.d_voltage, v.data(), num_neurons_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(network_state_.d_adaptation, a.data(), num_neurons_ * sizeof(float), cudaMemcpyHostToDevice);
    
    return true;
}

bool NeuralEngine::setSynapseStates(const std::vector<GPUSynapse>& synapses) {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> weights;
    
    rows.reserve(synapses.size());
    cols.reserve(synapses.size());
    weights.reserve(synapses.size());
    
    for(const auto& s : synapses) {
        if (s.active) {
             rows.push_back(s.post_neuron_idx);
             cols.push_back(s.pre_neuron_idx);
             weights.push_back(s.weight);
        }
    }
    
    synaptic_matrix_.setFromTriplets(rows, cols, weights);
    return true;
}

} // namespace neurogen
