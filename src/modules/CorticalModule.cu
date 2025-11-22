#include "modules/CorticalModule.h"
#include "engine/NetworkConfig.h"
#include "engine/NeuralEngine.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

CorticalModule::CorticalModule(const Config& config, int gpu_device_id) 
    : config_(config),
      current_dopamine_(0.0f),
      current_serotonin_(0.0f),
      current_inhibition_(config.modulation.inhibition_level),
      signal_mean_(0.0f),
      signal_variance_(0.0f) {
    
    // Instantiate the new Neural Engine (Phase 1 Refactor)
    neural_engine_ = std::make_unique<neurogen::NeuralEngine>(gpu_device_id);
    
    NetworkConfig net_config;
    net_config.num_neurons = config.num_neurons;
    net_config.percent_inhibitory = 0.2f;
    
    // Calculate synapse count based on fanout
    size_t fanout = static_cast<size_t>(std::max(1, config.fanout_per_neuron));
    net_config.num_synapses = static_cast<size_t>(config.num_neurons) * fanout;
    
    net_config.num_inputs = (config.num_inputs > 0)
        ? static_cast<size_t>(config.num_inputs)
        : static_cast<size_t>(std::max(1, config.num_neurons / 4));
    
    net_config.num_outputs = (config.num_outputs > 0)
        ? static_cast<size_t>(config.num_outputs)
        : static_cast<size_t>(std::max(1, config.num_neurons / 4));
        
    net_config.connection_density = static_cast<float>(fanout) / static_cast<float>(config.num_neurons);
    
    if (!neural_engine_->initialize(net_config)) {
        std::cerr << "Failed to init module " << config_.module_name << std::endl;
    }
    
    // Initialize working memory buffer
    working_memory_.resize(config.num_neurons / 10, 0.0f);  // 10% of neurons for working memory
    
    std::cout << "✓ Initialized module (NeuroGen 2.0): " << config_.module_name 
              << " with " << config.num_neurons << " neurons" << std::endl;
}

CorticalModule::~CorticalModule() {
    // unique_ptr handles cleanup
}

void CorticalModule::receiveInput(const std::vector<float>& input_vector) {
    // Store input for processing
    input_buffer_ = input_vector;
    
    // Update signal statistics for gating
    updateSignalStatistics(input_vector);
    
    // Apply neuromodulation to inputs
    std::vector<float> modulated_input = input_vector;
    for (size_t i = 0; i < modulated_input.size(); ++i) {
        // Dopamine increases signal strength, serotonin provides stability
        float modulation = 1.0f + (current_dopamine_ * config_.modulation.dopamine_sensitivity);
        modulation *= (1.0f + current_serotonin_ * config_.modulation.serotonin_sensitivity * 0.5f);
        modulation *= (1.0f - current_inhibition_);
        modulated_input[i] *= modulation;
    }
    
    // Use processInput which handles GPU transfer
    neural_engine_->processInput(modulated_input);
}

void CorticalModule::update(float dt_ms, float reward_signal) {
    // Apply neuromodulation effect on reward signal
    float modulated_reward = reward_signal * (1.0f + current_dopamine_ * config_.modulation.dopamine_sensitivity);
    
    // Step the physics of the neurons
    // Note: NeuralEngine::update takes (dt, reward)
    neural_engine_->update(dt_ms, modulated_reward);
    
    // Decay neuromodulators gradually
    current_dopamine_ *= 0.95f;  // 5% decay per update
    current_serotonin_ *= 0.98f;  // 2% decay per update
}

std::vector<float> CorticalModule::getOutputState() const {
    // Retrieve firing rates or voltages from the output layer
    // The new engine returns float vector of spikes/activities
    return neural_engine_->getNeuronOutputs();
}

void CorticalModule::setPlasticity(bool enabled) {
    // This would need a method in NeuralEngine to enable/disable learning
    // For Phase 1, we rely on the engine's internal configuration
    (void)enabled; // Suppress unused parameter warning
}

void CorticalModule::modulate(float dopamine, float serotonin) {
    // Apply neuromodulatory signals with sensitivity scaling
    current_dopamine_ += dopamine * config_.modulation.dopamine_sensitivity;
    current_serotonin_ += serotonin * config_.modulation.serotonin_sensitivity;
    
    // Clamp values to reasonable ranges
    current_dopamine_ = std::max(0.0f, std::min(2.0f, current_dopamine_));
    current_serotonin_ = std::max(0.0f, std::min(2.0f, current_serotonin_));
}

float CorticalModule::calculateSignalToNoise(const std::vector<float>& signal) const {
    if (signal.empty()) return 0.0f;
    
    // Calculate mean
    float mean = std::accumulate(signal.begin(), signal.end(), 0.0f) / signal.size();
    
    // Calculate variance
    float variance = 0.0f;
    for (float val : signal) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= signal.size();
    
    // Signal-to-noise ratio: mean / std_dev
    float std_dev = std::sqrt(variance);
    if (std_dev < 1e-6f) return 0.0f;
    
    return std::abs(mean) / std_dev;
}

std::vector<float> CorticalModule::gateSignal(const std::vector<float>& signal, float threshold) {
    float snr = calculateSignalToNoise(signal);
    
    std::vector<float> gated_signal = signal;
    
    // If SNR is below threshold, attenuate the signal
    if (snr < threshold) {
        float attenuation = snr / threshold;
        for (float& val : gated_signal) {
            val *= attenuation;
        }
    }
    
    return gated_signal;
}

void CorticalModule::setWorkingMemory(const std::vector<float>& memory_state) {
    working_memory_ = memory_state;
    if (working_memory_.size() != static_cast<size_t>(config_.num_neurons / 10)) {
        working_memory_.resize(config_.num_neurons / 10, 0.0f);
    }
}

std::vector<float> CorticalModule::getWorkingMemory() const {
    return working_memory_;
}

void CorticalModule::applyTopDownBias(float bias_strength) {
    // Apply a global bias to neuron excitability
    current_inhibition_ = config_.modulation.inhibition_level * (1.0f - bias_strength);
}

void CorticalModule::mapInputToNeurons(const std::vector<float>& input) {
    // This would map inputs to specific neurons
    // Implementation depends on the specific module type
    (void)input; // Suppress unused parameter warning
}

void CorticalModule::updateSignalStatistics(const std::vector<float>& signal) {
    if (signal.empty()) return;
    
    // Calculate running average of signal statistics
    float mean = std::accumulate(signal.begin(), signal.end(), 0.0f) / signal.size();
    signal_mean_ = 0.9f * signal_mean_ + 0.1f * mean;
    
    float variance = 0.0f;
    for (float val : signal) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= signal.size();
    signal_variance_ = 0.9f * signal_variance_ + 0.1f * variance;
}

neurogen::NeuralEngine* CorticalModule::getNeuralEngine() {
    return neural_engine_.get();
}

const neurogen::NeuralEngine* CorticalModule::getNeuralEngine() const {
    return neural_engine_.get();
}

std::vector<GPUNeuronState> CorticalModule::getNeuronStates() const {
    if (neural_engine_) {
        return neural_engine_->getNeuronStates();
    }
    return {};
}

std::vector<GPUSynapse> CorticalModule::getSynapseStates() const {
    if (neural_engine_) {
        return neural_engine_->getSynapseStates();
    }
    return {};
}

bool CorticalModule::setNeuronStates(const std::vector<GPUNeuronState>& neurons) {
    if (neural_engine_) {
        return neural_engine_->setNeuronStates(neurons);
    }
    return false;
}

bool CorticalModule::setSynapseStates(const std::vector<GPUSynapse>& synapses) {
    if (neural_engine_) {
        return neural_engine_->setSynapseStates(synapses);
    }
    return false;
}

void CorticalModule::restoreNeuromodulatorLevels(float dopamine, float serotonin) {
    current_dopamine_ = std::max(0.0f, std::min(2.0f, dopamine));
    current_serotonin_ = std::max(0.0f, std::min(2.0f, serotonin));
}
