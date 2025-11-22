#include "modules/InterModuleConnection.h"
#include "modules/CorticalModule.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

InterModuleConnection::InterModuleConnection(const Config& config)
    : config_(config),
      current_strength_(config.initial_strength),
      attention_modulation_(1.0f),
      pre_synaptic_trace_(0.0f),
      post_synaptic_trace_(0.0f),
      average_activity_(0.0f),
      total_transmitted_(0.0f),
      activation_count_(0) {
    
    if (!config_.source_module || !config_.target_module) {
        std::cerr << "⚠️  Warning: InterModuleConnection created with null module pointers" << std::endl;
    }
}

float InterModuleConnection::transmit(float dt) {
    if (!config_.source_module || !config_.target_module) {
        return 0.0f;
    }

    // Get output from source module
    std::vector<float> source_output = config_.source_module->getOutputState();
    
    if (source_output.empty()) {
        return 0.0f;
    }

    // Calculate average activity of source
    float source_activity = std::accumulate(source_output.begin(), source_output.end(), 0.0f) 
                           / source_output.size();

    // Apply connection strength, polarity, and attention modulation
    float signal_strength = source_activity * current_strength_ * attention_modulation_;
    
    // Inhibitory connections are negative
    if (!config_.is_excitatory) {
        signal_strength = -std::abs(signal_strength);
    }

    // Apply gating threshold
    if (std::abs(signal_strength) < config_.gating_threshold) {
        signal_strength *= (std::abs(signal_strength) / config_.gating_threshold);
    }

    // Scale output to match target module's expected input size
    std::vector<float> transmitted_signal(source_output.size());
    for (size_t i = 0; i < source_output.size(); ++i) {
        transmitted_signal[i] = source_output[i] * current_strength_ * attention_modulation_;
        if (!config_.is_excitatory) {
            transmitted_signal[i] = -std::abs(transmitted_signal[i]);
        }
    }

    // Send to target module
    config_.target_module->receiveInput(transmitted_signal);

    // Update statistics
    activation_count_++;
    total_transmitted_ += std::abs(signal_strength);
    average_activity_ = 0.99f * average_activity_ + 0.01f * std::abs(source_activity);

    // Update pre-synaptic trace for plasticity
    pre_synaptic_trace_ = TRACE_DECAY * pre_synaptic_trace_ + source_activity;

    return signal_strength;
}

void InterModuleConnection::updatePlasticity(float source_activity, float target_activity,
                                            float reward, float dt) {
    if (!config_.enable_plasticity) {
        return;
    }

    // Update traces
    pre_synaptic_trace_ = TRACE_DECAY * pre_synaptic_trace_ + source_activity;
    post_synaptic_trace_ = TRACE_DECAY * post_synaptic_trace_ + target_activity;

    // Hebbian learning: "neurons that fire together, wire together"
    float hebbian_term = pre_synaptic_trace_ * post_synaptic_trace_;

    // Reward modulation (dopaminergic gating)
    float reward_modulation = 1.0f + reward;

    // Weight update
    float delta_strength = config_.plasticity_rate * hebbian_term * reward_modulation * dt;

    // Apply update
    current_strength_ += delta_strength;

    // Bound the connection strength
    current_strength_ = std::max(STRENGTH_BOUNDS_MIN, 
                                std::min(STRENGTH_BOUNDS_MAX, current_strength_));
}

void InterModuleConnection::applyAttentionGating(float attention_strength) {
    // Attention modulates the effective connection strength
    attention_modulation_ = std::max(0.0f, std::min(1.0f, attention_strength));
}

void InterModuleConnection::setStrength(float strength) {
    current_strength_ = std::max(STRENGTH_BOUNDS_MIN, 
                                std::min(STRENGTH_BOUNDS_MAX, strength));
}

InterModuleConnection::Stats InterModuleConnection::getStats() const {
    return Stats{
        current_strength_,
        average_activity_,
        total_transmitted_,
        activation_count_,
        attention_modulation_,
        pre_synaptic_trace_,
        post_synaptic_trace_
    };
}

void InterModuleConnection::restoreState(float current_strength,
                                         float attention_modulation,
                                         float pre_synaptic_trace,
                                         float post_synaptic_trace,
                                         float average_activity,
                                         float total_transmitted,
                                         int activation_count) {
    setStrength(current_strength);
    attention_modulation_ = std::max(0.0f, std::min(1.0f, attention_modulation));
    pre_synaptic_trace_ = pre_synaptic_trace;
    post_synaptic_trace_ = post_synaptic_trace;
    average_activity_ = average_activity;
    total_transmitted_ = total_transmitted;
    activation_count_ = activation_count;
}

