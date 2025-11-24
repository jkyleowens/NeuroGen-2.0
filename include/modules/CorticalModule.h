#pragma once
#include <vector>
#include <string>
#include <memory>
#include "engine/GPUNeuralStructures.h"
#include "engine/NeuralEngine.h"

// Forward declaration ensuring namespace validity
namespace neurogen {
    class NeuralEngine;
}

class CorticalModule {
public:
    struct ModulationParams {
        float dopamine_sensitivity = 0.0f;
        float serotonin_sensitivity = 0.0f;
        float inhibition_level = 0.0f;
        float attention_threshold = 0.0f;
        float excitability_bias = 1.0f;
    };

    struct Config {
        std::string module_name;
        int num_neurons = 1024;
        bool enable_plasticity = true;
        float learning_rate = 0.01f;
        int fanout_per_neuron = 256;
        int num_inputs = 0;
        int num_outputs = 0;
        ModulationParams modulation;
    };

    CorticalModule(const Config& config, int gpu_device_id);
    ~CorticalModule();

    // Optimized: Uses internal buffer
    void receiveInput(const std::vector<float>& input_vector);

    void update(float dt_ms, float reward_signal);

    // Optimized: Returns const reference
    const std::vector<float>& getOutputState() const;

    void setPlasticity(bool enabled);
    void modulate(float dopamine, float serotonin);

    // Signal Processing
    float calculateSignalToNoise(const std::vector<float>& signal) const;
    
    // Novelty Gating (Predictive Coding)
    std::vector<float> gateSignal(const std::vector<float>& signal, float threshold);
    
    // Predictive Coding Support
    std::vector<float> calculatePredictionError(const std::vector<float>& prediction);
    
    // Error-driven Modulation
    void modulateWithError(const std::vector<float>& error_signal);
    
    // Gating Signal
    float getGatingSignal() const;

    void setWorkingMemory(const std::vector<float>& memory_state);
    std::vector<float> getWorkingMemory() const;
    void applyTopDownBias(float bias_strength);

    std::string getName() const { return config_.module_name; }
    const Config& getConfig() const { return config_; }

    neurogen::NeuralEngine* getNeuralEngine();
    const neurogen::NeuralEngine* getNeuralEngine() const;

    std::vector<GPUNeuronState> getNeuronStates() const;
    std::vector<GPUSynapse> getSynapseStates() const;
    bool setNeuronStates(const std::vector<GPUNeuronState>& neurons);
    bool setSynapseStates(const std::vector<GPUSynapse>& synapses);

    float getDopamineLevel() const { return current_dopamine_; }
    float getSerotoninLevel() const { return current_serotonin_; }
    void restoreNeuromodulatorLevels(float dopamine, float serotonin);

private:
    Config config_;
    std::unique_ptr<neurogen::NeuralEngine> neural_engine_;
    
    // Reusable buffers
    std::vector<float> input_buffer_;
    std::vector<float> modulation_buffer_;
    mutable std::vector<float> cached_output_state_;
    std::vector<float> previous_input_;
    std::vector<float> working_memory_;
    
    float current_dopamine_;
    float current_serotonin_;
    float current_inhibition_;
    
    float signal_mean_;
    float signal_variance_;
    
    void updateSignalStatistics(const std::vector<float>& signal);
};