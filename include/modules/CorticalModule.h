#pragma once
#include <vector>
#include <string>
#include <memory>
#include "engine/GPUNeuralStructures.h"
#include "engine/NeuralEngine.h" // New engine header

// Represents a distinct functional region of the brain
class CorticalModule {
public:
    // Neuromodulation parameters for biological realism
    struct ModulationParams {
        float dopamine_sensitivity = 0.0f;      // How sensitive to dopamine signals
        float serotonin_sensitivity = 0.0f;     // How sensitive to serotonin signals
        float inhibition_level = 0.0f;          // Base inhibitory tone
        float attention_threshold = 0.0f;       // Threshold for signal gating
        float excitability_bias = 1.0f;         // Global excitability modifier
    };

    struct Config {
        std::string module_name;
        int num_neurons = 1024;
        bool enable_plasticity = true;
        float learning_rate = 0.01f;
        int fanout_per_neuron = 256;            // Average outgoing synapses per neuron
        int num_inputs = 0;                     // Override input dimensionality (0 = auto)
        int num_outputs = 0;                    // Override output dimensionality (0 = auto)
        ModulationParams modulation;            // Neuromodulation parameters
    };

    CorticalModule(const Config& config, int gpu_device_id);
    ~CorticalModule();

    // 1. Input Processing: Receives signals from other modules or sensors
    // Converts high-level data into current injections for input neurons
    void receiveInput(const std::vector<float>& input_vector);

    // 2. The Core Step: Advances the CUDA simulation
    void update(float dt_ms, float reward_signal);

    // 3. Output Generation: Reads firing rates from output neurons
    // Returns a vector representation of this module's current thought/state
    std::vector<float> getOutputState() const;

    // 4. Plasticity Control: Modulate learning based on global state
    void setPlasticity(bool enabled);

    // 5. Neuromodulation: Apply dopamine, serotonin, and other modulators
    void modulate(float dopamine, float serotonin);

    // 6. Signal Gating: Calculate SNR and gate signals (for Thalamus)
    float calculateSignalToNoise(const std::vector<float>& signal) const;
    std::vector<float> gateSignal(const std::vector<float>& signal, float threshold);

    // 7. Working Memory: Store and retrieve persistent state (for PFC)
    void setWorkingMemory(const std::vector<float>& memory_state);
    std::vector<float> getWorkingMemory() const;

    // 8. Top-down control: Send bias signals to other modules
    void applyTopDownBias(float bias_strength);

    // 9. Get module name
    std::string getName() const { return config_.module_name; }

    // 10. Get module configuration
    const Config& getConfig() const { return config_; }

    // 11. Access underlying neural engine (for persistence and tooling)
    neurogen::NeuralEngine* getNeuralEngine();
    const neurogen::NeuralEngine* getNeuralEngine() const;

    // 12. Persistence: Access neuron/synapse state via wrapper
    // These methods delegate to NeuralEngine implementation
    std::vector<GPUNeuronState> getNeuronStates() const;
    std::vector<GPUSynapse> getSynapseStates() const;
    bool setNeuronStates(const std::vector<GPUNeuronState>& neurons);
    bool setSynapseStates(const std::vector<GPUSynapse>& synapses);

    // 13. Get neuromodulation state
    float getDopamineLevel() const { return current_dopamine_; }
    float getSerotoninLevel() const { return current_serotonin_; }

    // 14. Restore neuromodulator levels (used when loading checkpoints)
    void restoreNeuromodulatorLevels(float dopamine, float serotonin);

private:
    Config config_;
    std::unique_ptr<neurogen::NeuralEngine> neural_engine_; // Replaced NetworkCUDA
    
    // Buffer for inputs to avoid re-allocation
    std::vector<float> input_buffer_;
    
    // Working memory buffer for persistent state (PFC)
    std::vector<float> working_memory_;
    
    // Current neuromodulator levels
    float current_dopamine_;
    float current_serotonin_;
    float current_inhibition_;
    
    // Signal statistics for gating
    float signal_mean_;
    float signal_variance_;
    
    // Helper to map linear inputs to specific neuron indices
    void mapInputToNeurons(const std::vector<float>& input);
    
    // Update internal statistics
    void updateSignalStatistics(const std::vector<float>& signal);
};
