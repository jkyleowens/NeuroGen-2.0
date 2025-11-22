#pragma once
#include <memory>
#include <vector>
#include <string>

// Forward declaration
class CorticalModule;

/**
 * @brief Represents a connection between two cortical modules (the "Brain Bus")
 * 
 * Inter-module connections allow signals to flow between different brain regions
 * with biological properties like connection strength, excitatory/inhibitory nature,
 * attention-based gating, and Hebbian plasticity.
 */
class InterModuleConnection {
public:
    struct Config {
        std::string connection_name;
        CorticalModule* source_module;
        CorticalModule* target_module;
        float initial_strength;          // Initial connection weight
        bool is_excitatory;              // Excitatory (true) or inhibitory (false)
        float gating_threshold;          // Minimum signal strength to pass
        float plasticity_rate;           // How fast the connection learns
        bool enable_plasticity;          // Whether this connection can learn
    };

    InterModuleConnection(const Config& config);
    ~InterModuleConnection() = default;

    /**
     * @brief Transfer signal from source to target module
     * @param dt Time step in milliseconds
     * @return Actual signal strength transmitted
     */
    float transmit(float dt);

    /**
     * @brief Update connection strength based on Hebbian learning
     * @param source_activity Activity level of source module
     * @param target_activity Activity level of target module
     * @param reward Global reward signal for reinforcement
     * @param dt Time step
     */
    void updatePlasticity(float source_activity, float target_activity, 
                         float reward, float dt);

    /**
     * @brief Apply attention-based gating to the connection
     * @param attention_strength Attention weight (0-1)
     */
    void applyAttentionGating(float attention_strength);

    /**
     * @brief Get current connection strength
     */
    float getStrength() const { return current_strength_; }

    /**
     * @brief Set connection strength manually
     */
    void setStrength(float strength);

    /**
     * @brief Check if connection is excitatory
     */
    bool isExcitatory() const { return config_.is_excitatory; }

    /**
     * @brief Access connection configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Inspect internal modulation state
     */
    float getAttentionModulation() const { return attention_modulation_; }
    float getPreSynapticTrace() const { return pre_synaptic_trace_; }
    float getPostSynapticTrace() const { return post_synaptic_trace_; }

    /**
     * @brief Get connection statistics
     */
    struct Stats {
        float current_strength;
        float average_activity;
        float total_transmitted;
        int activation_count;
        float attention_modulation;
        float pre_synaptic_trace;
        float post_synaptic_trace;
    };
    Stats getStats() const;

    /**
     * @brief Restore connection dynamics from snapshot
     */
    void restoreState(float current_strength,
                      float attention_modulation,
                      float pre_synaptic_trace,
                      float post_synaptic_trace,
                      float average_activity,
                      float total_transmitted,
                      int activation_count);

private:
    Config config_;
    
    // Connection state
    float current_strength_;
    float attention_modulation_;
    
    // Hebbian learning traces
    float pre_synaptic_trace_;
    float post_synaptic_trace_;
    
    // Statistics
    float average_activity_;
    float total_transmitted_;
    int activation_count_;
    
    // Decay constants
    static constexpr float TRACE_DECAY = 0.95f;
    static constexpr float STRENGTH_BOUNDS_MIN = 0.01f;
    static constexpr float STRENGTH_BOUNDS_MAX = 2.0f;
};

