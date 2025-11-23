#pragma once
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <chrono>
#include "modules/CorticalModule.h"
#include "modules/InterModuleConnection.h"
#include "persistence/NetworkSnapshot.h"

/**
 * @brief Central coordinator for the modular brain architecture
 * 
 * The BrainOrchestrator manages all cortical modules and their interconnections,
 * orchestrates the cognitive cycle, distributes neuromodulatory signals, and
 * coordinates the overall "thinking" process of the neural system.
 */
class BrainOrchestrator {
public:
    enum class CognitivePhase {
        SENSATION,      // 0-50ms: Thalamic gating and input processing
        PERCEPTION,     // 50-150ms: Semantic processing in Wernicke's
        INTEGRATION,    // 150-300ms: PFC integration and memory retrieval
        SELECTION,      // 300-400ms: Basal ganglia action selection
        ACTION          // 400ms+: Output generation via Broca's
    };

    enum class ProcessingMode {
        SEQUENTIAL,     // Traditional: Full 5-phase cycle per token
        PIPELINED       // Streaming: Token input → working memory → parallel processing
    };

    struct Config {
        int gpu_device_id;
        float time_step_ms;              // Simulation time step (default: 1.0ms)
        bool enable_parallel_execution;  // Use CUDA streams for parallel modules
        bool enable_consolidation;       // Enable hippocampal replay
        float consolidation_interval_ms; // How often to run consolidation
        ProcessingMode processing_mode;  // Sequential vs Pipelined processing
        int max_pipeline_depth;          // Max tokens to accumulate before forcing output
    };

    BrainOrchestrator(const Config& config);
    ~BrainOrchestrator();

    /**
     * @brief Initialize all brain modules with their specific configurations
     */
    void initializeModules();

    /**
     * @brief Create the connectome (inter-module connections)
     */
    void createConnectome();

    /**
     * @brief Execute one complete cognitive cycle
     * @param input_embedding Token embedding to process
     * @return Output token probabilities (if in ACTION phase)
     */
    std::vector<float> cognitiveStep(const std::vector<float>& input_embedding);

    /**
     * @brief Route signals between all modules according to connectome
     */
    void routeSignals();

    /**
     * @brief Distribute global reward/dopamine signal to all modules
     * @param reward Reward prediction error (dopamine signal)
     */
    void distributeReward(float reward);

    /**
     * @brief Get the global workspace (shared activation patterns)
     * @return Combined activity from key modules
     */
    std::vector<float> getGlobalWorkspace();

    /**
     * @brief Update all inter-module connection plasticity
     * @param reward Global reward signal
     */
    void updateConnectionPlasticity(float reward);

    /**
     * @brief Trigger hippocampal consolidation (replay mechanism)
     */
    void consolidateMemory();

    /**
     * @brief Get current cognitive phase
     */
    CognitivePhase getCurrentPhase() const { return current_phase_; }

    /**
     * @brief Get module by name
     */
    CorticalModule* getModule(const std::string& name);

    /**
     * @brief Get system statistics
     */
    struct ModuleStats {
        float activity_level;
        float dopamine_level;
        float serotonin_level;
    };

    struct Stats {
        float total_time_ms;
        int cognitive_cycles;
        int tokens_processed;
        float average_reward;
        CognitivePhase current_phase;
        std::map<std::string, ModuleStats> module_stats;
    };
    Stats getStats() const;

    /**
     * @brief Capture the full neural network snapshot for persistence
     */
    persistence::BrainSnapshot captureSnapshot() const;

    /**
     * @brief Serialize the current state to a checkpoint file
     * @param file_path Destination file path
     * @return True on success
     */
    bool saveCheckpoint(const std::string& file_path) const;

    /**
     * @brief Restore orchestrator state from a checkpoint file
     * @param file_path Checkpoint file path
     * @return True on success
     */
    bool loadCheckpoint(const std::string& file_path);

    /**
     * @brief Reset the system state
     */
    void reset();

    /**
     * @brief Get output from Broca's area (language production)
     * @return Neural activity pattern from Broca's module
     */
    std::vector<float> getBrocaOutput();

    /**
     * @brief Modulate global neuromodulator levels
     * @param dopamine Dopamine level (reward/learning)
     * @param serotonin Serotonin level (mood/inhibition)
     * @param norepinephrine Norepinephrine level (arousal/attention)
     */
    void modulateGlobalState(float dopamine, float serotonin, float norepinephrine);

    /**
     * @brief Set processing mode
     * @param mode Sequential or pipelined processing
     */
    void setProcessingMode(ProcessingMode mode);

    /**
     * @brief Get current processing mode
     */
    ProcessingMode getProcessingMode() const { return processing_mode_; }

private:
    /**
     * @brief Pipeline state for streaming recurrent processing
     */
    struct PipelineState {
        // Working memory buffers (temporal context window)
        std::vector<float> wm_current;      // t=0 (just encoded)
        std::vector<float> wm_previous;     // t=-1 (being processed)
        std::vector<float> wm_context;      // t=-2 to t=-N (accumulated context)
        
        // Recurrent hidden states
        std::vector<float> pfc_hidden_state;
        std::vector<float> hippocampus_hidden_state;
        
        // Pipeline tracking
        int tokens_in_pipeline;
        bool output_ready;
        std::vector<float> pending_output;
        
        // Performance metrics
        float accumulated_processing_time;
        
        PipelineState() : tokens_in_pipeline(0), output_ready(false), 
                         accumulated_processing_time(0.0f) {}
    };
    Config config_;
    
    // The six core modules
    std::unique_ptr<CorticalModule> thalamus_;
    std::unique_ptr<CorticalModule> wernicke_;
    std::unique_ptr<CorticalModule> broca_;
    std::unique_ptr<CorticalModule> hippocampus_;
    std::unique_ptr<CorticalModule> pfc_;
    std::unique_ptr<CorticalModule> basal_ganglia_;
    
    // Inter-module connections (the connectome)
    std::vector<std::unique_ptr<InterModuleConnection>> connections_;
    
    // Map of module names to pointers for easy access
    std::map<std::string, CorticalModule*> module_map_;
    
    // Cognitive state
    CognitivePhase current_phase_;
    float phase_timer_;
    float total_time_;
    
    // Statistics
    int cognitive_cycles_;
    int tokens_processed_;
    float average_reward_;
    
    // Internal state for decision making
    bool should_generate_output_;
    std::vector<float> pending_input_;
    
    // Consolidation state
    float time_since_consolidation_;
    
    // Processing mode and pipeline state
    ProcessingMode processing_mode_;
    PipelineState pipeline_state_;
    
    // Helper functions for cognitive phases (sequential mode)
    void executeSensationPhase(const std::vector<float>& input);
    void executePerceptionPhase();
    void executeIntegrationPhase();
    void executeSelectionPhase();
    std::vector<float> executeActionPhase();
    
    // Helper to advance to next phase
    void advancePhase();
    
    // Calculate phase duration based on biological timing
    float getPhaseDuration(CognitivePhase phase) const;
    
    // Pipelined processing methods
    std::vector<float> pipelinedCognitiveStep(const std::vector<float>& input_embedding);
    void fastInputEncoding(const std::vector<float>& input);
    void parallelCorticalProcessing();
    std::vector<float> conditionalOutputGeneration();
    void updateRecurrentState();
};

