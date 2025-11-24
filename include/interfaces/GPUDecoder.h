#pragma once
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "interfaces/TokenEmbedding.h"

class GPUDecoder {
public:
    enum class SamplingStrategy {
        GREEDY,
        TEMPERATURE,
        TOP_K,
        TOP_P
    };

    struct Config {
        int vocab_size = 32000;
        int output_dim = 4096; // Size of neural output vector
        float temperature = 1.0f;
        int top_k = 0;
        float top_p = 0.0f;
        SamplingStrategy strategy = SamplingStrategy::TEMPERATURE;
        int gpu_device = 0;
    };

    GPUDecoder(const Config& config, std::shared_ptr<TokenEmbedding> embeddings);
    ~GPUDecoder();

    // Main decoding methods
    std::vector<float> decode(const std::vector<float>& neural_output);
    int decodeAndSample(const std::vector<float>& neural_output);
    std::pair<int, float> decodeAndSampleWithProb(const std::vector<float>& neural_output);
    std::string decodeToString(const std::vector<float>& neural_output);

    // NEW: Training methods for "Trainable Broca"
    std::vector<float> calculateErrorVector(const std::vector<float>& neural_output, int target_token_id);
    void updateWeights(const std::vector<float>& neural_output, int target_token_id, float learning_rate);

    // Persistence
    void saveWeights(const std::string& filepath);
    void loadWeights(const std::string& filepath);

private:
    Config config_;
    std::shared_ptr<TokenEmbedding> embeddings_;
    int gpu_device_;

    // CUDA resources
    void* cublas_handle_;
    void* stream_;
    
    // Device memory
    float* d_projection_matrix_;
    float* d_projection_bias_;
    float* d_logits_;
    float* d_probabilities_;
    float* d_neural_input_;
    
    // Host memory
    float* h_probabilities_;
    
    // RNG state
    void* d_rng_state_;

    void initializeGPU();
    void cleanupGPU();
    void initializeProjectionMatrix();
    
    // Helpers
    void projectToLogitsGPU(const float* d_input, float* d_output);
    void softmaxGPU(const float* d_logits, float* d_probs);
    int sampleTokenGPU(const float* d_probs);
    
    // Sampling implementations
    int greedySampleGPU(const float* d_probs);
    int temperatureSampleGPU(const float* d_probs);
    int topKSampleGPU(const float* d_probs);
    int topPSampleGPU(const float* d_probs);
    
    // Internal decode helper
    float* decodeGPU(const float* d_neural_output);
    int decodeAndSampleGPU(const float* d_neural_output);
};