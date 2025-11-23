/**
 * Python Bindings for NeuroGen 2.0
 * 
 * This file provides Python bindings using pybind11 to allow the
 * train_slimpajama.py script to interface with the C++ NeuroGen engine.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <fstream>
#include "modules/BrainOrchestrator.h"
#include "interfaces/TokenEmbedding.h"
#include "interfaces/OutputDecoder.h"
#include "interfaces/GPUDecoder.h"
#include <cuda_runtime.h>

namespace py = pybind11;

/**
 * Python-facing wrapper for the NeuroGen model
 */
class NeuroGenModel {
public:
    NeuroGenModel(int vocab_size, int embedding_dim, int gpu_device) 
        : vocab_size_(vocab_size), 
          embedding_dim_(embedding_dim),
          gpu_device_(gpu_device) {
        
        // Verify GPU is available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            throw std::runtime_error("No CUDA-capable GPU found!");
        }
        
        if (gpu_device_ >= device_count) {
            throw std::runtime_error("Invalid GPU device ID: " + std::to_string(gpu_device_));
        }
        
        // Set GPU device
        cudaSetDevice(gpu_device_);
        
        // Print GPU info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpu_device_);
        std::cout << "🎮 Using GPU: " << prop.name 
                  << " (Device " << gpu_device_ << ")" << std::endl;
        std::cout << "   Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "   Total Memory: " << (prop.totalGlobalMem / (1024*1024*1024.0)) << " GB" << std::endl;
        
        // Initialize Brain Orchestrator
        BrainOrchestrator::Config config;
        config.gpu_device_id = gpu_device_;
        config.time_step_ms = 1.0f;
        config.enable_parallel_execution = true;
        config.enable_consolidation = false; // Disable for training speed
        config.processing_mode = BrainOrchestrator::ProcessingMode::PIPELINED; // Enable pipelined mode for efficiency
        config.max_pipeline_depth = 8; // Accumulate up to 8 tokens of context
        
        brain_ = std::make_unique<BrainOrchestrator>(config);
        brain_->initializeModules();
        brain_->createConnectome();
        
        // Initialize embedding and decoder
        TokenEmbedding::Config embed_config;
        embed_config.vocab_size = vocab_size_;
        embed_config.embedding_dim = embedding_dim_;
        embed_config.use_random_init = true;
        embed_config.normalization = 1.0f;
        embed_config.vocab_file = "";
        
        embedding_ = std::make_shared<TokenEmbedding>(embed_config);
        embedding_->initialize();
        
        // Use GPU-accelerated decoder for 50-100x speedup!
        GPUDecoder::Config gpu_decoder_config;
        gpu_decoder_config.vocab_size = vocab_size_;
        gpu_decoder_config.output_dim = 32768; // Broca output size (MAXIMALLY SCALED for 4GB GPU - quadrupled)
        gpu_decoder_config.temperature = 1.0f;
        gpu_decoder_config.top_k = 50;
        gpu_decoder_config.top_p = 0.9f;
        gpu_decoder_config.strategy = GPUDecoder::SamplingStrategy::GREEDY; // Start with greedy for stability
        gpu_decoder_config.gpu_device = gpu_device_;
        
        gpu_decoder_ = std::make_shared<GPUDecoder>(gpu_decoder_config, embedding_);
        
        std::cout << "✅ NeuroGen model initialized with:" << std::endl;
        std::cout << "   Vocab Size: " << vocab_size_ << std::endl;
        std::cout << "   Embedding Dim: " << embedding_dim_ << std::endl;
        std::cout << "   GPU Device: " << gpu_device_ << std::endl;
    }
    
    /**
     * Train on a single batch (next-token prediction)
     * @param input_ids: List of input token IDs
     * @param target_ids: List of target token IDs (input shifted by 1)
     * @return: Tuple of (loss, accuracy)
     */
    std::pair<float, float> train_step(
        const std::vector<int>& input_ids, 
        const std::vector<int>& target_ids) {
        
        if (input_ids.size() != target_ids.size()) {
            throw std::runtime_error("Input and target sizes must match");
        }
        
        float total_loss = 0.0f;
        int correct_predictions = 0;
        int total_tokens = 0;
        
        // Process each token in sequence
        for (size_t i = 0; i < input_ids.size(); ++i) {
            // 1. Embed input token
            std::vector<float> embedded = embedding_->encodeById(input_ids[i]);
            
            // 2. Process through brain
            brain_->cognitiveStep(embedded);
            
            // 3. Get output from Broca (language production area)
            std::vector<float> brain_output = brain_->getBrocaOutput();
            
            // 4. Decode to vocabulary distribution using GPU decoder (50-100x faster!)
            int predicted_token = gpu_decoder_->decodeAndSample(brain_output);
            
            // 5. Compute loss (cross-entropy approximation via negative log likelihood)
            // For simplicity, using 0/1 loss for now
            float token_loss = (predicted_token == target_ids[i]) ? 0.0f : 1.0f;
            total_loss += token_loss;
            
            // 6. Track accuracy
            if (predicted_token == target_ids[i]) {
                correct_predictions++;
            }
            total_tokens++;
            
            // 7. Apply reward signal (dopamine modulation for learning)
            float reward = (predicted_token == target_ids[i]) ? 1.0f : -0.5f;
            brain_->modulateGlobalState(reward, 0.5f, 0.5f); // dopamine, serotonin, norepinephrine
        }
        
        float avg_loss = total_loss / std::max(total_tokens, 1);
        float accuracy = (float)correct_predictions / std::max(total_tokens, 1);
        
        return {avg_loss, accuracy};
    }
    
    /**
     * Generate text from a prompt
     * @param prompt_ids: List of prompt token IDs
     * @param max_length: Maximum generation length
     * @return: List of generated token IDs
     */
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_length) {
        std::vector<int> generated = prompt_ids;
        
        // Process prompt
        for (int token_id : prompt_ids) {
            std::vector<float> embedded = embedding_->encodeById(token_id);
            brain_->cognitiveStep(embedded);
        }
        
        // Generate new tokens
        for (int i = 0; i < max_length; ++i) {
            std::vector<float> brain_output = brain_->getBrocaOutput();
            int next_token = gpu_decoder_->decodeAndSample(brain_output);
            
            generated.push_back(next_token);
            
            // Feed generated token back as input
            std::vector<float> embedded = embedding_->encodeById(next_token);
            brain_->cognitiveStep(embedded);
            
            // Stop on EOS token (assuming token 50256 is EOS for GPT-2)
            if (next_token == 50256) break;
        }
        
        return generated;
    }
    
    /**
     * Save checkpoint (includes brain state, embeddings, and decoder)
     */
    void save_checkpoint(const std::string& path) {
        // Save brain orchestrator state
        if (!brain_->saveCheckpoint(path)) {
            throw std::runtime_error("Failed to save brain checkpoint");
        }
        
        // Save embeddings
        std::string embedding_path = path + ".embeddings";
        embedding_->saveEmbeddings(embedding_path);
        
        // Save decoder projection matrix
        std::string decoder_path = path + ".decoder";
        gpu_decoder_->saveWeights(decoder_path);
        
        std::cout << "✅ Checkpoint saved: " << path << std::endl;
    }
    
    /**
     * Load checkpoint (includes brain state, embeddings, and decoder)
     */
    void load_checkpoint(const std::string& path) {
        // Load brain orchestrator state
        if (!brain_->loadCheckpoint(path)) {
            throw std::runtime_error("Failed to load brain checkpoint");
        }
        
        // Load embeddings if they exist
        std::string embedding_path = path + ".embeddings";
        std::ifstream emb_check(embedding_path);
        if (emb_check.good()) {
            emb_check.close();
            embedding_->loadEmbeddings(embedding_path);
        } else {
            std::cout << "⚠️  No embedding file found, using current embeddings" << std::endl;
        }
        
        // Load decoder projection matrix if it exists
        std::string decoder_path = path + ".decoder";
        std::ifstream dec_check(decoder_path);
        if (dec_check.good()) {
            dec_check.close();
            gpu_decoder_->loadWeights(decoder_path);
        } else {
            std::cout << "⚠️  No decoder file found, using current weights" << std::endl;
        }
        
        std::cout << "✅ Checkpoint loaded: " << path << std::endl;
    }

private:
    int vocab_size_;
    int embedding_dim_;
    int gpu_device_;
    std::unique_ptr<BrainOrchestrator> brain_;
    std::shared_ptr<TokenEmbedding> embedding_;
    std::shared_ptr<GPUDecoder> gpu_decoder_;
};

// Python module definition
PYBIND11_MODULE(libneurogen, m) {
    m.doc() = "NeuroGen 2.0 - Bio-inspired Neural Language Model";
    
    py::class_<NeuroGenModel>(m, "NeuroGenModel")
        .def(py::init<int, int, int>(),
             py::arg("vocab_size"),
             py::arg("embedding_dim"),
             py::arg("gpu_device") = 0,
             "Initialize NeuroGen model\n\n"
             "Args:\n"
             "    vocab_size: Size of vocabulary\n"
             "    embedding_dim: Dimension of token embeddings\n"
             "    gpu_device: GPU device ID (default: 0)")
        .def("train_step", &NeuroGenModel::train_step,
             py::arg("input_ids"),
             py::arg("target_ids"),
             "Train on a batch with next-token prediction\n\n"
             "Args:\n"
             "    input_ids: List of input token IDs\n"
             "    target_ids: List of target token IDs\n\n"
             "Returns:\n"
             "    Tuple of (loss, accuracy)")
        .def("generate", &NeuroGenModel::generate,
             py::arg("prompt_ids"),
             py::arg("max_length") = 100,
             "Generate text from a prompt\n\n"
             "Args:\n"
             "    prompt_ids: List of prompt token IDs\n"
             "    max_length: Maximum generation length\n\n"
             "Returns:\n"
             "    List of generated token IDs")
        .def("save_checkpoint", &NeuroGenModel::save_checkpoint,
             py::arg("path"),
             "Save model checkpoint")
        .def("load_checkpoint", &NeuroGenModel::load_checkpoint,
             py::arg("path"),
             "Load model checkpoint");
    
    // Version info
    m.attr("__version__") = "2.0.0";
}

