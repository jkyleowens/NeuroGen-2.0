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
        gpu_decoder_config.output_dim = 8192; // Broca output size (Optimized for GTX 1650 4GB VRAM)
        gpu_decoder_config.temperature = 1.0f;
        gpu_decoder_config.top_k = 50;
        gpu_decoder_config.top_p = 0.9f;
        gpu_decoder_config.strategy = GPUDecoder::SamplingStrategy::TOP_K; // Use TOP_K sampling to satisfy the updated training algorithm requirements
        gpu_decoder_config.gpu_device = gpu_device_;
        
        gpu_decoder_ = std::make_shared<GPUDecoder>(gpu_decoder_config, embedding_);
        
        std::cout << "✅ NeuroGen model initialized with:" << std::endl;
        std::cout << "   Vocab Size: " << vocab_size_ << std::endl;
        std::cout << "   Embedding Dim: " << embedding_dim_ << std::endl;
        std::cout << "   GPU Device: " << gpu_device_ << std::endl;
        std::cout << "   🎯 Memory Optimized: GTX 1650 (4GB) configuration" << std::endl;
    }
    
    /**
     * Train on a single step (predict exactly ONE next token)
     * @param input_ids: List of input token IDs (context)
     * @param target_ids: List of target token IDs (same length, but we only use last one)
     * @return: Tuple of (loss, accuracy, predicted_token_id)
     */
    std::tuple<float, float, int> train_step(
        const std::vector<int>& input_ids, 
        const std::vector<int>& target_ids) {
        
        // Require non-empty input and at least one target token
        if (input_ids.empty() || target_ids.empty()) {
            throw std::runtime_error("Input and target must be non-empty");
        }
        
        // We no longer require input_ids.size() == target_ids.size().
        // Python now passes a full context sequence as input_ids and a
        // single next-token target as target_ids[0]. We always use the
        // last target token as the supervised label for this step.
        
        // 1) Run full context through embedding + brain
        for (size_t i = 0; i < input_ids.size(); ++i) {
            std::vector<float> embedded = embedding_->encodeById(input_ids[i]);
            brain_->cognitiveStep(embedded);
        }

        // 2) Get final Broca output and decode/sample with probability
        std::vector<float> brain_output = brain_->getBrocaOutput();
        auto token_and_prob = gpu_decoder_->decodeAndSampleWithProb(brain_output);
        int predicted_token = token_and_prob.first;
        float predicted_prob = token_and_prob.second;

        int target_token = target_ids.back();

        const float eps = 1e-8f;
        float nll = -std::log(std::max(predicted_prob, eps));
        float loss = nll;
        float accuracy = (predicted_token == target_token) ? 1.0f : 0.0f;

        float reward = -loss;
        brain_->modulateGlobalState(reward, 0.5f, 0.5f);

        return std::make_tuple(loss, accuracy, predicted_token);
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
             "Train on a batch with next-token prediction (single-step)\n\n"
             "Args:\n"
             "    input_ids: List of input token IDs\n"
             "    target_ids: List of target token IDs (same length as input_ids)\n\n"
             "Returns:\n"
             "    Tuple of (loss, accuracy, predicted_token_id)")
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

