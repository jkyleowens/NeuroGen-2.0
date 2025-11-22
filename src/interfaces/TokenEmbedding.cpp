#include "interfaces/TokenEmbedding.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>

TokenEmbedding::TokenEmbedding(const Config& config)
    : config_(config), next_token_id_(0) {
    
    embeddings_.reserve(config_.vocab_size);
    
    std::cout << "📝 Initializing TokenEmbedding with vocab_size=" 
              << config_.vocab_size << ", embedding_dim=" 
              << config_.embedding_dim << std::endl;
}

void TokenEmbedding::initialize() {
    // Initialize special tokens
    initializeSpecialTokens();
    
    if (config_.use_random_init) {
        // Random initialization for all tokens
        for (int i = next_token_id_; i < config_.vocab_size; ++i) {
            embeddings_.push_back(randomEmbedding());
        }
        std::cout << "✓ Initialized " << config_.vocab_size 
                  << " random embeddings" << std::endl;
    }
    
    // Load vocabulary if provided
    if (!config_.vocab_file.empty()) {
        loadVocabulary(config_.vocab_file);
    }
}

void TokenEmbedding::initializeSpecialTokens() {
    // Add special tokens
    addToken(UNK_TOKEN);  // ID 0
    addToken(PAD_TOKEN);  // ID 1
    addToken(BOS_TOKEN);  // ID 2
    addToken(EOS_TOKEN);  // ID 3
}

std::vector<float> TokenEmbedding::encode(const std::string& token) {
    int token_id = getTokenId(token);
    
    if (token_id < 0) {
        // Unknown token - return UNK embedding
        token_id = getTokenId(UNK_TOKEN);
    }
    
    return encodeById(token_id);
}

std::vector<float> TokenEmbedding::encodeById(int token_id) {
    if (token_id < 0 || token_id >= static_cast<int>(embeddings_.size())) {
        // Return zero embedding if invalid
        return std::vector<float>(config_.embedding_dim, 0.0f);
    }
    
    auto embedding = embeddings_[token_id];
    
    // Apply normalization if configured
    if (config_.normalization > 0.0f) {
        embedding = normalize(embedding);
    }
    
    return embedding;
}

std::vector<std::vector<float>> TokenEmbedding::encodeSequence(
    const std::vector<std::string>& tokens) {
    
    std::vector<std::vector<float>> sequence_embeddings;
    sequence_embeddings.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        sequence_embeddings.push_back(encode(token));
    }
    
    return sequence_embeddings;
}

int TokenEmbedding::getTokenId(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    return -1;  // Not found
}

std::string TokenEmbedding::getToken(int token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    return UNK_TOKEN;
}

int TokenEmbedding::addToken(const std::string& token) {
    // Check if token already exists
    int existing_id = getTokenId(token);
    if (existing_id >= 0) {
        return existing_id;
    }
    
    // Add new token
    int token_id = next_token_id_++;
    token_to_id_[token] = token_id;
    id_to_token_[token_id] = token;
    
    // Add random embedding for this token
    embeddings_.push_back(randomEmbedding());
    
    return token_id;
}

void TokenEmbedding::loadVocabulary(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "⚠️  Failed to open vocabulary file: " << filepath << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            addToken(line);
        }
    }
    
    file.close();
    std::cout << "✓ Loaded " << token_to_id_.size() 
              << " tokens from vocabulary file" << std::endl;
}

void TokenEmbedding::saveVocabulary(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "⚠️  Failed to open file for writing: " << filepath << std::endl;
        return;
    }
    
    // Write tokens in order of their IDs
    for (int i = 0; i < next_token_id_; ++i) {
        file << getToken(i) << "\n";
    }
    
    file.close();
    std::cout << "✓ Saved vocabulary to " << filepath << std::endl;
}

void TokenEmbedding::loadEmbeddings(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "⚠️  Failed to open embeddings file: " << filepath << std::endl;
        return;
    }
    
    // Read embeddings (simple binary format)
    embeddings_.clear();
    
    for (int i = 0; i < config_.vocab_size; ++i) {
        std::vector<float> embedding(config_.embedding_dim);
        file.read(reinterpret_cast<char*>(embedding.data()), 
                 config_.embedding_dim * sizeof(float));
        embeddings_.push_back(embedding);
    }
    
    file.close();
    std::cout << "✓ Loaded embeddings from " << filepath << std::endl;
}

void TokenEmbedding::saveEmbeddings(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "⚠️  Failed to open file for writing: " << filepath << std::endl;
        return;
    }
    
    // Write embeddings (simple binary format)
    for (const auto& embedding : embeddings_) {
        file.write(reinterpret_cast<const char*>(embedding.data()), 
                  embedding.size() * sizeof(float));
    }
    
    file.close();
    std::cout << "✓ Saved embeddings to " << filepath << std::endl;
}

std::vector<float> TokenEmbedding::normalize(const std::vector<float>& embedding) {
    // L2 normalization
    float sum_sq = 0.0f;
    for (float val : embedding) {
        sum_sq += val * val;
    }
    
    float norm = std::sqrt(sum_sq);
    if (norm < 1e-6f) {
        return embedding;  // Avoid division by zero
    }
    
    std::vector<float> normalized(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        normalized[i] = embedding[i] / norm;
    }
    
    return normalized;
}

std::vector<float> TokenEmbedding::randomEmbedding() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dist(0.0f, 0.1f);
    
    std::vector<float> embedding(config_.embedding_dim);
    for (int i = 0; i < config_.embedding_dim; ++i) {
        embedding[i] = dist(gen);
    }
    
    return normalize(embedding);
}

