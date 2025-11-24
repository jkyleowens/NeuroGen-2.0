#include "interfaces/GPUDecoder.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <random>
#include <fstream>
#include <vector>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t stat = call; \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error: " + std::to_string(stat)); \
        } \
    } while(0)

// ============================================================================
// GPU Kernels
// ============================================================================

/**
 * @brief Update decoder weights (Delta Rule)
 */
__global__ void updateDecoderWeightsKernel(float* weights, const float* inputs, 
                                   const float* probs, int target_idx, 
                                   int vocab_size, int input_dim, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = vocab_size * input_dim;
    
    if (idx >= total_elements) return;
    
    int row = idx / input_dim;
    int col = idx % input_dim;
    
    // Gradient: (Probability - Target)
    float target = (row == target_idx) ? 1.0f : 0.0f;
    float error = probs[row] - target; 
    
    // Update: W -= lr * error * input
    weights[idx] -= lr * error * inputs[col];
}

/**
 * @brief Softmax kernel with numerical stability
 */
__global__ void softmaxKernel(const float* __restrict__ logits, 
                               float* __restrict__ probs, 
                               int vocab_size) {
    extern __shared__ float shared_data[];
    float* shared_max = shared_data;
    float* shared_sum = shared_data + blockDim.x;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += stride) {
        local_max = fmaxf(local_max, logits[i]);
    }
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += stride) {
        float exp_val = expf(logits[i] - max_val);
        probs[i] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce to find total sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float sum_val = shared_sum[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < vocab_size; i += stride) {
        probs[i] /= sum_val;
    }
}

/**
 * @brief Temperature-scaled softmax kernel
 */
__global__ void temperatureSoftmaxKernel(const float* __restrict__ logits,
                                          float* __restrict__ probs,
                                          int vocab_size,
                                          float temperature) {
    extern __shared__ float shared_data[];
    float* shared_max = shared_data;
    float* shared_sum = shared_data + blockDim.x;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Find max
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += stride) {
        local_max = fmaxf(local_max, logits[i] / temperature);
    }
    shared_max[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += stride) {
        float exp_val = expf(logits[i] / temperature - max_val);
        probs[i] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float sum_val = shared_sum[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < vocab_size; i += stride) {
        probs[i] /= sum_val;
    }
}

/**
 * @brief Find argmax (greedy sampling)
 */
__global__ void argmaxKernel(const float* __restrict__ probs,
                              int* result,
                              int vocab_size) {
    extern __shared__ float shared_data[];
    int* shared_idx = (int*)(shared_data + blockDim.x);
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Find local max
    float local_max = -INFINITY;
    int local_idx = 0;
    for (int i = tid; i < vocab_size; i += stride) {
        if (probs[i] > local_max) {
            local_max = probs[i];
            local_idx = i;
        }
    }
    shared_data[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_data[tid + s] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *result = shared_idx[0];
    }
}

/**
 * @brief Compute cumulative sum (for sampling)
 */
__global__ void cumulativeSumKernel(const float* __restrict__ probs,
                                     float* __restrict__ cumsum,
                                     int vocab_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < vocab_size) {
        float sum = 0.0f;
        for (int i = 0; i <= tid; ++i) {
            sum += probs[i];
        }
        cumsum[tid] = sum;
    }
}

/**
 * @brief Binary search for sampling
 */
__global__ void binarySearchSampleKernel(const float* __restrict__ cumsum,
                                          float random_val,
                                          int* result,
                                          int vocab_size) {
    if (threadIdx.x == 0) {
        int left = 0;
        int right = vocab_size - 1;
        
        while (left < right) {
            int mid = (left + right) / 2;
            if (cumsum[mid] < random_val) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        *result = left;
    }
}

/**
 * @brief Initialize RNG states
 */
__global__ void initRNGKernel(curandState* states, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &states[tid]);
}

// ============================================================================
// GPUDecoder Implementation
// ============================================================================

GPUDecoder::GPUDecoder(const Config& config, std::shared_ptr<TokenEmbedding> embeddings)
    : config_(config),
      embeddings_(embeddings),
      gpu_device_(config.gpu_device),
      cublas_handle_(nullptr),
      stream_(nullptr),
      d_projection_matrix_(nullptr),
      d_projection_bias_(nullptr),
      d_logits_(nullptr),
      d_probabilities_(nullptr),
      d_neural_input_(nullptr),
      h_probabilities_(nullptr),
      d_rng_state_(nullptr) {
    
    initializeGPU();
    std::cout << "✓ GPU Decoder initialized (cuBLAS + CUDA)\n";
    std::cout << "  Vocab: " << config_.vocab_size << " | Output dim: " << config_.output_dim << "\n";
}

GPUDecoder::~GPUDecoder() {
    cleanupGPU();
}

void GPUDecoder::initializeGPU() {
    CUDA_CHECK(cudaSetDevice(gpu_device_));
    
    // FIX: Use local typed variables for creation, then assign to void* members
    cublasHandle_t handle;
    cudaStream_t stream;
    
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    
    // Store as member variables (implicit cast to void*)
    cublas_handle_ = handle;
    stream_ = stream;
    
    size_t matrix_size = static_cast<size_t>(config_.vocab_size) * config_.output_dim;
    CUDA_CHECK(cudaMalloc(&d_projection_matrix_, matrix_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_projection_bias_, config_.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_, config_.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probabilities_, config_.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_neural_input_, config_.output_dim * sizeof(float)));
    
    CUDA_CHECK(cudaMallocHost(&h_probabilities_, config_.vocab_size * sizeof(float)));
    
    // RNG Initialization
    CUDA_CHECK(cudaMalloc(&d_rng_state_, sizeof(curandState)));
    initRNGKernel<<<1, 1>>>(reinterpret_cast<curandState*>(d_rng_state_), time(nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    initializeProjectionMatrix();
}

void GPUDecoder::cleanupGPU() {
    if (d_projection_matrix_) cudaFree(d_projection_matrix_);
    if (d_projection_bias_) cudaFree(d_projection_bias_);
    if (d_logits_) cudaFree(d_logits_);
    if (d_probabilities_) cudaFree(d_probabilities_);
    if (d_neural_input_) cudaFree(d_neural_input_);
    if (h_probabilities_) cudaFreeHost(h_probabilities_);
    if (d_rng_state_) cudaFree(d_rng_state_);
    
    // FIX: Cast void* back to typed pointers for destruction
    if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    if (cublas_handle_) cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
}

void GPUDecoder::initializeProjectionMatrix() {
    size_t matrix_size = static_cast<size_t>(config_.vocab_size) * config_.output_dim;
    std::vector<float> h_matrix(matrix_size);
    
    float scale = sqrtf(2.0f / (config_.vocab_size + config_.output_dim));
    for (size_t i = 0; i < matrix_size; ++i) {
        h_matrix[i] = scale * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
    }
    
    CUDA_CHECK(cudaMemcpy(d_projection_matrix_, h_matrix.data(), 
                          matrix_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_projection_bias_, 0, config_.vocab_size * sizeof(float)));
}

void GPUDecoder::projectToLogitsGPU(const float* d_input, float* d_output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // FIX: Cast void* member to cublasHandle_t
    CUBLAS_CHECK(cublasSgemv(
        static_cast<cublasHandle_t>(cublas_handle_),
        CUBLAS_OP_N,
        config_.vocab_size,
        config_.output_dim,
        &alpha,
        d_projection_matrix_,
        config_.vocab_size,
        d_input,
        1,
        &beta,
        d_output,
        1
    ));
}

void GPUDecoder::softmaxGPU(const float* d_logits, float* d_probs) {
    int block_size = 256;
    int shared_mem_size = 2 * block_size * sizeof(float);
    
    // FIX: Cast void* member to cudaStream_t
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    
    if (config_.strategy == SamplingStrategy::TEMPERATURE && config_.temperature != 1.0f) {
        temperatureSoftmaxKernel<<<1, block_size, shared_mem_size, stream>>>(
            d_logits, d_probs, config_.vocab_size, config_.temperature
        );
    } else {
        softmaxKernel<<<1, block_size, shared_mem_size, stream>>>(
            d_logits, d_probs, config_.vocab_size
        );
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int GPUDecoder::sampleTokenGPU(const float* d_probs) {
    switch (config_.strategy) {
        case SamplingStrategy::GREEDY:
            return greedySampleGPU(d_probs);
        case SamplingStrategy::TEMPERATURE:
            return temperatureSampleGPU(d_probs);
        case SamplingStrategy::TOP_K:
            return topKSampleGPU(d_probs);
        case SamplingStrategy::TOP_P:
            return topPSampleGPU(d_probs);
        default:
            return greedySampleGPU(d_probs);
    }
}

int GPUDecoder::greedySampleGPU(const float* d_probs) {
    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    
    int block_size = 256;
    int shared_mem_size = block_size * (sizeof(float) + sizeof(int));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    
    argmaxKernel<<<1, block_size, shared_mem_size, stream>>>(
        d_probs, d_result, config_.vocab_size
    );
    
    int h_result;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(int), 
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_result));
    
    return h_result;
}

int GPUDecoder::temperatureSampleGPU(const float* d_probs) {
    curandState* state = reinterpret_cast<curandState*>(d_rng_state_);
    float random_val;
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    random_val = dis(gen);
    
    float* d_cumsum;
    CUDA_CHECK(cudaMalloc(&d_cumsum, config_.vocab_size * sizeof(float)));
    
    int block_size = 256;
    int num_blocks = (config_.vocab_size + block_size - 1) / block_size;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    
    cumulativeSumKernel<<<num_blocks, block_size, 0, stream>>>(
        d_probs, d_cumsum, config_.vocab_size
    );
    
    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    
    binarySearchSampleKernel<<<1, 1, 0, stream>>>(
        d_cumsum, random_val, d_result, config_.vocab_size
    );
    
    int h_result;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(d_cumsum));
    CUDA_CHECK(cudaFree(d_result));
    
    return h_result;
}

int GPUDecoder::topKSampleGPU(const float* d_probs) {
    return temperatureSampleGPU(d_probs);
}

int GPUDecoder::topPSampleGPU(const float* d_probs) {
    return temperatureSampleGPU(d_probs);
}

float* GPUDecoder::decodeGPU(const float* d_neural_output) {
    projectToLogitsGPU(d_neural_output, d_logits_);
    softmaxGPU(d_logits_, d_probabilities_);
    return d_probabilities_;
}

int GPUDecoder::decodeAndSampleGPU(const float* d_neural_output) {
    float* d_probs = decodeGPU(d_neural_output);
    return sampleTokenGPU(d_probs);
}

std::vector<float> GPUDecoder::decode(const std::vector<float>& neural_output) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_neural_input_, neural_output.data(),
                                config_.output_dim * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    
    float* d_probs = decodeGPU(d_neural_input_);
    
    CUDA_CHECK(cudaMemcpyAsync(h_probabilities_, d_probs,
                                config_.vocab_size * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    return std::vector<float>(h_probabilities_, h_probabilities_ + config_.vocab_size);
}

int GPUDecoder::decodeAndSample(const std::vector<float>& neural_output) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_neural_input_, neural_output.data(),
                                config_.output_dim * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    
    return decodeAndSampleGPU(d_neural_input_);
}

std::pair<int, float> GPUDecoder::decodeAndSampleWithProb(const std::vector<float>& neural_output) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    CUDA_CHECK(cudaMemcpyAsync(d_neural_input_, neural_output.data(),
                                config_.output_dim * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    float* d_probs = decodeGPU(d_neural_input_);
    int token_id = sampleTokenGPU(d_probs);

    CUDA_CHECK(cudaMemcpyAsync(h_probabilities_, d_probs,
                                config_.vocab_size * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float prob = 0.0f;
    if (token_id >= 0 && token_id < config_.vocab_size) {
        prob = h_probabilities_[token_id];
    }

    return std::make_pair(token_id, prob);
}

std::string GPUDecoder::decodeToString(const std::vector<float>& neural_output) {
    int token_id = decodeAndSample(neural_output);
    return std::to_string(token_id);
}

// ============================================================================
// NEW: Training Methods (Delta Rule Implementation)
// ============================================================================

std::vector<float> GPUDecoder::calculateErrorVector(const std::vector<float>& neural_output, int target_token_id) {
    // 1. Forward pass to get probabilities
    decode(neural_output); // This updates h_probabilities_
    
    // 2. Calculate Error Vector at Logits (CPU)
    std::vector<float> error_vector(config_.vocab_size);
    for (int i = 0; i < config_.vocab_size; ++i) {
        float target = (i == target_token_id) ? 1.0f : 0.0f;
        // Error = Prob - Target
        error_vector[i] = h_probabilities_[i] - target;
    }
    
    return error_vector;
}

void GPUDecoder::updateWeights(const std::vector<float>& neural_output, int target_token_id, float learning_rate) {
    // We launch the kernel to update weights directly on GPU.
    // Inputs:
    // - d_projection_matrix_ (Weights W)
    // - d_neural_input_ (Input activation x)
    // - d_probabilities_ (Predicted y)
    // - target_token_id (True label y_hat)
    
    // Note: d_neural_input_ and d_probabilities_ must be current.
    // If this is called immediately after a forward pass (like in cognitiveStep), they are.
    // To be safe, you can re-copy neural_output to d_neural_input_ if needed, 
    // but assuming they are fresh saves bandwidth.
    
    int total_weights = config_.vocab_size * config_.output_dim;
    int block_size = 256;
    int grid_size = (total_weights + block_size - 1) / block_size;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    
    updateDecoderWeightsKernel<<<grid_size, block_size, 0, stream>>>(
        d_projection_matrix_, 
        d_neural_input_, 
        d_probabilities_, 
        target_token_id, 
        config_.vocab_size, 
        config_.output_dim, 
        learning_rate
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void GPUDecoder::saveWeights(const std::string& filepath) {
    size_t matrix_size = config_.vocab_size * config_.output_dim;
    std::vector<float> h_projection_matrix(matrix_size);
    std::vector<float> h_projection_bias(config_.vocab_size);
    
    CUDA_CHECK(cudaMemcpy(h_projection_matrix.data(), d_projection_matrix_,
                          matrix_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_projection_bias.data(), d_projection_bias_,
                          config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::ofstream out(filepath, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file for writing: " + filepath);
    
    uint32_t magic = 0x44454344;
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&config_.vocab_size), sizeof(config_.vocab_size));
    out.write(reinterpret_cast<const char*>(&config_.output_dim), sizeof(config_.output_dim));
    out.write(reinterpret_cast<const char*>(h_projection_matrix.data()), matrix_size * sizeof(float));
    out.write(reinterpret_cast<const char*>(h_projection_bias.data()), config_.vocab_size * sizeof(float));
    out.close();
    std::cout << "✓ Saved decoder weights to " << filepath << std::endl;
}

void GPUDecoder::loadWeights(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file for reading: " + filepath);
    
    uint32_t magic, version;
    int vocab_size, output_dim;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    in.read(reinterpret_cast<char*>(&output_dim), sizeof(output_dim));
    
    size_t matrix_size = config_.vocab_size * config_.output_dim;
    std::vector<float> h_projection_matrix(matrix_size);
    in.read(reinterpret_cast<char*>(h_projection_matrix.data()), matrix_size * sizeof(float));
    
    std::vector<float> h_projection_bias(config_.vocab_size);
    in.read(reinterpret_cast<char*>(h_projection_bias.data()), config_.vocab_size * sizeof(float));
    in.close();
    
    CUDA_CHECK(cudaMemcpy(d_projection_matrix_, h_projection_matrix.data(),
                          matrix_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_projection_bias_, h_projection_bias.data(),
                          config_.vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "✓ Loaded decoder weights from " << filepath << std::endl;
}
