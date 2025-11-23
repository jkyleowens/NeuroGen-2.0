#include <iostream>
#include <memory>
#include <chrono>
#include <vector>
#include "modules/BrainOrchestrator.h"

void benchmarkMode(BrainOrchestrator::ProcessingMode mode, const std::string& mode_name) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmarking: " << mode_name << std::endl;
    std::cout << "========================================" << std::endl;
    
    BrainOrchestrator::Config config;
    config.gpu_device_id = 0;
    config.time_step_ms = 1.0f;
    config.enable_parallel_execution = true;
    config.enable_consolidation = false;
    config.processing_mode = mode;
    config.max_pipeline_depth = 8;
    
    auto brain = std::make_unique<BrainOrchestrator>(config);
    brain->initializeModules();
    brain->createConnectome();
    
    // Prepare input
    std::vector<float> input(512, 1.0f);
    
    // Warmup
    for (int i = 0; i < 100; ++i) {
        brain->cognitiveStep(input);
    }
    
    // Benchmark
    const int num_steps = 2000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_steps; ++i) {
        brain->cognitiveStep(input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    double hz = num_steps / elapsed.count();
    double ms_per_step = (elapsed.count() * 1000.0) / num_steps;
    
    std::cout << "\n📊 Results:" << std::endl;
    std::cout << "   Steps: " << num_steps << std::endl;
    std::cout << "   Time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "   Throughput: " << hz << " Hz" << std::endl;
    std::cout << "   Latency: " << ms_per_step << " ms/step" << std::endl;
}

int main() {
    std::cout << "🧠 NeuroGen 2.0: Pipeline vs Sequential Benchmark\n" << std::endl;
    
    // Test sequential mode
    benchmarkMode(BrainOrchestrator::ProcessingMode::SEQUENTIAL, "SEQUENTIAL MODE");
    
    // Test pipelined mode
    benchmarkMode(BrainOrchestrator::ProcessingMode::PIPELINED, "PIPELINED MODE");
    
    return 0;
}

