#include "persistence/CheckpointWriter.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <system_error>
#include <vector>

namespace persistence {

namespace {

template <typename T>
void writePrimitive(std::ostream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void writeBool(std::ostream& stream, bool value) {
    uint8_t as_byte = value ? 1u : 0u;
    writePrimitive(stream, as_byte);
}

void writeString(std::ostream& stream, const std::string& value) {
    uint32_t size = static_cast<uint32_t>(value.size());
    writePrimitive(stream, size);
    if (size > 0) {
        stream.write(value.data(), static_cast<std::streamsize>(size));
    }
}

void writeFloatVector(std::ostream& stream, const std::vector<float>& values) {
    uint64_t count = static_cast<uint64_t>(values.size());
    writePrimitive(stream, count);
    if (count > 0) {
        stream.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(count * sizeof(float)));
    }
}

template <typename T>
void writePodVector(std::ostream& stream, const std::vector<T>& values) {
    uint64_t count = static_cast<uint64_t>(values.size());
    writePrimitive(stream, count);
    if (count > 0) {
        stream.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(count * sizeof(T)));
    }
}

} // namespace

CheckpointWriter::CheckpointWriter(std::string output_path)
    : output_path_(std::move(output_path)) {}

bool CheckpointWriter::ensureParentDirectory() const {
    namespace fs = std::filesystem;
    fs::path path(output_path_);
    fs::path parent = path.parent_path();

    if (parent.empty()) {
        return true;
    }

    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec && !fs::exists(parent)) {
        std::cerr << "❌ Failed to create checkpoint directory \"" << parent
                  << "\": " << ec.message() << std::endl;
        return false;
    }
    return true;
}

bool CheckpointWriter::write(const BrainSnapshot& snapshot) {
    if (!ensureParentDirectory()) {
        return false;
    }

    std::ofstream stream(output_path_, std::ios::binary | std::ios::trunc);
    if (!stream.is_open()) {
        std::cerr << "❌ Failed to open checkpoint file for writing: "
                  << output_path_ << std::endl;
        return false;
    }

    std::vector<SectionLayout> sections(5);
    sections[0].descriptor.type = SectionType::Metadata;
    sections[1].descriptor.type = SectionType::Neurons;
    sections[2].descriptor.type = SectionType::Synapses;
    sections[3].descriptor.type = SectionType::Optimizer;
    sections[4].descriptor.type = SectionType::RandomState;

    uint64_t rng_seed = snapshot.rng_state.seeds.empty()
        ? snapshot.training_step
        : snapshot.rng_state.seeds.front();

    CheckpointHeader header = makeHeader(static_cast<uint32_t>(sections.size()),
                                         snapshot.training_step,
                                         snapshot.tokens_processed,
                                         rng_seed);

    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));

    std::vector<char> descriptor_padding(sections.size() * sizeof(SectionDescriptor), 0);
    stream.write(descriptor_padding.data(), static_cast<std::streamsize>(descriptor_padding.size()));

    sections[0].descriptor.offset_bytes = static_cast<uint64_t>(stream.tellp());
    sections[0].descriptor.length_bytes = writeMetadataSection(stream, snapshot, sections[0]);

    sections[1].descriptor.offset_bytes = static_cast<uint64_t>(stream.tellp());
    sections[1].descriptor.length_bytes = writeNeuronSection(stream, snapshot, sections[1]);

    sections[2].descriptor.offset_bytes = static_cast<uint64_t>(stream.tellp());
    sections[2].descriptor.length_bytes = writeSynapseSection(stream, snapshot, sections[2]);

    sections[3].descriptor.offset_bytes = static_cast<uint64_t>(stream.tellp());
    sections[3].descriptor.length_bytes = writeOptimizerSection(stream, snapshot, sections[3]);

    sections[4].descriptor.offset_bytes = static_cast<uint64_t>(stream.tellp());
    sections[4].descriptor.length_bytes = writeRandomStateSection(stream, snapshot, sections[4]);

    if (!stream.good()) {
        std::cerr << "❌ Error occurred while writing checkpoint payload: "
                  << output_path_ << std::endl;
        return false;
    }

    stream.seekp(sizeof(header), std::ios::beg);
    for (const auto& section : sections) {
        stream.write(reinterpret_cast<const char*>(&section.descriptor),
                     sizeof(section.descriptor));
    }

    stream.flush();
    if (!stream.good()) {
        std::cerr << "❌ Failed to finalize checkpoint file: " << output_path_ << std::endl;
        return false;
    }

    std::cout << "💾 Saved checkpoint to " << output_path_ << std::endl;
    return true;
}

uint64_t CheckpointWriter::writeMetadataSection(std::ostream& stream,
                                                const BrainSnapshot& snapshot,
                                                SectionLayout& layout) {
    const std::streampos start = stream.tellp();

    writePrimitive(stream, snapshot.training_step);
    writePrimitive(stream, snapshot.cognitive_cycles);
    writePrimitive(stream, snapshot.tokens_processed);
    writePrimitive(stream, snapshot.average_reward);
    writePrimitive(stream, snapshot.time_since_consolidation_ms);

    uint32_t module_count = static_cast<uint32_t>(snapshot.modules.size());
    writePrimitive(stream, module_count);

    for (const auto& module : snapshot.modules) {
        writePrimitive(stream, module.module_index);
        writeString(stream, module.config.module_name);
        writePrimitive(stream, module.config.num_neurons);
        writeBool(stream, module.config.enable_plasticity);
        writePrimitive(stream, module.config.learning_rate);
        writePrimitive(stream, module.config.fanout_per_neuron);
        writePrimitive(stream, module.config.num_inputs);
        writePrimitive(stream, module.config.num_outputs);
        writePrimitive(stream, module.config.dopamine_sensitivity);
        writePrimitive(stream, module.config.serotonin_sensitivity);
        writePrimitive(stream, module.config.inhibition_level);
        writePrimitive(stream, module.config.attention_threshold);
        writePrimitive(stream, module.config.excitability_bias);
        writePrimitive(stream, module.dopamine_level);
        writePrimitive(stream, module.serotonin_level);
        writeFloatVector(stream, module.working_memory);
    }

    uint32_t connection_count = static_cast<uint32_t>(snapshot.connections.size());
    writePrimitive(stream, connection_count);

    for (const auto& connection : snapshot.connections) {
        writeString(stream, connection.name);
        writeString(stream, connection.source_module);
        writeString(stream, connection.target_module);
        writeBool(stream, connection.is_excitatory);
        writeBool(stream, connection.plasticity_enabled);
        writePrimitive(stream, connection.current_strength);
        writePrimitive(stream, connection.gating_threshold);
        writePrimitive(stream, connection.plasticity_rate);
        writePrimitive(stream, connection.attention_modulation);
        writePrimitive(stream, connection.average_activity);
        writePrimitive(stream, connection.total_transmitted);
        writePrimitive(stream, connection.activation_count);
        writePrimitive(stream, connection.pre_synaptic_trace);
        writePrimitive(stream, connection.post_synaptic_trace);
    }

    layout.descriptor.record_count = snapshot.modules.size();
    const std::streampos end = stream.tellp();
    return static_cast<uint64_t>(end - start);
}

uint64_t CheckpointWriter::writeNeuronSection(std::ostream& stream,
                                              const BrainSnapshot& snapshot,
                                              SectionLayout& layout) {
    const std::streampos start = stream.tellp();
    uint32_t module_count = static_cast<uint32_t>(snapshot.modules.size());
    writePrimitive(stream, module_count);

    uint64_t total_neurons = 0;
    for (const auto& module : snapshot.modules) {
        writePrimitive(stream, module.module_index);
        uint64_t neuron_count = static_cast<uint64_t>(module.neurons.size());
        writePrimitive(stream, neuron_count);
        total_neurons += neuron_count;
        if (neuron_count > 0) {
            stream.write(reinterpret_cast<const char*>(module.neurons.data()),
                         static_cast<std::streamsize>(neuron_count * sizeof(GPUNeuronState)));
        }
    }

    layout.descriptor.record_count = total_neurons;
    const std::streampos end = stream.tellp();
    return static_cast<uint64_t>(end - start);
}

uint64_t CheckpointWriter::writeSynapseSection(std::ostream& stream,
                                               const BrainSnapshot& snapshot,
                                               SectionLayout& layout) {
    const std::streampos start = stream.tellp();
    uint32_t module_count = static_cast<uint32_t>(snapshot.modules.size());
    writePrimitive(stream, module_count);

    uint64_t total_synapses = 0;
    for (const auto& module : snapshot.modules) {
        writePrimitive(stream, module.module_index);
        uint64_t synapse_count = static_cast<uint64_t>(module.synapses.size());
        writePrimitive(stream, synapse_count);
        total_synapses += synapse_count;
        if (synapse_count > 0) {
            stream.write(reinterpret_cast<const char*>(module.synapses.data()),
                         static_cast<std::streamsize>(synapse_count * sizeof(GPUSynapse)));
        }
    }

    layout.descriptor.record_count = total_synapses;
    const std::streampos end = stream.tellp();
    return static_cast<uint64_t>(end - start);
}

uint64_t CheckpointWriter::writeOptimizerSection(std::ostream& stream,
                                                 const BrainSnapshot& snapshot,
                                                 SectionLayout& layout) {
    const std::streampos start = stream.tellp();
    const auto& blob = snapshot.optimizer_state.learning_state_blob;
    writePrimitive(stream, static_cast<uint64_t>(blob.size()));
    if (!blob.empty()) {
        stream.write(reinterpret_cast<const char*>(blob.data()),
                     static_cast<std::streamsize>(blob.size()));
    }
    layout.descriptor.record_count = blob.empty() ? 0 : 1;
    const std::streampos end = stream.tellp();
    return static_cast<uint64_t>(end - start);
}

uint64_t CheckpointWriter::writeRandomStateSection(std::ostream& stream,
                                                   const BrainSnapshot& snapshot,
                                                   SectionLayout& layout) {
    const std::streampos start = stream.tellp();
    writePodVector(stream, snapshot.rng_state.seeds);
    layout.descriptor.record_count = snapshot.rng_state.seeds.size();
    const std::streampos end = stream.tellp();
    return static_cast<uint64_t>(end - start);
}

} // namespace persistence


