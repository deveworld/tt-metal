// SPDX-License-Identifier: Apache-2.0
//
// Multi-core ternary matmul program factory.
// Reads packed 2-bit ternary weights, unpacks to bf16, runs dense matmul.
// Parallelizes over N dimension — each core handles a range of output tiles.

#include "ternary_matmul_simple_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::constants;

TernaryMatmulSimpleProgramFactory::cached_program_t TernaryMatmulSimpleProgramFactory::create(
    const TernaryMatmulParams& /*params*/,
    const TernaryMatmulInputs& inputs,
    std::vector<Tensor>& output_tensors) {

    auto& output = output_tensors.at(0);
    const auto& activation = inputs.input_tensor;
    const auto& packed_weight = inputs.weight_tensor;

    auto program = CreateProgram();

    // Dimensions
    auto act_shape = activation.padded_shape();
    auto out_shape = output.padded_shape();
    uint32_t M = act_shape[-2];
    uint32_t K = act_shape[-1];
    uint32_t N = out_shape[-1];
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    // Multi-core: use first row of cores, Nt must divide evenly
    auto device_grid = activation.device()->compute_with_storage_grid_size();
    uint32_t max_row_cores = device_grid.x;
    // Find largest divisor of Nt that fits in one row
    uint32_t num_cores = 1;
    for (uint32_t c = std::min(Nt, max_row_cores); c >= 1; --c) {
        if (Nt % c == 0) { num_cores = c; break; }
    }
    uint32_t nt_per_core = Nt / num_cores;

    // Single-row core range: (0,0) to (num_cores-1, 0)
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        cores.push_back({i, 0});
    }
    CoreRangeSet core_set(CoreRange({0, 0}, {num_cores - 1, 0}));

    // Data formats
    auto act_df = datatype_to_dataformat_converter(activation.dtype());
    auto out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t act_tile_bytes = tile_size(act_df);
    uint32_t out_tile_bytes = tile_size(out_df);
    constexpr uint32_t PACKED_TILE_BYTES = 256;

    // Circular buffers (same on all cores)
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(2 * act_tile_bytes, {{tt::CBIndex::c_0, act_df}})
            .set_page_size(tt::CBIndex::c_0, act_tile_bytes));
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(2 * act_tile_bytes, {{tt::CBIndex::c_1, act_df}})
            .set_page_size(tt::CBIndex::c_1, act_tile_bytes));
    auto packed_df = tt::DataFormat::RawUInt32;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(2 * PACKED_TILE_BYTES, {{tt::CBIndex::c_2, packed_df}})
            .set_page_size(tt::CBIndex::c_2, PACKED_TILE_BYTES));
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(2 * out_tile_bytes, {{tt::CBIndex::c_16, out_df}})
            .set_page_size(tt::CBIndex::c_16, out_tile_bytes));

    // Tensor accessors
    auto act_accessor = TensorAccessorArgs(*activation.buffer());
    auto packed_accessor = TensorAccessorArgs(*packed_weight.buffer());
    auto out_accessor = TensorAccessorArgs(*output.buffer());

    // Reader kernel (multi-core version)
    std::vector<uint32_t> reader_ct_args;
    auto act_ct = act_accessor.get_compile_time_args();
    auto packed_ct = packed_accessor.get_compile_time_args();
    reader_ct_args.insert(reader_ct_args.end(), act_ct.begin(), act_ct.end());
    reader_ct_args.insert(reader_ct_args.end(), packed_ct.begin(), packed_ct.end());

    auto reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_mc.cpp",
        core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    // Compute kernel — each core processes nt_per_core output tiles
    auto compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/ternary_mm_compute.cpp",
        core_set,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .compile_args = {Mt, Kt, nt_per_core}});

    // Writer kernel (multi-core version)
    std::vector<uint32_t> writer_ct_args = {static_cast<uint32_t>(tt::CBIndex::c_16)};
    auto out_ct = out_accessor.get_compile_time_args();
    writer_ct_args.insert(writer_ct_args.end(), out_ct.begin(), out_ct.end());

    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/writer_out_mc.cpp",
        core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    // Set per-core runtime args — each core handles nt_per_core N tiles
    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t nt_start = i * nt_per_core;
        uint32_t nt_count = std::min(nt_per_core, Nt - nt_start);

        SetRuntimeArgs(program, reader_id, cores[i], {
            activation.buffer()->address(),
            packed_weight.buffer()->address(),
            Kt, Nt, Mt, nt_start, nt_count
        });
        SetRuntimeArgs(program, writer_id, cores[i], {
            output.buffer()->address(),
            Mt, Nt, nt_start, nt_count
        });
    }

    return {std::move(program), {reader_id, compute_id, writer_id, cores[0]}};
}

void TernaryMatmulSimpleProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TernaryMatmulParams& /*params*/,
    const TernaryMatmulInputs& inputs,
    std::vector<Tensor>& output_tensors) {

    auto& output = output_tensors.at(0);
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    auto act_shape = inputs.input_tensor.padded_shape();
    auto out_shape = output.padded_shape();
    uint32_t Kt = act_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Nt = out_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Mt = act_shape[-2] / tt::constants::TILE_HEIGHT;

    // Update addresses for core 0 (shapes don't change for cached programs)
    SetRuntimeArgs(program, shared.reader_kernel_id, shared.core, {
        inputs.input_tensor.buffer()->address(),
        inputs.weight_tensor.buffer()->address(),
        Kt, Nt, Mt, 0u, Nt  // core 0 gets all tiles in single-use override
    });
    SetRuntimeArgs(program, shared.writer_kernel_id, shared.core, {
        output.buffer()->address(),
        Mt, Nt, 0u, Nt
    });
}

}  // namespace ttnn::experimental::prim
