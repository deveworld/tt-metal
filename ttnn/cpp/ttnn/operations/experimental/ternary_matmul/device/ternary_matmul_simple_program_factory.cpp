// SPDX-License-Identifier: Apache-2.0
//
// Simple single-core ternary matmul program factory.
// Reads packed 2-bit ternary weights, unpacks to bf16, runs dense matmul.
// Single core only — for correctness validation (Track A).

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

    // Multi-core: distribute Nt tiles across first row of compute grid.
    // With optimized loop order (mt→kt→nt), nt_per_core must be ≤ 8
    // (half-sync dest register limit). If not achievable, use legacy order.
    auto device_grid = activation.device()->compute_with_storage_grid_size();
    uint32_t max_row_cores = device_grid.x;  // already harvesting-aware
    constexpr uint32_t MAX_NT_PER_CORE = 8;  // dest register constraint
    uint32_t num_cores = 1;
    for (uint32_t c = std::min(Nt, max_row_cores); c >= 1; --c) {
        if (Nt % c == 0) { num_cores = c; break; }
    }
    uint32_t nt_per_core = Nt / num_cores;
    // Use optimized loop order when nt_per_core fits in dest registers
    bool use_fast_loop = (nt_per_core <= MAX_NT_PER_CORE);

    // Build per-core ranges (each core is its own range for safety)
    std::set<CoreRange> core_ranges;
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord c = {i, 0};
        cores.push_back(c);
        core_ranges.insert(CoreRange(c, c));
    }
    CoreRangeSet core_set(core_ranges);

    // Data formats
    auto act_df = datatype_to_dataformat_converter(activation.dtype());
    auto out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t act_tile_bytes = tile_size(act_df);
    uint32_t out_tile_bytes = tile_size(out_df);
    constexpr uint32_t PACKED_TILE_BYTES = 256;  // 32x32 * 2 bits / 8

    // Circular buffers
    // CB0: activation tiles (bf16)
    uint32_t cb0_tiles = 2;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(cb0_tiles * act_tile_bytes, {{tt::CBIndex::c_0, act_df}})
            .set_page_size(tt::CBIndex::c_0, act_tile_bytes));

    // CB1: unpacked weight tiles (bf16)
    uint32_t cb1_tiles = 2;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(cb1_tiles * act_tile_bytes, {{tt::CBIndex::c_1, act_df}})
            .set_page_size(tt::CBIndex::c_1, act_tile_bytes));

    // CB2: scratch for packed data (256 bytes per packed tile)
    // Use uint32 format — page_size must match packed tile size
    auto packed_df = tt::DataFormat::RawUInt32;
    uint32_t cb2_pages = 2;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(cb2_pages * PACKED_TILE_BYTES, {{tt::CBIndex::c_2, packed_df}})
            .set_page_size(tt::CBIndex::c_2, PACKED_TILE_BYTES));

    // CB16: output tiles (bf16)
    uint32_t cb16_tiles = 2;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(cb16_tiles * out_tile_bytes, {{tt::CBIndex::c_16, out_df}})
            .set_page_size(tt::CBIndex::c_16, out_tile_bytes));

    // === Reader kernel (fused: activation + packed weight) ===
    auto act_accessor = TensorAccessorArgs(*activation.buffer());
    auto packed_accessor = TensorAccessorArgs(*packed_weight.buffer());

    std::vector<uint32_t> reader_ct_args;
    auto act_ct = act_accessor.get_compile_time_args();
    auto packed_ct = packed_accessor.get_compile_time_args();
    reader_ct_args.insert(reader_ct_args.end(), act_ct.begin(), act_ct.end());
    reader_ct_args.insert(reader_ct_args.end(), packed_ct.begin(), packed_ct.end());

    // Select reader/compute kernels based on loop order
    const char* reader_kernel = use_fast_loop
        ? "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_mc.cpp"
        : "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_mc_legacy.cpp";
    const char* compute_kernel = use_fast_loop
        ? "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/ternary_mm_compute.cpp"
        : "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/ternary_mm_compute_legacy.cpp";

    auto reader_id = CreateKernel(
        program,
        reader_kernel,
        core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    // Per-core reader runtime args
    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t nt_start = i * nt_per_core;
        SetRuntimeArgs(program, reader_id, cores[i], {
            activation.buffer()->address(),
            packed_weight.buffer()->address(),
            Kt, Nt, Mt, nt_start, nt_per_core
        });
    }

    // === Compute kernel ===
    auto compute_id = CreateKernel(
        program,
        compute_kernel,
        core_set,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .compile_args = {Mt, Kt, nt_per_core}});

    // === Writer kernel ===
    auto out_accessor = TensorAccessorArgs(*output.buffer());
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

    // Per-core writer runtime args
    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t nt_start = i * nt_per_core;
        SetRuntimeArgs(program, writer_id, cores[i], {
            output.buffer()->address(),
            Mt, Nt, nt_start, nt_per_core
        });
    }

    return {std::move(program), {reader_id, compute_id, writer_id, cores}};
}

void TernaryMatmulSimpleProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TernaryMatmulParams& /*params*/,
    const TernaryMatmulInputs& inputs,
    std::vector<Tensor>& output_tensors) {

    auto& output = output_tensors.at(0);
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    // Update ALL cores' buffer addresses for program cache reuse
    for (const auto& core : shared.cores) {
        auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id, core);
        reader_args[0] = inputs.input_tensor.buffer()->address();
        reader_args[1] = inputs.weight_tensor.buffer()->address();

        auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id, core);
        writer_args[0] = output.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
