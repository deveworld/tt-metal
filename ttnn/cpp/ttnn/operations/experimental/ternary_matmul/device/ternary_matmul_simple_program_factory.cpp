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
    uint32_t max_cores = device_grid.x * device_grid.y;
    constexpr uint32_t MAX_NT_PER_CORE = 8;  // dest register constraint
    uint32_t num_cores = 1;
    for (uint32_t c = std::min(Nt, max_cores); c >= 1; --c) {
        if (Nt % c == 0) { num_cores = c; break; }
    }
    uint32_t nt_per_core = Nt / num_cores;
    bool use_fast_loop = (nt_per_core <= MAX_NT_PER_CORE);

    // Pick in0_block_w. Smaller blocks let reader/writer prefetch the next
    // K slice while compute consumes the current one (multi-iteration
    // pipelining). Cap at 16 — enough K work per call to amortize matmul_block
    // setup, while keeping CBs small for overlap.
    // Single K block per matmul: DST accumulation across cb_pop_front
    // didn't preserve partial sums correctly, so we fold the entire K
    // dimension into one acquire/commit window.
    uint32_t in0_block_w = Kt;

    // Build per-core ranges (each core is its own range for safety)
    std::set<CoreRange> core_ranges;
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord c = {i % device_grid.x, i / device_grid.x};
        cores.push_back(c);
        core_ranges.insert(CoreRange(c, c));
    }
    CoreRangeSet core_set(core_ranges);

    // Data formats
    auto act_df = datatype_to_dataformat_converter(activation.dtype());
    auto out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t act_tile_bytes = tile_size(act_df);
    uint32_t out_tile_bytes = tile_size(out_df);
    // Weight is pre-packed as BFP2_b tiles (320 bytes each). The tensor is
    // stored on host as uint32 but the bytes carry the exact bfp2 layout that
    // the Tensix UNPACKER hardware decodes to bf16 on the fly — no SW unpack.
    auto weight_df = tt::DataFormat::Bfp2_b;
    uint32_t weight_tile_bytes = tile_size(weight_df);

    // Circular buffers. cb_in1 exp init is one-shot via L1 cache (kernel
    // probes for the pattern), so 2× pipelining doesn't add init cost.
    uint32_t cb0_tiles = 2 * in0_block_w;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(cb0_tiles * act_tile_bytes, {{tt::CBIndex::c_0, act_df}})
            .set_page_size(tt::CBIndex::c_0, act_tile_bytes));

    uint32_t cb1_tiles = 2 * in0_block_w * nt_per_core;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(cb1_tiles * weight_tile_bytes, {{tt::CBIndex::c_1, weight_df}})
            .set_page_size(tt::CBIndex::c_1, weight_tile_bytes));

    // CB3: activation L1 cache for legacy reader (holds all Kt tiles)
    // Only needed for legacy loop; fast loop reads act once via CB0.
    if (!use_fast_loop) {
        uint32_t cb3_tiles = Kt;
        CreateCircularBuffer(program, core_set,
            CircularBufferConfig(cb3_tiles * act_tile_bytes, {{tt::CBIndex::c_3, act_df}})
                .set_page_size(tt::CBIndex::c_3, act_tile_bytes));
    }

    // CB16: output tiles (bf16)
    uint32_t cb16_tiles = 2;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(cb16_tiles * out_tile_bytes, {{tt::CBIndex::c_16, out_df}})
            .set_page_size(tt::CBIndex::c_16, out_tile_bytes));

    // === Reader (NCRISC, NOC_1): activation reads only ===
    auto act_accessor = TensorAccessorArgs(*activation.buffer());
    auto packed_accessor = TensorAccessorArgs(*packed_weight.buffer());
    auto out_accessor = TensorAccessorArgs(*output.buffer());

    std::vector<uint32_t> reader_ct_args;
    {
        auto act_ct = act_accessor.get_compile_time_args();
        reader_ct_args.insert(reader_ct_args.end(), act_ct.begin(), act_ct.end());
    }

    auto reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_act_only.cpp",
        core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    for (uint32_t i = 0; i < num_cores; ++i) {
        SetRuntimeArgs(program, reader_id, cores[i], {
            activation.buffer()->address(),
            Kt, Mt, in0_block_w
        });
    }

    // === Compute kernel ===
    auto compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/ternary_mm_compute.cpp",
        core_set,
        ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = true,
            .compile_args = {Mt, Kt, nt_per_core, in0_block_w}});

    // === Writer (BRISC, NOC_0): weight reads + output writes ===
    std::vector<uint32_t> writer_ct_args = {static_cast<uint32_t>(tt::CBIndex::c_16)};
    {
        auto packed_ct = packed_accessor.get_compile_time_args();
        auto out_ct = out_accessor.get_compile_time_args();
        writer_ct_args.insert(writer_ct_args.end(), packed_ct.begin(), packed_ct.end());
        writer_ct_args.insert(writer_ct_args.end(), out_ct.begin(), out_ct.end());
    }

    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/writer_ternary_weight_out.cpp",
        core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t nt_start = i * nt_per_core;
        SetRuntimeArgs(program, writer_id, cores[i], {
            output.buffer()->address(),
            packed_weight.buffer()->address(),
            Kt, Nt, Mt, nt_start, nt_per_core, in0_block_w
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

    // Update ALL cores' buffer addresses for program cache reuse.
    // Reader (NCRISC): only the activation address.
    // Writer (BRISC): output address + weight address.
    for (const auto& core : shared.cores) {
        auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id, core);
        reader_args[0] = inputs.input_tensor.buffer()->address();

        auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id, core);
        writer_args[0] = output.buffer()->address();
        writer_args[1] = inputs.weight_tensor.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
