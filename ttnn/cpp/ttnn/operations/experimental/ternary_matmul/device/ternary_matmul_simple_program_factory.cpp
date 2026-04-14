// SPDX-License-Identifier: Apache-2.0
//
// Multi-core ternary matmul program factory.
// Reads packed 2-bit (BFP2_b) ternary weights via the Tensix unpacker and
// runs a block matmul. Dual-NoC split: NCRISC/NOC_1 reads activations,
// BRISC/NOC_0 reads weights and writes outputs. Weight tensor stores only
// the 256-byte mantissa per tile; the constant 0x7F exponent block is
// synthesized in L1 by the writer kernel.

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

    // Single K block per matmul. Multi-block pipelining was tried but the
    // extra cb_wait/cb_pop sync cancelled the DMA/compute overlap gain —
    // compute is not the bottleneck at these shapes.
    uint32_t in0_block_w = Kt;

    // Multi-core: distribute Nt tiles across the compute grid.
    // Each matmul_block call has fixed overhead; raising nt_per_core widens
    // the output block (ct_dim) per call and amortises that overhead better
    // than spreading work over more cores that each do nt=1. We prefer
    // nt_per_core ∈ [2, 8]; cap is the half-sync dest register limit.
    auto device_grid = activation.device()->compute_with_storage_grid_size();
    uint32_t max_cores = device_grid.x * device_grid.y;
    constexpr uint32_t MAX_NT_PER_CORE = 8;
    constexpr uint32_t MIN_NT_PER_CORE = 2;
    uint32_t num_cores = 1;
    for (uint32_t c = std::min(Nt, max_cores); c >= 1; --c) {
        if (Nt % c == 0) {
            uint32_t npc = Nt / c;
            if (npc >= MIN_NT_PER_CORE && npc <= MAX_NT_PER_CORE) {
                num_cores = c;
                break;
            }
        }
    }
    // Fallback: largest divisor of Nt ≤ max_cores (when Nt is small or prime).
    if (num_cores == 1) {
        for (uint32_t c = std::min(Nt, max_cores); c >= 1; --c) {
            if (Nt % c == 0) { num_cores = c; break; }
        }
    }
    uint32_t nt_per_core = Nt / num_cores;

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

    // CB0 (activation): Kt tiles so compute can wait once for the full K.
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(Kt * act_tile_bytes, {{tt::CBIndex::c_0, act_df}})
            .set_page_size(tt::CBIndex::c_0, act_tile_bytes));

    // CB1 (weight): Kt × nt_per_core BFP2_b tiles.
    uint32_t cb1_tiles = Kt * nt_per_core;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(cb1_tiles * weight_tile_bytes, {{tt::CBIndex::c_1, weight_df}})
            .set_page_size(tt::CBIndex::c_1, weight_tile_bytes));

    // CB16: output tiles (bf16). Sized to hold all Nt tiles so compute can
    // push the full output block without waiting for the writer to drain.
    uint32_t cb16_tiles = nt_per_core + 2;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(cb16_tiles * out_tile_bytes, {{tt::CBIndex::c_16, out_df}})
            .set_page_size(tt::CBIndex::c_16, out_tile_bytes));

    // === Reader (NCRISC, NOC_1): activation reads only ===
    auto act_accessor = TensorAccessorArgs(*activation.buffer());
    auto packed_accessor = TensorAccessorArgs(*packed_weight.buffer());
    auto out_accessor = TensorAccessorArgs(*output.buffer());

    // Kt/Mt/in0_block_w are compile-time so the inner loops unroll and
    // tile_id arithmetic folds into constants.
    std::vector<uint32_t> reader_ct_args = {Kt, Mt, in0_block_w};
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

    // Activation address is shared across all cores — use common runtime args.
    SetCommonRuntimeArgs(program, reader_id, {activation.buffer()->address()});

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
    // Kt/Nt/Mt/nt_count/in0_block_w are compile-time; only nt_start varies
    // per core (can't be compile-time without one kernel per core).
    std::vector<uint32_t> writer_ct_args = {
        static_cast<uint32_t>(tt::CBIndex::c_16),
        Kt, Nt, Mt, nt_per_core, in0_block_w};
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

    // Output and weight DRAM addresses are shared — common runtime args.
    // Only nt_start differs per core, so that stays as a per-core arg.
    SetCommonRuntimeArgs(program, writer_id, {
        output.buffer()->address(),
        packed_weight.buffer()->address()
    });
    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t nt_start = i * nt_per_core;
        SetRuntimeArgs(program, writer_id, cores[i], {nt_start});
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

    // All buffer addresses are common runtime args — updating them is a
    // single write per kernel, not once per core. Per-core nt_start doesn't
    // depend on addresses so it stays as set at program creation.
    {
        auto& reader_common = GetCommonRuntimeArgs(program, shared.reader_kernel_id);
        reader_common[0] = inputs.input_tensor.buffer()->address();
    }
    {
        auto& writer_common = GetCommonRuntimeArgs(program, shared.writer_kernel_id);
        writer_common[0] = output.buffer()->address();
        writer_common[1] = inputs.weight_tensor.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
