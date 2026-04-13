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

    auto device_grid = activation.device()->compute_with_storage_grid_size();
    uint32_t max_cores_total = device_grid.x * device_grid.y;
    constexpr uint32_t MAX_NT_PER_CORE = 8;  // dest register constraint

    // Option A: largest rectangular cols × rows that divides Nt and gives
    // nt_per_core ≤ MAX_NT_PER_CORE. Enables activation multicast.
    uint32_t rect_cores = 1, rect_rows = 1, rect_cols = 1;
    for (uint32_t r = 1; r <= device_grid.y; ++r) {
        for (uint32_t c = 1; c <= device_grid.x; ++c) {
            uint32_t total = r * c;
            if (total > Nt) continue;
            if (Nt % total != 0) continue;
            if (Nt / total > MAX_NT_PER_CORE) continue;
            if (total > rect_cores) {
                rect_cores = total;
                rect_rows = r;
                rect_cols = c;
            }
        }
    }

    // Option B: largest any-shape layout (row-major over full grid). More
    // total cores → shorter compute tail but no multicast possible.
    uint32_t lshape_cores = 1;
    for (uint32_t c = std::min(Nt, max_cores_total); c >= 1; --c) {
        if (Nt % c == 0 && Nt / c <= MAX_NT_PER_CORE) { lshape_cores = c; break; }
    }

    // Pick rectangular only when it uses AT LEAST as many cores as the
    // L-shape; more cores means shorter compute wall time, which matters
    // more than activation-mcast savings in our shapes.
    bool use_mcast = (rect_cores >= lshape_cores) && (rect_cores >= 2);
    uint32_t num_cores = use_mcast ? rect_cores : lshape_cores;
    uint32_t nt_per_core = Nt / num_cores;
    bool use_fast_loop = (nt_per_core <= MAX_NT_PER_CORE);

    // Single K block per matmul.
    uint32_t in0_block_w = Kt;

    // Build per-core list. Rectangular layout packs cores into
    // (0..cols-1, 0..rows-1); L-shape layout uses device_grid.x row width.
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    CoreRangeSet core_set;
    if (use_mcast) {
        for (uint32_t r = 0; r < rect_rows; ++r) {
            for (uint32_t c = 0; c < rect_cols; ++c) {
                cores.push_back({c, r});
            }
        }
        core_set = CoreRangeSet(std::set<CoreRange>{
            CoreRange({0, 0}, {rect_cols - 1, rect_rows - 1})});
    } else {
        std::set<CoreRange> core_ranges;
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord c = {i % device_grid.x, i / device_grid.x};
            cores.push_back(c);
            core_ranges.insert(CoreRange(c, c));
        }
        core_set = CoreRangeSet(core_ranges);
    }

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

    // Circular buffers — single-K-block, 1× depth. No mt iteration so
    // pipelining across K isn't needed; this halves L1 usage.
    uint32_t cb0_tiles = in0_block_w;
    CreateCircularBuffer(program, core_set,
        CircularBufferConfig(cb0_tiles * act_tile_bytes, {{tt::CBIndex::c_0, act_df}})
            .set_page_size(tt::CBIndex::c_0, act_tile_bytes));

    uint32_t cb1_tiles = in0_block_w * nt_per_core;
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

    // === Reader kernel(s) ===
    auto act_accessor = TensorAccessorArgs(*activation.buffer());
    auto packed_accessor = TensorAccessorArgs(*packed_weight.buffer());
    auto out_accessor = TensorAccessorArgs(*output.buffer());

    KernelHandle reader_id = 0;
    KernelHandle sender_id = 0;
    KernelHandle receiver_id = 0;

    if (use_mcast) {
        // Allocate sender/receiver semaphores on every core so they live at
        // the same L1 offset across the grid.
        uint32_t sender_sem_id = CreateSemaphore(program, core_set, 0);
        uint32_t receiver_sem_id = CreateSemaphore(program, core_set, INVALID);

        // Multicast rectangle in physical NoC coords (the whole compute grid
        // we own). The sender sits at logical (0, 0).
        auto top_left_phys = activation.device()->worker_core_from_logical_core(
            CoreCoord{0, 0});
        auto bot_right_phys = activation.device()->worker_core_from_logical_core(
            CoreCoord{rect_cols - 1, rect_rows - 1});
        uint32_t mcast_x_start = top_left_phys.x;
        uint32_t mcast_y_start = top_left_phys.y;
        uint32_t mcast_x_end = bot_right_phys.x;
        uint32_t mcast_y_end = bot_right_phys.y;
        uint32_t num_mcast_dests = num_cores - 1;

        auto sender_phys = activation.device()->worker_core_from_logical_core(
            cores[0]);

        // Shared CT args for both sender and receiver readers: act accessor
        // followed by two semaphore IDs.
        std::vector<uint32_t> reader_ct_args;
        {
            auto act_ct = act_accessor.get_compile_time_args();
            reader_ct_args.insert(reader_ct_args.end(), act_ct.begin(), act_ct.end());
            reader_ct_args.push_back(sender_sem_id);
            reader_ct_args.push_back(receiver_sem_id);
        }

        CoreRangeSet sender_set(std::set<CoreRange>{CoreRange(cores[0], cores[0])});
        sender_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_act_sender.cpp",
            sender_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_ct_args});
        SetRuntimeArgs(program, sender_id, cores[0], {
            activation.buffer()->address(),
            Kt, Mt, in0_block_w,
            mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end,
            num_mcast_dests
        });

        // Receivers = all cores except cores[0]. Build a CoreRangeSet from
        // row (1..cols-1, 0) + rows (0..cols-1, 1..rows-1).
        std::set<CoreRange> recv_ranges;
        if (rect_cols >= 2) {
            recv_ranges.insert(CoreRange({1, 0}, {rect_cols - 1, 0}));
        }
        if (rect_rows >= 2) {
            recv_ranges.insert(CoreRange({0, 1}, {rect_cols - 1, rect_rows - 1}));
        }
        CoreRangeSet receiver_set(recv_ranges);

        receiver_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_act_receiver.cpp",
            receiver_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_ct_args});
        for (uint32_t i = 1; i < num_cores; ++i) {
            SetRuntimeArgs(program, receiver_id, cores[i], {
                Kt, Mt, in0_block_w,
                sender_phys.x, sender_phys.y
            });
        }
    } else {
        std::vector<uint32_t> reader_ct_args;
        {
            auto act_ct = act_accessor.get_compile_time_args();
            reader_ct_args.insert(reader_ct_args.end(), act_ct.begin(), act_ct.end());
        }
        reader_id = CreateKernel(
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

    return {std::move(program), {
        reader_id, sender_id, receiver_id,
        compute_id, writer_id,
        cores, use_mcast
    }};
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
    for (size_t i = 0; i < shared.cores.size(); ++i) {
        const auto& core = shared.cores[i];

        if (shared.use_mcast) {
            if (i == 0) {
                // Sender core: act_addr at position 0
                auto& args = GetRuntimeArgs(program, shared.sender_kernel_id, core);
                args[0] = inputs.input_tensor.buffer()->address();
            }
            // Receivers don't read activation, so no address update needed.
        } else {
            auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id, core);
            reader_args[0] = inputs.input_tensor.buffer()->address();
        }

        auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id, core);
        writer_args[0] = output.buffer()->address();
        writer_args[1] = inputs.weight_tensor.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
