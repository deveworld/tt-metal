// SPDX-License-Identifier: Apache-2.0
//
// Multi-core ternary matmul program factory with activation multicast.
// Uses BFP2_b weight format (HW unpack) + sender/receiver reader split so
// only one core reads activation from DRAM and multicasts to the rest.

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
    auto* device = activation.device();

    auto program = CreateProgram();

    auto act_shape = activation.padded_shape();
    auto out_shape = output.padded_shape();
    uint32_t M = act_shape[-2];
    uint32_t K = act_shape[-1];
    uint32_t N = out_shape[-1];
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    auto device_grid = device->compute_with_storage_grid_size();
    constexpr uint32_t MAX_NT_PER_CORE = 8;

    // Pick the largest rectangular cols × rows core grid such that:
    //   - cols * rows divides Nt
    //   - Nt / (cols*rows) ≤ MAX_NT_PER_CORE
    //   - cols ≤ device_grid.x, rows ≤ device_grid.y
    uint32_t cols = 1, rows = 1, num_cores = 1;
    for (uint32_t r = 1; r <= device_grid.y; ++r) {
        for (uint32_t c = 1; c <= device_grid.x; ++c) {
            uint32_t total = r * c;
            if (total > Nt) continue;
            if (Nt % total != 0) continue;
            if (Nt / total > MAX_NT_PER_CORE) continue;
            if (total > num_cores) {
                num_cores = total;
                cols = c;
                rows = r;
            }
        }
    }
    uint32_t nt_per_core = Nt / num_cores;
    uint32_t in0_block_w = Kt;  // single K block per matmul

    // Build per-core list following row-major order over the rectangle.
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            cores.push_back({c, r});
        }
    }

    // Sender = cores[0] = (0, 0). All others are receivers.
    CoreCoord sender_logical = cores[0];
    CoreRange sender_range(sender_logical, sender_logical);
    std::set<CoreRange> receiver_ranges;
    if (num_cores >= 2) {
        receiver_ranges.insert(CoreRange({0, 0}, {cols - 1, rows - 1}));
    }
    // The receiver kernel is launched on the full bounding rectangle minus
    // the sender — we use the same bounding rect for the multicast destination
    // with EXCLUDE_SRC. To launch the receiver kernel on every receiver core,
    // we place it on the full rect minus {sender}.
    std::set<CoreRange> all_ranges;
    all_ranges.insert(CoreRange({0, 0}, {cols - 1, rows - 1}));
    CoreRangeSet all_set(all_ranges);

    // Receiver core set = all minus sender (we express this as multiple ranges)
    std::set<CoreRange> recv_only_ranges;
    if (rows >= 1 && cols >= 2) {
        // Row 0: (1, 0) to (cols-1, 0)
        recv_only_ranges.insert(CoreRange({1, 0}, {cols - 1, 0}));
    }
    if (rows >= 2) {
        // Rows 1..rows-1: full width
        recv_only_ranges.insert(CoreRange({0, 1}, {cols - 1, rows - 1}));
    }
    CoreRangeSet receiver_set(recv_only_ranges);
    CoreRangeSet sender_set(std::set<CoreRange>{sender_range});
    uint32_t num_receivers = num_cores - 1;

    // Multicast destination rectangle in physical NoC coords.
    // Use the full bounding rect; sender is excluded via EXCLUDE_SRC mode.
    auto top_left_phys = device->worker_core_from_logical_core(CoreCoord{0, 0});
    auto bot_right_phys = device->worker_core_from_logical_core(CoreCoord{cols - 1, rows - 1});
    uint32_t mcast_x_start = top_left_phys.x;
    uint32_t mcast_y_start = top_left_phys.y;
    uint32_t mcast_x_end = bot_right_phys.x;
    uint32_t mcast_y_end = bot_right_phys.y;

    auto sender_phys = device->worker_core_from_logical_core(sender_logical);
    uint32_t sender_noc_x = sender_phys.x;
    uint32_t sender_noc_y = sender_phys.y;

    // Data formats
    auto act_df = datatype_to_dataformat_converter(activation.dtype());
    auto out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t act_tile_bytes = tile_size(act_df);
    uint32_t out_tile_bytes = tile_size(out_df);
    auto weight_df = tt::DataFormat::Bfp2_b;
    uint32_t weight_tile_bytes = tile_size(weight_df);

    // Circular buffers — created on ALL cores (sender + receivers)
    uint32_t cb0_tiles = 2 * in0_block_w;
    CreateCircularBuffer(program, all_set,
        CircularBufferConfig(cb0_tiles * act_tile_bytes, {{tt::CBIndex::c_0, act_df}})
            .set_page_size(tt::CBIndex::c_0, act_tile_bytes));

    uint32_t cb1_tiles = 2 * in0_block_w * nt_per_core;
    CreateCircularBuffer(program, all_set,
        CircularBufferConfig(cb1_tiles * weight_tile_bytes, {{tt::CBIndex::c_1, weight_df}})
            .set_page_size(tt::CBIndex::c_1, weight_tile_bytes));

    uint32_t cb16_tiles = 2;
    CreateCircularBuffer(program, all_set,
        CircularBufferConfig(cb16_tiles * out_tile_bytes, {{tt::CBIndex::c_16, out_df}})
            .set_page_size(tt::CBIndex::c_16, out_tile_bytes));

    // Semaphores (allocated on all cores so addresses are uniform)
    uint32_t sender_sem_id = CreateSemaphore(program, all_set, 0);
    uint32_t receiver_sem_id = CreateSemaphore(program, all_set, INVALID);

    // === Compile-time args (shared between sender and receiver readers) ===
    auto act_accessor = TensorAccessorArgs(*activation.buffer());
    auto packed_accessor = TensorAccessorArgs(*packed_weight.buffer());
    std::vector<uint32_t> reader_ct_args;
    {
        auto act_ct = act_accessor.get_compile_time_args();
        auto packed_ct = packed_accessor.get_compile_time_args();
        reader_ct_args.insert(reader_ct_args.end(), act_ct.begin(), act_ct.end());
        reader_ct_args.insert(reader_ct_args.end(), packed_ct.begin(), packed_ct.end());
        reader_ct_args.push_back(sender_sem_id);
        reader_ct_args.push_back(receiver_sem_id);
    }

    // === Sender reader kernel ===
    auto sender_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_mcast_sender.cpp",
        sender_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    SetRuntimeArgs(program, sender_id, sender_logical, {
        activation.buffer()->address(),
        packed_weight.buffer()->address(),
        Kt, Nt, Mt,
        /*nt_start=*/0, nt_per_core,
        mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end,
        num_receivers
    });

    // === Receiver reader kernel ===
    KernelHandle receiver_id = 0;
    if (num_receivers > 0) {
        receiver_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_mcast_receiver.cpp",
            receiver_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_ct_args});

        for (uint32_t i = 1; i < num_cores; ++i) {
            uint32_t nt_start = i * nt_per_core;
            SetRuntimeArgs(program, receiver_id, cores[i], {
                packed_weight.buffer()->address(),
                Kt, Nt, Mt, nt_start, nt_per_core,
                sender_noc_x, sender_noc_y
            });
        }
    }

    // === Compute kernel — runs on every core ===
    auto compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/ternary_mm_compute.cpp",
        all_set,
        ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = true,
            .compile_args = {Mt, Kt, nt_per_core, in0_block_w}});

    // === Writer kernel — pure output DMA, runs on every core ===
    auto out_accessor = TensorAccessorArgs(*output.buffer());
    std::vector<uint32_t> writer_ct_args = {static_cast<uint32_t>(tt::CBIndex::c_16)};
    {
        auto out_ct = out_accessor.get_compile_time_args();
        writer_ct_args.insert(writer_ct_args.end(), out_ct.begin(), out_ct.end());
    }
    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/writer_out_mc.cpp",
        all_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t nt_start = i * nt_per_core;
        SetRuntimeArgs(program, writer_id, cores[i], {
            output.buffer()->address(),
            Mt, Nt, nt_start, nt_per_core
        });
    }

    return {std::move(program), {sender_id, receiver_id, compute_id, writer_id, cores}};
}

void TernaryMatmulSimpleProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TernaryMatmulParams& /*params*/,
    const TernaryMatmulInputs& inputs,
    std::vector<Tensor>& output_tensors) {

    auto& output = output_tensors.at(0);
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    // Sender (cores[0]): act_addr, packed_addr at positions 0, 1
    {
        auto& args = GetRuntimeArgs(program, shared.sender_kernel_id, shared.cores[0]);
        args[0] = inputs.input_tensor.buffer()->address();
        args[1] = inputs.weight_tensor.buffer()->address();
    }

    // Receivers: only packed_addr (no activation read)
    if (shared.cores.size() > 1) {
        for (size_t i = 1; i < shared.cores.size(); ++i) {
            auto& args = GetRuntimeArgs(program, shared.receiver_kernel_id, shared.cores[i]);
            args[0] = inputs.weight_tensor.buffer()->address();
        }
    }

    // Writer: output_addr at position 0
    for (const auto& core : shared.cores) {
        auto& args = GetRuntimeArgs(program, shared.writer_kernel_id, core);
        args[0] = output.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
