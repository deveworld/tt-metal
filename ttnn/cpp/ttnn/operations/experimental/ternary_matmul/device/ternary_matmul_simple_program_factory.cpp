// SPDX-License-Identifier: Apache-2.0
//
// Multi-core ternary matmul program factory.
//
// Reads packed 2-bit (BFP2_b) ternary weights via the Tensix unpacker
// and runs a block matmul. RISC layout (production in0-sender pattern):
//   BRISC / NOC_0: activation reader (+ multicast sender/receiver)
//   NCRISC / NOC_1: writer = weight DMAs + cb1 exp init + output writes
//
// The weight tensor stores only the 256 B mantissa per tile in DRAM;
// the constant 0x7F exponent block is synthesized in L1 by the writer
// kernel at every launch.
//
// For shapes where a rectangular core layout matches or beats the
// L-shape core count, the factory picks multicast: core (0,0) runs
// the sender kernel (DRAM read → `noc_async_write_multicast` of the
// block → sem mcast), and the rest of the rectangle runs the receiver
// kernel (sem handshake only, no DRAM read). The sender must run on
// BRISC/NOC_0 — multicast writes originated from NCRISC on Blackhole
// corrupt the dispatcher command queue.

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

    // K-block pipelining: split Kt into 2 blocks so cb0/cb1 (sized for the
    // full Kt) become double-buffered — while compute drains block A, the
    // reader/writer can DMA block B. Previous attempt was perf-neutral but
    // was measured before the BRISC sender fix and the full QKV-to-ternary
    // integration; worth retrying now.
    uint32_t in0_block_w = (Kt % 4 == 0 && Kt >= 8) ? (Kt / 4)
                          : (Kt % 2 == 0 && Kt >= 4) ? (Kt / 2)
                          : Kt;

    // Multi-core: distribute Nt tiles across the compute grid.
    // Each matmul_block call has fixed overhead; raising nt_per_core widens
    // the output block (ct_dim) per call and amortises that overhead better
    // than spreading work over more cores that each do nt=1. We prefer
    // nt_per_core ∈ [2, 8]; cap is the half-sync dest register limit.
    auto device_grid = activation.device()->compute_with_storage_grid_size();
    uint32_t max_cores = device_grid.x * device_grid.y;
    constexpr uint32_t MAX_NT_PER_CORE = 8;
    constexpr uint32_t MIN_NT_PER_CORE = 2;

    // Option A (L-shape): largest Nt divisor ≤ max_cores with nt_per_core in
    // [MIN, MAX]. More cores → shorter compute tail. Can't multicast.
    uint32_t lshape_cores = 1;
    for (uint32_t c = std::min(Nt, max_cores); c >= 1; --c) {
        if (Nt % c == 0) {
            uint32_t npc = Nt / c;
            if (npc >= MIN_NT_PER_CORE && npc <= MAX_NT_PER_CORE) {
                lshape_cores = c;
                break;
            }
        }
    }
    if (lshape_cores == 1) {
        for (uint32_t c = std::min(Nt, max_cores); c >= 1; --c) {
            if (Nt % c == 0) { lshape_cores = c; break; }
        }
    }

    // Option B (rectangular): largest rows × cols ≤ grid with same Nt
    // divisibility and nt_per_core constraints. Rect enables activation
    // multicast — one core reads DRAM, broadcasts block to the rest.
    uint32_t rect_cores = 1, rect_rows = 1, rect_cols = 1;
    for (uint32_t r = 1; r <= device_grid.y; ++r) {
        for (uint32_t c = 1; c <= device_grid.x; ++c) {
            uint32_t total = r * c;
            if (total > Nt || total > max_cores) continue;
            if (Nt % total != 0) continue;
            uint32_t npc = Nt / total;
            if (npc < MIN_NT_PER_CORE || npc > MAX_NT_PER_CORE) continue;
            if (total > rect_cores) {
                rect_cores = total;
                rect_rows = r;
                rect_cols = c;
            }
        }
    }

    // Prefer multicast only when rectangular layout doesn't sacrifice cores.
    bool use_mcast = (rect_cores >= lshape_cores) && (rect_cores >= 2);
    uint32_t num_cores = use_mcast ? rect_cores : lshape_cores;
    uint32_t nt_per_core = Nt / num_cores;

    // Build core list + CoreRangeSet. Rect layout uses a single CoreRange;
    // L-shape builds one CoreRange per core (irregular placement safety).
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

    // === Reader kernel(s) (NCRISC, NOC_1): activation reads ===
    auto act_accessor = TensorAccessorArgs(*activation.buffer());
    auto packed_accessor = TensorAccessorArgs(*packed_weight.buffer());
    auto out_accessor = TensorAccessorArgs(*output.buffer());

    KernelHandle reader_id = 0;         // unicast reader
    KernelHandle sender_id = 0;         // mcast sender
    KernelHandle receiver_id = 0;       // mcast receivers

    if (!use_mcast) {
        std::vector<uint32_t> reader_ct_args = {Kt, Mt, in0_block_w};
        {
            auto act_ct = act_accessor.get_compile_time_args();
            reader_ct_args.insert(reader_ct_args.end(), act_ct.begin(), act_ct.end());
        }

        // Activation reader on BRISC/NOC_0 (production in0 sender pattern).
        reader_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_act_only.cpp",
            core_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_ct_args});

        SetCommonRuntimeArgs(program, reader_id, {activation.buffer()->address()});
    } else {
        // Sender on core[0]=(0,0); receivers on the rest of the rect.
        // Semaphores allocated on every core of core_set so they live at
        // the same L1 offset across the rectangle.
        uint32_t sender_sem_id = CreateSemaphore(program, core_set, 0);
        uint32_t receiver_sem_id = CreateSemaphore(
            program, core_set, static_cast<uint32_t>(INVALID));

        // Multicast rectangle in physical NoC coords.
        auto top_left_phys = activation.device()->worker_core_from_logical_core(
            CoreCoord{0, 0});
        auto bot_right_phys = activation.device()->worker_core_from_logical_core(
            CoreCoord{rect_cols - 1, rect_rows - 1});
        uint32_t mcast_x_start = top_left_phys.x;
        uint32_t mcast_y_start = top_left_phys.y;
        uint32_t mcast_x_end   = bot_right_phys.x;
        uint32_t mcast_y_end   = bot_right_phys.y;
        uint32_t num_mcast_dests = num_cores - 1;  // receivers only

        // Sender kernel (core 0 only).
        std::vector<uint32_t> sender_ct_args = {
            Kt, Mt, in0_block_w,
            mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end,
            num_mcast_dests,
            sender_sem_id, receiver_sem_id};
        {
            auto act_ct = act_accessor.get_compile_time_args();
            sender_ct_args.insert(sender_ct_args.end(), act_ct.begin(), act_ct.end());
        }
        CoreRangeSet sender_set(std::set<CoreRange>{CoreRange{{0, 0}, {0, 0}}});
        sender_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_act_sender.cpp",
            sender_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = sender_ct_args});
        SetCommonRuntimeArgs(program, sender_id, {activation.buffer()->address()});

        // Receiver kernel (all cores except (0,0)).
        if (num_mcast_dests > 0) {
            auto sender_phys = activation.device()->worker_core_from_logical_core(
                CoreCoord{0, 0});
            std::vector<uint32_t> recv_ct_args = {
                Kt, Mt, in0_block_w,
                static_cast<uint32_t>(sender_phys.x),
                static_cast<uint32_t>(sender_phys.y),
                sender_sem_id, receiver_sem_id};

            // All cores except (0,0). Use individual ranges so we don't
            // accidentally include the sender.
            std::set<CoreRange> recv_ranges;
            for (uint32_t i = 1; i < num_cores; ++i) {
                recv_ranges.insert(CoreRange(cores[i], cores[i]));
            }
            CoreRangeSet recv_set(recv_ranges);
            receiver_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/reader_ternary_act_receiver.cpp",
                recv_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = recv_ct_args});
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

    // Weight reads + output writes on NCRISC/NOC_1. The activation reader
    // (or mcast sender/receiver) lives on BRISC/NOC_0 — this split mirrors
    // production matmul where the in0 sender must run on BRISC so its
    // multicast writes use NOC_0 (multicast does not work reliably from
    // NCRISC on Blackhole).
    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ternary_matmul/device/kernels/writer_ternary_weight_out.cpp",
        core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
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

    return {std::move(program), {
        reader_id,
        sender_id,
        receiver_id,
        compute_id,
        writer_id,
        use_mcast,
        cores}};
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
    if (shared.use_mcast) {
        auto& sender_common = GetCommonRuntimeArgs(program, shared.sender_kernel_id);
        sender_common[0] = inputs.input_tensor.buffer()->address();
        // Receiver kernel has no common args — activation never goes through
        // it directly.
    } else {
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
