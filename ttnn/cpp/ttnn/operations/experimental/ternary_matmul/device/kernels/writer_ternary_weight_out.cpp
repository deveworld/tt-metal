// SPDX-License-Identifier: Apache-2.0
// writer_ternary_weight_out.cpp — NCRISC kernel handling the per-core
// weight DRAM reads, the L1 exponent-block synth for BFP2_b, and the
// output writes. Runs on NCRISC/NOC_1. The activation reader (and
// multicast sender/receiver) live on BRISC/NOC_0 so the two NoCs run
// in parallel.
//
// The weight CB uses BFP2_b format (320 B/tile, HW unpack). DRAM stores
// only the 256 B mantissa per tile; this kernel fills the constant 0x7F
// exponent block for every cb1 slot at kernel start (the compile-time
// slot count = Kt × nt_count lets the inner loop unroll), then DMAs the
// mantissa into bytes [64..319] of each slot per tile.
//
// Compile-time args:
//   0: out_cb_idx        (= cb_out)
//   1: Kt
//   2: Nt
//   3: Mt
//   4: nt_count
//   5: in0_block_w
//   6..: weight TensorAccessorArgs
//   following: out TensorAccessorArgs
//
// Common runtime args (shared by all cores):
//   0: out_addr      DRAM address of output tensor
//   1: packed_addr   DRAM base of mantissa-only weight tensor
//
// Per-core runtime args:
//   0: nt_start      first N tile assigned to this core

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

constexpr uint32_t BFP2_TILE_BYTES = 320;
constexpr uint32_t BFP2_EXP_BYTES  = 64;
constexpr uint32_t BFP2_MAN_BYTES  = 256;
constexpr uint32_t EXP_FILL_WORD   = 0x7F7F7F7Fu;

void kernel_main() {
    uint32_t out_addr    = get_common_arg_val<uint32_t>(0);
    uint32_t packed_addr = get_common_arg_val<uint32_t>(1);
    uint32_t nt_start    = get_arg_val<uint32_t>(0);

    constexpr uint32_t out_cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t Kt          = get_compile_time_arg_val(1);
    constexpr uint32_t Nt          = get_compile_time_arg_val(2);
    constexpr uint32_t Mt          = get_compile_time_arg_val(3);
    constexpr uint32_t nt_count    = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(5);
    constexpr auto weight_accessor_args = TensorAccessorArgs<6>();
    constexpr auto out_accessor_args =
        TensorAccessorArgs<weight_accessor_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_in1 = 1;

    const uint32_t out_page_bytes = get_local_cb_interface(out_cb_idx).fifo_page_size;
    const auto weight_tensor = TensorAccessor(weight_accessor_args, packed_addr, BFP2_MAN_BYTES);
    const auto out_tensor = TensorAccessor(out_accessor_args, out_addr, out_page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb1(cb_in1);
    experimental::CircularBuffer cbo(out_cb_idx);

    // Fill the 64-byte exponent block of every cb1 slot with 0x7F (encoded
    // exponent for 2^0 = 1) once per kernel launch. The mantissa DMAs below
    // only write bytes [64..319] of each tile, so the exp bytes must be
    // valid before compute's unpacker reads a slot.
    //
    // This runs unconditionally: between ternary_matmul invocations the CB1
    // L1 region can be reused by other kernels so we can't persist state.
    // The slot count is compile-time (Kt × nt_count), which lets the
    // compiler flatten both loops into straight-line stores.
    //
    // An earlier "probe + skip if already 0x7F" optimisation was unsafe:
    // on Blackhole L1 can come up with bytes that already look like 0x7F
    // at the probe address, so the init was occasionally skipped while
    // other slots still held garbage exp bytes → NaN matmul output.
    {
        constexpr uint32_t cb1_num_slots = Kt * nt_count;
        const uint32_t cb1_base = get_write_ptr(cb_in1);
        for (uint32_t slot = 0; slot < cb1_num_slots; ++slot) {
            volatile tt_l1_ptr uint32_t* exp_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    cb1_base + slot * BFP2_TILE_BYTES);
            #pragma GCC unroll 16
            for (uint32_t i = 0; i < BFP2_EXP_BYTES / 4; ++i) {
                exp_ptr[i] = EXP_FILL_WORD;
            }
        }
    }

    constexpr uint32_t num_k_blocks = Kt / in0_block_w;

    // Phase 1: per-K-block weight reads, mirroring reader_ternary_mc.cpp.
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            const uint32_t kt_base = kb * in0_block_w;

            cb1.reserve_back(in0_block_w * nt_count);

            for (uint32_t k = 0; k < in0_block_w; ++k) {
                uint32_t kt = kt_base + k;
                for (uint32_t nc = 0; nc < nt_count; ++nc) {
                    uint32_t nt = nt_start + nc;
                    uint32_t w_tile_id = kt * Nt + nt;
                    uint32_t slot_idx = k * nt_count + nc;
                    uint32_t offset = slot_idx * BFP2_TILE_BYTES + BFP2_EXP_BYTES;
                    noc.async_read(weight_tensor, cb1, BFP2_MAN_BYTES,
                                   {.page_id = w_tile_id},
                                   {.offset_bytes = offset});
                }
            }
            noc.async_read_barrier();
            cb1.push_back(in0_block_w * nt_count);
        }
    }

    // Phase 2: drain cb_out → DRAM.
    // No per-tile flushed: cb_out holds all Nt output tiles (sized
    // nt_per_core+2) and compute finishes before the writer starts draining,
    // so L1 slots are not reused. A single barrier at the end is enough.
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        cbo.wait_front(nt_count);
        for (uint32_t nc = 0; nc < nt_count; ++nc) {
            uint32_t nt = nt_start + nc;
            uint32_t tile_id = mt * Nt + nt;
            noc.async_write(cbo, out_tensor, out_page_bytes,
                            {.offset_bytes = nc * out_page_bytes},
                            {.page_id = tile_id});
        }
        noc.async_write_barrier();
        cbo.pop_front(nt_count);
    }
}
