// SPDX-License-Identifier: Apache-2.0
// writer_ternary_weight_out.cpp — BRISC kernel that handles BOTH the per-core
// weight DRAM reads (NOC_0) and the output writes (NOC_0). Reader (NCRISC,
// NOC_1) handles only the activation reads, so both NoCs run in parallel.
//
// The weight CB still uses the BFP2_b layout (320 B/tile). DRAM stores only
// the 256 B mantissa per tile; this kernel pre-fills the constant 0x7F
// exponent block once per program lifetime (L1 probe cached) and per tile
// DMAs only the 256 B mantissa into bytes [64..319].
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
//   2: init_exp      1 on first program launch (fill cb1 exp block), 0 after
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
    uint32_t init_exp    = get_common_arg_val<uint32_t>(2);
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
    // exponent for 2^0 = 1). The mantissa DMAs below only write bytes
    // [64..319] of each tile, so the exp bytes must be valid before
    // compute's unpacker reads a slot.
    //
    // Host drives init_exp: 1 on the first launch of a freshly cached
    // program, 0 on subsequent launches. Between launches of the same
    // cached program the CB1 L1 region belongs to us, so the exp bytes
    // remain intact from the previous fill. Skipping the re-init saves
    // ~cb1_num_slots × 16 volatile stores per launch on every core.
    //
    // (Previous attempts to self-detect first launch via an L1 probe
    // failed on Blackhole: un-touched L1 sometimes comes up with bytes
    // that already look like 0x7F, causing a false-positive skip and NaN
    // output. Driving it from host is reliable.)
    if (init_exp) {
        const uint32_t cb1_base = get_write_ptr(cb_in1);
        const uint32_t cb1_num_slots = get_local_cb_interface(cb_in1).fifo_num_pages;
        for (uint32_t slot = 0; slot < cb1_num_slots; ++slot) {
            volatile tt_l1_ptr uint32_t* exp_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    cb1_base + slot * BFP2_TILE_BYTES);
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
