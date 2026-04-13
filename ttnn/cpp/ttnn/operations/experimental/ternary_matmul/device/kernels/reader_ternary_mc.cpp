// SPDX-License-Identifier: Apache-2.0
// reader_ternary_mc.cpp — true 2-bit DRAM, BFP2_b L1 layout for HW unpack.
//
// DRAM holds 64 uint32 (256 bytes) of pure 2-bit mantissa per 32x32 tile.
// At kernel start, we initialize every cb_in1 slot's 64-byte exponent block
// with the constant 0x7F7F7F7F pattern (shared exp = 0 → scale = 2^0 = 1),
// which is the bfp2_b encoding for ternary {-1, 0, +1}. Per tile, we DMA
// only the 256-byte mantissa from DRAM into the slot's mantissa region.
// matmul_tiles then reads each cb_in1 slot as a full 320-byte BFP2_b tile
// and the Tensix unpacker decodes it natively.
//
// Runtime args:
//   0: act_addr
//   1: packed_addr     (mantissa-only DRAM base, 256 bytes/tile)
//   2: Kt
//   3: Nt
//   4: Mt
//   5: nt_start
//   6: nt_count
//   7: in0_block_w

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

constexpr uint32_t BFP2_TILE_BYTES = 320;
constexpr uint32_t BFP2_EXP_BYTES  = 64;
constexpr uint32_t BFP2_MAN_BYTES  = 256;
constexpr uint32_t EXP_FILL_WORD   = 0x7F7F7F7Fu;  // 4× exp byte = 127

void kernel_main() {
    uint32_t act_addr     = get_arg_val<uint32_t>(0);
    uint32_t packed_addr  = get_arg_val<uint32_t>(1);
    uint32_t Kt           = get_arg_val<uint32_t>(2);
    uint32_t Nt           = get_arg_val<uint32_t>(3);
    uint32_t Mt           = get_arg_val<uint32_t>(4);
    uint32_t nt_start     = get_arg_val<uint32_t>(5);
    uint32_t nt_count     = get_arg_val<uint32_t>(6);
    uint32_t in0_block_w  = get_arg_val<uint32_t>(7);

    constexpr auto act_accessor_args = TensorAccessorArgs<0>();
    constexpr auto weight_accessor_args =
        TensorAccessorArgs<act_accessor_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;

    const uint32_t act_page_bytes = get_local_cb_interface(cb_in0).fifo_page_size;
    // cb_in1 page size is BFP2_TILE_BYTES (320), but we only DMA mantissa.
    const auto act_tensor = TensorAccessor(act_accessor_args, act_addr, act_page_bytes);
    const auto weight_tensor = TensorAccessor(weight_accessor_args, packed_addr, BFP2_MAN_BYTES);

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_in0);
    experimental::CircularBuffer cb1(cb_in1);

    // One-shot init: fill every cb_in1 slot's exponent block with 0x7F.
    // Skip if the very first slot already shows the pattern — under trace
    // mode L1 contents survive across kernel launches, so the init only
    // needs to run on the first invocation of this program.
    const uint32_t cb1_base = get_write_ptr(cb_in1);
    const uint32_t cb1_num_slots = get_local_cb_interface(cb_in1).fifo_num_pages;
    {
        volatile tt_l1_ptr uint32_t* probe =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb1_base);
        if (*probe != EXP_FILL_WORD) {
            for (uint32_t slot = 0; slot < cb1_num_slots; ++slot) {
                volatile tt_l1_ptr uint32_t* exp_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        cb1_base + slot * BFP2_TILE_BYTES);
                for (uint32_t i = 0; i < BFP2_EXP_BYTES / 4; ++i) {
                    exp_ptr[i] = EXP_FILL_WORD;
                }
            }
        }
    }

    const uint32_t num_k_blocks = Kt / in0_block_w;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            const uint32_t kt_base = kb * in0_block_w;

            cb0.reserve_back(in0_block_w);
            cb1.reserve_back(in0_block_w * nt_count);

            for (uint32_t k = 0; k < in0_block_w; ++k) {
                uint32_t act_tile_id = mt * Kt + (kt_base + k);
                noc.async_read(act_tensor, cb0, act_page_bytes,
                               {.page_id = act_tile_id},
                               {.offset_bytes = k * act_page_bytes});
            }

            // Mantissa reads: skip the first 64 bytes of every tile slot.
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
            cb0.push_back(in0_block_w);
            cb1.push_back(in0_block_w * nt_count);
        }
    }
}
