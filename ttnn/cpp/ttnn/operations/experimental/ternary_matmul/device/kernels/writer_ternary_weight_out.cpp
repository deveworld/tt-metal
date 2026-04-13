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
// Runtime args:
//   0: out_addr      DRAM address of output tensor
//   1: packed_addr   DRAM base of mantissa-only weight tensor
//   2: Kt
//   3: Nt
//   4: Mt
//   5: nt_start
//   6: nt_count
//   7: in0_block_w
//
// Compile-time args:
//   0: out_cb_idx        (= cb_out)
//   1..: weight TensorAccessorArgs
//   following: out TensorAccessorArgs

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
    uint32_t out_addr     = get_arg_val<uint32_t>(0);
    uint32_t packed_addr  = get_arg_val<uint32_t>(1);
    uint32_t Kt           = get_arg_val<uint32_t>(2);
    uint32_t Nt           = get_arg_val<uint32_t>(3);
    uint32_t Mt           = get_arg_val<uint32_t>(4);
    uint32_t nt_start     = get_arg_val<uint32_t>(5);
    uint32_t nt_count     = get_arg_val<uint32_t>(6);
    uint32_t in0_block_w  = get_arg_val<uint32_t>(7);

    constexpr uint32_t out_cb_idx = get_compile_time_arg_val(0);
    constexpr auto weight_accessor_args = TensorAccessorArgs<1>();
    constexpr auto out_accessor_args =
        TensorAccessorArgs<weight_accessor_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_in1 = 1;

    const uint32_t out_page_bytes = get_local_cb_interface(out_cb_idx).fifo_page_size;
    const auto weight_tensor = TensorAccessor(weight_accessor_args, packed_addr, BFP2_MAN_BYTES);
    const auto out_tensor = TensorAccessor(out_accessor_args, out_addr, out_page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb1(cb_in1);
    experimental::CircularBuffer cbo(out_cb_idx);

    // One-shot 0x7F exp init (cached via L1 probe).
    {
        const uint32_t cb1_base = get_write_ptr(cb_in1);
        const uint32_t cb1_num_slots = get_local_cb_interface(cb_in1).fifo_num_pages;
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
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nc = 0; nc < nt_count; ++nc) {
            uint32_t nt = nt_start + nc;
            uint32_t tile_id = mt * Nt + nt;

            cbo.wait_front(1);
            noc.async_write(cbo, out_tensor, out_page_bytes,
                            {}, {.page_id = tile_id});
            noc.async_writes_flushed();
            cbo.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
