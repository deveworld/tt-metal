// SPDX-License-Identifier: Apache-2.0
// reader_ternary_mc.cpp — Bfp2_b hardware-unpack variant.
//
// Each core owns a contiguous N-tile range. Reader DMAs activation tiles
// (bf16) into cb_in0 and weight tiles (Bfp2_b, 320 bytes each) directly into
// cb_in1. The Tensix unpacker decodes bfp2 → bf16 at matmul_tiles time, so
// no software unpack is needed.
//
// Runtime args:
//   0: act_addr      - DRAM address of activation tensor
//   1: packed_addr   - DRAM address of bfp2 weight tensor
//   2: Kt            - total K tiles
//   3: Nt            - total N tiles
//   4: Mt            - total M tiles
//   5: nt_start      - first N tile for this core
//   6: nt_count      - number of N tiles for this core

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t act_addr    = get_arg_val<uint32_t>(0);
    uint32_t packed_addr = get_arg_val<uint32_t>(1);
    uint32_t Kt          = get_arg_val<uint32_t>(2);
    uint32_t Nt          = get_arg_val<uint32_t>(3);
    uint32_t Mt          = get_arg_val<uint32_t>(4);
    uint32_t nt_start    = get_arg_val<uint32_t>(5);
    uint32_t nt_count    = get_arg_val<uint32_t>(6);

    constexpr auto act_accessor_args = TensorAccessorArgs<0>();
    constexpr auto weight_accessor_args =
        TensorAccessorArgs<act_accessor_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;

    const uint32_t act_page_bytes = get_local_cb_interface(cb_in0).fifo_page_size;
    const uint32_t weight_page_bytes = get_local_cb_interface(cb_in1).fifo_page_size;

    const auto act_tensor = TensorAccessor(act_accessor_args, act_addr, act_page_bytes);
    const auto weight_tensor = TensorAccessor(weight_accessor_args, packed_addr, weight_page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_in0);
    experimental::CircularBuffer cb1(cb_in1);

    // Loop order: mt → kt → nt
    // Activation A[mt, kt] is read ONCE per kt, reused across all nt.
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            uint32_t act_tile_id = mt * Kt + kt;
            cb0.reserve_back(1);
            noc.async_read(act_tensor, cb0, act_page_bytes,
                           {.page_id = act_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb0.push_back(1);

            for (uint32_t nc = 0; nc < nt_count; ++nc) {
                uint32_t nt = nt_start + nc;
                uint32_t w_tile_id = kt * Nt + nt;
                cb1.reserve_back(1);
                noc.async_read(weight_tensor, cb1, weight_page_bytes,
                               {.page_id = w_tile_id}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb1.push_back(1);
            }
        }
    }
}
