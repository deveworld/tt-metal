// SPDX-License-Identifier: Apache-2.0
// reader_ternary_mc.cpp — Bfp2_b hardware-unpack variant, batched reads.
//
// Reader issues one full in0_block_w worth of activation and weight reads
// (activation: in0_block_w tiles, weight: in0_block_w × nt_count tiles) and
// then does a single noc_async_read_barrier before pushing them all at once.
// This amortizes barrier overhead across a full K-block.

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
    uint32_t in0_block_w = get_arg_val<uint32_t>(7);

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

    const uint32_t num_k_blocks = Kt / in0_block_w;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            const uint32_t kt_base = kb * in0_block_w;

            // Reserve full batch upfront
            cb0.reserve_back(in0_block_w);
            cb1.reserve_back(in0_block_w * nt_count);

            // Issue all activation reads for this K-block
            for (uint32_t k = 0; k < in0_block_w; ++k) {
                uint32_t act_tile_id = mt * Kt + (kt_base + k);
                noc.async_read(act_tensor, cb0, act_page_bytes,
                               {.page_id = act_tile_id},
                               {.offset_bytes = k * act_page_bytes});
            }

            // Issue all weight reads for this K-block (kt-major, nt-minor
            // ordering matches matmul_block layout)
            for (uint32_t k = 0; k < in0_block_w; ++k) {
                uint32_t kt = kt_base + k;
                for (uint32_t nc = 0; nc < nt_count; ++nc) {
                    uint32_t nt = nt_start + nc;
                    uint32_t w_tile_id = kt * Nt + nt;
                    uint32_t offset = (k * nt_count + nc) * weight_page_bytes;
                    noc.async_read(weight_tensor, cb1, weight_page_bytes,
                                   {.page_id = w_tile_id},
                                   {.offset_bytes = offset});
                }
            }

            // Single barrier for the whole K-block
            noc.async_read_barrier();
            cb0.push_back(in0_block_w);
            cb1.push_back(in0_block_w * nt_count);
        }
    }
}
