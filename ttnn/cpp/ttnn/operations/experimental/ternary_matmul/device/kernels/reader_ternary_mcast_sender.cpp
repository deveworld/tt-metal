// SPDX-License-Identifier: Apache-2.0
// reader_ternary_mcast_sender.cpp — NCRISC on the single sender core.
//
// Reads all Kt activation tiles from DRAM once, multicasts them to every
// receiver core's cb_in0, then signals the receiver semaphore. Afterwards,
// reads its own nt_count weight tiles from DRAM for the local compute work.
//
// Runtime args:
//   0: act_addr
//   1: packed_addr
//   2: Kt
//   3: Nt
//   4: Mt               (must be 1 for decode)
//   5: nt_start         (sender's own slice)
//   6: nt_count
//   7: mcast_noc_x_start (physical NoC)
//   8: mcast_noc_y_start
//   9: mcast_noc_x_end
//  10: mcast_noc_y_end
//  11: num_receivers    (total receivers excluding sender)
//
// Compile-time args:
//   [act TensorAccessor args], [weight TensorAccessor args], named cb_in0,
//   sender_sem_id (named), receiver_sem_id (named).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t act_addr             = get_arg_val<uint32_t>(0);
    uint32_t packed_addr          = get_arg_val<uint32_t>(1);
    uint32_t Kt                   = get_arg_val<uint32_t>(2);
    uint32_t Nt                   = get_arg_val<uint32_t>(3);
    uint32_t Mt                   = get_arg_val<uint32_t>(4);
    uint32_t nt_start             = get_arg_val<uint32_t>(5);
    uint32_t nt_count             = get_arg_val<uint32_t>(6);
    uint32_t mcast_x_start        = get_arg_val<uint32_t>(7);
    uint32_t mcast_y_start        = get_arg_val<uint32_t>(8);
    uint32_t mcast_x_end          = get_arg_val<uint32_t>(9);
    uint32_t mcast_y_end          = get_arg_val<uint32_t>(10);
    uint32_t num_receivers        = get_arg_val<uint32_t>(11);

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

    constexpr uint32_t sender_sem_ct_idx =
        weight_accessor_args.next_compile_time_args_offset();
    constexpr uint32_t receiver_sem_ct_idx = sender_sem_ct_idx + 1;
    experimental::Semaphore<> sender_sem(get_compile_time_arg_val(sender_sem_ct_idx));
    experimental::Semaphore<> receiver_sem(get_compile_time_arg_val(receiver_sem_ct_idx));

    // set_multicast reads from the local copy and propagates to receivers,
    // so the source value must be VALID up-front.
    if (num_receivers > 0) {
        receiver_sem.set(VALID);
    }

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        // Wait for every receiver to signal readiness (no-op if num_receivers=0)
        if (num_receivers > 0) {
            sender_sem.wait(num_receivers);
            sender_sem.set(0);
        }

        // Reserve space in local cb_in0 for the full K slice.
        cb0.reserve_back(Kt);

        // Issue all Kt activation reads in parallel, one barrier.
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            uint32_t act_tile_id = mt * Kt + kt;
            noc.async_read(act_tensor, cb0, act_page_bytes,
                           {.page_id = act_tile_id},
                           {.offset_bytes = kt * act_page_bytes});
        }
        noc.async_read_barrier();

        // Multicast the whole Kt-tile block to every receiver's cb_in0
        // (skip entirely when there are no receivers).
        if (num_receivers > 0) {
            const uint32_t cb0_wr_ptr = cb0.get_write_ptr();
            const uint64_t mcast_dst_addr = get_noc_multicast_addr(
                mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end,
                cb0_wr_ptr, noc.get_noc_id());
            noc_async_write_multicast(
                cb0_wr_ptr, mcast_dst_addr,
                Kt * act_page_bytes, num_receivers, false, noc.get_noc_id());
            noc_async_writes_flushed(noc.get_noc_id());

            receiver_sem.set_multicast(
                noc, mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end,
                num_receivers);
        }

        // Local push — sender has the data already from its DRAM read.
        cb0.push_back(Kt);

        // Weight reads for sender's own nt_count N tiles. kt-major order
        // (matches matmul_block's expectation of in0_block_w * nt_count).
        cb1.reserve_back(Kt * nt_count);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            for (uint32_t nc = 0; nc < nt_count; ++nc) {
                uint32_t nt = nt_start + nc;
                uint32_t w_tile_id = kt * Nt + nt;
                uint32_t offset = (kt * nt_count + nc) * weight_page_bytes;
                noc.async_read(weight_tensor, cb1, weight_page_bytes,
                               {.page_id = w_tile_id},
                               {.offset_bytes = offset});
            }
        }
        noc.async_read_barrier();
        cb1.push_back(Kt * nt_count);
    }
}
