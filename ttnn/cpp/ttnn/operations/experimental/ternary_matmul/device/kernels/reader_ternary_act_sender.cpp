// SPDX-License-Identifier: Apache-2.0
// reader_ternary_act_sender.cpp — BRISC kernel on the single sender core.
//
// Reads all Kt activation tiles from DRAM into local cb_in0, then multicasts
// the block to every receiver's cb_in0 region, then signals the receiver
// semaphore. Receivers pick up the data without any DRAM read. Producer
// side of activation-multicast pattern used by production matmul kernels.
//
// MUST run on BRISC/NOC_0. Multicast writes originated from NCRISC on
// Blackhole hang and corrupt the dispatcher CQ regardless of rectangle
// size, contiguity, explicit noc= arg, or barrier choice. Production
// `reader_bmm_tile_layout_in0_sender_padding.cpp` runs on RISCV_0 for
// the same reason.
//
// Compile-time args:
//   0: Kt
//   1: Mt
//   2: in0_block_w          (equal to Kt under current config)
//   3: mcast_x_start        (physical NoC) rectangle that receives mcast
//   4: mcast_y_start
//   5: mcast_x_end
//   6: mcast_y_end
//   7: num_mcast_dests      (receiver count, excludes sender)
//   8: sender_sem_id
//   9: receiver_sem_id
//  10..: activation TensorAccessorArgs
//
// Common runtime args:
//   0: act_addr             DRAM address of activation tensor

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t act_addr = get_common_arg_val<uint32_t>(0);

    constexpr uint32_t Kt              = get_compile_time_arg_val(0);
    constexpr uint32_t Mt              = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_w     = get_compile_time_arg_val(2);
    constexpr uint32_t mcast_x_start   = get_compile_time_arg_val(3);
    constexpr uint32_t mcast_y_start   = get_compile_time_arg_val(4);
    constexpr uint32_t mcast_x_end     = get_compile_time_arg_val(5);
    constexpr uint32_t mcast_y_end     = get_compile_time_arg_val(6);
    constexpr uint32_t num_mcast_dests = get_compile_time_arg_val(7);
    constexpr uint32_t sender_sem_id   = get_compile_time_arg_val(8);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(9);
    constexpr auto act_accessor_args = TensorAccessorArgs<10>();

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t num_k_blocks = Kt / in0_block_w;

    const uint32_t act_page_bytes = get_local_cb_interface(cb_in0).fifo_page_size;
    const auto act_tensor = TensorAccessor(act_accessor_args, act_addr, act_page_bytes);

    experimental::CircularBuffer cb0(cb_in0);

    const uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    const uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t* sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);

    // The sender's own receiver_sem slot needs VALID so that
    // noc_semaphore_set_multicast broadcasts that value out.
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr), VALID);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            const uint32_t kt_base = kb * in0_block_w;

            // Wait for every receiver to signal "I'm ready for a new block"
            // (they each atomically increment our sender_sem by 1).
            if constexpr (num_mcast_dests > 0) {
                noc_semaphore_wait(sender_sem_ptr, num_mcast_dests);
                noc_semaphore_set(sender_sem_ptr, 0);
            }

            // Reserve + DMA the activation block from DRAM into local cb_in0.
            cb0.reserve_back(in0_block_w);
            const uint32_t cb0_wr_ptr = get_write_ptr(cb_in0);
            for (uint32_t k = 0; k < in0_block_w; ++k) {
                uint32_t tile_id = mt * Kt + (kt_base + k);
                uint64_t src_noc_addr = get_noc_addr(tile_id, act_tensor);
                noc_async_read(src_noc_addr,
                               cb0_wr_ptr + k * act_page_bytes,
                               act_page_bytes);
            }
            noc_async_read_barrier();

            // Multicast the whole block to every receiver's cb_in0.
            // This kernel runs on BRISC/NOC_0 so all NoC calls use the
            // default noc_index = 0.
            if constexpr (num_mcast_dests > 0) {
                const uint64_t mcast_dest_addr = get_noc_multicast_addr(
                    mcast_x_start, mcast_y_start,
                    mcast_x_end,   mcast_y_end,
                    cb0_wr_ptr);
                const uint32_t block_bytes = in0_block_w * act_page_bytes;

                noc_async_write_multicast(
                    cb0_wr_ptr,
                    mcast_dest_addr,
                    block_bytes,
                    num_mcast_dests,
                    /*linked=*/false);
                // Ensure data arrives before the sem write so receivers
                // can't see VALID before the cb0 data is valid.
                noc_async_write_barrier();

                const uint64_t mcast_sem_addr = get_noc_multicast_addr(
                    mcast_x_start, mcast_y_start,
                    mcast_x_end,   mcast_y_end,
                    receiver_sem_addr);
                noc_semaphore_set_multicast(
                    receiver_sem_addr,
                    mcast_sem_addr,
                    num_mcast_dests);
                noc_async_write_barrier();
            }

            cb0.push_back(in0_block_w);
        }
    }
}
