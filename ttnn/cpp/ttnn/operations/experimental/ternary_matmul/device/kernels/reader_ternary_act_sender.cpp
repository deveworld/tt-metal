// SPDX-License-Identifier: Apache-2.0
// reader_ternary_act_sender.cpp — NCRISC kernel on the single sender core.
//
// Reads all Kt activation tiles from DRAM into local cb_in0, then multicasts
// the same data to every receiver core's cb_in0 region, then signals the
// receiver semaphore. Receivers pick up the data without any DRAM read.
//
// Uses the raw noc_async_write_multicast + noc_semaphore_set_multicast API
// (matches production matmul pattern; experimental::Noc wrapper caused
// completion queue corruption in earlier attempts).
//
// Runtime args:
//   0: act_addr
//   1: Kt
//   2: Mt
//   3: in0_block_w           (equal to Kt under current config)
//   4: mcast_dest_noc_x_start (physical NoC)
//   5: mcast_dest_noc_y_start
//   6: mcast_dest_noc_x_end
//   7: mcast_dest_noc_y_end
//   8: num_mcast_dests        (receivers only, excludes sender)
//
// Compile-time args:
//   act TensorAccessorArgs, then sender_sem_id, receiver_sem_id.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "experimental/noc.h"

void kernel_main() {
    uint32_t act_addr          = get_arg_val<uint32_t>(0);
    uint32_t Kt                = get_arg_val<uint32_t>(1);
    uint32_t Mt                = get_arg_val<uint32_t>(2);
    uint32_t in0_block_w       = get_arg_val<uint32_t>(3);
    uint32_t mcast_x_start     = get_arg_val<uint32_t>(4);
    uint32_t mcast_y_start     = get_arg_val<uint32_t>(5);
    uint32_t mcast_x_end       = get_arg_val<uint32_t>(6);
    uint32_t mcast_y_end       = get_arg_val<uint32_t>(7);
    uint32_t num_mcast_dests   = get_arg_val<uint32_t>(8);

    constexpr auto act_accessor_args = TensorAccessorArgs<0>();
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(
        act_accessor_args.next_compile_time_args_offset());
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(
        act_accessor_args.next_compile_time_args_offset() + 1);

    constexpr uint32_t cb_in0 = 0;
    const uint32_t act_page_bytes = get_local_cb_interface(cb_in0).fifo_page_size;
    const auto act_tensor = TensorAccessor(act_accessor_args, act_addr, act_page_bytes);

    experimental::CircularBuffer cb0(cb_in0);

    // Semaphore L1 addresses (allocated on every core at the same offset).
    const uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    const uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t* sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);

    // Set the sender's own receiver_sem L1 slot to VALID so that
    // noc_semaphore_set_multicast can broadcast that value.
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr), VALID);

    const uint32_t num_k_blocks = Kt / in0_block_w;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            const uint32_t kt_base = kb * in0_block_w;

            // Wait for every receiver to signal "I'm ready for a new block"
            // (they each atomically increment our sender_sem by 1).
            if (num_mcast_dests > 0) {
                noc_semaphore_wait(sender_sem_ptr, num_mcast_dests);
                noc_semaphore_set(sender_sem_ptr, 0);
            }

            // Reserve space + DMA the activation block from DRAM into local cb_in0.
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

            // Multicast the whole block to every receiver's cb_in0 slot.
            // EXCLUDE_SRC (default) skips the sender even if it's inside
            // the rectangle.
            if (num_mcast_dests > 0) {
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
#ifdef ARCH_BLACKHOLE
                noc_async_writes_flushed();
#endif

                // Signal data ready on every receiver.
                const uint64_t mcast_sem_addr = get_noc_multicast_addr(
                    mcast_x_start, mcast_y_start,
                    mcast_x_end,   mcast_y_end,
                    receiver_sem_addr);
                noc_semaphore_set_multicast(
                    receiver_sem_addr,
                    mcast_sem_addr,
                    num_mcast_dests);
            }

            cb0.push_back(in0_block_w);
        }
    }
}
