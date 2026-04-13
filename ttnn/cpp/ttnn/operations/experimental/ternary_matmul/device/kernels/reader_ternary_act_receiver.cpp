// SPDX-License-Identifier: Apache-2.0
// reader_ternary_act_receiver.cpp — NCRISC on every non-sender core.
//
// Does NOT read activation from DRAM. Instead, signals the sender (by
// atomically incrementing the sender's sender_sem), then spin-waits on
// the local receiver_sem until the sender multicasts the data + sets
// the sem to VALID. Then pushes cb_in0 so compute can consume.
//
// Runtime args:
//   0: Kt
//   1: Mt
//   2: in0_block_w
//   3: sender_noc_x   (physical)
//   4: sender_noc_y
//
// Compile-time args (layout matches sender's):
//   act TensorAccessorArgs (unused but kept for arg-offset parity),
//   sender_sem_id, receiver_sem_id.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t Kt           = get_arg_val<uint32_t>(0);
    uint32_t Mt           = get_arg_val<uint32_t>(1);
    uint32_t in0_block_w  = get_arg_val<uint32_t>(2);
    uint32_t sender_noc_x = get_arg_val<uint32_t>(3);
    uint32_t sender_noc_y = get_arg_val<uint32_t>(4);

    constexpr auto act_accessor_args = TensorAccessorArgs<0>();
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(
        act_accessor_args.next_compile_time_args_offset());
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(
        act_accessor_args.next_compile_time_args_offset() + 1);

    constexpr uint32_t cb_in0 = 0;
    experimental::CircularBuffer cb0(cb_in0);

    const uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);

    // Remote sender_sem address in the sender's L1.
    const uint64_t sender_sem_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(sender_sem_id));

    const uint32_t num_k_blocks = Kt / in0_block_w;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            // Reserve space for the multicast destination before signaling
            // the sender — the sender's multicast write lands at this L1
            // address.
            cb0.reserve_back(in0_block_w);

            // Reset our local sem, then tell the sender we're ready.
            noc_semaphore_set(receiver_sem_ptr, INVALID);
            noc_semaphore_inc(sender_sem_noc_addr, 1);

            // Spin until the sender's multicast sets our sem to VALID.
            noc_semaphore_wait(receiver_sem_ptr, VALID);

            cb0.push_back(in0_block_w);
        }
    }
}
