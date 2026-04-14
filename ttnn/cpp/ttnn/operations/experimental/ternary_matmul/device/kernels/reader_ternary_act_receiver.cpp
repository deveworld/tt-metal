// SPDX-License-Identifier: Apache-2.0
// reader_ternary_act_receiver.cpp — BRISC kernel on every non-sender core.
//
// Does NOT read activation from DRAM. Instead, signals the sender (by
// atomically incrementing the sender's sender_sem), then spin-waits on
// the local receiver_sem until the sender multicasts the data + sets
// the sem to VALID. Then pushes cb_in0 so compute can consume. Consumer
// side of the activation-multicast pattern.
//
// Runs on BRISC/NOC_0 to match the sender (see
// reader_ternary_act_sender.cpp for why NCRISC-originated multicast is
// unreliable on Blackhole).
//
// Compile-time args (shape + semaphore config):
//   0: Kt
//   1: Mt
//   2: in0_block_w
//   3: sender_noc_x        (physical NoC x of the sender core)
//   4: sender_noc_y
//   5: sender_sem_id
//   6: receiver_sem_id
//
// No runtime args — every receiver core is identical.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t Kt              = get_compile_time_arg_val(0);
    constexpr uint32_t Mt              = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_w     = get_compile_time_arg_val(2);
    constexpr uint32_t sender_noc_x    = get_compile_time_arg_val(3);
    constexpr uint32_t sender_noc_y    = get_compile_time_arg_val(4);
    constexpr uint32_t sender_sem_id   = get_compile_time_arg_val(5);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(6);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t num_k_blocks = Kt / in0_block_w;

    experimental::CircularBuffer cb0(cb_in0);

    const uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);

    const uint64_t sender_sem_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(sender_sem_id));

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            // Reserve space for the multicast destination before signaling
            // the sender — the sender's multicast write lands at the current
            // cb0 write pointer.
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
