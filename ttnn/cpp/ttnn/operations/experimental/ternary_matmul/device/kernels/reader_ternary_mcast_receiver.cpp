// SPDX-License-Identifier: Apache-2.0
// reader_ternary_mcast_receiver.cpp — NCRISC on receiver cores.
//
// Does NOT read activation from DRAM. Instead, signals the sender that it's
// ready, then waits for the sender to multicast Kt activation tiles into
// cb_in0. Reads its own nt_count weight tiles from DRAM after the activation
// arrives.
//
// Runtime args:
//   0: packed_addr
//   1: Kt
//   2: Nt
//   3: Mt
//   4: nt_start
//   5: nt_count
//   6: sender_noc_x   (physical)
//   7: sender_noc_y
//
// Compile-time args:
//   [act TensorAccessor args], [weight TensorAccessor args],
//   sender_sem_id, receiver_sem_id

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t packed_addr     = get_arg_val<uint32_t>(0);
    uint32_t Kt              = get_arg_val<uint32_t>(1);
    uint32_t Nt              = get_arg_val<uint32_t>(2);
    uint32_t Mt              = get_arg_val<uint32_t>(3);
    uint32_t nt_start        = get_arg_val<uint32_t>(4);
    uint32_t nt_count        = get_arg_val<uint32_t>(5);
    uint32_t sender_noc_x    = get_arg_val<uint32_t>(6);
    uint32_t sender_noc_y    = get_arg_val<uint32_t>(7);

    // Activation accessor still needs CT args for kernel build symmetry; we
    // don't actually read it.
    constexpr auto act_accessor_args = TensorAccessorArgs<0>();
    constexpr auto weight_accessor_args =
        TensorAccessorArgs<act_accessor_args.next_compile_time_args_offset()>();
    constexpr uint32_t sender_sem_ct_idx =
        weight_accessor_args.next_compile_time_args_offset();
    constexpr uint32_t receiver_sem_ct_idx = sender_sem_ct_idx + 1;

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;

    const uint32_t weight_page_bytes = get_local_cb_interface(cb_in1).fifo_page_size;
    const auto weight_tensor = TensorAccessor(weight_accessor_args, packed_addr, weight_page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_in0);
    experimental::CircularBuffer cb1(cb_in1);
    experimental::Semaphore<> sender_sem(get_compile_time_arg_val(sender_sem_ct_idx));
    experimental::Semaphore<> receiver_sem(get_compile_time_arg_val(receiver_sem_ct_idx));

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        // Reserve cb_in0 space so write-pointer matches sender's expectation.
        cb0.reserve_back(Kt);

        // Receiver-side handshake: reset our sem, tell sender we're ready.
        receiver_sem.set(INVALID);
        sender_sem.up(noc, sender_noc_x, sender_noc_y, 1);

        // Block until sender multicasts data + sets our sem to VALID.
        receiver_sem.wait(VALID);

        // Activation block is now in our cb_in0 — make it visible to compute.
        cb0.push_back(Kt);

        // Read our own weight tiles (kt-major).
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
