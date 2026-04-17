// SPDX-License-Identifier: Apache-2.0
// reader_ternary_norm_receiver.cpp — BRISC receiver for fused RMSNorm +
// ternary matmul. Receives multicasted activation into cb_raw, fills
// constant tiles (scaler, eps), and independently reads gamma from DRAM.
//
// The sender multicasts ALL Kt activation tiles in one round (the compute
// kernel needs them all upfront for sum-of-squares reduction).
//
// Runs on BRISC/NOC_0 to match sender (see reader_ternary_act_sender.cpp).
//
// Compile-time args:
//   0: Kt
//   1: Mt
//   2: in0_block_w  (unused, keeps layout aligned)
//   3: sender_noc_x
//   4: sender_noc_y
//   5: sender_sem_id
//   6: receiver_sem_id
//   7..: gamma TensorAccessorArgs
//
// Common runtime args:
//   0: gamma_addr
//   1: scaler_val_u32
//   2: eps_val_u32

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t gamma_addr  = get_common_arg_val<uint32_t>(0);
    uint32_t scaler_u32  = get_common_arg_val<uint32_t>(1);
    uint32_t eps_u32     = get_common_arg_val<uint32_t>(2);

    constexpr uint32_t Kt              = get_compile_time_arg_val(0);
    constexpr uint32_t Mt              = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_w     = get_compile_time_arg_val(2);
    constexpr uint32_t sender_noc_x    = get_compile_time_arg_val(3);
    constexpr uint32_t sender_noc_y    = get_compile_time_arg_val(4);
    constexpr uint32_t sender_sem_id   = get_compile_time_arg_val(5);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(6);

    constexpr auto gamma_accessor_args = TensorAccessorArgs<7>();

    constexpr uint32_t cb_raw    = 2;  // c_2
    constexpr uint32_t cb_gamma  = 3;  // c_3
    constexpr uint32_t cb_scaler = 6;  // c_6
    constexpr uint32_t cb_eps    = 7;  // c_7

    const uint32_t gamma_page_bytes = get_local_cb_interface(cb_gamma).fifo_page_size;
    const auto gamma_tensor = TensorAccessor(gamma_accessor_args, gamma_addr, gamma_page_bytes);

    experimental::CircularBuffer cb_raw_buf(cb_raw);
    experimental::CircularBuffer cb_gamma_buf(cb_gamma);

    const uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);

    const uint64_t sender_sem_noc_addr =
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(sender_sem_id));

    // Fill constant tiles (scaler = 1/K, epsilon).
    fill_cb_with_value(cb_scaler, scaler_u32);
    fill_cb_with_value(cb_eps, eps_u32);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        // Reserve space for ALL Kt tiles before signaling sender.
        // Sender's multicast lands at our current cb_raw write pointer.
        cb_raw_buf.reserve_back(Kt);

        // Reset local sem, then tell sender we're ready.
        noc_semaphore_set(receiver_sem_ptr, INVALID);
        noc_semaphore_inc(sender_sem_noc_addr, 1);

        // Spin until sender's multicast sets our sem to VALID.
        noc_semaphore_wait(receiver_sem_ptr, VALID);

        cb_raw_buf.push_back(Kt);

        // Read gamma tiles independently from DRAM (one at a time).
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_gamma_buf.reserve_back(1);
            const uint32_t gamma_wr_ptr = get_write_ptr(cb_gamma);
            uint64_t gamma_noc_addr = get_noc_addr(kt, gamma_tensor);
            noc_async_read(gamma_noc_addr, gamma_wr_ptr, gamma_page_bytes);
            noc_async_read_barrier();
            cb_gamma_buf.push_back(1);
        }
    }
}
