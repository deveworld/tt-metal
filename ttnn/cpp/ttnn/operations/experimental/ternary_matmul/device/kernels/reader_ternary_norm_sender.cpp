// SPDX-License-Identifier: Apache-2.0
// reader_ternary_norm_sender.cpp — BRISC sender for fused RMSNorm + ternary
// matmul. Reads raw activation from DRAM, multicasts ALL Kt tiles to every
// receiver's cb_raw, fills constant tiles (scaler, eps), and streams gamma.
//
// The compute kernel needs all Kt tiles upfront for the sum-of-squares
// reduction, so we send the entire activation block in one multicast round
// (unlike the standard sender which pipelines K-blocks).
//
// MUST run on BRISC/NOC_0 (see reader_ternary_act_sender.cpp for why).
//
// Compile-time args:
//   0: Kt
//   1: Mt
//   2: in0_block_w  (unused by sender, keeps arg layout aligned)
//   3: mcast_x_start
//   4: mcast_y_start
//   5: mcast_x_end
//   6: mcast_y_end
//   7: num_mcast_dests
//   8: sender_sem_id
//   9: receiver_sem_id
//  10..: activation TensorAccessorArgs
//  following: gamma TensorAccessorArgs
//
// Common runtime args:
//   0: act_addr
//   1: gamma_addr
//   2: scaler_val_u32 (bit-cast float 1/K)
//   3: eps_val_u32    (bit-cast float epsilon)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t act_addr    = get_common_arg_val<uint32_t>(0);
    uint32_t gamma_addr  = get_common_arg_val<uint32_t>(1);
    uint32_t scaler_u32  = get_common_arg_val<uint32_t>(2);
    uint32_t eps_u32     = get_common_arg_val<uint32_t>(3);

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

    constexpr auto act_accessor_args   = TensorAccessorArgs<10>();
    constexpr auto gamma_accessor_args =
        TensorAccessorArgs<act_accessor_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_raw    = 2;  // c_2
    constexpr uint32_t cb_gamma  = 3;  // c_3
    constexpr uint32_t cb_scaler = 6;  // c_6
    constexpr uint32_t cb_eps    = 7;  // c_7

    const uint32_t act_page_bytes   = get_local_cb_interface(cb_raw).fifo_page_size;
    const uint32_t gamma_page_bytes = get_local_cb_interface(cb_gamma).fifo_page_size;

    const auto act_tensor   = TensorAccessor(act_accessor_args, act_addr, act_page_bytes);
    const auto gamma_tensor = TensorAccessor(gamma_accessor_args, gamma_addr, gamma_page_bytes);

    experimental::CircularBuffer cb_raw_buf(cb_raw);
    experimental::CircularBuffer cb_gamma_buf(cb_gamma);

    const uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    const uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t* sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);

    // Pre-set sender's own receiver_sem to VALID for multicast broadcast.
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr), VALID);

    // Fill constant tiles (scaler = 1/K, epsilon).
    fill_cb_with_value(cb_scaler, scaler_u32);
    fill_cb_with_value(cb_eps, eps_u32);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        // Wait for every receiver to signal ready.
        if constexpr (num_mcast_dests > 0) {
            noc_semaphore_wait(sender_sem_ptr, num_mcast_dests);
            noc_semaphore_set(sender_sem_ptr, 0);
        }

        // Read ALL Kt activation tiles from DRAM into local cb_raw.
        cb_raw_buf.reserve_back(Kt);
        const uint32_t cb_raw_wr_ptr = get_write_ptr(cb_raw);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            uint32_t tile_id = mt * Kt + kt;
            uint64_t src_noc_addr = get_noc_addr(tile_id, act_tensor);
            noc_async_read(src_noc_addr,
                           cb_raw_wr_ptr + kt * act_page_bytes,
                           act_page_bytes);
        }
        noc_async_read_barrier();

        // Multicast all Kt tiles to every receiver's cb_raw.
        if constexpr (num_mcast_dests > 0) {
            const uint64_t mcast_dest_addr = get_noc_multicast_addr(
                mcast_x_start, mcast_y_start,
                mcast_x_end,   mcast_y_end,
                cb_raw_wr_ptr);
            const uint32_t block_bytes = Kt * act_page_bytes;

            noc_async_write_multicast(
                cb_raw_wr_ptr,
                mcast_dest_addr,
                block_bytes,
                num_mcast_dests,
                /*linked=*/false);
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

        cb_raw_buf.push_back(Kt);

        // Batch all Kt gamma reads into one barrier instead of Kt
        // per-tile barriers (saves Kt × NoC-barrier latency on the
        // critical path because compute Phase 2 pops one gamma per tile).
        cb_gamma_buf.reserve_back(Kt);
        const uint32_t gamma_base_wr_ptr = get_write_ptr(cb_gamma);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            uint64_t gamma_noc_addr = get_noc_addr(kt, gamma_tensor);
            noc_async_read(
                gamma_noc_addr,
                gamma_base_wr_ptr + kt * gamma_page_bytes,
                gamma_page_bytes);
        }
        noc_async_read_barrier();
        cb_gamma_buf.push_back(Kt);
    }
}
