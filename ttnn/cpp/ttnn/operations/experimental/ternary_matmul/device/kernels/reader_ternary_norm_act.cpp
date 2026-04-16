// SPDX-License-Identifier: Apache-2.0
// reader_ternary_norm_act.cpp — BRISC reader for the fused RMSNorm + ternary
// matmul kernel. Reads raw activation into cb_raw, fills constant tiles
// (scaler, epsilon) in L1, and streams gamma tiles into cb_gamma one at a
// time so the compute kernel can normalise the activation in-line.
//
// Runs on BRISC (RISCV_0 / NOC_0) — same as the non-fused reader, matching
// the production in0-sender pattern required for Blackhole multicast.
//
// Compile-time args:
//   0: Kt
//   1: Mt
//   2: in0_block_w           (unused by reader, but keeps arg layout aligned)
//   3..: activation TensorAccessorArgs
//   following: gamma TensorAccessorArgs
//
// Common runtime args (shared by all cores):
//   0: act_addr              DRAM/L1 address of raw activation
//   1: gamma_addr            DRAM/L1 address of gamma weight
//   2: scaler_val_u32        bit-cast of float 1/K
//   3: eps_val_u32           bit-cast of float epsilon

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t act_addr    = get_common_arg_val<uint32_t>(0);
    uint32_t gamma_addr  = get_common_arg_val<uint32_t>(1);
    uint32_t scaler_u32  = get_common_arg_val<uint32_t>(2);
    uint32_t eps_u32     = get_common_arg_val<uint32_t>(3);

    constexpr uint32_t Kt          = get_compile_time_arg_val(0);
    constexpr uint32_t Mt          = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(2);

    constexpr auto act_accessor_args   = TensorAccessorArgs<3>();
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

    experimental::Noc noc;
    experimental::CircularBuffer cb_raw_buf(cb_raw);
    experimental::CircularBuffer cb_gamma_buf(cb_gamma);

    // ================================================================
    // Fill constant tiles: cb_scaler (1/K) and cb_eps (epsilon)
    // ================================================================
    // Each tile has 32×32 = 1024 bf16 elements = 2048 bytes. We fill
    // every element with the same value so reduce_tile and add_tiles
    // produce correct results regardless of which positions are active.
    {
        // cb_scaler: tile filled with 1/K (bf16)
        // Convert float to bf16: truncate lower 16 bits of float32.
        uint16_t scaler_bf16 = static_cast<uint16_t>(scaler_u32 >> 16);
        uint32_t scaler_word = (static_cast<uint32_t>(scaler_bf16) << 16) |
                                static_cast<uint32_t>(scaler_bf16);

        cb_reserve_back(cb_scaler, 1);
        volatile tt_l1_ptr uint32_t* scaler_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t i = 0; i < 1024 / 2; ++i) {  // 512 uint32 words = 1024 bf16
            scaler_ptr[i] = scaler_word;
        }
        cb_push_back(cb_scaler, 1);

        // cb_eps: tile filled with epsilon (bf16)
        uint16_t eps_bf16 = static_cast<uint16_t>(eps_u32 >> 16);
        uint32_t eps_word = (static_cast<uint32_t>(eps_bf16) << 16) |
                             static_cast<uint32_t>(eps_bf16);

        cb_reserve_back(cb_eps, 1);
        volatile tt_l1_ptr uint32_t* eps_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_eps));
        for (uint32_t i = 0; i < 1024 / 2; ++i) {
            eps_ptr[i] = eps_word;
        }
        cb_push_back(cb_eps, 1);
    }

    // ================================================================
    // Read all Kt raw activation tiles into cb_raw (one shot)
    // ================================================================
    // The compute kernel needs all Kt tiles for the sum-of-squares
    // reduction (accessed by tile index). We read them all upfront.
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        cb_raw_buf.reserve_back(Kt);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            uint32_t act_tile_id = mt * Kt + kt;
            noc.async_read(act_tensor, cb_raw_buf, act_page_bytes,
                           {.page_id = act_tile_id},
                           {.offset_bytes = kt * act_page_bytes});
        }
        noc.async_read_barrier();
        cb_raw_buf.push_back(Kt);

        // ============================================================
        // Stream gamma tiles one at a time into cb_gamma
        // ============================================================
        // The compute kernel pops cb_gamma one tile per normalization
        // step, so we push one tile at a time. The double-buffered CB
        // (2 tiles) lets the DMA stay one tile ahead of compute.
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_gamma_buf.reserve_back(1);
            noc.async_read(gamma_tensor, cb_gamma_buf, gamma_page_bytes,
                           {.page_id = kt},
                           {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_gamma_buf.push_back(1);
        }
    }
}
