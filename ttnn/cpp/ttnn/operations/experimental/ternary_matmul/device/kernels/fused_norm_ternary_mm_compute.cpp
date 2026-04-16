// SPDX-License-Identifier: Apache-2.0
//
// fused_norm_ternary_mm_compute.cpp — Fused RMSNorm + ternary matmul compute
// kernel. Eliminates the separate RMSNorm kernel launch by computing the norm
// in-line before the matmul, using only standard Tensix compute APIs.
//
// Phase 1 (RMSNorm reduction):
//   Square each activation tile, accumulate element-wise across Kt tiles,
//   reduce to scalar, multiply by 1/K scaler → mean(x²), add eps, rsqrt.
//
// Phase 2 (Normalize + gamma):
//   For each tile: multiply raw activation by the rsqrt scalar (broadcast),
//   then multiply by gamma. Pack normed tiles to cb_in0 for the matmul.
//
// Phase 3 (Matmul):
//   Standard block matmul: cb_in0 × cb_in1 (BFP2_b weights) → cb_out.
//   K-block pipelining with in0_block_w tiles per block.
//
// While TRISC runs phases 1-2, NCRISC concurrently DMAs weights into cb_in1.
// By the time phase 3 starts, weight data is already in L1.
//
// CB Layout:
//   c_0  (cb_in0):   normed activation for matmul (Kt tiles)
//   c_1  (cb_in1):   BFP2_b weights (Kt × nt_per_core tiles)
//   c_2  (cb_raw):   raw activation from DRAM/L1 (Kt tiles)
//   c_3  (cb_gamma): gamma' weights, streamed 1 tile at a time
//   c_4  (cb_sq):    temp for one squared tile (1 tile)
//   c_5  (cb_sqsum): running sum of squares / rsqrt scalar (2 tiles)
//   c_6  (cb_scaler):1/K scaler tile (1 tile, never popped)
//   c_7  (cb_eps):   epsilon tile (1 tile, never popped)
//   c_16 (cb_out):   matmul output
//
// Compile-time args:
//   0: Mt
//   1: Kt
//   2: Nt          (= nt_per_core)
//   3: in0_block_w (K block width, must divide Kt)

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/bcast.h"

#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(3);
    constexpr uint32_t num_k_blocks = Kt / in0_block_w;

    constexpr auto cb_in0    = tt::CBIndex::c_0;
    constexpr auto cb_in1    = tt::CBIndex::c_1;
    constexpr auto cb_raw    = tt::CBIndex::c_2;
    constexpr auto cb_gamma  = tt::CBIndex::c_3;
    constexpr auto cb_sq     = tt::CBIndex::c_4;
    constexpr auto cb_sqsum  = tt::CBIndex::c_5;
    constexpr auto cb_scaler = tt::CBIndex::c_6;
    constexpr auto cb_eps    = tt::CBIndex::c_7;
    constexpr auto cb_out    = tt::CBIndex::c_16;

    // ==================================================================
    // Phase 1: RMSNorm — compute rsqrt(mean(x²) + eps)
    // ==================================================================

    // Wait for all raw activation tiles and the constant tiles.
    cb_wait_front(cb_raw, Kt);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    // Step 1a: Square each tile and accumulate element-wise.
    // After this loop, cb_sqsum has one tile where each element [i][j] =
    // sum_{t=0}^{Kt-1}( raw[t][i][j]² ).
    for (uint32_t t = 0; t < Kt; ++t) {
        // Square: DST[0] = raw[t] * raw[t]  (element-wise)
        tile_regs_acquire();
        mul_tiles_init(cb_raw, cb_raw);
        mul_tiles(cb_raw, cb_raw, t, t, 0);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_sq, 1);
        pack_tile(0, cb_sq);
        cb_push_back(cb_sq, 1);
        tile_regs_release();

        // Accumulate into cb_sqsum.
        if (t == 0) {
            // First tile: copy to cb_sqsum directly.
            tile_regs_acquire();
            cb_wait_front(cb_sq, 1);
            cb_reserve_back(cb_sqsum, 1);

            copy_tile_to_dst_init_short(cb_sq);
            copy_tile(cb_sq, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_sqsum);

            cb_pop_front(cb_sq, 1);
            cb_push_back(cb_sqsum, 1);
            tile_regs_release();
        } else {
            // Subsequent tiles: add to running sum.
            tile_regs_acquire();
            cb_wait_front(cb_sqsum, 1);
            cb_wait_front(cb_sq, 1);
            cb_reserve_back(cb_sqsum, 1);

            add_tiles_init(cb_sqsum, cb_sq);
            add_tiles(cb_sqsum, cb_sq, 0, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_sqsum);

            cb_pop_front(cb_sqsum, 1);
            cb_pop_front(cb_sq, 1);
            cb_push_back(cb_sqsum, 1);
            tile_regs_release();
        }
    }

    // Step 1b: Reduce the sum tile to a scalar, multiplied by 1/K.
    // reduce_tile with PoolType::SUM and REDUCE_SCALAR computes:
    //   DST[0] = sum_over_elements( sqsum_tile * scaler_tile )
    //          = (1/K) * total_sum_of_squares = mean(x²)
    {
        tile_regs_acquire();
        cb_wait_front(cb_sqsum, 1);

        reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sqsum, cb_scaler, cb_sqsum);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sqsum, cb_scaler, 0, 0, 0);
        reduce_uninit();

        // Pack mean scalar back to cb_sqsum (reuse).
        cb_pop_front(cb_sqsum, 1);
        cb_reserve_back(cb_sqsum, 1);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_sqsum);
        cb_push_back(cb_sqsum, 1);
        tile_regs_release();
    }

    // Step 1c: Add epsilon and compute rsqrt.
    // DST[0] = rsqrt(mean(x²) + eps) = 1 / RMS(x)
    {
        tile_regs_acquire();
        cb_wait_front(cb_sqsum, 1);

        add_tiles_init(cb_sqsum, cb_eps);
        add_tiles(cb_sqsum, cb_eps, 0, 0, 0);

        rsqrt_tile_init();
        rsqrt_tile(0);

        // Store rsqrt scalar in cb_sqsum (reuse as cb_rsqrt).
        cb_pop_front(cb_sqsum, 1);
        cb_reserve_back(cb_sqsum, 1);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_sqsum);
        cb_push_back(cb_sqsum, 1);
        tile_regs_release();
    }

    // cb_sqsum now has the rsqrt scalar tile (1 tile, kept for phase 2).

    // ==================================================================
    // Phase 2: Normalize — raw[t] * rsqrt_scalar * gamma[t] → cb_in0
    // ==================================================================
    // The scaler CB (1/K) in the reduce step already handled the 1/K
    // division, so cb_sqsum holds rsqrt(mean(x²) + eps) = 1/RMS.
    // Standard gamma is used directly — no pre-scaling needed.

    cb_wait_front(cb_sqsum, 1);  // rsqrt scalar stays in CB throughout

    for (uint32_t t = 0; t < Kt; ++t) {
        tile_regs_acquire();

        // DST[0] = raw[t] * rsqrt_scalar  (broadcast scalar multiply)
        mul_tiles_bcast_scalar_init_short(cb_raw, cb_sqsum);
        mul_tiles_bcast_scalar(cb_raw, cb_sqsum, t, 0, 0);

        // DST[0] *= gamma'[t]  (element-wise, gamma in-place via dest reuse)
        cb_wait_front(cb_gamma, 1);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL,
                                     EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL,
                                EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_in0, 1);
        pack_tile(0, cb_in0);
        cb_push_back(cb_in0, 1);
        tile_regs_release();

        cb_pop_front(cb_gamma, 1);
    }

    // Clean up phase 1-2 CBs.
    cb_pop_front(cb_raw, Kt);
    cb_pop_front(cb_sqsum, 1);
    // cb_scaler and cb_eps are never popped (persistent constants).

    // ==================================================================
    // Phase 3: Block matmul — cb_in0 × cb_in1 → cb_out
    // ==================================================================
    // Identical to ternary_mm_compute.cpp. By this point, NCRISC has had
    // the entire norm phase to DMA weights into cb_in1.

    mm_block_init(cb_in0, cb_in1, cb_out,
                  /*transpose=*/0,
                  /*ct_dim=*/Nt,
                  /*rt_dim=*/1,
                  /*kt_dim=*/in0_block_w);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        tile_regs_acquire();

        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            cb_wait_front(cb_in0, in0_block_w);
            cb_wait_front(cb_in1, in0_block_w * Nt);

            uint32_t in0_idx = 0;
            uint32_t in1_idx = 0;
            for (uint32_t k = 0; k < in0_block_w; ++k) {
                matmul_block(cb_in0, cb_in1,
                             in0_idx, in1_idx,
                             /*idst=*/0,
                             /*transpose=*/0,
                             /*ct_dim=*/Nt,
                             /*rt_dim=*/1,
                             /*kt_dim=*/in0_block_w);
                in0_idx += 1;
                in1_idx += Nt;
            }

            cb_pop_front(cb_in0, in0_block_w);
            cb_pop_front(cb_in1, in0_block_w * Nt);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, Nt);
        pack_tile_block(0, cb_out, Nt);
        cb_push_back(cb_out, Nt);

        tile_regs_release();
    }
}
