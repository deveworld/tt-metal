// SPDX-License-Identifier: Apache-2.0
//
// ternary_mm_compute.cpp — Block matmul compute kernel for ternary (bfp2) weights.
//
// Uses matmul_block with an inner K-block of in0_block_w tiles per call,
// massively reducing call overhead vs tile-by-tile matmul_tiles. Works with
// Bfp2_b weight CB so the Tensix unpacker decodes on the fly.
//
// Compile-time args:
//   0: Mt
//   1: Kt
//   2: Nt          (= nt_per_core, also used as ct_dim)
//   3: in0_block_w (K block width, must divide Kt)

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"

#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(3);
    constexpr uint32_t num_k_blocks = Kt / in0_block_w;

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

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

            // One matmul_block call does in0_block_w × Nt tile-ops and
            // accumulates into DST[0..Nt-1].
            matmul_block(cb_in0, cb_in1,
                         /*in0_tile_index=*/0,
                         /*in1_tile_index=*/0,
                         /*idst=*/0,
                         /*transpose=*/0,
                         /*ct_dim=*/Nt,
                         /*rt_dim=*/1,
                         /*kt_dim=*/in0_block_w);

            cb_pop_front(cb_in0, in0_block_w);
            cb_pop_front(cb_in1, in0_block_w * Nt);
        }

        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_reserve_back(cb_out, 1);
            pack_tile(nt, cb_out);
            cb_push_back(cb_out, 1);
        }

        tile_regs_release();
    }
}
