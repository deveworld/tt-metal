// SPDX-License-Identifier: Apache-2.0
//
// ternary_mm_compute.cpp — Multi-K-block matmul with PACKER_L1_ACC.
//
// For each K block:
//   1. wait for in0/in1 tiles
//   2. acquire DST, run matmul_block × in0_block_w
//   3. commit/wait, pack partial sum to cb_out's L1 region with l1_acc
//      (overwrite on first block, accumulate on subsequent blocks)
//   4. release DST, pop CBs
// After all K blocks, push the accumulated cb_out to the writer.
//
// Compile-time args:
//   0: Mt
//   1: Kt
//   2: Nt          (= nt_per_core, also used as ct_dim)
//   3: in0_block_w (K block width, must divide Kt)

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#include "llk_pack_api.h"
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
        // Reserve cb_out slots once; we will overwrite/accumulate them
        // across K blocks via PACKER_L1_ACC.
        cb_reserve_back(cb_out, Nt);

        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            cb_wait_front(cb_in0, in0_block_w);
            cb_wait_front(cb_in1, in0_block_w * Nt);

            tile_regs_acquire();

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

            tile_regs_commit();
            tile_regs_wait();

            // First K block: overwrite cb_out. Subsequent: accumulate.
            if (kb == 0) {
                PACK((llk_pack_reconfig_l1_acc(0)));
            } else if (kb == 1) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
            pack_tile_block(0, cb_out, Nt);

            tile_regs_release();
            cb_pop_front(cb_in0, in0_block_w);
            cb_pop_front(cb_in1, in0_block_w * Nt);
        }

        // Disable l1_acc so subsequent matmul calls (in next program launch)
        // don't accumulate stale data.
        PACK((llk_pack_reconfig_l1_acc(0)));

        cb_push_back(cb_out, Nt);
    }
}
