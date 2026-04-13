// SPDX-License-Identifier: Apache-2.0
//
// ternary_mm_compute.cpp — Blocked matmul compute kernel for ternary weights.
//
// Loop order: mt → kt → nt (matches reader)
// Uses dest[0..Nt-1] to accumulate N tiles across K dimension.
// Activation tile is consumed once per kt (shared across all nt).
//
// Compile-time args:
//   arg 0: Mt — number of M-dimension tile rows
//   arg 1: Kt — number of K-dimension tiles (inner dimension)
//   arg 2: Nt — number of N-dimension tile columns (per core)

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

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // activation tiles (bfloat16)
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // unpacked weight tiles (bfloat16)
    constexpr auto cb_out = tt::CBIndex::c_16;   // output tiles

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        // Accumulate all K tiles into dest[0..Nt-1]
        tile_regs_acquire();

        for (uint32_t kt = 0; kt < Kt; ++kt) {
            // Wait for activation + the whole batch of Nt weight tiles at once
            // (reader pushes all Nt after a single barrier).
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, Nt);

            for (uint32_t nt = 0; nt < Nt; ++nt) {
                matmul_tiles(cb_in0, cb_in1, 0, nt, nt);
            }

            cb_pop_front(cb_in1, Nt);
            cb_pop_front(cb_in0, 1);
        }

        // Pack all Nt output tiles
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
