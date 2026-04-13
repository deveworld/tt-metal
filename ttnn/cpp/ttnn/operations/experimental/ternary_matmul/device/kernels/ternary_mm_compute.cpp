// SPDX-License-Identifier: Apache-2.0
//
// ternary_mm_compute.cpp — Blocked matmul compute kernel for ternary weights.
//
// Single-phase version. Reader (NCRISC) pushes activation + packed bytes to
// cb_scratch; writer (BRISC) consumes cb_scratch, unpacks to cb_in1. Compute
// consumes cb_in0 (activation) and cb_in1 (unpacked weight) as usual.
//
// Compile-time args:
//   0: Mt
//   1: Kt
//   2: Nt (= nt_per_core)

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

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        tile_regs_acquire();

        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(cb_in0, 1);
            for (uint32_t nt = 0; nt < Nt; ++nt) {
                cb_wait_front(cb_in1, 1);
                matmul_tiles(cb_in0, cb_in1, 0, 0, nt);
                cb_pop_front(cb_in1, 1);
            }
            cb_pop_front(cb_in0, 1);
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
