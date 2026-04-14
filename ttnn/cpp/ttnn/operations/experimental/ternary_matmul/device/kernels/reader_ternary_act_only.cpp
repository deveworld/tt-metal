// SPDX-License-Identifier: Apache-2.0
// reader_ternary_act_only.cpp — NCRISC kernel that ONLY reads the activation
// tiles from DRAM into cb_in0. Weight reads are handled by the writer kernel
// on BRISC so the two NoCs (NOC_1 + NOC_0) operate in parallel.
//
// Compile-time args:
//   0: Kt
//   1: Mt
//   2: in0_block_w
//   3..: activation TensorAccessorArgs
//
// Common runtime args (shared by all cores):
//   0: act_addr

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t act_addr = get_common_arg_val<uint32_t>(0);

    constexpr uint32_t Kt          = get_compile_time_arg_val(0);
    constexpr uint32_t Mt          = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = Kt / in0_block_w;

    constexpr auto act_accessor_args = TensorAccessorArgs<3>();

    constexpr uint32_t cb_in0 = 0;

    const uint32_t act_page_bytes = get_local_cb_interface(cb_in0).fifo_page_size;
    const auto act_tensor = TensorAccessor(act_accessor_args, act_addr, act_page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_in0);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            const uint32_t kt_base = kb * in0_block_w;
            cb0.reserve_back(in0_block_w);
            for (uint32_t k = 0; k < in0_block_w; ++k) {
                uint32_t act_tile_id = mt * Kt + (kt_base + k);
                noc.async_read(act_tensor, cb0, act_page_bytes,
                               {.page_id = act_tile_id},
                               {.offset_bytes = k * act_page_bytes});
            }
            noc.async_read_barrier();
            cb0.push_back(in0_block_w);
        }
    }
}
