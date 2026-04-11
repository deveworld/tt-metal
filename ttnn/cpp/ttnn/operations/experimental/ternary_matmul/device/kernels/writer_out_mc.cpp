// SPDX-License-Identifier: Apache-2.0
// writer_out_mc.cpp - Multi-core writer for ternary matmul output.
// Each core writes its assigned N tile range.
//
// Runtime args:
//   0: dst_addr   - DRAM address of output tensor
//   1: Mt         - total M tiles
//   2: Nt_total   - total N tiles in output (for tile_id calculation)
//   3: nt_start   - first N tile for this core
//   4: nt_count   - number of N tiles for this core

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    uint32_t Mt         = get_arg_val<uint32_t>(1);
    uint32_t Nt_total   = get_arg_val<uint32_t>(2);
    uint32_t nt_start   = get_arg_val<uint32_t>(3);
    uint32_t nt_count   = get_arg_val<uint32_t>(4);

    constexpr uint32_t out_cb_idx = get_compile_time_arg_val(0);
    constexpr auto out_accessor_args = TensorAccessorArgs<1>();

    const uint32_t page_bytes = get_local_cb_interface(out_cb_idx).fifo_page_size;
    const auto out_tensor = TensorAccessor(out_accessor_args, dst_addr, page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out(out_cb_idx);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nc = 0; nc < nt_count; ++nc) {
            uint32_t nt = nt_start + nc;
            uint32_t tile_id = mt * Nt_total + nt;

            cb_out.wait_front(1);
            noc.async_write(cb_out, out_tensor, page_bytes,
                            {}, {.page_id = tile_id});
            noc.async_writes_flushed();
            cb_out.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
