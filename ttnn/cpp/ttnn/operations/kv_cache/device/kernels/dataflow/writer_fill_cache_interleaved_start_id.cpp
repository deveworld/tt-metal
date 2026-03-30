// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row_num = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t batch_start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t num_heads = get_compile_time_arg_val(1);
    constexpr uint32_t input_Ht = get_compile_time_arg_val(2);
    constexpr uint32_t cache_HtWt = get_compile_time_arg_val(3);
    constexpr uint32_t cache_CHtWt = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr auto dst_args = TensorAccessorArgs<6>();

    const uint32_t rows_per_batch = num_heads * input_Ht;
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_out);
    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    for (uint32_t row_id = start_row_num; row_id < start_row_num + num_rows; ++row_id) {
        const uint32_t batch_idx = row_id / rows_per_batch;
        const uint32_t row_in_batch = row_id % rows_per_batch;
        const uint32_t head_idx = row_in_batch / input_Ht;
        const uint32_t seq_idx = row_in_batch % input_Ht;
        uint32_t cache_tile_id =
            batch_start_id + batch_idx * cache_CHtWt + head_idx * cache_HtWt + seq_idx * Wt;

        for (uint32_t tile = 0; tile < Wt; ++tile) {
            cb.wait_front(1);
            noc.async_write(cb, s, page_bytes, {}, {.page_id = cache_tile_id});
            noc.async_writes_flushed();
            cb.pop_front(1);
            cache_tile_id += 1;
        }
    }
    noc.async_write_barrier();
}
