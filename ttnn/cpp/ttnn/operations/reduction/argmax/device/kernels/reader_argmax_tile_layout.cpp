// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_tile_layout.hpp"
#include "argmax_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

#include <stdint.h>

void kernel_main() {
    // Compile time args
    // -----------------
    constexpr uint32_t src_cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(1);

    constexpr uint32_t src_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(3);

    constexpr uint32_t tile_height = get_compile_time_arg_val(4);
    constexpr uint32_t tile_width = get_compile_time_arg_val(5);

    // Input padded size (last two dims) in tiles
    constexpr uint32_t input_height = get_compile_time_arg_val(6);
    constexpr uint32_t input_width = get_compile_time_arg_val(7);

    // Input logical size (last two dims) in data elements
    constexpr uint32_t logical_height = get_compile_time_arg_val(8);
    constexpr uint32_t logical_width = get_compile_time_arg_val(9);

    // Size of all dims combined, excluding the last two dims.
    constexpr uint32_t outer_dim_size = get_compile_time_arg_val(10);

    constexpr bool reduce_all = (bool)get_compile_time_arg_val(11);
    constexpr bool keepdim = (bool)get_compile_time_arg_val(12);

    constexpr uint32_t num_c_time_args = 13;

    // Runtime args
    // ------------
    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(1);

    // Tensor Accessors
    // ----------------
    constexpr auto s_src_args = TensorAccessorArgs<num_c_time_args>();
    constexpr auto s_dst_args = TensorAccessorArgs<s_src_args.next_compile_time_args_offset()>();

    auto s_src = TensorAccessor(s_src_args, src_base_addr, src_page_size);
    auto s_dst = TensorAccessor(s_dst_args, dst_base_addr, dst_page_size);

    using dst_accessor_type = decltype(s_dst);

    // CB for input data.
    const uint32_t src_cb_addr = get_write_ptr(src_cb_idx);
    constexpr DataFormat src_data_format = get_dataformat(src_cb_idx);

    // CB for output data.
    const uint32_t dst_cb_addr = get_write_ptr(dst_cb_idx);

    auto default_val = get_default_value<src_data_format>();
    // C++ type representation of the src/dst data formats
    using src_element_type = decltype(default_val);

    // This assumes the reduction is along the 'W' dimension
    src_element_type max_values[tile_height] = {default_val};
    uint32_t arg_max[tile_height] = {0};

    // Only ROW_MAJOR output layout is supported.
    //
    // When keep_dim==true
    // Each output row contains 1 element. We will accumulate tile_height output rows,
    // and then write them out in one pass.
    //
    // When keep_dim=false
    // Each output row contains logical_height elements. We will accumulate values
    // for a single output row, then write all of them out in a single noc message.

    // Number of data elements in one output page (ROW_MAJOR layout)
    constexpr uint32_t output_page_elements = keepdim ? 1 : logical_height;

    // Array for accumulating final argmax values. Used only when keepdim==true.
    uint32_t accumulated_arg_max[tile_height] = {0};

    constexpr uint32_t tile_height_rem = logical_height % tile_height;
    constexpr uint32_t tile_width_rem = logical_width % tile_width;
    constexpr uint32_t face_height_rem = logical_height % face_height;
    constexpr uint32_t face_width_rem = logical_width % face_width;

    const InputContext input_ctx(
        tile_height,
        tile_width,
        input_height,
        input_width,
        logical_height,
        logical_width,
        tile_height_rem,
        tile_width_rem,
        face_height_rem,
        face_width_rem,
        src_data_format,
        src_cb_addr);

    OutputContext output_ctx((uint32_t*)accumulated_arg_max, tile_height, dst_cb_addr, output_page_elements, keepdim);

    // -------------------------------------------------------------------------
    // Fast path: single-row bf16 argmax (input_height==1, dim=-1).
    // Batch-read ONLY row 0 from every tile (64 bytes per 2048-byte tile)
    // into L1, then scan contiguously. Avoids 4000+ per-tile NOC barriers.
    // -------------------------------------------------------------------------
    if constexpr (src_data_format == DataFormat::Float16_b && !reduce_all &&
                  input_height == 1 && outer_dim_size == 1) {
        // Each 32×32 bf16 tile stores 4 faces (16×16 each, 512 bytes/face).
        // Row 0 data lives at face0 offset 0 (16 uint16) and face1 offset 512 (16 uint16).
        constexpr uint32_t face_bytes = face_height * face_width * sizeof(uint16_t);  // 512
        constexpr uint32_t row0_elems_per_face = face_width;  // 16
        constexpr uint32_t row0_bytes = row0_elems_per_face * sizeof(uint16_t);  // 32

        // L1 buffer: 64 bytes per tile (face0_row0 + face1_row0)
        constexpr uint32_t per_tile_l1 = row0_bytes * 2;  // 64
        // Use src CB as scratch — it's sized for at least 1 tile (2048 bytes).
        // We need input_width × 64 bytes. Check if it fits in CB.
        // If not, fall through to the generic path below.
        constexpr uint32_t total_l1_needed = input_width * per_tile_l1;
        constexpr bool fits_in_cb = (total_l1_needed <= src_page_size * 4);
        // Use a secondary L1 region or CB. For safety, cap at 256 KB.
        constexpr bool can_use_fast_path = fits_in_cb || (total_l1_needed <= 256 * 1024);

        if constexpr (can_use_fast_path) {
            // Issue ALL reads upfront — 2 small reads per tile (face0_row0, face1_row0).
            // NOC requests to different banks pipeline automatically.
            uint32_t l1_write = src_cb_addr;
            for (uint32_t j = 0; j < input_width; j++) {
                const uint64_t tile_addr = get_noc_addr(j, s_src);
                // face0, row 0: first row0_bytes of tile
                noc_async_read(tile_addr, l1_write, row0_bytes);
                l1_write += row0_bytes;
                // face1, row 0: at offset face_bytes from tile start
                noc_async_read(tile_addr + face_bytes, l1_write, row0_bytes);
                l1_write += row0_bytes;
            }
            noc_async_read_barrier();

            // Scan all values in L1 with branchless orderable comparison.
            const uint16_t* data = reinterpret_cast<const uint16_t*>(src_cb_addr);
            const uint32_t total_elems = input_width * tile_width;  // input_width * 32
            uint16_t best_ord = 0;
            uint32_t best_i = 0;
            for (uint32_t idx = 0; idx < total_elems; idx++) {
                uint16_t raw = data[idx];
                uint16_t ord = (raw & 0x8000) ? (uint16_t)(~raw) : (uint16_t)(raw | 0x8000);
                if (ord > best_ord) {
                    best_ord = ord;
                    best_i = idx;
                }
            }
            // Convert flat index back: every 32 elements spans one tile
            // (16 from face0 + 16 from face1), so tile = best_i / 32,
            // within-tile col = (best_i % 32 < 16) ? (best_i % 32) : (16 + best_i % 32 - 16)
            // Actually the data is laid out as [face0_row0(16), face1_row0(16)] per tile,
            // so within-tile col = best_i % 32 for the first 16, then 16 + (best_i%32 - 16).
            // Simplifies to: tile_idx = best_i / 32, col = best_i % 32.
            // Global index = tile_idx * tile_width + col.
            uint32_t tile_idx = best_i / tile_width;
            uint32_t col = best_i % tile_width;
            uint32_t global_idx = tile_idx * tile_width + col;
            // Clamp to logical width
            if (global_idx >= logical_width) global_idx = 0;

            volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_cb_addr);
            out[0] = global_idx;
            const uint64_t dst_noc_addr = get_noc_addr(0, s_dst);
            noc_async_write(dst_cb_addr, dst_noc_addr, dst_page_size);
            noc_async_write_barrier();
            return;
        }
    }

    // -------------------------------------------------------------------------
    // Generic path: per-tile read + process (original code)
    // -------------------------------------------------------------------------

    // Iterate over the initial dimensions combined together
    for (uint32_t outer_index = 0; outer_index < outer_dim_size; outer_index++) {
        for (uint32_t i = 0; i < input_height; i++) {
            for (uint32_t row = 0; row < tile_height; row++) {
                max_values[row] = default_val;
                arg_max[row] = 0;
            }

            const uint32_t units_generated =
                (tile_height_rem == 0 || i < input_height - 1) ? tile_height : tile_height_rem;

            for (uint32_t j = 0; j < input_width; j++) {
                constexpr uint32_t inner_size = input_height * input_width;
                const int src_tile_id = outer_index * inner_size + i * input_width + j;

                const uint64_t src_noc_addr = get_noc_addr(src_tile_id, s_src);
                noc_async_read(src_noc_addr, src_cb_addr, src_page_size);
                noc_async_read_barrier();

                uint32_t tile_rows_processed = 0;
                process_input_tile<src_element_type, src_data_format>(
                    input_ctx, j, i, max_values, arg_max, tile_height, tile_rows_processed);
                ASSERT(tile_rows_processed == units_generated);
            }

            collect_row_major_output<keepdim>(arg_max, units_generated, output_ctx);

            if (output_ctx.collected_count >= output_page_elements) {
                write_to_output<dst_accessor_type, keepdim>(s_dst, output_ctx);
            }
        }
    }
}

void get_face_data_range(
    uint32_t& data_rows,
    uint32_t& data_cols,
    uint32_t tile_x,
    uint32_t tile_y,
    uint32_t face_id,
    const InputContext& ctx) {
    const bool is_bottom_tile = tile_y == (ctx.input_height - 1);
    const bool is_right_most_tile = tile_x == (ctx.input_width - 1);

    // Initialize the range as full face
    data_rows = face_height;
    data_cols = face_width;

    if (!ctx.has_padding) {
        return;
    }

    if (!is_bottom_tile && !is_right_most_tile) {
        // Only marginal tiles may contain the padding
        return;
    }

    const bool is_right_face = (face_id == 1 || face_id == 3);
    const bool is_bottom_face = (face_id == 2 || face_id == 3);

    const uint32_t height_rem = ctx.tile_h_rem;
    if (is_bottom_tile && height_rem != 0) {
        if (is_bottom_face) {
            const bool skip_bottom_face = height_rem < face_height;
            if (skip_bottom_face) {
                data_rows = 0;
                data_cols = 0;
                return;
            }
            data_rows = ctx.face_h_rem;
        } else {
            // One of the upper faces
            if (height_rem < face_height) {
                data_rows = height_rem;
            }
        }
    }

    const uint32_t width_rem = ctx.tile_w_rem;
    if (is_right_most_tile && width_rem != 0) {
        if (is_right_face) {
            const bool skip_right_face = width_rem < face_width;
            if (skip_right_face) {
                data_rows = 0;
                data_cols = 0;
                return;
            }
            data_cols = ctx.face_w_rem;
        } else {
            if (width_rem < face_width) {
                data_cols = width_rem;
            }
        }
    }
}
