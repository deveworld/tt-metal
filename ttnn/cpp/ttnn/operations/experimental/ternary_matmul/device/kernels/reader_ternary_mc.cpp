// SPDX-License-Identifier: Apache-2.0
// reader_ternary_mc.cpp - Multi-core fused reader for packed ternary matmul.
// Each core handles a contiguous range of N tiles [nt_start, nt_start+nt_count).
//
// Runtime args:
//   0: act_addr      - DRAM address of activation tensor
//   1: packed_addr   - DRAM address of packed weight tensor
//   2: Kt            - total K tiles
//   3: Nt            - total N tiles (for weight tile indexing)
//   4: Mt            - total M tiles
//   5: nt_start      - first N tile for this core
//   6: nt_count      - number of N tiles for this core

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

constexpr uint32_t PACKED_TILE_BYTES = 256;

constexpr uint16_t BF16_ZERO    = 0x0000u;
constexpr uint16_t BF16_POS_ONE = 0x3F80u;
constexpr uint16_t BF16_NEG_ONE = 0xBF80u;

static constexpr uint16_t LUT[4] = {
    BF16_ZERO, BF16_POS_ONE, BF16_NEG_ONE, BF16_ZERO,
};

inline void unpack_byte(uint8_t packed, uint16_t* dst) {
    dst[0] = LUT[(packed >> 6) & 0x03];
    dst[1] = LUT[(packed >> 4) & 0x03];
    dst[2] = LUT[(packed >> 2) & 0x03];
    dst[3] = LUT[packed & 0x03];
}

void kernel_main() {
    uint32_t act_addr    = get_arg_val<uint32_t>(0);
    uint32_t packed_addr = get_arg_val<uint32_t>(1);
    uint32_t Kt          = get_arg_val<uint32_t>(2);
    uint32_t Nt          = get_arg_val<uint32_t>(3);
    uint32_t Mt          = get_arg_val<uint32_t>(4);
    uint32_t nt_start    = get_arg_val<uint32_t>(5);
    uint32_t nt_count    = get_arg_val<uint32_t>(6);

    constexpr auto act_accessor_args = TensorAccessorArgs<0>();
    constexpr auto packed_accessor_args = TensorAccessorArgs<2>();

    constexpr uint32_t cb_in0    = 0;
    constexpr uint32_t cb_in1    = 1;
    constexpr uint32_t cb_scratch = 2;

    const uint32_t act_page_bytes = get_local_cb_interface(cb_in0).fifo_page_size;
    const uint32_t scratch_page_bytes = get_local_cb_interface(cb_scratch).fifo_page_size;

    const auto act_tensor = TensorAccessor(act_accessor_args, act_addr, act_page_bytes);
    const auto packed_tensor = TensorAccessor(packed_accessor_args, packed_addr, scratch_page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_in0);
    experimental::CircularBuffer cb1(cb_in1);
    experimental::CircularBuffer cb_s(cb_scratch);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nc = 0; nc < nt_count; ++nc) {
            uint32_t nt = nt_start + nc;
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                // Read activation tile A[mt, kt]
                uint32_t act_tile_id = mt * Kt + kt;
                cb0.reserve_back(1);
                noc.async_read(act_tensor, cb0, act_page_bytes,
                               {.page_id = act_tile_id}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb0.push_back(1);

                // Read packed weight tile B[kt, nt]
                uint32_t w_tile_id = kt * Nt + nt;
                cb_s.reserve_back(1);
                noc.async_read(packed_tensor, cb_s, scratch_page_bytes,
                               {.page_id = w_tile_id}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb_s.push_back(1);

                // Unpack packed data → bf16 tile
                cb_s.wait_front(1);
                cb1.reserve_back(1);
                uint32_t l1_scratch_rd = get_local_cb_interface(cb_scratch).fifo_rd_ptr;
                uint32_t l1_weight = get_local_cb_interface(cb_in1).fifo_wr_ptr;
                const uint8_t* src = reinterpret_cast<const uint8_t*>(l1_scratch_rd);
                uint16_t* dst = reinterpret_cast<uint16_t*>(l1_weight);

                for (uint32_t i = 0; i < PACKED_TILE_BYTES; ++i) {
                    unpack_byte(src[i], &dst[i * 4]);
                }

                cb_s.pop_front(1);
                cb1.push_back(1);
            }
        }
    }
}
