// SPDX-License-Identifier: Apache-2.0
// writer_ternary_dual.cpp — BRISC kernel with two jobs:
//   Phase 1: consume packed bytes from cb_scratch (produced by NCRISC reader),
//            unpack them to bf16, push to cb_in1 (consumed by compute).
//   Phase 2: write cb_out tiles back to DRAM.
// The NCRISC reader only does DRAM reads; unpack happens entirely on BRISC.
// This pins unpack to one thread and lets reader + writer + compute run
// concurrently without any K-range splitting.
//
// Runtime args:
//   0: out_addr — DRAM address of output tensor
//   1: Kt
//   2: Nt
//   3: Mt
//   4: nt_start
//   5: nt_count

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

constexpr uint32_t PACKED_TILE_BYTES = 256;

constexpr uint16_t BF16_ZERO    = 0x0000u;
constexpr uint16_t BF16_POS_ONE = 0x3F80u;
constexpr uint16_t BF16_NEG_ONE = 0xBF80u;

static constexpr uint16_t CODE_LUT[4] = {
    BF16_ZERO, BF16_POS_ONE, BF16_NEG_ONE, BF16_ZERO,
};

static uint64_t BYTE_LUT[256];

inline void init_byte_lut() {
    for (uint32_t b = 0; b < 256; ++b) {
        uint64_t v = 0;
        v |= ((uint64_t)CODE_LUT[(b >> 6) & 0x3]) << 0;
        v |= ((uint64_t)CODE_LUT[(b >> 4) & 0x3]) << 16;
        v |= ((uint64_t)CODE_LUT[(b >> 2) & 0x3]) << 32;
        v |= ((uint64_t)CODE_LUT[b & 0x3])        << 48;
        BYTE_LUT[b] = v;
    }
}

inline void unpack_tile_fast(const uint8_t* src, uint64_t* dst) {
    for (uint32_t i = 0; i < PACKED_TILE_BYTES; ++i) {
        dst[i] = BYTE_LUT[src[i]];
    }
}

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t Kt       = get_arg_val<uint32_t>(1);
    uint32_t Nt       = get_arg_val<uint32_t>(2);
    uint32_t Mt       = get_arg_val<uint32_t>(3);
    uint32_t nt_start = get_arg_val<uint32_t>(4);
    uint32_t nt_count = get_arg_val<uint32_t>(5);

    constexpr auto out_accessor_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_in1     = 1;
    constexpr uint32_t cb_scratch = 2;
    constexpr uint32_t cb_out     = 16;

    const uint32_t out_page_bytes = get_local_cb_interface(cb_out).fifo_page_size;
    const auto out_tensor = TensorAccessor(out_accessor_args, out_addr, out_page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb1(cb_in1);
    experimental::CircularBuffer cb_s(cb_scratch);
    experimental::CircularBuffer cbo(cb_out);

    init_byte_lut();

    // Phase 1: consume packed bytes from cb_scratch, unpack to cb_in1.
    // Reader pushes Kt × nt_count packed tiles; we emit one bf16 tile per
    // packed tile, in the same order.
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            for (uint32_t nc = 0; nc < nt_count; ++nc) {
                cb_s.wait_front(1);
                cb1.reserve_back(1);

                uint32_t l1_scratch_rd = get_local_cb_interface(cb_scratch).fifo_rd_ptr;
                uint32_t l1_weight = get_local_cb_interface(cb_in1).fifo_wr_ptr;
                const uint8_t* src = reinterpret_cast<const uint8_t*>(l1_scratch_rd);
                uint64_t* dst = reinterpret_cast<uint64_t*>(l1_weight);

                unpack_tile_fast(src, dst);

                cb_s.pop_front(1);
                cb1.push_back(1);
            }
        }
    }

    // Phase 2: cb_out → DRAM
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nc = 0; nc < nt_count; ++nc) {
            uint32_t nt = nt_start + nc;
            uint32_t tile_id = mt * Nt + nt;
            cbo.wait_front(1);
            noc.async_write(cbo, out_tensor, out_page_bytes,
                            {}, {.page_id = tile_id});
            noc.async_writes_flushed();
            cbo.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
