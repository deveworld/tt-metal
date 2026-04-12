// SPDX-License-Identifier: Apache-2.0
// reader_ternary_mc_legacy.cpp - Legacy loop order (mt→nt→kt) with activation
// L1 caching. Reads Kt activation tiles ONCE per M-row into CB3 cache,
// then copies from cache to CB0 for each N-tile reuse.
//
// DRAM reads reduced from Kt × nt_count to Kt per M-row (nt_count× savings).

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

// Fast L1-to-L1 word-aligned copy
inline void l1_copy(uint32_t dst_addr, uint32_t src_addr, uint32_t bytes) {
    volatile uint32_t* d = reinterpret_cast<volatile uint32_t*>(dst_addr);
    volatile uint32_t* s = reinterpret_cast<volatile uint32_t*>(src_addr);
    uint32_t words = bytes >> 2;
    for (uint32_t i = 0; i < words; ++i) {
        d[i] = s[i];
    }
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

    constexpr uint32_t cb_in0     = 0;
    constexpr uint32_t cb_in1     = 1;
    constexpr uint32_t cb_scratch = 2;
    constexpr uint32_t cb_act_cache = 3;

    const uint32_t act_page_bytes = get_local_cb_interface(cb_in0).fifo_page_size;
    const uint32_t scratch_page_bytes = get_local_cb_interface(cb_scratch).fifo_page_size;

    const auto act_tensor = TensorAccessor(act_accessor_args, act_addr, act_page_bytes);
    const auto packed_tensor = TensorAccessor(packed_accessor_args, packed_addr, scratch_page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_in0);
    experimental::CircularBuffer cb1(cb_in1);
    experimental::CircularBuffer cb_s(cb_scratch);

    // Get cache base address (CB3 is pre-reserved as raw L1 buffer)
    // CB3 has Kt pages of act_page_bytes. We use fifo_wr_ptr as base.
    uint32_t cache_base = get_local_cb_interface(cb_act_cache).fifo_wr_ptr;

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        // Phase 1: Read ALL Kt activation tiles from DRAM into L1 cache.
        // Use cb0 as intermediate: DRAM → cb0 → cache (L1 copy).
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            uint32_t act_tile_id = mt * Kt + kt;
            cb0.reserve_back(1);
            noc.async_read(act_tensor, cb0, act_page_bytes,
                           {.page_id = act_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb0.push_back(1);

            // Copy from cb0 to cache
            cb0.wait_front(1);
            uint32_t src = get_local_cb_interface(cb_in0).fifo_rd_ptr;
            uint32_t dst = cache_base + kt * act_page_bytes;
            l1_copy(dst, src, act_page_bytes);
            cb0.pop_front(1);
        }

        // Phase 2: For each N-tile, reuse cached activation tiles
        for (uint32_t nc = 0; nc < nt_count; ++nc) {
            uint32_t nt = nt_start + nc;
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                // Copy cached activation to cb_in0 (L1→L1)
                cb0.reserve_back(1);
                uint32_t src = cache_base + kt * act_page_bytes;
                uint32_t dst = get_local_cb_interface(cb_in0).fifo_wr_ptr;
                l1_copy(dst, src, act_page_bytes);
                cb0.push_back(1);

                // Read packed weight tile B[kt, nt] from DRAM
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
                const uint8_t* wsrc = reinterpret_cast<const uint8_t*>(l1_scratch_rd);
                uint16_t* wdst = reinterpret_cast<uint16_t*>(l1_weight);

                for (uint32_t i = 0; i < PACKED_TILE_BYTES; ++i) {
                    unpack_byte(wsrc[i], &wdst[i * 4]);
                }

                cb_s.pop_front(1);
                cb1.push_back(1);
            }
        }
    }
}
