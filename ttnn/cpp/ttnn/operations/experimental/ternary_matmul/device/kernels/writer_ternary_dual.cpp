// SPDX-License-Identifier: Apache-2.0
// writer_ternary_dual.cpp — BRISC kernel that runs in parallel with the
// NCRISC reader. Splits K-dimension unpack work in half:
//   reader (NCRISC): kt in [0, kt_split)           → cb_in0,  cb_in1
//   writer (BRISC):  kt in [kt_split, Kt)          → cb_in0b, cb_in1b
// After the main loop, this kernel also writes cb_out back to DRAM.
//
// Runtime args:
//   0: act_addr   — DRAM address of activation tensor
//   1: packed_addr — DRAM address of packed weight tensor
//   2: out_addr   — DRAM address of output tensor
//   3: Kt
//   4: Nt
//   5: Mt
//   6: nt_start
//   7: nt_count
//   8: kt_split   — K-index at which reader hands off to writer

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
    uint32_t act_addr    = get_arg_val<uint32_t>(0);
    uint32_t packed_addr = get_arg_val<uint32_t>(1);
    uint32_t out_addr    = get_arg_val<uint32_t>(2);
    uint32_t Kt          = get_arg_val<uint32_t>(3);
    uint32_t Nt          = get_arg_val<uint32_t>(4);
    uint32_t Mt          = get_arg_val<uint32_t>(5);
    uint32_t nt_start    = get_arg_val<uint32_t>(6);
    uint32_t nt_count    = get_arg_val<uint32_t>(7);
    uint32_t kt_split    = get_arg_val<uint32_t>(8);

    constexpr auto act_accessor_args = TensorAccessorArgs<0>();
    constexpr auto packed_accessor_args =
        TensorAccessorArgs<act_accessor_args.next_compile_time_args_offset()>();
    constexpr auto out_accessor_args =
        TensorAccessorArgs<packed_accessor_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_in0b    = 4;   // activation, second K half
    constexpr uint32_t cb_in1b    = 5;   // unpacked weight, second K half
    constexpr uint32_t cb_scratch_b = 6; // packed scratch, second K half
    constexpr uint32_t cb_out     = 16;

    const uint32_t act_page_bytes = get_local_cb_interface(cb_in0b).fifo_page_size;
    const uint32_t scratch_page_bytes = get_local_cb_interface(cb_scratch_b).fifo_page_size;
    const uint32_t out_page_bytes = get_local_cb_interface(cb_out).fifo_page_size;

    const auto act_tensor = TensorAccessor(act_accessor_args, act_addr, act_page_bytes);
    const auto packed_tensor = TensorAccessor(packed_accessor_args, packed_addr, scratch_page_bytes);
    const auto out_tensor = TensorAccessor(out_accessor_args, out_addr, out_page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb0b(cb_in0b);
    experimental::CircularBuffer cb1b(cb_in1b);
    experimental::CircularBuffer cb_sb(cb_scratch_b);
    experimental::CircularBuffer cbo(cb_out);

    init_byte_lut();

    // Phase 1: Handle K tiles [kt_split, Kt), producing cb_in0b + cb_in1b
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kt = kt_split; kt < Kt; ++kt) {
            uint32_t act_tile_id = mt * Kt + kt;
            cb0b.reserve_back(1);
            noc.async_read(act_tensor, cb0b, act_page_bytes,
                           {.page_id = act_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb0b.push_back(1);

            for (uint32_t nc = 0; nc < nt_count; ++nc) {
                uint32_t nt = nt_start + nc;
                uint32_t w_tile_id = kt * Nt + nt;
                cb_sb.reserve_back(1);
                noc.async_read(packed_tensor, cb_sb, scratch_page_bytes,
                               {.page_id = w_tile_id}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb_sb.push_back(1);

                cb_sb.wait_front(1);
                cb1b.reserve_back(1);
                // TEMP DIAG: writer skips unpack too
                cb_sb.pop_front(1);
                cb1b.push_back(1);
            }
        }
    }

    // Phase 2: Write output from cb_out to DRAM
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
