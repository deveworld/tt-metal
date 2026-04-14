// SPDX-License-Identifier: Apache-2.0
//
// Multi-core ternary matmul program factory.
// Uses BFP2_b weights (Tensix unpacker decodes on the fly) with a dual-NoC
// reader/writer split. When the chosen core layout is rectangular, the
// reader is split into a sender (DRAM read + noc_async_write_multicast)
// and receivers (semaphore wait) so activation DRAM bandwidth is shared.

#pragma once

#include <tt-metalium/program.hpp>
#include "ternary_matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct TernaryMatmulSimpleProgramFactory {
    struct shared_variables_t {
        // For non-mcast: reader_kernel_id is set, sender/receiver are 0.
        // For mcast: sender_kernel_id is set; receiver_kernel_id may be 0
        // if num_cores == 1 (no receivers).
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle sender_kernel_id;
        tt::tt_metal::KernelHandle receiver_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        bool use_mcast;
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const TernaryMatmulParams& params,
        const TernaryMatmulInputs& inputs,
        std::vector<Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TernaryMatmulParams& params,
        const TernaryMatmulInputs& inputs,
        std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::experimental::prim
