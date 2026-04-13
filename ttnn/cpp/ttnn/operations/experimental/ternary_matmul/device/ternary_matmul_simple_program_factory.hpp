// SPDX-License-Identifier: Apache-2.0
//
// Simple single-core program factory for ternary matmul.
// Uses reader_ternary_fused.cpp for packed 2-bit weight reads.
// For Track A correctness validation before multi-core optimization.

#pragma once

#include <tt-metalium/program.hpp>
#include "ternary_matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct TernaryMatmulSimpleProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;      // non-multicast path
        tt::tt_metal::KernelHandle sender_kernel_id;      // mcast sender (cores[0])
        tt::tt_metal::KernelHandle receiver_kernel_id;    // mcast receivers
        tt::tt_metal::KernelHandle compute_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        std::vector<CoreCoord> cores;
        bool use_mcast;
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
