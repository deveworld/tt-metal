// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::experimental::prim {

struct TernaryMatmulConfig {
    uint32_t M_block_size{};
    uint32_t K_block_size{};
    uint32_t N_block_size{};
    uint32_t subblock_h{};
    uint32_t subblock_w{};

    tt::tt_metal::CoreCoord compute_with_storage_grid_size = {0, 0};
};

struct TernaryMatmulParams {
    std::optional<TernaryMatmulConfig> config;
    std::optional<operations::unary::UnaryWithParam> fused_activation;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;

    // Fused addcmul: ternary_a + scalar * matmul_output * ternary_b
    std::optional<float> fused_ternary_scalar;

    DeviceComputeKernelConfig compute_kernel_config;
    int32_t chunks = 1;  // Number of output tensors to split into (default 1 for backward compat)
    int32_t dim = -1;    // Dimension to split along (default -1)

    // Packed ternary weight mode: weight is uint32 ROW_MAJOR with 2-bit packed ternary values.
    // Selects TernaryMatmulSimpleProgramFactory (single-core, reader unpacks to bf16).
    bool use_packed_ternary = false;

    // Fused RMSNorm: when norm_epsilon is set and norm_weight is provided in
    // TernaryMatmulInputs, the kernel computes RMSNorm(input) * gamma before
    // the matmul, eliminating a separate kernel launch.
    std::optional<float> norm_epsilon;
};

struct TernaryMatmulInputs {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<Tensor> bias_tensor;
    std::optional<Tensor> optional_input_tensor;  // for StridedAllGatherTernaryMatmul

    // Fused addcmul: ternary_a + scalar * matmul_output * ternary_b
    std::optional<Tensor> fused_ternary_input_a;  // residual/base (broadcast like bias)
    std::optional<Tensor> fused_ternary_input_b;  // gate/multiplier (full MxN shape)

    // Fused RMSNorm gamma weight (1D, shape [K]). When provided together with
    // norm_epsilon in TernaryMatmulParams, the compute kernel fuses the norm.
    std::optional<Tensor> norm_weight;
};

}  // namespace ttnn::experimental::prim
