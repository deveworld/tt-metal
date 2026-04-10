// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/ternary_matmul/device/ternary_matmul_device_operation_types.hpp"

namespace ttnn::operations::experimental::ternary_matmul {

// Re-export the config type for backward compatibility
using TernaryMatmulConfig = ttnn::experimental::prim::TernaryMatmulConfig;

}  // namespace ttnn::operations::experimental::ternary_matmul

namespace ttnn::experimental {

ttnn::Tensor ternary_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const ttnn::experimental::prim::TernaryMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
}
