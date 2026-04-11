// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ternary_matmul.hpp"
#include "device/ternary_matmul_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental {

ttnn::Tensor ternary_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const ttnn::experimental::prim::TernaryMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    bool use_packed_ternary) {
    // Call device operation with chunks=1 (default), which returns a vector with 1 element
    auto outputs = ttnn::prim::ternary_matmul(
        input_tensor,
        weight_tensor,
        bias_tensor,
        std::move(fused_activation),
        config,
        memory_config,
        dtype,
        compute_kernel_config,
        /*chunks=*/1,
        /*dim=*/-1,
        /*fused_ternary_scalar=*/std::nullopt,
        /*fused_ternary_input_a=*/std::nullopt,
        /*fused_ternary_input_b=*/std::nullopt,
        use_packed_ternary);

    // Extract and return the single output
    TT_FATAL(outputs.size() == 1, "Expected single output from ternary_matmul, got {}", outputs.size());
    return outputs[0];
}

}  // namespace ttnn::experimental
