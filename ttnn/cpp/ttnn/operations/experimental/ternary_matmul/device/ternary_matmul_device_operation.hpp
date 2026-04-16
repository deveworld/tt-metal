// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ternary_matmul_device_operation_types.hpp"
#include "ternary_matmul_program_factory.hpp"
#include "ternary_matmul_simple_program_factory.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/ternary_matmul/device/ternary_matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct TernaryMatmulDeviceOperation {
    using operation_attributes_t = TernaryMatmulParams;
    using tensor_args_t = TernaryMatmulInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<TernaryMatmulProgramFactory, TernaryMatmulSimpleProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        const std::optional<Tensor>& bias_tensor,
        std::optional<operations::unary::UnaryWithParam> fused_activation,
        const std::optional<const TernaryMatmulConfig>& config,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<const DataType> dtype,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config,
        int32_t chunks = 1,
        int32_t dim = -1,
        std::optional<float> fused_ternary_scalar = std::nullopt,
        const std::optional<Tensor>& fused_ternary_input_a = std::nullopt,
        const std::optional<Tensor>& fused_ternary_input_b = std::nullopt,
        const std::optional<Tensor>& norm_weight = std::nullopt,
        std::optional<float> norm_epsilon = std::nullopt);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> ternary_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<Tensor>& bias_tensor,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const experimental::prim::TernaryMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    int32_t chunks = 1,
    int32_t dim = -1,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<Tensor>& fused_ternary_input_a = std::nullopt,
    const std::optional<Tensor>& fused_ternary_input_b = std::nullopt,
    bool use_packed_ternary = false,
    const std::optional<Tensor>& norm_weight = std::nullopt,
    std::optional<float> norm_epsilon = std::nullopt);

}  // namespace ttnn::prim
