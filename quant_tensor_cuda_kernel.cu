#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void quant_tensor_pertensor_forward_kernel(
    const scalar_t *tensor,
    const scalar_t *scale,
    const scalar_t *zero_point,
    scalar_t *quantized_tensor,
    const int tensor_numel,
    const int qmin,
    const int qmax,
    const bool asymmetric, const bool simulate)
{
    scalar_t s = scale[0];
    scalar_t zp = asymmetric ? zero_point[0] : 0;
    // index
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < tensor_numel)
    {
        float o = floorf(tensor[ind] / s + 0.5);

        if (asymmetric)
        {
            o += zp;
        }
        o = fmin(fmax(o, qmin), qmax);
        // printf("ind %d, s %f, tensor[ind], %f, o %d, qmin %d, qmax %d\n", ind, s, tensor[ind], o, qmin, qmax);
        if (simulate)
        {
            if (asymmetric)
            {
                quantized_tensor[ind] = (o - zp) * s;
            }
            else
            {
                quantized_tensor[ind] = o * s;
            }
        }
        else
        {
            quantized_tensor[ind] = o;
        }
    }
}

template <typename scalar_t>
__global__ void quant_tensor_g0_forward_kernel(

    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> tensor,
    const scalar_t *scale,
    const scalar_t *zero_point,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> quantized_tensor,
    const int n_per_channel,
    const int qmin,
    const int qmax,
    const bool asymmetric,
    const bool simulate)
{
    const int c = blockIdx.x;
    scalar_t s = scale[c];
    scalar_t zp = asymmetric ? zero_point[0] : 0;

    // index
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < n_per_channel)
    {
        float o = floorf(tensor[c][ind] / s + 0.5);

        if (asymmetric)
        {
            o += zp;
        }
        o = fmin(fmax(o, qmin), qmax);
        // printf("ind %d, s %f, tensor[ind], %f, o %d, qmin %d, qmax %d\n", ind, s, tensor[ind], o, qmin, qmax);
        if (simulate)
        {
            if (asymmetric)
            {
                quantized_tensor[c][ind] = (o - zp) * s;
            }
            else
            {
                quantized_tensor[c][ind] = o * s;
            }
        }
        else
        {
            quantized_tensor[c][ind] = o;
        }
    }
}

torch::Tensor quant_tensor_cuda_forward(
    torch::Tensor tensor,
    torch::Tensor scale,
    torch::Tensor zero_point,
    const int qmin,
    const int qmax,
    bool asymmetric = false,
    bool simulate = true)
{
    const auto tensor_numel = tensor.numel();
    // const auto tensor_size = tensor.size();
    const auto scale_numel = scale.numel();
    // const auto scale_size = scale.size();
    const auto ndim = tensor.dim();
    auto data_type = tensor.type();

    if (scale_numel == 1)
    {
        // per-tensor quantization
        auto quantized_tensor = torch::zeros_like(tensor);
        const int threads = 32;
        const dim3 blocks((tensor_numel + threads - 1) / threads);
        auto tensor_ptr = tensor.data_ptr<float>();
        auto scale_ptr = scale.data_ptr<float>();
        auto zero_point_ptr = zero_point.data_ptr<float>();
        auto quantized_tensor_ptr = quantized_tensor.data_ptr<float>();

        quant_tensor_pertensor_forward_kernel<float><<<blocks, threads>>>(
            tensor_ptr,
            scale_ptr,
            zero_point_ptr,
            quantized_tensor_ptr,
            tensor_numel,
            qmin,
            qmax,
            asymmetric,
            simulate);
        return quantized_tensor;
    }
    else if (ndim > 1)
    {

        if (scale_numel == tensor.size(0))
        {
            // per channel (one-dimensional) quantization
            auto tensor_view = tensor.view({scale_numel, -1});
            const int threads = 32;
            const int n_per_channel = tensor_view.size(1);
            const dim3 blocks(scale_numel, (n_per_channel + threads - 1) / threads);
            auto quantized_tensor = torch::zeros_like(tensor_view);

            quant_tensor_g0_forward_kernel<float><<<blocks, threads>>>(
                tensor_view.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                scale.data_ptr<float>(),
                zero_point.data_ptr<float>(),
                quantized_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                n_per_channel,
                qmin,
                qmax,
                asymmetric,
                simulate);
            quantized_tensor = quantized_tensor.view_as(tensor);
            return quantized_tensor;
        }
    }

    // use original aten
    at::Tensor integer = tensor.div(scale).add_(0.5).floor_();
    if (asymmetric)
    {
        integer.add_(zero_point);
    }
    auto out = integer.clamp_(qmin, qmax);
    if (simulate)
    {
        if (asymmetric)
        {
            out = out.sub_(zero_point);
        }
        out = out.mul_(scale);
    }
    else
    {
        out = out.to(torch::kInt);
    }
    return out;
}
