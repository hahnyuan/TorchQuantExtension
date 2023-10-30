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
    const int asymmetric,
    const bool simulate)
{
    scalar_t s = scale[0];
    scalar_t zp = zero_point[0];
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
    auto quantized_tensor = torch::zeros_like(tensor);
    if (scale_numel == 1)
    {
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
    }
    // else if (ndim == 4 && tensor_size[1] == scale_size[1] && scale_size[1] == scale_numel)
    // {
    // }
    else
    {
        // use original aten
        // integer = tensor.div(scale).add_(0.5).floor_()
        //     if self.asymmetric:
        //         integer.add_(zero_point)
        //     out = integer.clamp_(self.qmin, self.qmax)
        //     if simulate:
        //         if self.asymmetric:
        //             out = out.sub_(zero_point)
        //         out = out.mul_(scale)
        //     else:
        //         out = out.long()
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
        quantized_tensor = out;
        }
    return quantized_tensor;
}

template <typename scalar_t>
__global__ void quant_tensor_4d_g1_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> tensor,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> scale,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> zero_point,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> quantized_tensor)
{
}