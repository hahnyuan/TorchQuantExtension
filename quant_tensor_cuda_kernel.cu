#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#define N_PER_THREAD 16

template <typename scalar_t>
__global__ void quant_tensor_pertensor_forward_kernel(
    const scalar_t *tensor,
    const scalar_t *scale,
    const scalar_t *zero_point,
    scalar_t *quantized_tensor,
    const long tensor_numel,
    const long qmin,
    const long qmax,
    const bool asymmetric, const bool simulate)
{
    scalar_t s = scale[0];
    scalar_t zp = asymmetric ? zero_point[0] : 0;
    // index
    for (long i = 0; i < N_PER_THREAD; i++)
    {
        const long ind = i * (blockDim.x * gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
        // const long ind = blockIdx.x * blockDim.x + threadIdx.x;
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
}

template <typename scalar_t>
__global__ void quant_tensor_pertensor_sym_sim_fast_kernel(
    const scalar_t *tensor,
    const scalar_t *scale,
    scalar_t *quantized_tensor,
    const long tensor_numel,
    const long qmin,
    const long qmax)
{
    // Experimental fast kernel for symmetric quantization
    scalar_t s = scale[0];
    // index
    for (long i = 0; i < N_PER_THREAD; i++)
    {
        const long ind = i * (blockDim.x * gridDim.x) + blockIdx.x * blockDim.x + threadIdx.x;

        if (ind < tensor_numel)
        {
            float o = floorf(tensor[ind] / s + 0.5);
            o = fmin(fmax(o, qmin), qmax);
            quantized_tensor[ind] = o * s;
        }
    }
}

template <typename scalar_t>
__global__ void quant_tensor_g0_forward_kernel(

    const scalar_t *tensor,
    const scalar_t *scale,
    const scalar_t *zero_point,
    scalar_t *quantized_tensor,
    const long scale_num,
    const long n_per_channel,
    const long qmin,
    const long qmax,
    const bool asymmetric,
    const bool simulate)
{
    // index
    long ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < n_per_channel)
    {
        for (int c = 0; c < scale_num; c++)
        {
            scalar_t s = scale[c];
            scalar_t zp = asymmetric ? zero_point[c] : 0;
            float o = floorf(tensor[ind] / s + 0.5);

            if (asymmetric)
            {
                o += zp;
            }
            o = fmin(fmax(o, qmin), qmax);
            // printf("ind %d, s %f, tensor[ind], %f, o %d, qmin %d, qmax %d\nblock info blockIdx.x=%d, blockIdx.y=%d, blockDim.x=%d, threadIdx.x=%d, ind=%d, n_per_channel=%d, c=%d\n", ind, s, tensor[ind], o, qmin, qmax, blockIdx.x, blockIdx.y, blockDim.x, threadIdx.x, ind, n_per_channel, c);
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
            ind += n_per_channel;
        }
    }
}

torch::Tensor quant_tensor_cuda_forward(
    torch::Tensor tensor,
    torch::Tensor scale,
    torch::Tensor zero_point,
    const long qmin,
    const long qmax,
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
        const long threads = 64;
        const dim3 blocks((tensor_numel + (threads * N_PER_THREAD) - 1) / (threads * N_PER_THREAD));
        auto tensor_ptr = tensor.data_ptr<float>();
        auto scale_ptr = scale.data_ptr<float>();
        auto zero_point_ptr = zero_point.data_ptr<float>();
        auto quantized_tensor_ptr = quantized_tensor.data_ptr<float>();
        if (simulate && !asymmetric)
        {
            quant_tensor_pertensor_sym_sim_fast_kernel<float><<<blocks, threads>>>(
                tensor_ptr,
                scale_ptr,
                quantized_tensor_ptr,
                tensor_numel,
                qmin,
                qmax);
            return quantized_tensor;
        }
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
            // printf("quant_tensor_g0_forward_kernel scale_numel %d", scale_numel);
            // per channel (one-dimensional) quantization
            auto tensor_view = tensor.view({scale_numel, -1});
            const long threads = 32;
            const long n_per_channel = tensor_view.size(1);
            const dim3 blocks((n_per_channel + threads - 1) / threads);
            auto quantized_tensor = torch::zeros_like(tensor);

            quant_tensor_g0_forward_kernel<float><<<blocks, threads>>>(
                tensor_view.data_ptr<float>(),
                scale.data_ptr<float>(),
                zero_point.data_ptr<float>(),
                quantized_tensor.data_ptr<float>(),
                scale_numel,
                n_per_channel,
                qmin,
                qmax,
                asymmetric,
                simulate);
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
    return out;
}
