#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor quant_tensor_cuda_forward(
    torch::Tensor tensor,
    torch::Tensor scale,
    torch::Tensor zero_point,
    const int qmin,
    const int qmax,
    bool asymmetric = false,
    bool simulate = true);

// TODO
// std::vector<torch::Tensor> quant_tensor_backward

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor quant_tensor_forward(
    torch::Tensor tensor,
    torch::Tensor scale,
    torch::Tensor zero_point,
    const int qmin,
    const int qmax,
    bool asymmetric = false,
    bool simulate = true)
{
    CHECK_INPUT(tensor);
    CHECK_INPUT(scale);
    CHECK_INPUT(zero_point);

    // printf("qmin = %d, qmax = %d\n", qmin, qmax);

    return quant_tensor_cuda_forward(tensor, scale, zero_point, qmin, qmax, asymmetric, simulate);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_tensor_forward", &quant_tensor_forward, "Quant tensor forward");
}