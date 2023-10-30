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
// std::vector<torch::Tensor> quant_tensor_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell,
//     torch::Tensor new_cell,
//     torch::Tensor input_gate,
//     torch::Tensor output_gate,
//     torch::Tensor candidate_cell,
//     torch::Tensor X,
//     torch::Tensor gate_weights,
//     torch::Tensor weights);

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

// TODO
// std::vector<torch::Tensor> lltm_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell,
//     torch::Tensor new_cell,
//     torch::Tensor input_gate,
//     torch::Tensor output_gate,
//     torch::Tensor candidate_cell,
//     torch::Tensor X,
//     torch::Tensor gate_weights,
//     torch::Tensor weights)
// {
//     CHECK_INPUT(grad_h);
//     CHECK_INPUT(grad_cell);
//     CHECK_INPUT(input_gate);
//     CHECK_INPUT(output_gate);
//     CHECK_INPUT(candidate_cell);
//     CHECK_INPUT(X);
//     CHECK_INPUT(gate_weights);
//     CHECK_INPUT(weights);

//     return lltm_cuda_backward(
//         grad_h,
//         grad_cell,
//         new_cell,
//         input_gate,
//         output_gate,
//         candidate_cell,
//         X,
//         gate_weights,
//         weights);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_tensor_forward", &quant_tensor_forward, "Quant tensor forward");
    // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}