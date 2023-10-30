import math
import torch
import torch.nn.functional as F

# Our module!
import quant_cuda_tools
import time

# quant_tensor_forward(
#     torch::Tensor tensor,
#     torch::Tensor scale,
#     torch::Tensor zero_point,
#     const int qmin,
#     const int qmax,
#     bool asymmetric = false,
#     bool simulate = true)

# write test code for quant_cuda_tools.quant_tensor_forward

tensor=torch.randn(1,128, 56,56).cuda()
scale=torch.ones(1).cuda()*0.1
zero_point=torch.zeros(1).cuda()
qmin=-127
qmax=128
asymmetric = False
simulate = True




# qtensor=quant_cuda_tools.quant_tensor_forward(torch.randn(10, 10).cuda(), torch.randn(10, 10).cuda())

def raw_torch_quant_tensor_forward(tensor, scale, zero_point, qmin, qmax, asymmetric = False, simulate = True):
    s=scale[0]
    zp=zero_point[0]
    # index
    o = (tensor / s + 0.5).floor()
    if asymmetric:
        o += zp
    o = torch.clamp(o, qmin, qmax)
    if simulate:
        if asymmetric:
            quantized_tensor = (o - zp) * s
        else:
            quantized_tensor = o * s
    else:
        quantized_tensor = o
    return quantized_tensor

# speed test for quant_cuda_tools.quant_tensor_forward, with warm up
for i in range(10):
    qtensor=quant_cuda_tools.quant_tensor_forward(tensor, scale, zero_point, qmin, qmax, asymmetric, simulate)

st=time.time()
for i in range(1000):
    qtensor=quant_cuda_tools.quant_tensor_forward(tensor, scale, zero_point, qmin, qmax, asymmetric, simulate)
ed=time.time()
print("quant_cuda_tools.quant_tensor_forward time:", ed-st)

# speed test for raw_torch_quant_tensor_forward, with warm up
for i in range(10):
    qtensor=raw_torch_quant_tensor_forward(tensor, scale, zero_point, qmin, qmax, asymmetric, simulate)

st=time.time()
for i in range(1000):
    qtensor=raw_torch_quant_tensor_forward(tensor, scale, zero_point, qmin, qmax, asymmetric, simulate)
ed=time.time()
print("raw_torch_quant_tensor_forward time:", ed-st)

# print(qtensor)
raw_qtensor=raw_torch_quant_tensor_forward(tensor, scale, zero_point, qmin, qmax, asymmetric, simulate)
# print(raw_qtensor)
assert torch.allclose(qtensor, raw_qtensor)
