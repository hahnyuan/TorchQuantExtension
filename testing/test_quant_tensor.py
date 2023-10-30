import math
import torch
import torch.nn.functional as F

# Our module!
import torch_quant_ext
import time

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

tensor=torch.randn(1,128, 56,56).cuda()
scale=torch.ones(1).cuda()*0.1
zero_point=torch.ones(1).cuda()
qmin=-127
qmax=128
for asymmetric in [False,True]:
    for simulate in [False,True]:

        # speed test for quant_cuda_tools.quant_tensor_forward, with warm up
        for i in range(10):
            qtensor=torch_quant_ext.quant_tensor_forward(tensor, scale, zero_point, qmin, qmax, asymmetric, simulate)

        st=time.time()
        for i in range(1000):
            qtensor=torch_quant_ext.quant_tensor_forward(tensor, scale, zero_point, qmin, qmax, asymmetric, simulate)
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
