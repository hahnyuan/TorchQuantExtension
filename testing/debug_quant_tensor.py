import math
import torch
import torch.nn.functional as F

# Our module!
import torch_quant_ext
import time


def raw_torch_quant_tensor_forward(
    tensor, scale, zero_point, qmin, qmax, asymmetric=False, simulate=True
):
    s = scale
    zp = zero_point
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


shape = [32, 128, 56, 56]
shape = [2, 3, 2, 1]

tensor = torch.randn(shape).cuda()
qmin = -128
qmax = 127
# for scale_shape in [(1,), (shape[0], 1, 1, 1), (1, shape[1], 1, 1)]:
for scale_shape in [(1, shape[1], 1, 1)]:
    print(f"====== scale_shape: {scale_shape} =====")
    scale = torch.randn(scale_shape).cuda() * 0.1
    zero_point = torch.randn(scale_shape).cuda()

    for asymmetric in [False, True]:
        for simulate in [False, True]:
            # speed test for quant_cuda_tools.quant_tensor_forward, with warm up
            for i in range(1):
                qtensor = torch_quant_ext.quant_tensor_forward(
                    tensor, scale, zero_point, qmin, qmax, asymmetric, simulate
                )

            for i in range(1):
                raw_qtensor = raw_torch_quant_tensor_forward(
                    tensor, scale, zero_point, qmin, qmax, asymmetric, simulate
                )

            # print(qtensor)
            print(qtensor.view(-1)[:10])
            print(raw_qtensor.view(-1)[:10])
            # print((raw_qtensor != qtensor).float().mean())
            assert torch.allclose(qtensor, raw_qtensor, atol=1e-2, rtol=1e-2)
