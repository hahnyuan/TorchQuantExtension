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


shape = [512, 128, 56, 56]

tensor = torch.randn(shape).cuda()
qmin = -127
qmax = 128
for scale_shape in [(1,), (shape[0], 1, 1, 1), (1, shape[1], 1, 1)]:
    print(f"====== scale_shape: {scale_shape} =====")
    scale = torch.randn(scale_shape).cuda() * 0.1
    zero_point = torch.randn(scale_shape).cuda()

    for asymmetric in [False, True]:
        for simulate in [False, True]:
            # speed test for quant_cuda_tools.quant_tensor_forward, with warm up
            for i in range(100):
                qtensor = torch_quant_ext.quant_tensor_forward(
                    tensor, scale, zero_point, qmin, qmax, asymmetric, simulate
                )

            st = time.time()
            for i in range(1000):
                qtensor = torch_quant_ext.quant_tensor_forward(
                    tensor, scale, zero_point, qmin, qmax, asymmetric, simulate
                )
            ed = time.time()
            print("quant_cuda_tools.quant_tensor_forward time:", ed - st)

            # speed test for raw_torch_quant_tensor_forward, with warm up
            for i in range(100):
                raw_qtensor = raw_torch_quant_tensor_forward(
                    tensor, scale, zero_point, qmin, qmax, asymmetric, simulate
                )

            st = time.time()
            for i in range(1000):
                raw_qtensor = raw_torch_quant_tensor_forward(
                    tensor, scale, zero_point, qmin, qmax, asymmetric, simulate
                )
            ed = time.time()
            print("raw_torch_quant_tensor_forward time:", ed - st)

            # print(qtensor)
            # print(qtensor.view(-1)[:10], raw_qtensor.view(-1)[:10])
            # print((raw_qtensor != qtensor).float().mean())
            assert torch.allclose(qtensor, raw_qtensor, atol=1e-2, rtol=1e-2)
