# Pytorch extension for quantization with high-efficient CUDA kernels

## Introduction
This is a Pytorch extension for quantization with high-efficient CUDA kernels. The kernels are implemented in CUDA and C++ for speed. The extension can be used as a drop-in replacement for the quantization functions in Pytorch.

Now Implemented:
- [x] quant_tensor_forward
    - [x] per-tensor quantization
    - [x] per-channel (dim=0) quantization
    - [x] per-channel (dim!=1) quantization
    - [x] multi-dimensional (contiguous dims) quantization
    - [x] symmetric and asymmetric quantization
    - [x] simulate quantization
    - [x] hardware quantization

TODO:
- quant_tensor_forward
    - [ ] schedule space search
    - [ ] optimize for hardware quantization
    - [ ] multi-dimensional (non-contiguous dims) quantization
- [ ] quant_tensor_backward

## Install
```bash
python setup.py install
```

## Usage

Function `torch_quant_ext.quant_tensor_forward` arguments:
- `tensor`: input tensor, must be float32 and on CUDA
- `scale`: a scale factor for quantization, must be float32 and on CUDA
- `zero_point`: a zero point for quantization, must be int32 and on CUDA when `asymmetric` is True
- `qmin`: minimum quantized value, int
- `qmax`: maximum quantized value, int
- `asymmetric`: whether to use asymmetric quantization
- `simulate`: whether to simulate quantization, if True, the function returns a simulated float32 tensor, otherwise, the function returns a integer tensor.

Example
```python
import torch
import torch_quant_ext
tensor=torch.randn(1,128, 56,56).cuda()
scale=torch.ones(1).cuda()*0.1
zero_point=torch.zeros(1).cuda()
qmin=-128
qmax=127
asymmetric = False
simulate = True
qtensor=torch_quant_ext.quant_tensor_forward(tensor, scale, zero_point, qmin, qmax, asymmetric, simulate)
```

## Benchmark

You can run the benchmark with the following command:
```bash
python testing/test_quant_tensor.py
```

In 3090, the benchmark result now is:
```
 input tensor shape = [32, 128, 56, 56]
====== scale_shape: (1,) =====
quant_cuda_tools.quant_tensor_forward time: 0.16011714935302734
raw_torch_quant_tensor_forward time: 0.496074914932251
quant_cuda_tools.quant_tensor_forward time: 0.1484525203704834
raw_torch_quant_tensor_forward time: 0.6212563514709473
quant_cuda_tools.quant_tensor_forward time: 0.148956298828125
raw_torch_quant_tensor_forward time: 0.6214470863342285
quant_cuda_tools.quant_tensor_forward time: 0.1492767333984375
raw_torch_quant_tensor_forward time: 0.869920015335083
====== scale_shape: (32, 1, 1, 1) =====
quant_cuda_tools.quant_tensor_forward time: 0.14829182624816895
raw_torch_quant_tensor_forward time: 0.4974837303161621
quant_cuda_tools.quant_tensor_forward time: 0.14850831031799316
raw_torch_quant_tensor_forward time: 0.6215956211090088
quant_cuda_tools.quant_tensor_forward time: 0.14849543571472168
raw_torch_quant_tensor_forward time: 0.6217432022094727
quant_cuda_tools.quant_tensor_forward time: 0.14897656440734863
raw_torch_quant_tensor_forward time: 0.8700988292694092
====== scale_shape: (1, 128, 1, 1) =====
quant_cuda_tools.quant_tensor_forward time: 0.16698098182678223
raw_torch_quant_tensor_forward time: 0.5074722766876221
quant_cuda_tools.quant_tensor_forward time: 0.16730976104736328
raw_torch_quant_tensor_forward time: 0.6299552917480469
quant_cuda_tools.quant_tensor_forward time: 0.1678450107574463
raw_torch_quant_tensor_forward time: 0.6304922103881836
quant_cuda_tools.quant_tensor_forward time: 0.16791033744812012
raw_torch_quant_tensor_forward time: 0.8754174709320068
====== scale_shape: (1, 128, 56, 1) =====
quant_cuda_tools.quant_tensor_forward time: 0.20274829864501953
raw_torch_quant_tensor_forward time: 0.5258545875549316
quant_cuda_tools.quant_tensor_forward time: 0.20163941383361816
raw_torch_quant_tensor_forward time: 0.6451809406280518
quant_cuda_tools.quant_tensor_forward time: 0.20253539085388184
raw_torch_quant_tensor_forward time: 0.6454493999481201
quant_cuda_tools.quant_tensor_forward time: 0.20097780227661133
raw_torch_quant_tensor_forward time: 0.8841354846954346
====== scale_shape: (32, 128, 1, 1) =====
quant_cuda_tools.quant_tensor_forward time: 0.20096588134765625
raw_torch_quant_tensor_forward time: 0.5245876312255859
quant_cuda_tools.quant_tensor_forward time: 0.2000885009765625
raw_torch_quant_tensor_forward time: 0.643810510635376
quant_cuda_tools.quant_tensor_forward time: 0.20007944107055664
raw_torch_quant_tensor_forward time: 0.6438231468200684
quant_cuda_tools.quant_tensor_forward time: 0.20105504989624023
raw_torch_quant_tensor_forward time: 0.8828990459442139
```