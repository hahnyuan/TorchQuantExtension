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
 asymmetric False, simulate False>>
quant_cuda_tools.quant_tensor_forward time: 0.1618516445159912
raw_torch_quant_tensor_forward time: 0.49678897857666016
 asymmetric False, simulate True>>
quant_cuda_tools.quant_tensor_forward time: 0.14883160591125488
raw_torch_quant_tensor_forward time: 0.6218023300170898
 asymmetric True, simulate False>>
quant_cuda_tools.quant_tensor_forward time: 0.14836812019348145
raw_torch_quant_tensor_forward time: 0.6216340065002441
 asymmetric True, simulate True>>
quant_cuda_tools.quant_tensor_forward time: 0.14945650100708008
raw_torch_quant_tensor_forward time: 0.8701145648956299
====== scale_shape: (32, 1, 1, 1) =====
 asymmetric False, simulate False>>
quant_cuda_tools.quant_tensor_forward time: 0.14828157424926758
raw_torch_quant_tensor_forward time: 0.4974396228790283
 asymmetric False, simulate True>>
quant_cuda_tools.quant_tensor_forward time: 0.14822173118591309
raw_torch_quant_tensor_forward time: 0.6218113899230957
 asymmetric True, simulate False>>
quant_cuda_tools.quant_tensor_forward time: 0.14885473251342773
raw_torch_quant_tensor_forward time: 0.6221122741699219
 asymmetric True, simulate True>>
quant_cuda_tools.quant_tensor_forward time: 0.1488347053527832
raw_torch_quant_tensor_forward time: 0.8702168464660645
====== scale_shape: (1, 128, 1, 1) =====
 asymmetric False, simulate False>>
quant_cuda_tools.quant_tensor_forward time: 0.16342616081237793
raw_torch_quant_tensor_forward time: 0.5069057941436768
 asymmetric False, simulate True>>
quant_cuda_tools.quant_tensor_forward time: 0.1657116413116455
raw_torch_quant_tensor_forward time: 0.6297616958618164
 asymmetric True, simulate False>>
quant_cuda_tools.quant_tensor_forward time: 0.1664135456085205
raw_torch_quant_tensor_forward time: 0.6302330493927002
 asymmetric True, simulate True>>
quant_cuda_tools.quant_tensor_forward time: 0.167433500289917
raw_torch_quant_tensor_forward time: 0.8754758834838867
====== scale_shape: (1, 128, 56, 1) =====
 asymmetric False, simulate False>>
quant_cuda_tools.quant_tensor_forward time: 0.20211529731750488
raw_torch_quant_tensor_forward time: 0.5255229473114014
 asymmetric False, simulate True>>
quant_cuda_tools.quant_tensor_forward time: 0.2008955478668213
raw_torch_quant_tensor_forward time: 0.645756721496582
 asymmetric True, simulate False>>
quant_cuda_tools.quant_tensor_forward time: 0.20218849182128906
raw_torch_quant_tensor_forward time: 0.6454365253448486
 asymmetric True, simulate True>>
quant_cuda_tools.quant_tensor_forward time: 0.20259952545166016
raw_torch_quant_tensor_forward time: 0.8846399784088135
====== scale_shape: (32, 128, 1, 1) =====
 asymmetric False, simulate False>>
quant_cuda_tools.quant_tensor_forward time: 0.2020270824432373
raw_torch_quant_tensor_forward time: 0.5252349376678467
 asymmetric False, simulate True>>
quant_cuda_tools.quant_tensor_forward time: 0.20096182823181152
raw_torch_quant_tensor_forward time: 0.6446208953857422
 asymmetric True, simulate False>>
quant_cuda_tools.quant_tensor_forward time: 0.20227742195129395
raw_torch_quant_tensor_forward time: 0.6445820331573486
 asymmetric True, simulate True>>
quant_cuda_tools.quant_tensor_forward time: 0.20257925987243652
raw_torch_quant_tensor_forward time: 0.8831808567047119
```