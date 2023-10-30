# Pytorch extension for quantization with high-efficient CUDA kernels

## Introduction
This is a Pytorch extension for quantization with high-efficient CUDA kernels. The kernels are implemented in CUDA and C++ for speed. The extension can be used as a drop-in replacement for the quantization functions in Pytorch.

Now Implemented:
- [x] quant_tensor_forward
    - [x] per-tensor quantization
    - [x] per-channel (dim=0) quantization
    - [x] symmetric and asymmetric quantization
    - [x] simulate quantization
    - [x] hardware quantization

TODO:
- quant_tensor_forward
    - [ ] optimize for per-tensor quantization (one thread process multiple elements)
    - [ ] optimize for hardware quantization
    - [ ] multi-dimensional quantization
    - [ ] per-channel (dim>0) quantization
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

The function returns a quantized tensor.

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
