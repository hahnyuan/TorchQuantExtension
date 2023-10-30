# Pytorch extension for quantization with high-efficient CUDA kernels

## Introduction
This is a Pytorch extension for quantization with high-efficient CUDA kernels. The kernels are implemented in CUDA and C++ for speed. The extension can be used as a drop-in replacement for the quantization functions in Pytorch.

## Install
```bash
python setup.py install
```

## Usage
```python
import torch
import torch_quant_ext
tensor=torch.randn(1,128, 56,56).cuda()
scale=torch.ones(1).cuda()*0.1
zero_point=torch.zeros(1).cuda()
qmin=-127
qmax=128
asymmetric = False
simulate = True
qtensor=torch_quant_ext.quant_tensor_forward(tensor, scale, zero_point, qmin, qmax, asymmetric, simulate)
```
