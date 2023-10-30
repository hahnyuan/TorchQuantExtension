# Pytorch extension for quantization with high-efficient CUDA kernels

## Introduction
This is a Pytorch extension for quantization with high-efficient CUDA kernels. The kernels are implemented in CUDA and C++ for speed. The extension can be used as a drop-in replacement for the quantization functions in Pytorch.

Now Implemented:
- [x] quant_tensor_forward
    - [x] per-tensor quantization
    - [x] per-channel (dim=0) quantization
    - [x] per-channel (dim=1) quantization
    - [x] symmetric and asymmetric quantization
    - [x] simulate quantization
    - [x] hardware quantization

TODO:
- quant_tensor_forward
    - [ ] schedule space search
    - [ ] optimize for hardware quantization
    - [ ] multi-dimensional quantization
    - [ ] per-channel (dim>1) quantization
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
====== scale_shape: (1,) =====
quant_cuda_tools.quant_tensor_forward time: 0.1658310890197754
raw_torch_quant_tensor_forward time: 0.5000412464141846
quant_cuda_tools.quant_tensor_forward time: 0.14867734909057617
raw_torch_quant_tensor_forward time: 0.6218092441558838
quant_cuda_tools.quant_tensor_forward time: 0.1491851806640625
raw_torch_quant_tensor_forward time: 0.6219408512115479
quant_cuda_tools.quant_tensor_forward time: 0.14947509765625
raw_torch_quant_tensor_forward time: 0.8702328205108643
====== scale_shape: (32, 1, 1, 1) =====
quant_cuda_tools.quant_tensor_forward time: 0.15418529510498047
raw_torch_quant_tensor_forward time: 0.5008578300476074
quant_cuda_tools.quant_tensor_forward time: 0.15454864501953125
raw_torch_quant_tensor_forward time: 0.6247196197509766
quant_cuda_tools.quant_tensor_forward time: 0.15569067001342773
raw_torch_quant_tensor_forward time: 0.6252713203430176
quant_cuda_tools.quant_tensor_forward time: 0.15563368797302246
raw_torch_quant_tensor_forward time: 0.8722531795501709
====== scale_shape: (1, 128, 1, 1) =====
quant_cuda_tools.quant_tensor_forward time: 0.4161872863769531
raw_torch_quant_tensor_forward time: 0.4959299564361572
quant_cuda_tools.quant_tensor_forward time: 0.5526113510131836
raw_torch_quant_tensor_forward time: 0.6204173564910889
quant_cuda_tools.quant_tensor_forward time: 0.5526134967803955
raw_torch_quant_tensor_forward time: 0.6205401420593262
quant_cuda_tools.quant_tensor_forward time: 0.8255910873413086
raw_torch_quant_tensor_forward time: 0.8694519996643066
```