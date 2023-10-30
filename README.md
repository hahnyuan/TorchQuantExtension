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
quant_cuda_tools.quant_tensor_forward time: 0.16434168815612793
raw_torch_quant_tensor_forward time: 0.49608373641967773
quant_cuda_tools.quant_tensor_forward time: 0.14842557907104492
raw_torch_quant_tensor_forward time: 0.6216511726379395
quant_cuda_tools.quant_tensor_forward time: 0.1490321159362793
raw_torch_quant_tensor_forward time: 0.6217072010040283
quant_cuda_tools.quant_tensor_forward time: 0.14930939674377441
raw_torch_quant_tensor_forward time: 0.8700952529907227
====== scale_shape: (32, 1, 1, 1) =====
quant_cuda_tools.quant_tensor_forward time: 0.14836525917053223
raw_torch_quant_tensor_forward time: 0.49754834175109863
quant_cuda_tools.quant_tensor_forward time: 0.14853239059448242
raw_torch_quant_tensor_forward time: 0.6219451427459717
quant_cuda_tools.quant_tensor_forward time: 0.14851784706115723
raw_torch_quant_tensor_forward time: 0.621967077255249
quant_cuda_tools.quant_tensor_forward time: 0.14893770217895508
raw_torch_quant_tensor_forward time: 0.8701951503753662
====== scale_shape: (1, 128, 1, 1) =====
quant_cuda_tools.quant_tensor_forward time: 0.1670219898223877
raw_torch_quant_tensor_forward time: 0.5074267387390137
quant_cuda_tools.quant_tensor_forward time: 0.1672518253326416
raw_torch_quant_tensor_forward time: 0.630223274230957
quant_cuda_tools.quant_tensor_forward time: 0.16779541969299316
raw_torch_quant_tensor_forward time: 0.6305780410766602
quant_cuda_tools.quant_tensor_forward time: 0.16786408424377441
raw_torch_quant_tensor_forward time: 0.8754994869232178
```