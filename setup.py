from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='torch_quant_ext',
      ext_modules=[cpp_extension.CppExtension('torch_quant_ext', ['quant_tensor_cuda.cpp','quant_tensor_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      version='0.1.0',
      )