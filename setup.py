from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='quant_cuda_tools',
      ext_modules=[cpp_extension.CppExtension('quant_cuda_tools', ['quant_tensor_cuda.cpp','quant_tensor_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})