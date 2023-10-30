from setuptools import setup, Extension
from torch.utils import cpp_extension

ext_module = cpp_extension.CppExtension(
    "torch_quant_ext",
    ["quant_tensor_cuda.cpp", "quant_tensor_cuda_kernel.cu"],
    # extra_compile_args={"cxx": ["-g"], "nvcc": ["-g", "-G"]},
)
setup(
    name="torch_quant_ext",
    ext_modules=[ext_module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    version="0.1.1",
)
