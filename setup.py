from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='offload_adam',
    ext_modules=[
        CppExtension(
            'offload_adam',
            ['offload_adam_bindings.c'],
            extra_compile_args=['-mavx512f', '-O3']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
