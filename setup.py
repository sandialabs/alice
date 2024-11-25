""" Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

Written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.

This algorithm implements methods described in the paper, "Curvature in the Looking-Glass:
Optimal Methods to Exploit Curvature of Expectation in the Loss Landscape."
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

arch_list = 'Ampere;Ada;Hopper'

os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list

setup(
    name='alice',
    version='0.1.1',
    ext_modules=[
        CUDAExtension(
            name='alice11_fused_cuda',
            sources=['./src/alice/alice11_fused_cuda/alice11_fused.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

