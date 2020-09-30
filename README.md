# GemmKernels

_Flexible and performant GEMM kernels in Julia_

[![Pipeline Status][cicd-image]][cicd-url]

This package contains a framework to instantiate flexible, performant GEMM (General Matrix Multiplication) kernels.

It decomposes GEMM kernels into orthogonal components:

- _Params_ determine the tiling size and launch configuration of the GEMM kernel. The tiling sizes are specified in _logical_ coordinates, i.e. with a meaning specified by the user.
- _Layouts_ convert the logical coordinates of tiles to physical offsets in memory.
- _Transforms_ are used to apply any arbitrary Julia functor to the GEMM's inputs or outputs. They are applied after every load, and before every store.
- _Operators_ are responsible to perform the matrix multiplication itself. They load tiles from shared memory, perform the matrix multiplication, and store the resultant tile back to shared memory.
- _Epilogues_ copy tiles of the resultant matrix to global memory, and can be used to implement arbitrary post-processing, such as adding a bias vector to the resultant matrix.

Each of these components corresponds to a set of functions with a predetermined interface.
These functions can be customised by the user through Julia's multiple dispatch functionality.

The package includes 2 user-facing interfaces: a fully-featured interface (see e.g. `test/matmul.jl`) and a BLAS-like interface (see e.g. `test/blas.jl`).

The documentation is still a WIP, but you can get an idea of the usage of this package using the examples in `test/` and `benchmark/`.

## Quick Start

The package can be installed using Julia's build-in package manager.
Open the Julia REPL, type `]` to enter Pkg-mode, and run:

```
pkg> add https://github.com/thomasfaingnaert/GemmKernels.jl
```

Note that you need a sufficiently recent NVIDIA GPU (Volta or later) that contains Tensor Cores to use this package.

## Project Status

At the moment, the package only contains GEMM kernels for CUDA-enabled NVIDIA GPUs and targets Tensor Cores exclusively.

It contains the necessary components for mixed-precision GEMMs using WMMA, GEMMs exploiting operation fusion with elementwise operations or bias vectors, diagonal matrices, and matrices of complex/dual numbers.

## Performance

![Performance Graph][performance-graph]

The above figure shows the performance of a mixed-precision multiplication of two FP16 matrices, resulting in an FP32 resultant matrix, for different memory layouts.
We compare our kernels with the state-of-the-art libraries cuBLAS and CUTLASS on an RTX 2080 Ti.

## Citation

For more details on the implementation and performance results, please see our accompanying paper (pre-print available on [arXiv][arxiv-paper]).
The `CITATION.bib` file in the root of this repository contains a citation.

[cicd-image]: https://gitlab.com/JuliaGPU/GemmKernels.jl/badges/master/pipeline.svg
[cicd-url]: https://gitlab.com/JuliaGPU/GemmKernels.jl/commits/master
[performance-graph]: media/performance-wmma-gemm.png
[arxiv-paper]: https://arxiv.org/abs/2009.12263
