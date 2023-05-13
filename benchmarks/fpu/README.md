# How to use

```bash
# From the root of GemmKernels.jl

# For GemmKernels.jl
julia --project benchmarks/fpu/benchmark.jl GEMMKERNELS gemmkernels.csv
# For CUBLAS
julia --project benchmarks/fpu/benchmark.jl CUBLAS cublas.csv
# For CUTLASS
julia --project benchmarks/fpu/benchmark.jl CUTLASS cutlass.csv
```