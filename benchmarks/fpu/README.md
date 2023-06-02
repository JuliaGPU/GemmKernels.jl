# How to use

Set the environment variables `PATH_TO_JULIA`, `PATH_TO_NCU`, and `PATH_TO_CUTLASS`. Note that cuTLASS is optional if you just want to benchmark GemmKernels.jl against cuBLAS. Just remove the `PATH_TO_CUTLASS` line from `benchmark.jl` in that case.

```bash
# From the root of GemmKernels.jl

# For GemmKernels.jl
julia --project benchmarks/fpu/benchmark.jl GEMMKERNELS gemmkernels.csv
# For CUBLAS
julia --project benchmarks/fpu/benchmark.jl CUBLAS cublas.csv
# For CUTLASS
julia --project benchmarks/fpu/benchmark.jl CUTLASS cutlass.csv
```

If Nsight Compute throws a 'No kernels were profiled.' error, try adding this to the beginning of the command.

```bash
LD_LIBRARY_PATH=$($PATH_TO_JULIA -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))')
```