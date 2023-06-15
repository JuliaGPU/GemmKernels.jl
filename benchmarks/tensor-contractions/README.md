# How to use

Set the environment variables `PATH_TO_JULIA`, `PATH_TO_NCU`.

```bash
# From the root of GemmKernels.jl

julia --project benchmarks/tensor-contractions/benchmark.jl GEMMKERNELS gettGemmKernels.csv
julia --project benchmarks/tensor-contractions/benchmark.jl CUTENSOR gettCuTensor.csv GETT
julia --project benchmarks/tensor-contractions/benchmark.jl CUTENSOR tgettCuTensor.csv TGETT
julia --project benchmarks/tensor-contractions/benchmark.jl CUTENSOR ttgtCuTensor.csv TTGT
julia --project benchmarks/tensor-contractions/benchmark.jl CUTENSOR defaultCuTensor.csv DEFAULT
```

If Nsight Compute throws a 'No kernels were profiled.' error, try adding this to the beginning of the command.

```bash
LD_LIBRARY_PATH=$($PATH_TO_JULIA -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))')
```