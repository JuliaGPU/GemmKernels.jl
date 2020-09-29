# Benchmarking

## Setup

```bash
# Set JULIA_PATH if you want to use a source build of Julia
# export JULIA_PATH=~/src/julia

export CUTLASS_PROF_PATH=~/src/cutlass/build/tools/profiler/cutlass_profiler
```

## WMMA

```bash
cd wmma/

for file in *.jl; do
    ./profile-julia.sh $file
done

./profile-cutlass.sh

julia plot.jl
```

## Diagonal matrices

### Run times

```bash
cd diagonal/

julia --project=../../
include("benchmark.jl")

bench = @benchmark bench_cublas($a_cublas, $b, $c, $M, $N, $K, $transpose_a, $transpose_b)
summarise_bench(bench)

bench = @benchmark bench_gemmkernels($a_gk, $b, $c, $M, $N, $K, $transpose_a, $transpose_b)
summarise_bench(bench)
```

### Kernel statistics

```bash
cd diagonal/

nv-nsight-cu-cli --mode=launch $(which julia) --project=../../
include("benchmark.jl")

bench_cublas(a_cublas, b, c, M, N, K, transpose_a, transpose_b, 10)

bench_gemmkernels(a_gk, b, c, M, N, K, transpose_a, transpose_b, 10)
```

## Operator fusion

TODO

## Complex and Dual numbers

TODO

