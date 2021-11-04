# Benchmarking

## Setup

```bash
# Set JULIA_PATH if you want to use a source build of Julia
# export JULIA_PATH=~/src/julia

export CUTLASS_PROF_PATH=~/src/cutlass/build/tools/profiler/cutlass_profiler
export CUTLASS_EXAMPLES_BUILD_PATH=~/src/cutlass/build/examples
```

## WMMA

```bash
cd wmma/

for file in gemmkernels.jl cublas.jl; do
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
include("benchmark.jl");

bench = @benchmark bench_cublas($a_cublas, $b, $c, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

bench = @benchmark bench_gemmkernels($a_gk, $b, $c, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);
```

### Kernel statistics

```bash
cd diagonal/

nv-nsight-cu-cli --mode=launch $(which julia) --project=../../
include("benchmark.jl")

bench_cublas(a_cublas, b, c, M, N, K, transpose_a, transpose_b, 10);

bench_gemmkernels(a_gk, b, c, M, N, K, transpose_a, transpose_b, 10);
```

## Operator fusion

```bash
cd operator-fusion/

julia --project=../../
include("benchmark.jl")

###############

bench = @benchmark bench_cublas($a, $b, $c, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

bench = @benchmark bench_gemmkernels($a, $b, $c, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

###############

bench = @benchmark bench_cublas_relu($a, $b, $c, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

bench = @benchmark bench_gemmkernels_relu($a, $b, $c, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

###############

bench = @benchmark bench_cublas_bias($a, $b, $c, $bias, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

bench = @benchmark bench_gemmkernels_bias($a, $b, $c, $bias, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

###############

bench = @benchmark bench_cublas_biasrelu($a, $b, $c, $bias, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

bench = @benchmark bench_gemmkernels_biasrelu($a, $b, $c, $bias, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

###############

bench = @benchmark bench_cublas_biasrelutwice($a, $b, $c, $bias, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

bench = @benchmark bench_gemmkernels_biasrelutwice($a, $b, $c, $bias, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

###############

bench = @benchmark bench_cublas_biasrelutwice_ab_elop($a, $b, $c, $bias, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);

bench = @benchmark bench_gemmkernels_biasrelutwice_ab_elop($a, $b, $c, $bias, $M, $N, $K, $transpose_a, $transpose_b); summarise_bench(bench);
```

## Complex and Dual numbers

```bash
cd complex-dual/

for file in gemmkernels_*.jl cudajl_complex.jl; do
    ./profile-julia.sh $file
done

./profile-cutlass.sh

julia plot.jl
```
