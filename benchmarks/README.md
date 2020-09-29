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

julia plot.jl wmma
```

## Diagonal matrices

TODO

## Operator fusion

TODO

## Complex and Dual numbers

TODO

