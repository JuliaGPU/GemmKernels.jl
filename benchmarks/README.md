# Benchmarking

## WMMA

```bash
export JULIA_PATH=~/src/julia
export CUTLASS_PROF_PATH=~/src/cutlass/build/tools/profiler/cutlass_profiler

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

