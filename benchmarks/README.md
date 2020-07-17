# Benchmarking

## WMMA

```bash
export JULIA_PATH=~/src/julia

for file in wmma/*.jl; do
    ./profile.sh $file
done

julia plot.jl wmma
```
