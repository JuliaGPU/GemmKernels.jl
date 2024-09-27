#!/bin/bash
#PBS -N profile
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=26:00:00

cd /data/gent/428/vsc42867/julia/dev/GemmKernels

module load CUDA/12.5.0

export LD_LIBRARY_PATH=
export JULIA_CUDA_USE_COMPAT=false

julia --project -e 'using InteractiveUtils; using CUDA; using Pkg; versioninfo(); CUDA.versioninfo(); Pkg.status(mode=PKGMODE_MANIFEST)'
ncu ./tuning/tune.sh -u --no-systemd
