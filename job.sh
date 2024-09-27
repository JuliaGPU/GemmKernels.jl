#!/bin/bash
julia --project -e 'using InteractiveUtils; using CUDA; using Pkg; versioninfo(); CUDA.versioninfo(); Pkg.status(mode=PKGMODE_MANIFEST)'
~/bin/store/nvidia-nsight-compute-2024.3/ncu --profile-from-start off -o prof ./tuning/tune.sh -u --no-systemd
