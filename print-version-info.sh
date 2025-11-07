#!/usr/bin/env bash
set -Eeuo pipefail

# Host
echo "================================================================="
echo "$(hostname)"
echo "================================================================="

# Driver
cat /proc/driver/nvidia/version
julia --project=tuning -e 'using CUDA; println("CUDA toolkit supported by driver: ", CUDA.driver_version())'

# CUDA toolkit
julia --project=tuning -e 'using CUDA; println("CUDA toolkit version used: ", CUDA.runtime_version())'

# OS version
cat /etc/os-release | grep -F 'PRETTY_NAME'

# CPU specs
cat /proc/cpuinfo | grep -F 'model name' | uniq

# GPU
nvidia-smi -L
