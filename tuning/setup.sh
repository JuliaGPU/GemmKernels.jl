#!/usr/bin/env bash

# set-up that needs to be run as root

GPU_ID=$1
GPU_CLOCK=$2
MEM_CLOCK=$3
PID=$4

# check that we're root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root."
    exit 1
fi

# determine validity of settings
if [[ "$GPU_CLOCK" == "-1" ]]; then
    GPU_CLOCK="$(nvidia-smi -i $GPU_ID --query-supported-clocks=graphics --format=csv | sort -rn | head -1 | cut -f1 -d' ')"
fi
if [[ "$MEM_CLOCK" == "-1" ]]; then
    MEM_CLOCK="$(nvidia-smi -i $GPU_ID --query-supported-clocks=memory --format=csv | sort -rn | head -1 | cut -f1 -d' ')"
fi
echo "Locking GPU $GPU_ID clock speeds to $GPU_CLOCK MHz (GPU) / $MEM_CLOCK MHz (Mem)..."
if ! nvidia-smi -i $GPU_ID --query-supported-clocks=graphics,memory --format=csv | grep -F "$GPU_CLOCK MHz, $MEM_CLOCK MHz"; then
    echo "Unsupported combination of clock speeds!"
    exit 1
fi

# configure the GPU
nvidia-smi -i $GPU_ID -pm 1
nvidia-smi -i $GPU_ID --lock-gpu-clocks=$GPU_CLOCK
nvidia-smi -i $GPU_ID --lock-memory-clocks=$MEM_CLOCK

# wait for as long as the main script is running
echo "Waiting for PID $PID to finish..."
while kill -0 $PID 2>/dev/null; do
    sleep 1
done

# reset the GPU
echo "Unlocking GPU clock speeds..."
nvidia-smi -i $GPU_ID --reset-gpu-clocks
nvidia-smi -i $GPU_ID --reset-memory-clocks
