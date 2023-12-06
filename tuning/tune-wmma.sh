#!/usr/bin/env bash
set -Eeuo pipefail

GPU_CLOCK=915
MEM_CLOCK=5001

usage()
{
    cat <<EOF >&2
Usage: $0 [OPTIONS]

Tune WMMA Parameters.

Options:
-h, --help                 Show this help.
-gc, --gpu-clock speed     Change the frequency the GPU core clock is locked to
                           before benchmarking, in MHz (default 915 MHz).
-mc, --memory-clock speed  Change the frequency the GPU memory clock is locked to
                           before benchmarking, in MHz (default 5001 MHz).
EOF
}

positional=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage; exit 0
            ;;
        -gc|--gpu-clock)
            shift
            GPU_CLOCK=$1
            shift
            ;;
        -mc|--memory-clock)
            shift
            MEM_CLOCK=$1
            shift
            ;;
        -*)
            echo "Unknown command-line option '$1'."
            echo "Try '$0 --help' for more information."
            exit 1
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done
set -- "${positional[@]}"

if [[ $# -ne 0 ]]; then
    echo "Expected 0 positional arguments, but got $#."
    echo "Try '$0 --help' for more information."
    exit 1
fi

echo "Locking GPU clock speeds to $GPU_CLOCK MHz (GPU) / $MEM_CLOCK MHz (Mem)..."

if ! nvidia-smi --query-supported-clocks=graphics,memory --format=csv | grep -F "$GPU_CLOCK MHz, $MEM_CLOCK MHz"; then
    echo "Unsupported combination of clock speeds!"
    exit 1
fi

# Prompt for sudo
sudo -v &>/dev/null

# Sudo keep-alive
while true; do
    sleep 300
    sudo -n true
    kill -0 "$$" || exit
done &> /dev/null &

sudo nvidia-smi -pm 1
sudo nvidia-smi --lock-gpu-clocks=$GPU_CLOCK
sudo nvidia-smi --lock-memory-clocks=$MEM_CLOCK

cd "$( dirname "${BASH_SOURCE[0]}" )"
cd ..

echo "+++ :julia: Instantiating project"
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=tuning -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

until julia --project=tuning -e '
    using CUDA, Distributed

    # determine how many workers to use
    memory_usage = 5*2^30
    cpu_memory = Sys.free_memory()
    gpu_memory = CUDA.available_memory()
    workers = min(
        floor(Int, cpu_memory / memory_usage),
        floor(Int, gpu_memory / memory_usage),
        Sys.CPU_THREADS
    )
    println("+++ :julia: Tuning using $workers workers")

    # launch workers
    using Distributed
    env = [
        "JULIA_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => string(memory_usage),
    ]
    exeflags = [
        "--project=$(Base.active_project())",
        "--heap-size-hint=$memory_usage"
    ]
    addprocs(workers; exeflags, env)

    using Distributed
    @everywhere push!(LOAD_PATH, @__DIR__)
    @everywhere include("tuning/tune-wmma.jl")' "$@"; do

    echo "Tuning script crashed. Resuming after 1 second..." >&2
    sleep 1
done

echo "Unlocking GPU clock speeds..."
sudo nvidia-smi --reset-gpu-clocks
sudo nvidia-smi --reset-memory-clocks
