#!/usr/bin/env bash
set -Eeuo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

GPU_ID=0
GPU_CLOCK=-1
MEM_CLOCK=-1

usage()
{
    cat <<EOF >&2
Usage: $0 [OPTIONS]

Tune WMMA Parameters.

Options:
-h, --help                 Show this help.
-i id                      Specify which GPU to target.
-gc, --gpu-clock speed     Change the frequency the GPU core clock is locked to
                           before benchmarking, in MHz (default the max frequency).
-mc, --memory-clock speed  Change the frequency the GPU memory clock is locked to
                           before benchmarking, in MHz (default the max frequency).
EOF
}

positional=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage; exit 0
            ;;
        -i)
            shift
            GPU_ID=$1
            shift
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

# set-up GPUs
sudo -b ./setup.sh $GPU_ID $GPU_CLOCK $MEM_CLOCK $$
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "+++ :julia: Instantiating project"
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

julia --project tune.jl "$@"
