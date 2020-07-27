#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./profile.sh <script>
if [[ $# < 1 ]]; then
    echo "Usage: $0 <script>" 1>&2
    exit 1
fi

# Julia path must be set
if [[ -z ${JULIA_PATH+x} ]]; then
    echo "Please set JULIA_PATH to the root path of the Julia repository." 1>&2
    echo "Example: export JULIA_PATH=~/src/julia/" 1>&2
    exit 1
fi

cd "$( dirname "${BASH_SOURCE[0]}" )"

SCRIPT="$1"

for a_layout in n t; do
    for b_layout in n t; do
        OUTPUT_FILE="${SCRIPT%.*}-${a_layout}${b_layout}.csv"

        printf "N,runtime\n" >$OUTPUT_FILE

        for i in {7..14}; do
            N=$((2**i))

            # runtime is in nanoseconds
            runtime=$(LD_LIBRARY_PATH=${JULIA_PATH}/usr/lib nv-nsight-cu-cli --profile-from-start off -f --csv --units base --metrics 'gpu__time_duration.avg' ${JULIA_PATH}/julia $SCRIPT $N $N $N $a_layout $b_layout 2>/dev/null | grep 'gpu__time_duration' | awk -F',' '{print $NF}' | sed 's/"//g' | paste -sd ',')

            printf "$N,\"$runtime\"\n" >>$OUTPUT_FILE
        done
    done
done
