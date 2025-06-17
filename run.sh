#!/usr/bin/env bash
set -Eeuo pipefail

function success() {
	for file in tuning/*.pdf; do
		curl -T "$file" -H "Filename: $file" -H "Title: Benchmark finished on $(hostname)" ntfy.thomasfaingnaert.be/benchmarks-wo
	done
}

function failure() {
	curl -d "Benchmark failed on $(hostname)" -H "Title: Benchmark failed on $(hostname)"  -H "Priority: urgent" ntfy.thomasfaingnaert.be/benchmarks-wo
}

rm -rf ~/.julia/scratchspaces/*
rm -f tuning/best-configs.bin

if [[ "$(hostname)" == "cyclops" ]]; then
	(./tuning/tune.sh -u && GK_PLOT_ONLY=1 ./tuning/tune.sh -u -i 1) && success || failure
else
	./tuning/tune.sh && success || failure
fi
