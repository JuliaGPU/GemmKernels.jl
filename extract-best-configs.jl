#!/usr/bin/env julia

using DataFrames
using Serialization

function main()
    df = open("tuning/best-configs.bin") do io
        deserialize(io)
    end

    println("idx,gemmkernels_times,baseline_times")
    for (i, row) in enumerate(eachrow(df))
        println("$(join(row.gemmkernels_times, ";")),$(join(row.baseline_times,";"))")
    end
end

isinteractive() || main()
