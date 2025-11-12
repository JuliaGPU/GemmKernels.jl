#!/usr/bin/env julia

using DataFrames
using Serialization

function main()
    open("tuning/data-params-best-configs.csv", "w") do outf
        df = open("tuning/best-configs.bin") do io
            deserialize(io)
        end

        println(outf, "host;tc;implementation;metric_name;metric_value;metric_unit")

        for (i, row) in enumerate(eachrow(df))
            for (col, val) in pairs(row)
                pretty_val = val

                if isa(val, Array)
                    pretty_val = join(val, ",")
                end

                println(outf, "$(gethostname());$i;gk;$col;$pretty_val;")
            end
        end
    end
end

isinteractive() || main()
