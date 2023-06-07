using JSON

file = ARGS[1]
@show file

function main()
    open(file, "a") do io
        write(io, "impl,contraction,result\n")
    end

    fp = open("benchmarks/tensor-contractions/benchmark-suite.json", "r")

    jsonData = JSON.parse(read(fp, String))

    for el in jsonData
        parseableName = el["parseableName"]

        tensorModes = Vector{Vector{Int}}(undef, 0)
        for tensor in split(parseableName, "-")
            tensorMode = Vector{Int}(undef, 0)

            for mode in split(tensor, ".")
                push!(tensorMode, parse(Int, mode))
            end

            push!(tensorModes, tensorMode)
        end

        extents = Tuple(x for x in el["extents"])

        # return (tensorModes, extents)
        tensorModes = repr(tensorModes)
        extents = repr(extents)

        println(el["name"])

        cmd = `/usr/local/NVIDIA-Nsight-Compute-2023.1/ncu --profile-from-start off -f --csv --print-units base --metrics 'gpu__time_duration.avg' /home/wjvermeu/julia-versions/julia-1.8.3/bin/julia --project  /home/wjvermeu/GemmKernels.jl/benchmarks/tensor-contractions/contraction.jl $tensorModes $extents`

        result = read(
            pipeline(
                cmd, 
                `grep 'gpu__time_duration.avg'`,
                `awk -F',' '{print $NF}'`,
                `sed 's/"//g'`,
                `paste -sd ','`
            ), 
            String
        )[1:end-1]

        open(file, "a") do io
            write(io, "GK", ",", el["name"], ",\"", result, "\"\n")
        end

        cmd = `/usr/local/NVIDIA-Nsight-Compute-2023.1/ncu --profile-from-start off -f --csv --print-units base --metrics 'gpu__time_duration.avg' /home/wjvermeu/julia-versions/julia-1.8.3/bin/julia --project  /home/wjvermeu/GemmKernels.jl/benchmarks/tensor-contractions/cutensor.jl $tensorModes $extents`

        result = read(
            pipeline(
                cmd, 
                `grep 'gpu__time_duration.avg'`,
                `awk -F',' '{print $NF}'`,
                `sed 's/"//g'`,
                `paste -sd ','`
            ), 
            String
        )[1:end-1]

        open(file, "a") do io
            write(io, "CT", ",", el["name"], ",\"", result, "\"\n")
        end
    end

    nothing
end

isinteractive() || main()