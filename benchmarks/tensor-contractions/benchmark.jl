using JSON

PATH_TO_NCU = get(ENV, "PATH_TO_NCU", "ncu")
PATH_TO_JULIA = get(ENV, "PATH_TO_JULIA", "julia")

PWD = ENV["PWD"]
RELATIVE_PATH = "benchmarks/tensor-contractions"

implementation = ARGS[1]
file = ARGS[2]
algorithm = get(ARGS, 3, "DEFAULT")

file = PWD * "/" * RELATIVE_PATH * "/" * file

function main()
    @show implementation, file, algorithm

    # Open JSON with TC configs
    # These configs hold the TC name, a parseable representation of that name, and the extents of 
    # the dimensions of the parseable representation. The tensor contractions are source from
    # the TCCG benchmark suite.
    fp = open(PWD * "/" * RELATIVE_PATH * "/" * "benchmark-suite.json", "r")
    jsonData = JSON.parse(read(fp, String))
    close(fp)

    # Write CSV header
    open(file, "a") do io
        write(io, "contraction,result\n")
    end

    for el in jsonData
        # Get arguments from JSON
        parseableName = el["parseableName"]
        extents = Tuple(x for x in el["extents"])

        # Parsing the parseable name
        tensorModes = Vector{Vector{Int}}(undef, 0)
        for tensor in split(parseableName, "-")
            tensorMode = Vector{Int}(undef, 0)

            for mode in split(tensor, ".")
                push!(tensorMode, parse(Int, mode))
            end

            push!(tensorModes, tensorMode)
        end

        # Get string representations of arguments to pass to Julia script
        tensorModes = repr(tensorModes)
        extents = repr(extents)

        # Print the tensor contraction name to keep track of progress
        println(el["name"])

        # Prepare the command to run depending on the implementation
        if (implementation == "GEMMKERNELS")
            cmd = `$PATH_TO_NCU
                --profile-from-start off -f --csv --print-units base --metrics 'gpu__time_duration.avg' 
                $PATH_TO_JULIA --project 
                $PWD/$RELATIVE_PATH/contraction.jl 
                $tensorModes $extents`
        elseif (implementation == "CUTENSOR")
            cmd = `$PATH_TO_NCU
                --profile-from-start off -f --csv --print-units base --metrics 'gpu__time_duration.avg' 
                $PATH_TO_JULIA --project 
                $PWD/$RELATIVE_PATH/cutensor.jl 
                $tensorModes $extents $algorithm`
        end

        # Run the command and parse the result
        result = read(
            pipeline(
                cmd, 
                `grep 'gpu__time_duration.avg'`,
                `awk -F'",' '{print $NF}'`,
                `sed 's/"//g'`,
                `sed 's/,//g'`,
                `paste -sd ','`
            ), 
            String
        )[1:end-1]

        # Write results to CSV
        open(file, "a") do io
            write(io, el["name"], ",\"", result, "\"\n")
        end
        @show result
    end

    nothing
end

isinteractive() || main()