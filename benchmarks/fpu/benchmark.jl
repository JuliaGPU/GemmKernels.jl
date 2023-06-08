using JSON

PATH_TO_NCU = get(ENV, "PATH_TO_NCU", "ncu")
PATH_TO_JULIA = get(ENV, "PATH_TO_JULIA", "julia")
PATH_TO_CUTLASS = get(ENV, "PATH_TO_CUTLASS", "cutlass")

PWD = ENV["PWD"]
RELATIVE_PATH = "./benchmarks/fpu"

implementation = ARGS[1]
file = ARGS[2]

file = PWD * "/" * RELATIVE_PATH * "/" * file

function main()
    # Open JSON with GEMM configs
    # These configs hold the GEMM shapes, operator shapes, compute types and data types
    fp = open(PWD * "/" * RELATIVE_PATH * "/" * "gemm-configs.json", "r")
    jsonData = JSON.parse(read(fp, String))
    close(fp)

    # Write CSV header
    open(file, "a") do io
        write(io, "N,results\n")
    end

    # For every GEMM configuration in the JSON
    for el in jsonData
        # Get arguments from JSON
        N = el["N"]
        GEMM = el["GEMM"]
        BLOCK = el["BLOCK"]
        OPERATOR = el["OPERATOR"]
        COMPUTE_TYPE = el["COMPUTE_TYPE"]
        DATA_TYPE = el["DATA_TYPE"]
        CUTLASS_KERNEL = el["CUTLASS_KERNEL"]

        WARPS_PER_BLOCK = haskey(el, "WARPS_PER_BLOCK") ? el["WARPS_PER_BLOCK"] : ""
        COMPUTE_WARP = haskey(el, "COMPUTE_WARP") ? el["COMPUTE_WARP"] : ""

        # Give arguments to Julia script
        if (implementation == "GEMMKERNELS")
            cmd = `$PATH_TO_NCU 
                --profile-from-start off -f --csv --print-units base --metrics 'gpu__time_duration.avg'
                $PATH_TO_JULIA --project 
                $PWD/$RELATIVE_PATH/fpu.jl
                $GEMM $BLOCK $OPERATOR $COMPUTE_TYPE $DATA_TYPE $WARPS_PER_BLOCK $COMPUTE_WARP`
            
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
        elseif (implementation == "CUBLAS")
            cmd = `$PATH_TO_NCU 
                --profile-from-start off -f --csv --print-units base --metrics 'gpu__time_duration.avg'
                $PATH_TO_JULIA --project 
                $PWD/$RELATIVE_PATH/cublas.jl
                $GEMM $COMPUTE_TYPE $DATA_TYPE`

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
        elseif (implementation == "CUTLASS")
            GEMM = eval(Meta.parse(GEMM))
            GEMM = NamedTuple{(:M, :N, :K)}(GEMM)

            cmd = `$PATH_TO_NCU
                --csv --print-units base --metrics 'gpu__time_duration.avg' -k Kernel2
                $PATH_TO_CUTLASS
                --operation=Gemm --gemm_kind=gemm --m=$(GEMM.M) --n=$(GEMM.N) --k=$(GEMM.K)
                --kernels=$(CUTLASS_KERNEL)
                --warmup-iterations=0 --profiling-iterations=10 --verification-enabled=false --alpha=2 --beta=3`

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
        end

        # Write results to CSV
        open(file, "a") do io
            write(io, string(N), ",\"", result, "\"\n")
        end
    end

    nothing
end

isinteractive() || main()