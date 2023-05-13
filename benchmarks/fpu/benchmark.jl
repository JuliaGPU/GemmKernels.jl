using JSON

PATH_TO_NCU = haskey(ENV, "PATH_TO_NCU") ? ENV["PATH_TO_NCU"] : "/usr/local/NVIDIA-Nsight-Compute-2023.1/ncu"
PATH_TO_JULIA = haskey(ENV, "PATH_TO_JULIA") ? ENV["PATH_TO_JULIA"] : "/home/wjvermeu/julia-versions/julia-1.8.3/bin/julia"
PATH_TO_CUTLASS = haskey(ENV, "PATH_TO_CUTLASS") ? ENV["PATH_TO_CUTLASS"] : "/home/wjvermeu/cutlass/build/tools/profiler/cutlass_profiler"

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
                    `awk -F',' '{print $NF}'`,
                    `sed 's/"//g'`,
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
                    `awk -F',' '{print $NF}'`,
                    `sed 's/"//g'`,
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
                --kernels=cutlass_simt_sgemm_128x128_8x2_nn_align1 
                --warmup-iterations=0 --profiling-iterations=10 --verification-enabled=false`

            @show cmd

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

            @show result
        end

        # Write results to CSV
        open(file, "a") do io
            write(io, string(N), ",\"", result, "\"\n")
        end
    end

    nothing
end

isinteractive() || main()