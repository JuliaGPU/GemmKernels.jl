using CSV
using DataFrames
using Plots.PlotMeasures
using Statistics
using StatsPlots

function getTFLOPS(arr, N)
    arr = split(arr, ',')
    arr = parse.(Float64, arr)
    arr = mean(arr)
    return 2 * (2^N)^3 / (arr * 1e3)
end

function main()
    df_gemmkernels = DataFrame(CSV.File("gemmkernels.csv"))
    df_cublas = DataFrame(CSV.File("cublas.csv"))
    # df_cutlass = DataFrame(CSV.File("cutlass.csv"))

    N = df_gemmkernels[!, :N]
    gemmkernels = getTFLOPS.(df_gemmkernels[!, :results], N)
    cublas = getTFLOPS.(df_cublas[!, :results], N)
    # cutlass = getTFLOPS.(df_cutlass[!, :results], N)

    p = plot(
        [gemmkernels cublas],
        # [gemmkernels cublas cutlass],
        marker=:circle,
        xticks=(1 : 8, N),
        xlabel="N",
        ylabel="TFLOPS",
        # label=["GemmKernels.jl" "CUBLAS"],
        label=["GemmKernels.jl" "CUBLAS" "CUTLASS"],
        xlims=(0, 9),
        ylims=(0, 12),
    )

    savefig(p, "gemm.png")
end

isinteractive() || main(ARGS[1])