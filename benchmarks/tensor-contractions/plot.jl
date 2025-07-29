using CSV
using DataFrames
using Plots.PlotMeasures
using Statistics
using StatsPlots

function getTFLOPS(arr, N; no_runs=10, debug=false)
    arr = split(arr, ',')
    arr = parse.(Float64, arr)

    new_arr = Vector{Float64}(undef, no_runs)
    no_kernels = Int(length(arr) / no_runs)
    for i in 1 : no_runs
        new_arr[i] = sum(arr[(i-1) * no_kernels + 1 : i * no_kernels])
    end

    arr = mean(new_arr)

    # return arr / 1e3 (in μs)
    return 2 * (2^N)^3 / (arr * 1e3) # in TFLOPS
end

function main()
    df_gemmkernels = DataFrame(CSV.File("gettGemmKernels.csv"))
    df_cutensor_gett = DataFrame(CSV.File("gettCuTensor.csv"))
    df_cutensor_tgett = DataFrame(CSV.File("tgettCuTensor.csv"))
    df_cutensor_ttgt = DataFrame(CSV.File("ttgtCuTensor.csv"))

    GK = getTFLOPS.(df_gemmkernels[!, :result], 11)
    CT_GETT = getTFLOPS.(df_cutensor_gett[!, :result], 11)
    CT_TGETT = getTFLOPS.(df_cutensor_tgett[!, :result], 11)
    CT_TTGT = getTFLOPS.(df_cutensor_ttgt[!, :result], 11)

    ticklabel = df_gemmkernels[!, :contraction]

    # These lines can be used to sort the data by ascending TFLOPS of GemmKernels.jl
    new_sort = sortperm(GK)
    GK = GK[new_sort]
    CT_GETT = CT_GETT[new_sort]
    CT_TGETT = CT_TGETT[new_sort]
    CT_TTGT = CT_TTGT[new_sort]
    ticklabel = df_gemmkernels[!, :contraction][new_sort]

    p = plot(
        [GK CT_GETT CT_TGETT], 
        size=(800, 400), 
        marker=[:c :d :ut], 
        label=["GemmKernels.jl" "cuTENSOR GETT" "cuTENSOR TGETT" "cuTENSOR TTGT"],
        thickness=10,
        linewidth=2,
        xticks=(1 : length(GK), ticklabel),
        xrotation=90,
        bottom_margin=80px,
        yticks=(0 : 5 : 45),
        ylims=(0, 45)
    )

    savefig(p, "gett.pdf")
end

isinteractive() || main()