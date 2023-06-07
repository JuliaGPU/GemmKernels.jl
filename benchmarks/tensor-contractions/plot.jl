using CSV
using DataFrames
using Plots.PlotMeasures
using Statistics
using StatsPlots

function getMean(arr)
    arr = split(arr, ',')
    arr = parse.(Float64, arr)
    arr = mean(arr)
    return arr / 1e3
end

function main(file)
    df = DataFrame(CSV.File(file))

    GKIdx = findall(x -> x == "GK", df[!, :impl])
    CTIdx = findall(x -> x == "CT", df[!, :impl])

    GK = getMean.(df[!, :result][GKIdx])
    CT = getMean.(df[!, :result][CTIdx])

    ticklabel = df[!, :contraction][GKIdx]

    p = groupedbar(
        [GK CT], 
        group=repeat(["0 - GemmKernels.jl", "1 - CUTENSOR"], inner=length(GKIdx)),
        xticks=(1 : length(GKIdx), ticklabel),
        bar_position = :dodge, 
        bar_width=0.7, 
        ylabel="GPU time (μs)",
        size=(1200,500), 
        xrotation=90, 
        bottom_margin=100px, 
        left_margin=30px,
        xlims=(0,49), 
        ylims=(0,5000)
    )

    savefig(p, file * ".pdf")
end

isinteractive() || main(ARGS[1])