using CSV
using DataFrames
using Plots
using Statistics

pyplot()

function convert_to_array(str)
    els = split(str, ',')

    if "n/a" in els
        @warn "Benchmark contains 'n/a' values. These will be ignored."
        filter!(el -> el != "n/a", els)
    end

    return parse.(Float64, els)
end

labels = Dict(
              "gemmkernels" => "Our implementation",
              "cublas" => "cuBLAS",
              "cutlass" => "CUTLASS",
              "cudajl" => "CUDA.jl",
             )

seriesnumbers = Dict(
              "gemmkernels" => 1,
              "cublas" => 2,
              "cutlass" => 3,
              "cudajl" => 4,
             )

markershapes = Dict(
                    "nn" => :circle,
                    "tt" => :cross,
                    "tn" => :diamond,
                    "nt" => :dtriangle,
                   )

for file in readdir()
    if isfile(file) && splitext(basename(file))[2] == ".csv"
        filename = splitext(basename(file))[1]

        implementation = split(filename, "-")[1] # e.g. gemmkernels
        layout = split(filename, "-")[2] # e.g. nn

        label = layout == "nn" ? labels[implementation] : ""
        seriesnr = seriesnumbers[implementation]
        markershape = markershapes[layout]

        df = DataFrame(CSV.File(file))

        N = df[!, :N]
        runtime_arr = convert_to_array.(df[!, :runtime]) .* 1e3 # in ps
        flops_factor = 2
        tflops = [flops_factor * N[i] ^ 3 ./ runtime_arr[i] for i = 1 : length(N)]

        plot!(N, mean.(tflops), seriescolor=seriesnr, label=label, xscale=:log2, markershape=markershape, ribbon=std.(tflops), fillalpha=.5)
    end
end

title!("Performance of mixed-precision GEMM")
xlabel!("N")
ylabel!("TFLOPS")
savefig("plot.pdf")
