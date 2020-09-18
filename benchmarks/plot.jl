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

titles = Dict(
              "wmma" => "Performance of mixed-precision GEMM",
              "complex-wmma" => "Performance of mixed-precision complex GEMM",
              "dual-wmma" => "Performance of mixed-precision dual GEMM",
             )

flops_factors = Dict(
                     "wmma" => 2,
                     "complex-wmma" => 8,
                     "dual-wmma" => 6,
                    )

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
                    "tt" => :dtriangle,
                   )

dir = ARGS[1] # e.g. wmma
cd(dir)

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
        tflops = [flops_factors[dir] * N[i] ^ 3 ./ runtime_arr[i] for i = 1 : length(N)]

        plot!(N, mean.(tflops), seriescolor=seriesnr, label=label, xscale=:log2, markershape=markershape, ribbon=std.(tflops), fillalpha=.5)
    end
end

title!(titles[dir])
xlabel!("N")
ylabel!("TFLOPS")
savefig("plot.pdf")
