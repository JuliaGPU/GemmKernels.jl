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
              "gemmkernels_complex" => "Our implementation (complex)",
              "gemmkernels_dual" => "Our implementation (dual)",
              "cutlass" => "CUTLASS (complex)",
              "cudajl_complex" => "CUDA.jl (complex)",
             )

seriesnumbers = Dict(
              "gemmkernels_complex" => 1
              "gemmkernels_dual" => 2,
              "cutlass" => 3,
              "cudajl_complex" => 4,
             )

for file in readdir()
    if isfile(file) && splitext(basename(file))[2] == ".csv"
        filename = splitext(basename(file))[1]

        implementation = filename # e.g. gemmkernels_complex

        label = labels[implementation]
        seriesnr = seriesnumbers[implementation]
        markershape = :circle

        df = DataFrame(CSV.File(file))

        N = df[!, :N]
        runtime_arr = convert_to_array.(df[!, :runtime]) .* 1e3 # in ps
        flops_factor = flops_factors[implementation]
        tflops = [flops_factor * N[i] ^ 3 ./ runtime_arr[i] for i = 1 : length(N)]

        plot!(N, mean.(tflops), seriescolor=seriesnr, label=label, xscale=:log2, markershape=markershape, ribbon=std.(tflops), fillalpha=.5)
    end
end

title!("Performance of complex/dual GEMMs")
xlabel!("N")
ylabel!("TFLOPS")
savefig("plot.pdf")
