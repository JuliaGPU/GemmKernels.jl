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
             )

labels = Dict(
              "cublas-nn" => "cuBLAS (NN)",
              "cublas-tt" => "cuBLAS (TT)",
              "cublas-tn" => "cuBLAS (TN)",
              "cublas-nt" => "cuBLAS (NT)",
              "gemmkernels-nn" => "Our implementation (NN)",
              "gemmkernels-tt" => "Our implementation (TT)",
              "gemmkernels-tn" => "Our implementation (TN)",
              "gemmkernels-nt" => "Our implementation (NT)",
             )

dir = ARGS[1]
cd(dir)

for file in readdir()
    if isfile(file) && splitext(basename(file))[2] == ".csv"
        label = labels[splitext(basename(file))[1]]

        df = DataFrame(CSV.File(file))

        N = df[!, :N]
        runtime_arr = convert_to_array.(df[!, :runtime]) .* 1e3 # in ps
        tflops = [2 * N[i] ^ 3 ./ runtime_arr[i] for i = 1 : length(N)]

        plot!(N, mean.(tflops), label=label, xscale=:log2, markershape=:circle, ribbon=std.(tflops), fillalpha=.5)
    end
end

title!(titles[dir])
xlabel!("N")
ylabel!("TFLOPS")
savefig("plot.pdf")
