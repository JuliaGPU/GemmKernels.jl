using GemmKernels
using Test

macro test_if(label, expr)
    return quote
        if isempty(ARGS) || $(label) in ARGS
            $(esc(expr))
        else
            nothing
        end
    end
end

@testset "GemmKernels.jl" begin
    include("tiling.jl")
    include("matmul.jl")
    include("blas.jl")
end
