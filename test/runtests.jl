using GemmKernels
using Test

@testset "GemmKernels.jl" begin
    #= include("tiling.jl") =#
    #= include("matmul.jl") =#
    include("blas.jl")
end
