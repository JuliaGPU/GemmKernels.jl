using GemmKernels
using Test

import CUDA
import InteractiveUtils

InteractiveUtils.versioninfo()
CUDA.versioninfo()
@info "" CUDA.capability.(CUDA.devices())

macro test_if(label, expr)
    return quote
        if isempty(ARGS) || $(label) in ARGS
            $(esc(expr))
        else
            nothing
        end
    end
end

CUDA.allowscalar(false)

@testset "GemmKernels.jl" begin
    include("tiling.jl")
    include("matmul.jl")
    include("blas.jl")
end
