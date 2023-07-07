module GemmKernels

using CUDA
using LinearAlgebra

# utilities
include("tiling.jl")
include("array.jl")
include("utils.jl")

# framework
include("config.jl")
include("epilogue.jl")
include("kernel.jl")
include("layout.jl")
include("operator.jl")
include("transform.jl")

# instantiations
include("matmul.jl")

end
