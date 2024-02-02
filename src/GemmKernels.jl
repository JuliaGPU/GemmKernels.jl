module GemmKernels

using CUDA
using LinearAlgebra

# utilities
include("vectorisation.jl")
include("bitarrayindex.jl")
include("tiling.jl")
include("array.jl")
include("utils.jl")

# framework
include("mma.jl")
include("layout.jl")
include("operator.jl")
include("config.jl")
include("epilogue.jl")
include("kernel.jl")
include("transform.jl")

# instantiations
include("matmul.jl")

end
