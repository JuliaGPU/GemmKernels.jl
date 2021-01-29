using GemmKernels

using Pkg
Pkg.add(PackageSpec(name="XUnit", rev="9b756fcda72d813dbf017f8400d7c55251ef7d1b"))

using XUnit

import CUDA
import InteractiveUtils

@info "Julia details\n\n" * sprint(io->InteractiveUtils.versioninfo(io))
@info "CUDA details\n\n" * sprint(io->CUDA.versioninfo(io))

CUDA.allowscalar(false)

@testset runner=ParallelTestRunner() "GemmKernels.jl" begin
    include("tiling.jl")
    include("matmul.jl")
    include("blas.jl")
end
