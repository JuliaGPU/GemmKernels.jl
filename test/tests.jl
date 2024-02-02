using Distributed, XUnit, Dates
using CUDA, GemmKernels
CUDA.allowscalar(false)

if myid() == 1
    using InteractiveUtils
    @info "Julia details:\n" * sprint(io->InteractiveUtils.versioninfo(io))
    @info "CUDA details:\n" * sprint(io->CUDA.versioninfo(io))
end

t0 = now()
try
    @testset runner=DistributedTestRunner() "GemmKernels.jl" begin
        include("tiling.jl")
        # include("matmul.jl")
        # include("blas.jl")
        # include("examples.jl")
        # include("bitarrayindex.jl")
    end
finally
    if myid() == 1
        t1 = now()
        elapsed = canonicalize(Dates.CompoundPeriod(t1-t0))
        println("Testing finished in $elapsed")
    end
end

