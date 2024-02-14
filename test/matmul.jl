using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra
using Random

include("../configs/configs.jl")

@testset "Matrix multiplication" begin
    @testcase "$( cf.name )" for cf in get_configs()
        try
            reference_mul!, a, b, c, d = generate_inputs(cf)
            rand!(a)
            rand!(b)
            rand!(c)
            d .= 0

            run_gemm(cf, a, b, c, d)
            reference_mul!(c, a, b)
            @test verify(cf, c, d)
        catch err
            # Count tests with config errors as "broken".
            if isa(err, GemmKernels.ConfigError)
                @test true skip=true
            else
                rethrow()
            end
        end
    end
end
