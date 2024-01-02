using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra

include("../configs/configs.jl")

@testset "Matrix multiplication" begin
    @testcase "$( cf.name )" for cf in get_configs()
        try
            c_h, a, b, c, d = generate_inputs(cf)
            run_gemm(cf, a, b, c, d)
            @test verify(cf, c_h, d)
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
