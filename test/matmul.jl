using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra
using Random

include("../configs/configs.jl")

@testset "Matrix multiplication" begin
    @testcase "$( problem )" for (problem, params...) in get_configs()
        f = Base.retry(delays=ExponentialBackOff(first_delay=1, max_delay=120, n=5), check=(s,err) -> isa(err, CUDA.OutOfGPUMemoryError)) do
            try
                data = allocate_data(problem)
                initialize_data(problem, data...)

                reference_result = calculate_reference(problem, data...)
                result = execute(problem, data...; params...)

                @test verify(problem, reference_result, result)
            catch err
                # Ignore Config Errors.
                isa(err, GemmKernels.ConfigError) || rethrow()
            end
        end

        f()
    end
end
