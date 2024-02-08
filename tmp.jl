include("configs/configs.jl")

using CUDA
using Test
using GemmKernels

function test()
    for cf in get_configs()
        c_h, a, b, c, d = generate_inputs(cf)
        run_gemm(cf, a, b, c, d)
        @test verify(cf, c_h, d)
    end
end
