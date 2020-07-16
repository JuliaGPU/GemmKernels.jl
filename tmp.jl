using CUDA
using GemmKernels
using Test

function benc()
    M = 128
    N = 128
    K = 128

    a_h = transpose(rand(Float16, (M, K)) / sqrt(Float16(K)))
    b_h = transpose(rand(Float16, (K, N)) / sqrt(Float16(K)))
    c_h = rand(Float32, (M, N))

    a   = CuArray(a_h)
    b   = CuArray(b_h)
    c   = CuArray(c_h)
    d   = similar(c)

    conf = GemmKernels.get_config(
                                gemm_shape = (M = M, N = N, K = K),
                                operator = Operator.WMMAOp{16, 16, 16},

                                global_a_layout = Layout.AlignedRowMajor{Float16},
                                global_b_layout = Layout.AlignedRowMajor{Float16},

                                global_c_layout = Layout.AlignedColMajor{Float32},
                                global_d_layout = Layout.AlignedColMajor{Float32},
                                )

    GemmKernels.matmul(a, b, c, d, conf)

    @test all(isapprox.(Float32.(transpose(a_h)) * Float32.(transpose(b_h)) + c_h, Array(d); rtol = sqrt(eps(Float16))))
end
