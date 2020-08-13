using CUDA
using GemmKernels
using LinearAlgebra
using Test

function run_gemm()
    M = 2048;
    N = 2048;
    K = 2048;

    transpose_a = false;
    transpose_b = false;

    a_h = rand(Float16, M);
    b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
    c_h = rand(Float32, (M, N))

    # Transpose input if necessary
    a_h = transpose_a ? transpose(a_h) : a_h
    b_h = transpose_b ? transpose(b_h) : b_h

    a   = CuArray(a_h)
    b   = CuArray(b_h)
    c   = CuArray(c_h)
    d   = similar(c)

    conf = GemmKernels.get_config(
        gemm_shape = (M = M, N = N, K = K),
        operator = Operator.WMMAOp{16, 16, 16},
        global_a_layout = transpose_a ? Layout.AlignedRowMajor{Float16} : Layout.Diagonal{Float16},
        global_b_layout = transpose_b ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},

        global_c_layout = Layout.AlignedColMajor{Float32},
        global_d_layout = Layout.AlignedColMajor{Float32},

        shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},
        shared_b_layout = transpose_b ? Layout.Padded{Layout.AlignedRowMajor{Float16}, 8} : Layout.Padded{Layout.AlignedColMajor{Float16}, 8},

        is_a_col_major = !transpose_a,
        is_b_col_major = !transpose_b,
                            )

    GemmKernels.matmul(a, b, c, d, conf)

    # Transpose outputs, if necessary
    new_a_h = transpose_a ? transpose(a_h) : a_h
    new_b_h = transpose_b ? transpose(b_h) : b_h

    # TODO: remove; debug only
    got = Array(d);
    expected = Float32.(Diagonal(new_a_h)) * Float32.(new_b_h) + c_h

    xr = 1 : 16
    yr = 1 : 16

    display("text/plain", got[xr, yr])
    println()
    display("text/plain", expected[xr, yr])
    println()

    @test all(isapprox.(Float32.(Diagonal(new_a_h)) * Float32.(new_b_h) + c_h, Array(d); rtol = sqrt(eps(Float16))))
end
