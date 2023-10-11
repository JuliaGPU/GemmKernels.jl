using CUDA, GemmKernels
using LinearAlgebra

function main()
    M = K = N = 4096

    A = CUDA.rand(Float32, M, K)
    B = CUDA.rand(Float32, K, N)
    C = CUDA.zeros(Float32, M, N)

    # pow2-sized, 128-bit aligned inputs, so we can use aligned layouts.
    # we don't have transposed inputs, so everything is column major.
    @assert stride(A, 2) % 16 == 0
    global_a_layout = Layout.UnsafeAlignedColMajor{eltype(A)}
    @assert stride(B, 2) % 16 == 0
    global_b_layout = Layout.UnsafeAlignedColMajor{eltype(B)}
    # we want to do a simple C = A * B, so no need to load C first.
    global_c_layout = Layout.Zero{eltype(C)}
    @assert stride(C, 2) % 16 == 0
    global_d_layout = Layout.UnsafeAlignedColMajor{eltype(C)}

    # shared layouts are similar.
    # the frequently-accessed a/b shmems are padded to avoid bank conflicts.
    shared_a_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{eltype(A)}, 8}
    shared_b_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{eltype(B)}, 8}
    shared_c_layout = shared_d_layout = Layout.UnsafeAlignedColMajor{eltype(C)}

    # we use the FPU operator, for compatibility
    compute_type = promote_type(eltype(A), eltype(B))
    operator = Operator.FPUOp{8, 8, 1, compute_type, eltype(C)}

    # we use the single-stage kernel, for simplicity
    kernel = Kernel.matmul_singlestage

    # the block shape can be determined by a heuristic, but we'll just hard code it here.
    # note that it exactly covers the inputs, so that we can use the unsafe layouts.
    block_shape = (M = 128, N = 128, K = 32)
    @assert M % block_shape.M == 0
    @assert N % block_shape.N == 0
    @assert K % block_shape.K == 0

    conf = GemmKernels.get_config(;
        gemm_shape = (; M, N, K), block_shape,
        operator = Operator.FPUOp{8, 8, 1, 4, 8, 1, compute_type, eltype(C)},

        global_a_layout, global_b_layout, global_c_layout, global_d_layout,
        shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout,

        is_a_col_major = true,
        is_b_col_major = true
    )

    GemmKernels.matmul(conf, A, B, C, C; kernel)

    @assert Array(C) ≈ Array(A) * Array(B)
end

isinteractive() || main()
