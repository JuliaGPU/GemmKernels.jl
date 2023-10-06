# List of configurations to use for testing and benchmarking.

using GemmKernels
using LinearAlgebra
using ForwardDiff

GEMM_SIZES = [128, 256, 2048]

struct Configuration
    name           # Human-readable name of the configuration.
    config         # GemmKernels.Config instance to use.
    alpha          # Value of alpha
    beta           # Value of beta
    a_type         # Type of the A matrix on host and in GMEM.
    b_type         # Type of the B matrix on host and in GMEM.
    c_type         # Type of the C matrix on host and in GMEM.
    d_type         # Type of the D matrix on host and in GMEM.
    transpose_a    # Whether or not A is transposed
    transpose_b    # Whether or not B is transposed
    calc_reference # mul!-like function to calculate the reference
    epilogue       # The epilogue to use.
    verify         # Verify function to use.
    kernel         # The kernel function to use.
end

function get_custom_mul!(element_update)
    (C, A, B, alpha, beta) -> begin
        M = size(C, 1)
        N = size(C, 2)
        K = size(A, 2)

        for i in 1:M
            for j in 1:N
                res = beta * C[i, j]

                for k in 1:K
                    res = alpha * element_update(A[i, k], B[k, j], res)
                end

                C[i, j] = res
            end
        end
    end
end

# Generate input matrices.
function generate_inputs(cf::Configuration)
    M = cf.config.matmul_shape.M
    N = cf.config.matmul_shape.N
    K = cf.config.matmul_shape.K

    a_h = rand(cf.a_type, (M, K))
    b_h = rand(cf.b_type, (K, N))
    c_h = rand(cf.c_type, (M, N))

    a_h = cf.transpose_a ? transpose(a_h) : a_h
    b_h = cf.transpose_b ? transpose(b_h) : b_h

    a = CuArray(a_h)
    b = CuArray(b_h)
    c = CuArray(c_h)
    d = similar(c)

    new_a_h = cf.transpose_a ? transpose(a_h) : a_h
    new_b_h = cf.transpose_b ? transpose(b_h) : b_h

    (cf.calc_reference)(c_h, new_a_h, new_b_h, cf.alpha, cf.beta)
    c_h, a, b, c, d
end

# Get the kernel info.
function get_info(cf::Configuration, a, b, c, d)
    alpha = cf.alpha
    beta = cf.beta
    a_transf = (alpha == one(alpha)) ? Transform.Elementwise(identity) : Transform.Elementwise(x -> x * alpha)
    c_transf = (beta == one(beta)) ? Transform.Elementwise(identity) : Transform.Elementwise(x -> x * beta)

    info = GemmKernels.Information()
    GemmKernels.matmul(cf.config, a, b, c, d;
                       transform_shared_to_regs_a = a_transf,
                       transform_shared_to_regs_c = c_transf,
                       epilogue = cf.epilogue,
                       kernel = cf.kernel,
                       info = info)
    info
end

# Run the GEMM.
function run_gemm(cf::Configuration, a, b, c, d)
    alpha = cf.alpha
    beta = cf.beta
    a_transf = (alpha == one(alpha)) ? Transform.Elementwise(identity) : Transform.Elementwise(x -> x * alpha)
    c_transf = (beta == one(beta)) ? Transform.Elementwise(identity) : Transform.Elementwise(x -> x * beta)

    GemmKernels.matmul(cf.config, a, b, c, d;
                       transform_shared_to_regs_a = a_transf,
                       transform_shared_to_regs_c = c_transf,
                       epilogue = cf.epilogue,
                       kernel = cf.kernel)
end

# Verify results.
function verify(cf::Configuration, c_h, d)
    cf.verify(c_h, d)
end

function verify_default(c_h, d)
    isapprox(c_h, Array(d))
end

function verify_bias(c_h, d, bias)
    c_h .+ Array(bias) ≈ Array(d)
end

function verify_dual(c_h, d)
    c_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, c_h)
    d_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, Array(d))
    isapprox(c_dual, d_dual)
end

function get_configs()
    rv = []

    # FPU Op
    for (A_type, B_type, CD_type) in [
            (Float16, Float16, Float32),
            (Float32, Float32, Float32),
            (Float32, Float32, Float64),
            (Float64, Float64, Float64),
            (Int16, Int16, Int16),
            (Int32, Int32, Int32),
            (Int64, Int64, Int64)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [(8, 16, 2)],
        N in GEMM_SIZES

        # XXX: Should we do non-square matrices as well?
        M = K = N

        name = "FPU GEMM $(A_type)*$(B_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"
        compute_type = promote_type(A_type, B_type)

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        block_shape = (M = 64, N = 64, K = 32),
                                        operator = Operator.FPUOp{OP_M, OP_N, OP_K, compute_type, CD_type},
                                        global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{A_type} : Layout.UnsafeAlignedColMajor{A_type},
                                        global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{B_type} : Layout.UnsafeAlignedColMajor{B_type},

                                        global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                        global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        push!(rv, Configuration(name,
                                conf,
                                convert(compute_type, 2),
                                convert(compute_type, 3),
                                A_type,
                                B_type,
                                CD_type,
                                CD_type,
                                transpose_a,
                                transpose_b,
                                mul!,
                                Epilogue.Default(),
                                verify_default,
                                Kernel.matmul_pipelined))
    end

    # FPU Op shapes
    for (A_type, B_type, CD_type) in [
            (Float32, Float32, Float32)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [
            (4, 8, 1),
            (8, 8, 1),
            (4, 16, 1),
            (4, 8, 2),
            (8, 16, 2)],
        N in [128]

        # We'll only test square matrices.
        M = K = N

        name = "FPU GEMM $(A_type)*$(B_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"
        compute_type = promote_type(A_type, B_type)

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        block_shape = (M = 128, N = 64, K = 32),
                                        operator = Operator.FPUOp{OP_M, OP_N, OP_K, compute_type, CD_type},
                                        global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{A_type} : Layout.UnsafeAlignedColMajor{A_type},
                                        global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{B_type} : Layout.UnsafeAlignedColMajor{B_type},

                                        global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                        global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        push!(rv, Configuration(name,
                                conf,
                                convert(compute_type, 2),
                                convert(compute_type, 3),
                                A_type,
                                B_type,
                                CD_type,
                                CD_type,
                                transpose_a,
                                transpose_b,
                                mul!,
                                Epilogue.Default(),
                                verify_default,
                                Kernel.matmul_pipelined))
    end

    # Tropical GEMM
    for (A_type, B_type, CD_type, min_dimension) in [
            (Float32, Float32, Float32, 128)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [(8, 16, 2)],
        (M, N, K) in min_dimension .* [
            [1, 1, 1],
            [2, 2, 1],
            [1, 1, 2],
            [2, 2, 2]]

        name = "Tropical GEMM $(A_type)*$(B_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"
        compute_type = promote_type(A_type, B_type)

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        block_shape = (M = 64, N = 64, K = 32),
                                        operator = Operator.TropicalFPUOp{OP_M, OP_N, OP_K, compute_type, CD_type},
                                        global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{A_type} : Layout.UnsafeAlignedColMajor{A_type},
                                        global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{B_type} : Layout.UnsafeAlignedColMajor{B_type},

                                        global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                        global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        push!(rv, Configuration(name,
                                conf,
                                convert(compute_type, 1),
                                convert(compute_type, 1),
                                A_type,
                                B_type,
                                CD_type,
                                CD_type,
                                transpose_a,
                                transpose_b,
                                get_custom_mul!((a, b, c) -> max(a + b, c)),
                                Epilogue.Default(),
                                verify_default,
                                Kernel.matmul_pipelined))
    end

    # WMMA GEMM
    for (AB_type, CD_type, min_dimension) in [
        (Float16, Float16, 256),
        (Float16, Float32, 128)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [(16, 16, 16)],
        (M, N, K) in vcat(min_dimension .* [
            [1, 1, 1],
            [2, 2, 1],
            [1, 1, 2],
            [2, 2, 2]], [[2048, 2048, 2048]])

        name = "WMMA GEMM $(AB_type)*$(AB_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        operator = Operator.WMMAOp{OP_M, OP_N, OP_K, AB_type, CD_type},
                                        global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},
                                        global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},

                                        global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                        global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        push!(rv, Configuration(name,
                                conf,
                                convert(AB_type, 2),
                                convert(AB_type, 3),
                                AB_type,
                                AB_type,
                                CD_type,
                                CD_type,
                                transpose_a,
                                transpose_b,
                                mul!,
                                Epilogue.Default(),
                                verify_default,
                                Kernel.matmul_pipelined))
    end

    # WMMA GEMM + bias
    for (AB_type, CD_type, min_dimension) in [
        (Float16, Float32, 128)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [(16, 16, 16)],
        (M, N, K) in vcat(min_dimension .* [
            [1, 1, 1],
            [2, 2, 2]], [[4096, 4096, 4096]])

        name = "WMMA GEMM+bias $(AB_type)*$(AB_type)+$(CD_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"

        # Bias vector: this vector contains 1 element per column of the result matrix.
        # This bias element is added to all elements in one column of the D matrix.
        # D is a M x N matrix, so the bias is an N-element vector.
        bias = CuArray(rand(CD_type, (1, N)))

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        operator = Operator.WMMAOp{OP_M, OP_N, OP_K, AB_type, CD_type},
                                        global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},
                                        global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},

                                        global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                        global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        push!(rv, Configuration(name,
                                conf,
                                convert(AB_type, 1),
                                convert(AB_type, 1),
                                AB_type,
                                AB_type,
                                CD_type,
                                CD_type,
                                transpose_a,
                                transpose_b,
                                mul!,
                                Epilogue.Bias(pointer(bias)),
                                (c_h, d) -> verify_bias(c_h, d, bias),
                                Kernel.matmul_pipelined))
    end

    # WMMA Diagonal GEMM
    for (AB_type, CD_type, min_dimension) in [
        (Float16, Float32, 128)],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [(16, 16, 16)],
        (M, N, K) in vcat(min_dimension .* [
            [1, 1, 1],
            [2, 2, 2]], [[4096, 4096, 4096]])

        @assert M == K "Diagonal only supports square A matrix (A == M)"

        transpose_a = false
        name = "WMMA diagonal GEMM diag($(AB_type))*$(AB_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"

        # NOTE: the testing framework will generate an MxK A matrix,
        # but we only use the first M elements in the first column,
        # and interpret that as the diagonal elements.

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        operator = Operator.WMMAOp{OP_M, OP_N, OP_K, AB_type, CD_type},
                                        global_a_layout = Layout.Diagonal{AB_type},
                                        global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},

                                        global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                        global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        shared_a_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{Float16}, 8},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )


        push!(rv, Configuration(name,
                                conf,
                                convert(AB_type, 1),
                                convert(AB_type, 1),
                                AB_type,
                                AB_type,
                                CD_type,
                                CD_type,
                                transpose_a,
                                transpose_b,
                                (C, A, B, alpha, beta) -> mul!(C, Diagonal(A[1:M,1]), B, true, true),
                                Epilogue.Default(),
                                verify_default,
                                Kernel.matmul_singlestage))
    end

    # WMMA Complex GEMM
    for (AB_type, CD_type) in [(Float16, Float32)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [(16, 16, 16)],
        (M, N, K) in [
            (128, 128, 128),
            (256, 256, 256),
            (2048, 2048, 2048)]
        name = "WMMA Complex GEMM $(AB_type)*$(AB_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        operator = Operator.WMMAComplexOp{OP_M, OP_N, OP_K},

                                        global_a_layout = transpose_a ? Layout.InterleavedRowMajor{Float16} : Layout.InterleavedColMajor{Float16},
                                        global_b_layout = transpose_b ? Layout.InterleavedRowMajor{Float16} : Layout.InterleavedColMajor{Float16},
                                        global_c_layout = Layout.InterleavedColMajor{Float32},
                                        global_d_layout = Layout.InterleavedColMajor{Float32},

                                        shared_a_layout = transpose_a ? Layout.Padded{Layout.SplitRowMajor{Float16}, 8} : Layout.Padded{Layout.SplitColMajor{Float16}, 8},
                                        shared_b_layout = transpose_b ? Layout.Padded{Layout.SplitRowMajor{Float16}, 8} : Layout.Padded{Layout.SplitColMajor{Float16}, 8},
                                        shared_c_layout = Layout.SplitColMajor{Float32},
                                        shared_d_layout = Layout.SplitColMajor{Float32},

                                        warps_per_block = 8,

                                        compute_warp = (M = 16, N = 32, K = 16),

                                        block_shape = (M = 64, N = 64, K = 32),

                                        mem_a_warp = transpose_a ? (M = 4, K = 32) : (M = 64, K = 2),
                                        mem_b_warp = transpose_b ? (K = 2, N = 64) : (K = 32, N = 4),
                                        mem_cd_warp = (M = 64, N = 1),

                                        mem_a_thread = transpose_a ? (M = 1, K = 4) : (M = 4, K = 1),
                                        mem_b_thread = transpose_b ? (K = 1, N = 4) : (K = 4, N = 1),
                                        mem_cd_thread = (M = 2, N = 1),

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b
                                        )

        push!(rv, Configuration(name,
                                conf,
                                convert(Complex{AB_type}, 1),
                                convert(Complex{AB_type}, 1),
                                Complex{AB_type},
                                Complex{AB_type},
                                Complex{CD_type},
                                Complex{CD_type},
                                transpose_a,
                                transpose_b,
                                mul!,
                                Epilogue.Default(),
                                verify_default,
                                Kernel.matmul_pipelined))
    end

    # WMMA Dual GEMM
    for (AB_type, CD_type) in [(Float16, Float32)],
        transpose_a = [false],
        transpose_b = [false],
        (OP_M, OP_N, OP_K) in [(16, 16, 16)],
        (M, N, K) in [
            (128, 128, 128),
            (256, 256, 256),
            (2048, 2048, 2048)]
        name = "WMMA Dual GEMM d$(AB_type)*d$(AB_type)=d$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        operator = Operator.WMMADualOp{OP_M, OP_N, OP_K},

                                        global_a_layout = Layout.InterleavedColMajor{Float16},
                                        global_b_layout = Layout.InterleavedColMajor{Float16},
                                        global_c_layout = Layout.InterleavedColMajor{Float32},
                                        global_d_layout = Layout.InterleavedColMajor{Float32},

                                        shared_a_layout = Layout.Padded{Layout.SplitColMajor{Float16}, 8},
                                        shared_b_layout = Layout.Padded{Layout.SplitColMajor{Float16}, 8},
                                        shared_c_layout = Layout.SplitColMajor{Float32},
                                        shared_d_layout = Layout.SplitColMajor{Float32},

                                        warps_per_block = 8,

                                        compute_warp = (M = 16, N = 32, K = 16),

                                        block_shape = (M = 64, N = 64, K = 32),

                                        mem_a_warp = (M = 64, K = 2),
                                        mem_b_warp = (K = 32, N = 4),
                                        mem_cd_warp = (M = 64, N = 1),

                                        mem_a_thread = (M = 4, K = 1),
                                        mem_b_thread = (K = 4, N = 1),
                                        mem_cd_thread = (M = 2, N = 1)
                                        )

        dual_conv(M) = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, M)

        push!(rv, Configuration(name,
                                conf,
                                convert(Complex{AB_type}, 1),
                                convert(Complex{AB_type}, 1),
                                Complex{AB_type},
                                Complex{AB_type},
                                Complex{CD_type},
                                Complex{CD_type},
                                transpose_a,
                                transpose_b,
                                (C, A, B, alpha, beta) -> mul!(dual_conv(C), dual_conv(Complex{Float32}.(A)), dual_conv(Complex{Float32}.(B)), true, true),
                                Epilogue.Default(),
                                verify_dual,
                                Kernel.matmul_pipelined))
    end

    rv
end
