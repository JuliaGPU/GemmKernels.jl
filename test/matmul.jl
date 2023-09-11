using CUDA
using ForwardDiff
using GemmKernels
import Octavian, LinearAlgebra

# for large, non-BLAS-compatible matrices, use Octavian.
matmul!(C, A, B, alpha=true, beta=false) = LinearAlgebra.mul!(C, A, B, alpha, beta)
function matmul!(C::Array,
                 A::Union{Array, LinearAlgebra.Transpose{<:Any, <:Array},
                                 LinearAlgebra.Adjoint{<:Any, <:Array}},
                 B::Union{Array, LinearAlgebra.Transpose{<:Any, <:Array},
                                 LinearAlgebra.Adjoint{<:Any, <:Array}},
                 alpha::Bool=true, beta::Bool=false)
    supported = eltype(C) <: LinearAlgebra.BlasFloat &&
                eltype(A) <: LinearAlgebra.BlasFloat &&
                eltype(B) <: LinearAlgebra.BlasFloat &&
                eltype(C) == eltype(A) == eltype(B)
    if !supported && (sizeof(C) > 2^20 || sizeof(A) > 2^20 || sizeof(B) > 2^20)
        Octavian.matmul!(C, A, B, alpha, beta)
    else
        LinearAlgebra.mul!(C, A, B, alpha, beta)
    end
end

################################################################################

@testset "Matmul API" begin
    @testset "FPU GEMM $(A_type)*$(B_type)=$(CD_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))" for
        (A_type, B_type, CD_type, min_dimension) in [
            (Float16, Float16, Float32, 128), (Float32, Float32, Float32, 128), (Float32, Float32, Float64, 128), (Float64, Float64, Float64, 128),
            (Int16, Int16, Int16, 128), (Int32, Int32, Int32, 128), (Int64, Int64, Int64, 128),
        ],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [(8, 16, 2)]

        compute_type = promote_type(A_type, B_type)

        @testcase "(M = $M, N = $N, K = $K)" for (M, N, K) in vcat(min_dimension.*[[1,1,1], [2, 2, 1], [1, 1, 2], [2, 2, 2]], [[2048, 2048, 2048]])
            alpha = convert(compute_type, 2)
            beta  = convert(CD_type, 3)

            if A_type <: Integer
                a_h = rand(A_type, (M, K))
                b_h = rand(B_type, (K, N))
            else
                a_h = rand(A_type, (M, K)) / sqrt(A_type(K))
                b_h = rand(B_type, (K, N)) / sqrt(B_type(K))
            end
            c_h = rand(CD_type, (M, N))

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a   = CuArray(a_h)
            b   = CuArray(b_h)
            c   = CuArray(c_h)
            d   = similar(c)

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

            GemmKernels.matmul(conf, a, b, c, d;
                               transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                               transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                               kernel = Kernel.matmul_pipelined)

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            matmul!(c_h, new_a_h, new_b_h, alpha, beta)
            if A_type <: Integer
                @test c_h ≈ Array(d)
            else
                @test c_h ≈ Array(d) rtol=sqrt(eps(A_type))
            end
        end
    end

    @testset "FPU GEMM OPERATOR SHAPE ($(OP_M), $(OP_N), $(OP_K)) (NN, NT, TN, TT)" for (OP_M, OP_N, OP_K) in [
            (4, 8, 1), (8, 8, 1), (4, 16, 1), (4, 8, 2), (8, 16, 2)
        ]
        @testcase "NN, NT, TN, TT" for (transpose_a, transpose_b) in [(false, false), (false, true), (true, false), (true, true)]
            (M, N, K) = (128, 128, 128)
            (A_type, B_type, CD_type) = (Float32, Float32, Float32)

            compute_type = promote_type(A_type, B_type)

            alpha = convert(compute_type, 2)
            beta  = convert(CD_type, 3)

            a_h = rand(A_type, (M, K)) / sqrt(A_type(K))
            b_h = rand(B_type, (K, N)) / sqrt(B_type(K))
            c_h = rand(CD_type, (M, N))

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a   = CuArray(a_h)
            b   = CuArray(b_h)
            c   = CuArray(c_h)
            d   = similar(c)

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

            GemmKernels.matmul(conf, a, b, c, d;
                                transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                                transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                                kernel = Kernel.matmul_pipelined
                                )

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            matmul!(c_h, new_a_h, new_b_h, alpha, beta)
            @test c_h ≈ Array(d) rtol=sqrt(eps(A_type))
        end
    end

    @testset "TROPICAL GEMM $(A_type)*$(B_type)=$(CD_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))" for
        (A_type, B_type, CD_type, min_dimension) in [(Float32, Float32, Float32, 128)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [(8, 16, 2)]

        compute_type = promote_type(A_type, B_type)

        @testcase "(M = $M, N = $N, K = $K)" for (M, N, K) in vcat(min_dimension.*[[1,1,1], [2, 2, 1], [1, 1, 2], [2, 2, 2]])
            a_h = rand(A_type, (M, K)) / sqrt(A_type(K))
            b_h = rand(B_type, (K, N)) / sqrt(B_type(K))
            c_h = rand(CD_type, (M, N))
            d_h = similar(c_h)

            for i in 1 : M
                for j in 1 : N
                    d_h[i, j] = c_h[i, j]
                    for k in 1 : K
                        d_h[i, j] = max(a_h[i, k] + b_h[k, j], d_h[i, j])
                    end
                end
            end

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a   = CuArray(a_h)
            b   = CuArray(b_h)
            c   = CuArray(c_h)
            d   = similar(c)

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

            GemmKernels.matmul(conf, a, b, c, d; kernel = Kernel.matmul_pipelined)

            @test d_h ≈ Array(d) rtol=sqrt(eps(A_type))
        end
    end


    @testset "WMMA GEMM $(AB_type)*$(AB_type)=$(CD_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' ))" for transpose_a = [false, true],
        transpose_b = [false, true],
        (AB_type, CD_type, min_dimension) in [(Float16, Float16, 256), (Float16, Float32, 128)]
        @testcase "(M = $M, N = $N, K = $K)" for (M, N, K) in vcat(min_dimension.*[[1,1,1], [2,2,1], [1,1,2], [2,2,2]], [[2048, 2048, 2048]])
            alpha = convert(AB_type, 2)
            beta  = convert(CD_type, 3)

            a_h = rand(AB_type, (M, K)) / sqrt(AB_type(K))
            b_h = rand(AB_type, (K, N)) / sqrt(AB_type(K))
            c_h = rand(CD_type, (M, N))

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a   = CuArray(a_h)
            b   = CuArray(b_h)
            c   = CuArray(c_h)
            d   = similar(c)

            conf = GemmKernels.get_config(
                                          gemm_shape = (M = M, N = N, K = K),
                                          operator = Operator.WMMAOp{16, 16, 16, AB_type, CD_type},
                                          global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},
                                          global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},

                                          global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                          global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                          is_a_col_major = !transpose_a,
                                          is_b_col_major = !transpose_b,
                                         )

            GemmKernels.matmul(conf, a, b, c, d;
                               transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                               transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                               kernel = Kernel.matmul_pipelined
                              )

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            matmul!(c_h, new_a_h, new_b_h, alpha, beta)
            @test c_h ≈ Array(d) rtol=sqrt(eps(AB_type))
        end
    end

    @testset "WMMA GEMM ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) + bias" for transpose_a = [false, true],
        transpose_b = [false, true]

        @testcase "(M = $M, N = $N, K = $K)" for (M, N, K) in [(128, 128, 128), (256, 256, 256), (4096, 4096, 4096)]
            a_h = rand(Float16, (M, K)) / sqrt(Float16(K))
            b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
            c_h = rand(Float32, (M, N))

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a   = CuArray(a_h)
            b   = CuArray(b_h)
            c   = CuArray(c_h)
            d   = similar(c)

            # Bias vector: this vector contains 1 element per column of the result matrix.
            # This bias element is added to all elements in one column of the D matrix.
            # D is a M x N matrix, so the bias is an N-element vector.
            bias = CuArray(rand(Float32, (1, N)))

            # Custom epilogue to add bias
            ep = Epilogue.Bias(pointer(bias))

            conf = GemmKernels.get_config(
                                          gemm_shape = (M = M, N = N, K = K),
                                          operator = Operator.WMMAOp{16, 16, 16, Float16, Float32},
                                          global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{Float16} : Layout.UnsafeAlignedColMajor{Float16},
                                          global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{Float16} : Layout.UnsafeAlignedColMajor{Float16},

                                          global_c_layout = Layout.UnsafeAlignedColMajor{Float32},
                                          global_d_layout = Layout.UnsafeAlignedColMajor{Float32},

                                          is_a_col_major = !transpose_a,
                                          is_b_col_major = !transpose_b,
                                         )

            GemmKernels.matmul(conf, a, b, c, d;
                               epilogue = ep,
                               kernel = Kernel.matmul_pipelined
                              )

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            matmul!(c_h, new_a_h, new_b_h, true, true)
            @test c_h .+ Array(bias) ≈ Array(d) rtol=sqrt(eps(Float16))
        end
    end

    @testset "WMMA GEMM (A = diagonal, B = $( !transpose_b ? 'N' : 'T' ))" for transpose_b = [false, true]
        @testcase "(M = $M, N = $N, K = $K)" for (M, N, K) in [(128, 128, 128), (256, 256, 256), (4096, 4096, 4096)]
            @assert M == K "Diagonal only supports square A matrix (M == K)"

            transpose_a = false

            a_h = rand(Float16, M)
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
                                          operator = Operator.WMMAOp{16, 16, 16, Float16, Float32},
                                          global_a_layout = Layout.Diagonal{Float16},
                                          global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{Float16} : Layout.UnsafeAlignedColMajor{Float16},

                                          global_c_layout = Layout.UnsafeAlignedColMajor{Float32},
                                          global_d_layout = Layout.UnsafeAlignedColMajor{Float32},

                                          shared_a_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{Float16}, 8},

                                          is_a_col_major = !transpose_a,
                                          is_b_col_major = !transpose_b,
                                         )

            GemmKernels.matmul(conf, a, b, c, d)

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            matmul!(c_h, Diagonal(new_a_h), new_b_h, true, true)
            @test c_h ≈ Array(d) rtol=sqrt(eps(Float16))
        end
    end

    @testset "WMMA Complex GEMM ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' ))" for transpose_a = [false, true],
        transpose_b = [false, true]

        @testcase "(M = $M, N = $N, K = $K)" for (M, N, K) = [(128, 128, 128), (256, 256, 256), (2048, 2048, 2048)]
            a_h = rand(Complex{Float16}, (M, K)) / sqrt(Float16(K))
            b_h = rand(Complex{Float16}, (K, N)) / sqrt(Float16(K))
            c_h = rand(Complex{Float32}, (M, N))

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a = CuArray(a_h)
            b = CuArray(b_h)
            c = CuArray(c_h)
            d = similar(c)

            conf = GemmKernels.get_config(
                                          gemm_shape = (M = M, N = N, K = K),
                                          operator = Operator.WMMAComplexOp{16, 16, 16},

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

            GemmKernels.matmul(conf, a, b, c, d;
                               kernel = Kernel.matmul_pipelined)

            new_a_h = a_h
            new_b_h = b_h

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(new_a_h) : new_a_h
            new_b_h = transpose_b ? transpose(new_b_h) : new_b_h

            matmul!(c_h, new_a_h, new_b_h, true, true)
            @test c_h ≈ Array(d) rtol=sqrt(eps(Float16))
        end
    end

    @testset "WMMA Dual GEMM" begin
        @testcase "(M = $M, N = $N, K = $K)" for (M, N, K) in [(128, 128, 128), (256, 256, 256), (2048, 2048, 2048)]
            a_h = rand(Complex{Float16}, (M, K)) / sqrt(Float16(K))
            b_h = rand(Complex{Float16}, (K, N)) / sqrt(Float16(K))
            c_h = rand(Complex{Float32}, (M, N))

            a = CuArray(a_h)
            b = CuArray(b_h)
            c = CuArray(c_h)
            d = similar(c)

            conf = GemmKernels.get_config(
                                          gemm_shape = (M = M, N = N, K = K),
                                          operator = Operator.WMMADualOp{16, 16, 16},

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

            GemmKernels.matmul(conf, a, b, c, d;
                               kernel = Kernel.matmul_pipelined)

            a_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, Complex{Float32}.(a_h))
            b_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, Complex{Float32}.(b_h))
            c_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, c_h)
            d_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, Array(d))

            matmul!(c_dual, a_dual, b_dual, true, true)
            @test c_dual ≈ d_dual rtol=sqrt(eps(Float16))
        end
    end
end

################################################################################
