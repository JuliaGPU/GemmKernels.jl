using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra
using Test

function main()
    @testset "FPU GEMM $(A_type)*$(B_type)+$(CD_type)=$(CD_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))" for (A_type, B_type, CD_type, min_dimension) in [(Float16, Float16, Float32, 128), (Float32, Float32, Float32, 128), (Float32, Float32, Float64, 128), (Float64, Float64, Float64, 128)], transpose_a = [false, true], 
        transpose_b = [false, true], 
        (OP_M, OP_N, OP_K) in [(8, 16, 2)]
        @testset "(M = $M, N = $N, K = $K)" for (M, N, K) in vcat(min_dimension.*[[1,1,1], [2, 2, 1], [1, 1, 2], [2, 2, 2]], [[2048, 2048, 2048]])
            alpha = convert(A_type, 2)
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
                                            # TODO: Does not work with N = 64, investigate.
                                            block_shape = (M = 64, N = 64, K = 32),
                                            operator = Operator.FPUOp{OP_M, OP_N, OP_K, CD_type, A_type},
                                            global_a_layout = transpose_a ? Layout.AlignedRowMajor{A_type} : Layout.AlignedColMajor{A_type},
                                            global_b_layout = transpose_b ? Layout.AlignedRowMajor{B_type} : Layout.AlignedColMajor{B_type},

                                            global_c_layout = Layout.AlignedColMajor{CD_type},
                                            global_d_layout = Layout.AlignedColMajor{CD_type},

                                            is_a_col_major = !transpose_a,
                                            is_b_col_major = !transpose_b,
                                            )

            GemmKernels.matmul(a, b, c, d, conf;
                                transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                                transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                                kernel = Kernel.matmul_pipelined
                                )

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h
            
            @test all(isapprox.(alpha * CD_type.(new_a_h) * CD_type.(new_b_h) + beta * c_h, Array(d); rtol = sqrt(eps(A_type))))
        end
    end
end

function main2()
    @testset "FPU GEMM OPERATOR SHAPE ($(OP_M), $(OP_N), $(OP_K)) (NN, NT, TN, TT)" for (OP_M, OP_N, OP_K) in [
            (4, 8, 1), (4, 8, 2), (4, 8, 4), (4, 8, 8), (4, 8, 16),
            (8, 8, 1), (8, 8, 2), (8, 8, 4), (8, 8, 8),
            (8, 16, 1), (8, 16, 2), (8, 16, 4), (8, 16, 8), 
            (16, 16, 1), (16, 16, 2), (16, 16, 4), 
            (16, 8, 1), (16, 8, 2), (16, 8, 4),
        ]
        @testset "NN, NT, TN, TT" for (transpose_a, transpose_b) in [(false, false), (false, true), (true, false), (true, true)]
            (M, N, K) = (128, 128, 128)
            (A_type, B_type, CD_type) = (Float32, Float32, Float32)
            
            alpha = convert(A_type, 2)
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
                                            # TODO: Does not work with N = 64, investigate.
                                            block_shape = (M = 128, N = 64, K = 32),
                                            operator = Operator.FPUOp{OP_M, OP_N, OP_K, CD_type, A_type},
                                            global_a_layout = transpose_a ? Layout.AlignedRowMajor{A_type} : Layout.AlignedColMajor{A_type},
                                            global_b_layout = transpose_b ? Layout.AlignedRowMajor{B_type} : Layout.AlignedColMajor{B_type},

                                            global_c_layout = Layout.AlignedColMajor{CD_type},
                                            global_d_layout = Layout.AlignedColMajor{CD_type},

                                            is_a_col_major = !transpose_a,
                                            is_b_col_major = !transpose_b,
                                            )

            GemmKernels.matmul(a, b, c, d, conf;
                                transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                                transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                                kernel = Kernel.matmul_pipelined
                                )

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h
            
            @test all(isapprox.(alpha * CD_type.(new_a_h) * CD_type.(new_b_h) + beta * c_h, Array(d); rtol = sqrt(eps(A_type))))
        end
    end
end

isinteractive() || main()