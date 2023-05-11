using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra
using Test

function main()
    # @testset "FPU GEMM $(A_type)*$(B_type)+$(CD_type)=$(CD_type) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), 1)" 
    for transpose_a = [true], 
        transpose_b = [true], 
        (A_type, B_type, CD_type, min_dimension) in [(Float32, Float32, Float32, 128)],
        (OP_M, OP_N) in [(16, 16)]
        # @testset "(M = $M, N = $N, K = $K)" 
        for (M, N, K) in vcat(min_dimension.*[[1,1,1]])
            
            alpha = convert(A_type, 0)
            beta  = convert(CD_type, 3)

            a_h = rand(A_type, (M, K)) / sqrt(A_type(K))
            b_h = rand(B_type, (K, N)) / sqrt(B_type(K))
            c_h = rand(CD_type, (M, N))

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            c_h = transpose(c_h)

            a   = CuArray(a_h)
            b   = CuArray(b_h)
            c   = CuArray(c_h)
            d   = similar(c)

            # conf = GemmKernels.get_config(
            #                                 gemm_shape = (M = M, N = N, K = K),
            #                                 # TODO: Does not work with N = 64, investigate.
            #                                 block_shape = (M = 128, N = 64, K = 32),
            #                                 operator = Operator.FPUOp{OP_M, OP_N, 1, CD_type},
            #                                 global_a_layout = transpose_a ? Layout.AlignedRowMajor{A_type} : Layout.AlignedColMajor{A_type},
            #                                 global_b_layout = transpose_b ? Layout.AlignedRowMajor{B_type} : Layout.AlignedColMajor{B_type},

            #                                 global_c_layout = Layout.AlignedRowMajor{CD_type},
            #                                 global_d_layout = Layout.AlignedRowMajor{CD_type},

            #                                 is_a_col_major = !transpose_a,
            #                                 is_b_col_major = !transpose_b,
            #                                 )

            # GemmKernels.matmul(a, b, c, d, conf;
            #                     transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
            #                     transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
            #                     kernel = Kernel.matmul_singlestage
            #                     )

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            new_c_h = transpose(c_h)

            display((alpha * CD_type.(new_a_h) * CD_type.(new_b_h) + beta * new_c_h)[1:10, 1:10]) 
            # display(Array(d)[1:10, 1:10])
            
            # @test all(isapprox.(alpha * CD_type.(new_a_h) * CD_type.(new_b_h) + beta * new_c_h, Array(d); rtol = sqrt(eps(A_type))))
        end
    end
end

function main2()
    @testset "WMMA GEMM ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) + bias" for transpose_a = [false, true],

        transpose_b = [false, true]

        @testset "(M = $M, N = $N, K = $K)" for (M, N, K) in [(128, 128, 128), (256, 256, 256), (4096, 4096, 4096)]
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
                                          operator = Operator.WMMAOp{16, 16, 16, Float32},
                                          global_a_layout = transpose_a ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},
                                          global_b_layout = transpose_b ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},

                                          global_c_layout = Layout.AlignedColMajor{Float32},
                                          global_d_layout = Layout.AlignedColMajor{Float32},

                                          is_a_col_major = !transpose_a,
                                          is_b_col_major = !transpose_b,
                                         )

            GemmKernels.matmul(a, b, c, d, conf;
                               epilogue = ep,
                               kernel = Kernel.matmul_pipelined
                              )

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            @test all(isapprox.(Float32.(new_a_h) * Float32.(new_b_h) + c_h .+ Array(bias), Array(d); rtol = sqrt(eps(Float16))))
        end
    end


end

isinteractive() || main()