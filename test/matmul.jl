using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra

################################################################################

@testset "Matmul API" begin
    @test_if "wmma" @testset "WMMA GEMM ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' ))" for transpose_a = [false, true],
        transpose_b = [false, true]

        @testset "(M = $M, N = $N, K = $K)" for (M, N, K) in [(128, 128, 128), (256, 256, 128), (128, 128, 256), (256, 256, 256), (2048, 2048, 2048)]
            alpha = 2
            beta  = 3

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

            conf = GemmKernels.get_config(
                                          gemm_shape = (M = M, N = N, K = K),
                                          operator = Operator.WMMAOp{16, 16, 16},
                                          global_a_layout = transpose_a ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},
                                          global_b_layout = transpose_b ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},

                                          global_c_layout = Layout.AlignedColMajor{Float32},
                                          global_d_layout = Layout.AlignedColMajor{Float32},

                                          is_a_col_major = !transpose_a,
                                          is_b_col_major = !transpose_b,
                                         )

            GemmKernels.matmul(a, b, c, d, conf;
                               transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                               transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta))

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            @test all(isapprox.(alpha * Float32.(new_a_h) * Float32.(new_b_h) + beta * c_h, Array(d); rtol = sqrt(eps(Float16))))
        end
    end

    @test_if "bias" @testset "WMMA GEMM ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) + bias" for transpose_a = [false, true],
        transpose_b = [false, true]

        @testset "(M = $M, N = $N, K = $K)" for (M, N, K) in [(128, 128, 128), (256, 256, 256)]
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
                                          operator = Operator.WMMAOp{16, 16, 16},
                                          global_a_layout = transpose_a ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},
                                          global_b_layout = transpose_b ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},

                                          global_c_layout = Layout.AlignedColMajor{Float32},
                                          global_d_layout = Layout.AlignedColMajor{Float32},

                                          is_a_col_major = !transpose_a,
                                          is_b_col_major = !transpose_b,
                                         )

            GemmKernels.matmul(a, b, c, d, conf;
                               epilogue = ep)

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            @test all(isapprox.(Float32.(new_a_h) * Float32.(new_b_h) + c_h .+ Array(bias), Array(d); rtol = sqrt(eps(Float16))))
        end
    end

    @test_if "diagonal" @testset "WMMA GEMM (A = diagonal, B = $( !transpose_b ? 'N' : 'T' ))" for transpose_b = [false, true]
        @testset "(M = $M, N = $N, K = $K)" for (M, N, K) in [(128, 128, 128), (256, 256, 256)]
            @assert M == K "Diagonal only supports square A matrix (M == K)"

            transpose_a = false

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
                                          global_a_layout = Layout.Diagonal{Float16},
                                          global_b_layout = transpose_b ? Layout.AlignedRowMajor{Float16} : Layout.AlignedColMajor{Float16},

                                          global_c_layout = Layout.AlignedColMajor{Float32},
                                          global_d_layout = Layout.AlignedColMajor{Float32},

                                          shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},

                                          is_a_col_major = !transpose_a,
                                          is_b_col_major = !transpose_b,
                                         )

            GemmKernels.matmul(a, b, c, d, conf)

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(a_h) : a_h
            new_b_h = transpose_b ? transpose(b_h) : b_h

            @test all(isapprox.(Float32.(Diagonal(new_a_h)) * Float32.(new_b_h) + c_h, Array(d); rtol = sqrt(eps(Float16))))
        end
    end

    @test_if "complex" @testset "WMMA Complex GEMM ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' ))" for transpose_a = [false, true],
        transpose_b = [false, true]

        @testset "(M = $M, N = $N, K = $K)" for (M, N, K) = [(128, 128, 128), (256, 256, 256)]
            a_h = rand(Complex{Float16}, (M, K)) / sqrt(Float16(K));
            b_h = rand(Complex{Float16}, (K, N)) / sqrt(Float16(K));
            c_h = rand(Complex{Float32}, (M, N));

            # Transpose input if necessary
            a_h = transpose_a ? transpose(a_h) : a_h
            b_h = transpose_b ? transpose(b_h) : b_h

            a = CuArray(a_h);
            b = CuArray(b_h);
            c = CuArray(c_h);
            d = similar(c);

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

            GemmKernels.matmul(a, b, c, d, conf;)

            new_a_h = a_h
            new_b_h = b_h

            # Transpose outputs, if necessary
            new_a_h = transpose_a ? transpose(new_a_h) : new_a_h
            new_b_h = transpose_b ? transpose(new_b_h) : new_b_h

            # TODO: Figure out why changing this to a * b + c = d instead of a * b = d - c
            # makes tests fail for CC (see #19).
            @test all(isapprox.(new_a_h * new_b_h, Array(d) - c_h; rtol=sqrt(eps(Float16))));
        end
    end

    @test_if "dual" @testset "WMMA Dual GEMM" begin
        @testset "(M = $M, N = $N, K = $K)" for (M, N, K) in [(128, 128, 128), (256, 256, 256)]
            a_h = rand(Complex{Float16}, (M, K)) / sqrt(Float16(K));
            b_h = rand(Complex{Float16}, (K, N)) / sqrt(Float16(K));
            c_h = rand(Complex{Float32}, (M, N));

            a = CuArray(a_h);
            b = CuArray(b_h);
            c = CuArray(c_h);
            d = similar(c);

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

            GemmKernels.matmul(a, b, c, d, conf)

            a_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, Complex{Float32}.(a_h))
            b_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, Complex{Float32}.(b_h))
            c_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, c_h)
            d_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, Array(d))

            @test all(isapprox.(a_dual * b_dual + c_dual, d_dual; rtol=sqrt(eps(Float16))));
        end
    end
end

################################################################################
