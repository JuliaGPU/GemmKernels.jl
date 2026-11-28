using CUDA
using GemmKernels
using GemmKernels.Tensors.GETTLayout

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

# The address arithmetic of the GETT layouts (createGETTLayout's `indexing`): the lean
# layouts must compute exactly the offsets of the original ones, and :default must still give
# the original layout types (so that the generated code does not change).

# reference: the original formula, sum over the modes of (G ÷ div) % mod * stride
function reference_offset(G1, G2, divT1, modT1, strides1, divT2, modT2, strides2)
    offset = 1
    for i in eachindex(divT1)
        offset += ((G1 ÷ divT1[i]) % modT1[i]) * strides1[i]
    end
    for i in eachindex(divT2)
        offset += ((G2 ÷ divT2[i]) % modT2[i]) * strides2[i]
    end
    return offset
end

@testset "GETT indexing" begin
    @testcase "lean offsets equal the original ones" begin
        # (extents, (modes of the first dimension, modes of the second dimension))
        # (the lean layouts are only used for tensors with fewer than 2^31 elements; UInt64
        # checks the formula on the larger ones)
        layouts = [([5136, 5120], ([1], [2])),
                   ([72, 72, 72, 72], ([1, 3], [4, 2])),
                   ([72, 72, 72, 72, 72, 72], ([1, 3], [6, 5])),
                   ([24, 16, 16, 24, 16, 24], ([2, 4], [6, 5, 1])),
                   ([312, 296, 96, 962], ([1], [4, 3]))]
        for (extent, dims) in layouts
            (divT1, divT2, modT1, modT2, strides1, strides2, _, _) =
                precomputeGETTLayoutConstants(extent, dims, false)
            size1, size2 = prod(extent[dims[1]]), prod(extent[dims[2]])
            index_types = prod(extent) <= typemax(Int32) ? (UInt64, UInt32) : (UInt64,)
            for IT in index_types, _ in 1:2000
                G1, G2 = rand(0:size1-1), rand(0:size2-1)
                @test GETTLayout.lean_offset(IT, G1, G2, Val(divT1), Val(modT1), Val(strides1),
                                             Val(divT2), Val(modT2), Val(strides2)) ==
                      reference_offset(G1, G2, divT1, modT1, strides1, divT2, modT2, strides2)
            end
        end
    end

    @testcase "layout types" begin
        args = (Float16, [72, 72, 72, 72], ([1, 3], [4, 2]), true, false)
        @test GETTLayout.createGETTLayout(args...) <: GETTLayout.GETTLayoutColMajor
        @test GETTLayout.createGETTLayout(args...; indexing = :default) ==
              GETTLayout.createGETTLayout(args...)
        @test GETTLayout.createGETTLayout(args...; indexing = :lean32) <:
              GETTLayout.GETTLayoutLeanColMajor{Float16, UInt32}
        @test_throws ArgumentError GETTLayout.createGETTLayout(args...; indexing = :lean64)
    end

    if CUDA.functional()
        kernels = Any[Kernel.matmul_singlestage, Kernel.matmul_pipelined]
        capability(device()) ≥ v"8.0" && push!(kernels, Kernel.matmul_pipelined_async(3))

        # a GEMM-like contraction (one mode per dimension), and TCCG TC22's pattern (two)
        contractions = [("1.2-1.3-3.2", [256, 128, 192], [1], [2], [3], true, true),
                        ("1.2.3.4-1.5.3.6-2.6.4.5", [32, 16, 32, 16, 16, 16], [1, 3], [2, 4], [6, 5], true, false)]
        @testcase "$name, $kernel, indexing = $indexing" for (name, extents, PERM_M, PERM_N, PERM_K, a_col, b_col) in contractions,
                                                               kernel in kernels, indexing in ["default", "lean32"]
            problem = WMMATensorContraction(; name, extents, data_type = Float16, compute_type = Float16,
                                              accumulate_type = Float32, zero_c = true)
            data = allocate_data(problem)
            initialize_data(problem, data...)
            reference = Array(calculate_reference(problem, data...))
            params = (; BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                        OP_M = 16, OP_N = 16, OP_K = 16, kernel,
                        is_A_col_major = a_col, is_B_col_major = b_col, is_D_col_major = true,
                        PERM_M, PERM_N, PERM_K, cta_swizzle = CTASwizzle.Identity,
                        INDEXING = indexing)
            args = try
                prepare(problem, data...; params...)
            finally
                GemmKernels.Tensors.OVERRIDE_do_override = false
            end
            initialize_data(problem, data...; params...)
            result = Array(execute(problem, data...; args...))
            @test verify(problem, reference, result)
        end
    end
end
