using CUDA
using ForwardDiff
using GemmKernels
using LinearAlgebra
using Statistics
using Printf

gemm_shape = eval(Meta.parse(ARGS[1]))
gemm_names = (:M, :N, :K)
gemm_shape = NamedTuple{gemm_names}(gemm_shape)

block_shape = eval(Meta.parse(ARGS[2]))
block_names = (:M, :N, :K)
block_shape = NamedTuple{block_names}(block_shape)

operator_shape = eval(Meta.parse(ARGS[3]))
operator_names = (:M, :N, :K)
operator_shape = NamedTuple{operator_names}(operator_shape)

compute_type = eval(Meta.parse(ARGS[4]))
data_type = eval(Meta.parse(ARGS[5]))

warps_per_block = 8
compute_warp = (M = block_shape.M ÷ 4, N = block_shape.N ÷ 2, K = operator_shape.K)

if (size(ARGS) == 7)
    warps_per_block = eval(Meta.parse(ARGS[6]))
    compute_warp = eval(Meta.parse(ARGS[7]))
end

function main()
    # ----
    # GEMM
    # ----

    alpha = rand(compute_type)

    a = CuArray(rand(compute_type, (gemm_shape.M, gemm_shape.K)) / sqrt(compute_type(gemm_shape.K)))
    b = CuArray(rand(compute_type, (gemm_shape.K, gemm_shape.N)) / sqrt(compute_type(gemm_shape.K)))

    beta = rand(data_type)
    c = CuArray(rand(data_type, (gemm_shape.M, gemm_shape.N)))
    d = similar(c)

    # ------
    # Config
    # ------

    conf = GemmKernels.get_config(
        gemm_shape = gemm_shape,
        block_shape = block_shape,
        operator = Operator.FPUOp{operator_shape.M, operator_shape.N, operator_shape.K, data_type, compute_type},

        global_a_layout = Layout.AlignedColMajor{compute_type},
        global_b_layout = Layout.AlignedColMajor{compute_type},

        global_c_layout = Layout.AlignedColMajor{data_type},
        global_d_layout = Layout.AlignedColMajor{data_type},

        is_a_col_major = true,
        is_b_col_major = true,

        warps_per_block = warps_per_block,
        compute_warp = compute_warp,
    )

    # ------
    # Matmul
    # ------

    CUDA.@profile begin
        for i = 1 : 10
            CUDA.@sync begin
                GemmKernels.matmul(
                    a, b, c, d, conf;
                    transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
                    transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
                    kernel = Kernel.matmul_pipelined
                )
            end
        end
    end
end

isinteractive() || main()