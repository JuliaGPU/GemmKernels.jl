using CUDA
using GemmKernels.TensorPlan
using Test

modes = eval(Meta.parse(ARGS[1]))
extents = eval(Meta.parse(ARGS[2]))

A = CuArray(rand(Float16, extents[modes[2]]) / sqrt(Float16(2048)))
B = CuArray(rand(Float16, extents[modes[3]]) / sqrt(Float16(2048)))
D = CuArray(rand(Float16, extents[modes[1]]))

algo = CUDA.CUTENSOR.CUTENSOR_ALGO_GETT

plan = CUDA.CUTENSOR.plan_contraction(
    A, modes[2], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    B, modes[3], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    D, modes[1], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    algo = algo,
    compute_type = Float16
)

CUDA.CUTENSOR.contraction!(
    1,
    A, modes[2], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    B, modes[3], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    1,
    D, modes[1], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
    compute_type = Float16,
    plan = plan
)

CUDA.@profile begin
    for i = 1 : 10
        CUDA.@sync begin
            CUDA.CUTENSOR.contraction!(
                1,
                A, modes[2], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                B, modes[3], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                1,
                D, modes[1], CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                CUDA.CUTENSOR.CUTENSOR_OP_IDENTITY,
                compute_type = Float16,
                plan = plan
            )
        end
    end
end

