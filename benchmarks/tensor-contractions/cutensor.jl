using CUDA
using cuTENSOR
using Test

modes = eval(Meta.parse(ARGS[1]))
extents = eval(Meta.parse(ARGS[2]))
algorithm = ARGS[3]

algo = cuTENSOR.CUTENSOR_ALGO_DEFAULT_PATIENT

if (algorithm == "GETT")
    algo = cuTENSOR.CUTENSOR_ALGO_GETT
elseif (algorithm == "TGETT")
    algo = cuTENSOR.CUTENSOR_ALGO_TGETT
elseif (algorithm == "TTGT")
    algo = cuTENSOR.CUTENSOR_ALGO_TTGT
end

A = CuArray(rand(Float16, extents[modes[2]]) / sqrt(Float16(2048)))
B = CuArray(rand(Float16, extents[modes[3]]) / sqrt(Float16(2048)))
D = CuArray(rand(Float16, extents[modes[1]]))


plan = cuTENSOR.plan_contraction(
    A, modes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
    B, modes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
    D, modes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
    cuTENSOR.CUTENSOR_OP_IDENTITY,
    algo = algo,
    compute_type = Float16
)

cuTENSOR.contraction!(
    1,
    A, modes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
    B, modes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
    1,
    D, modes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
    cuTENSOR.CUTENSOR_OP_IDENTITY,
    compute_type = Float16,
    plan = plan
)

CUDA.@profile begin
    for i = 1 : 10
        CUDA.@sync begin
            cuTENSOR.contraction!(
                1,
                A, modes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
                B, modes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
                1,
                D, modes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
                cuTENSOR.CUTENSOR_OP_IDENTITY,
                compute_type = Float16,
                plan = plan
            )
        end
    end
end

