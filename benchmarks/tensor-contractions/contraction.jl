using CUDA
using GemmKernels.TensorPlan
using Test

modes = eval(Meta.parse(ARGS[1]))
extents = eval(Meta.parse(ARGS[2]))

A = CuArray(rand(Float16, extents[modes[2]]) / sqrt(Float16(2048)))
B = CuArray(rand(Float16, extents[modes[3]]) / sqrt(Float16(2048)))
C = CuArray(rand(Float16, extents[modes[1]]))
D = CuArray(zeros(Float16, extents[modes[1]]))

plan = TensorPlan.ContractionPlan(
    A, modes[2],
    B, modes[3],
    D, modes[1],
    D, modes[1],
)

TensorPlan.contraction!(plan, Float16(1.0), A, B, Float16(1.0), C, D)

CUDA.@profile begin
    for i = 1 : 10
        CUDA.@sync begin
            TensorPlan.contraction!(plan, Float16(1.0), A, B, Float16(1.0), 1, D)
        end
    end
end