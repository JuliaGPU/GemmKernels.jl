using CUDA
using cuTENSOR
using GemmKernels.Operator
using GemmKernels.Tensors
using Test
using JSON

function testContraction(extents, tensorModes, operator, dataType)
    A = CuArray(rand(dataType, extents[tensorModes[2]]) / sqrt(dataType(2048)))
    B = CuArray(rand(dataType, extents[tensorModes[3]]) / sqrt(dataType(2048)))
    C = CuArray(rand(dataType, extents[tensorModes[1]]))
    D = CuArray(zeros(dataType, extents[tensorModes[1]]))

    plan = Tensors.ContractionPlan(
        A, tensorModes[2],
        B, tensorModes[3],
        C, tensorModes[1],
        D, tensorModes[1];
        operator = operator,
    )

    Tensors.contraction!(plan, 1, A, B, 1, C, D)
    D1 = Array(D)

    # CUTENSOR
    algo = cuTENSOR.CUTENSOR_ALGO_GETT

    plan = cuTENSOR.plan_contraction(
        A, tensorModes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
        B, tensorModes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
        C, tensorModes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cuTENSOR.CUTENSOR_OP_IDENTITY,
        algo = algo,
        compute_type = dataType
    )

    cuTENSOR.contraction!(
        1,
        A, tensorModes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
        B, tensorModes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
        1,
        C, tensorModes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cuTENSOR.CUTENSOR_OP_IDENTITY,
        compute_type = dataType,
        plan = plan
    )
    D2 = Array(C)

    return all(isapprox.(Array(D1), Array(D2); rtol = sqrt(eps(dataType))))
end

@testset "Tensor Contraction" begin
    fp = open("./benchmark-suite.json", "r")

    jsonData = JSON.parse(read(fp, String))

    @test_if "contraction" @testset "TCCG benchmark suite against cuTENSOR with $(operator)" for (operator, dataType) in [(Operator.WMMAOp, Float16) , (Operator.FPUOp, Float32)] 
        for el in jsonData
            parseableName = el["parseableName"]

            tensorModes = Vector{Vector{Int}}(undef, 0)
            for tensor in split(parseableName, "-")
                tensorMode = Vector{Int}(undef, 0)

                for mode in split(tensor, ".")
                    push!(tensorMode, parse(Int, mode))
                end

                push!(tensorModes, tensorMode)
            end

            extents = Tuple(x for x in el["extents"])

            name = el["name"]
            @testset "$name" begin
                @test testContraction(extents, tensorModes, operator, dataType)
            end
        end
    end
end