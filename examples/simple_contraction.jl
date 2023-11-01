using CUDA
using cuTENSOR
using GemmKernels.Operator
using GemmKernels.Tensors


function main()
    parseableName = "1.2.3.4.5-5.3.2.6.1-6.4"

    tensorModes = Vector{Vector{Int}}(undef, 0)
    for tensor in split(parseableName, "-")
        tensorMode = Vector{Int}(undef, 0)

        for mode in split(tensor, ".")
            push!(tensorMode, parse(Int, mode))
        end

        push!(tensorModes, tensorMode)
    end

    extents = Tuple(x for x in [8, 8, 4, 2048, 8, 2048])

    operator = Operator.WMMAOp
    dataType = Float16

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