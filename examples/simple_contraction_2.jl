using CUDA
using cuTENSOR
using GemmKernels.Operator
using GemmKernels.Tensors


function main()
    # ABCDEF - DEGC - GFAB
    parseableName = "1.2.3.4.5.6-4.5.7.3-7.6.1.2"

    tensorModes = Vector{Vector{Int}}(undef, 0)
    for tensor in split(parseableName, "-")
        tensorMode = Vector{Int}(undef, 0)

        for mode in split(tensor, ".")
            push!(tensorMode, parse(Int, mode))
        end

        push!(tensorModes, tensorMode)
    end

    extents = Tuple(x for x in [
        8,
        16,
        16,
        16,
        8,
        16,
        2048
    ])

    operator = Operator.WMMAOp
    dataType = Float16

    A = CuArray(rand(dataType, extents[tensorModes[2]]) / sqrt(dataType(2048)))
    B = CuArray(rand(dataType, extents[tensorModes[3]]) / sqrt(dataType(2048)))
    C = CuArray(rand(dataType, extents[tensorModes[1]]))
    D = CuArray(zeros(dataType, extents[tensorModes[1]]))

    plan = Tensors.ContractionPlan(
        TensorDescriptor(A), tensorModes[2],
        TensorDescriptor(B), tensorModes[3],
        TensorDescriptor(C), tensorModes[1],
        TensorDescriptor(D), tensorModes[1];
        operator = operator,
        computeType=eltype(A),
        dataType=eltype(C)
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
        (A), tensorModes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
        (B), tensorModes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
        1,
        (C), tensorModes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cuTENSOR.CUTENSOR_OP_IDENTITY,
        compute_type = dataType,
        plan = plan
    )
    D2 = Array(C)

    display(D1[1:8, 1:8, 1, 1, 1, 1])
    display(D2[1:8, 1:8, 1, 1, 1, 1])

    return all(isapprox.(Array(D1), Array(D2); rtol = sqrt(eps(dataType))))
end