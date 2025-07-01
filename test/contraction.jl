using CUDA
using cuTENSOR
using GemmKernels.Operator
using GemmKernels.Tensors
using JSON

function testContraction(extents, tensorModes)
    # For the sake of simplicity, we pad the extents of the tensors to be a multiple of 512. This
    # allows for a broad range of possible block shapes in the GEMM.
    padded_extents = copy(extents)
    for (idx1, idx2) in [(1, 2), (3, 2), (1, 3)]
        intersection = intersect(tensorModes[idx1], tensorModes[idx2])

        if prod(extents[intersection]) % 512 == 0
            continue
        end

        padded_extents[intersection[1]] = Int64(ceil(extents[intersection[1]] / 512) * 512)
    end

    # Casting the extents to tuples.
    extents = Tuple(extents)
    padded_extents = Tuple(padded_extents)
    
    # Specific kernel configurations.
    operator = Operator.WMMAOp
    dataType = Float16
    computeType = Float16
    accumulateType = Float32

    # K-dimension length.
    K = prod(extents[intersect(tensorModes[2], tensorModes[3])])

    # Creating random tensors.
    A_host = rand(dataType, extents[tensorModes[2]]) / sqrt(dataType(K))
    B_host = rand(dataType, extents[tensorModes[3]]) / sqrt(dataType(K))
    C_host = rand(dataType, extents[tensorModes[1]])

    # ------------------------------
    # GemmKernels.jl implementation.
    # ------------------------------

    # Creating tensors on the GPU with the padded extents.
    A = CuArray(zeros(dataType, padded_extents[tensorModes[2]]))
    B = CuArray(zeros(dataType, padded_extents[tensorModes[3]]))
    C = CuArray(zeros(dataType, padded_extents[tensorModes[1]]))
    D = CuArray(zeros(dataType, padded_extents[tensorModes[1]]))

    # Copying the random tensors to the GPU.
    A[(1:extent for extent in extents[tensorModes[2]])...] = A_host
    B[(1:extent for extent in extents[tensorModes[3]])...] = B_host
    C[(1:extent for extent in extents[tensorModes[1]])...] = C_host

    # Creating the tensor contraction plan, similar to the cuTENSOR API.
    plan = Tensors.ContractionPlan(
        TensorDescriptor(A), tensorModes[2],
        TensorDescriptor(B; unaryOp=CUDA.cos), tensorModes[3],
        TensorDescriptor(C), tensorModes[1],
        TensorDescriptor(D), tensorModes[1];
        operator=operator,
        computeType=computeType,
        accumulateType=accumulateType
    )

    # Executing the tensor contraction.
    Tensors.contraction!(plan, 1, A, B, 1, C, D)

    # Copying the result back to the CPU while removing the padding.
    D1 = Array(D[(1:extent for extent in extents[tensorModes[1]])...])

    # ------------------------------
    # cuTENSOR implementation.
    # ------------------------------

    # cuTENSOR does not need padding, so we directly use the host tensors.
    A = CuArray(A_host)
    B = cos.(CuArray(B_host))
    C = CuArray(C_host)

    # Creating the cuTENSOR contraction plan.
    plan = cuTENSOR.plan_contraction(
        A, tensorModes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
        B, tensorModes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
        C, tensorModes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cuTENSOR.CUTENSOR_OP_IDENTITY,
        algo=cuTENSOR.CUTENSOR_ALGO_GETT,
        compute_type=computeType
    )

    # Executing the tensor contraction.
    cuTENSOR.contract!(
        1,
        (A), tensorModes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
        (B), tensorModes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
        1,
        (C), tensorModes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cuTENSOR.CUTENSOR_OP_IDENTITY,
        compute_type=Float32,
        plan=plan
    )

    # Copying the result back to the CPU.
    D2 = Array(C)

    return all(isapprox.(Array(D1), Array(D2); rtol = sqrt(eps(dataType))))
end

@testset "Tensor Contraction" begin
    fp = open("./benchmark-suite.json", "r")

    jsonData = JSON.parse(read(fp, String))

    @testset "TCCG benchmark suite against cuTENSOR" begin
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

            extents = Vector{Int}(el["extents"])

            name = el["name"]
            @testset "$name" begin
                @test testContraction(extents, tensorModes)
            end
        end
    end
end