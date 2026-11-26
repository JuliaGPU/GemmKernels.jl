using CUDA
using cuTENSOR
using GemmKernels.Operator
using GemmKernels.Tensors
using JSON

# Pad the extents of the tensors such that the M, N, and K dimensions are a multiple of 512.
# This allows for a broad range of possible block shapes in the GEMM.
function pad_extents_512(extents, tensorModes)
    padded_extents = copy(extents)
    for (idx1, idx2) in [(1, 2), (3, 2), (1, 3)]
        intersection = intersect(tensorModes[idx1], tensorModes[idx2])

        if prod(extents[intersection]) % 512 == 0
            continue
        end

        padded_extents[intersection[1]] = Int64(ceil(extents[intersection[1]] / 512) * 512)
    end
    return padded_extents
end

# Compute D = α ⋅ opA(A) ⋅ opB(B) + β ⋅ C with GemmKernels (using the default, heuristic
# kernel parameters) and with cuTENSOR, and compare the results.
function testContraction(extents, tensorModes; α = 1, β = 1, opA = identity, opB = CUDA.cos)
    padded_extents = pad_extents_512(extents, tensorModes)

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
        α,
        A, TensorDescriptor(A; unaryOp=opA), tensorModes[2],
        B, TensorDescriptor(B; unaryOp=opB), tensorModes[3],
        β,
        C, TensorDescriptor(C), tensorModes[1],
        D, TensorDescriptor(D), tensorModes[1];
        operator=operator,
        computeType=computeType,
        accumulateType=accumulateType
    )

    # Executing the tensor contraction.
    Tensors.contraction!(plan, α, A, B, β, C, D)

    # Copying the result back to the CPU while removing the padding.
    D1 = Array(D[(1:extent for extent in extents[tensorModes[1]])...])

    # ------------------------------
    # cuTENSOR implementation.
    # ------------------------------

    # cuTENSOR does not need padding, so we directly use the host tensors.
    # The elementwise operations are applied separately.
    A = opA.(CuArray(A_host))
    B = opB.(CuArray(B_host))
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
        α,
        (A), tensorModes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
        (B), tensorModes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
        β,
        (C), tensorModes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cuTENSOR.CUTENSOR_OP_IDENTITY,
        compute_type=Float32,
        plan=plan
    )

    # Copying the result back to the CPU.
    D2 = Array(C)

    return all(isapprox.(Array(D1), Array(D2); rtol = sqrt(eps(dataType))))
end

function parse_modes(parseableName)
    tensorModes = Vector{Vector{Int}}(undef, 0)
    for tensor in split(parseableName, "-")
        push!(tensorModes, [parse(Int, mode) for mode in split(tensor, ".")])
    end
    return tensorModes
end

@testset "Tensor Contraction" begin
    jsonData = JSON.parse(read(joinpath(@__DIR__, "benchmark-suite.json"), String))

    @testset "TCCG benchmark suite against cuTENSOR" begin
        @testcase "$(el["name"])" for el in jsonData
            tensorModes = parse_modes(el["parseableName"])
            extents = Vector{Int}(el["extents"])
            @test testContraction(extents, tensorModes)
        end
    end

    @testcase "α with an elementwise operation on A" begin
        # D = α ⋅ opA(A) ⋅ B + β ⋅ C. Applying α before the elementwise operation would
        # compute opA(α ⋅ A) instead, which differs for any operation that is not
        # positively homogeneous (unlike e.g. ReLU), such as this affine one.
        tensorModes = parse_modes("1.2.3-2.4.1-4.3")
        @test testContraction([312, 312, 24, 312], tensorModes;
                              α = 2, β = 1, opA = x -> x + one(x), opB = identity)
    end

    @testcase "Tropical contraction" begin
        # D[m, n] = max(C[m, n], maxₖ A[m, k] + B[k, n]), with C = -Inf.
        M = N = K = 512
        A = CUDA.rand(Float32, M, K)
        B = CUDA.rand(Float32, K, N)
        C = CUDA.fill(-Inf32, M, N)
        D = CUDA.zeros(Float32, M, N)

        plan = Tensors.ContractionPlan(
            1,
            A, TensorDescriptor(A), [1, 3],
            B, TensorDescriptor(B), [3, 2],
            1,
            C, TensorDescriptor(C), [1, 2],
            D, TensorDescriptor(D), [1, 2];
            operator=Operator.FPUOp,
            computeType=Float32,
            accumulateType=Float32
        )
        # (+, max) re-plans the contraction with Operator.TropicalFPUOp.
        Tensors.contraction!(plan, 1, A, B, 1, C, D, +, max)

        A_host, B_host = Array(A), Array(B)
        reference = [maximum(A_host[m, k] + B_host[k, n] for k in 1:K) for m in 1:M, n in 1:N]
        @test Array(D) ≈ reference
    end

    @testcase "Unsupported descriptors are rejected" begin
        A = CUDA.rand(Float16, 512, 512)
        B = CUDA.rand(Float16, 512, 512)
        C = CUDA.rand(Float16, 512, 512)
        D = CUDA.rand(Float16, 512, 512)
        plan(; descA = TensorDescriptor(A), modeC = [1, 2], descC = TensorDescriptor(C)) =
            Tensors.ContractionPlan(
                1,
                A, descA, [1, 3],
                B, TensorDescriptor(B), [3, 2],
                1,
                C, descC, modeC,
                D, TensorDescriptor(D), [1, 2];
                computeType=Float16, accumulateType=Float32)

        # C must have the same modes and data type as D.
        @test_throws ArgumentError plan(modeC = [2, 1])
        @test_throws ArgumentError plan(descC = TensorDescriptor(CUDA.rand(Float32, 512, 512)))

        # Only dense column-major tensors are supported.
        @test_throws ArgumentError plan(descA = TensorDescriptor(A; stride = [1, 1024]))
    end
end
