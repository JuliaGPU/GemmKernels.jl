using CUDA
using cuTENSOR
using GemmKernels.Operator
using GemmKernels.Tensors
using JSON

# Pad the extents of the tensors such that the M, N, and K dimensions of the equivalent GEMM
# are multiples of 512, which allows for a broad range of block shapes. Like `pad_extents`
# in configs/configs.jl, we pick the smallest padding of *all* extents of a dimension whose
# product is a multiple of 512, rather than padding a single extent to a multiple of 512:
# for extents of 72, that keeps tensors ~15x smaller (0.4 instead of 10 GiB for TC19-30).
function pad_extents_512(extents, tensorModes)
    padded_extents = copy(extents)
    for (idx1, idx2) in [(1, 2), (3, 2), (1, 3)]
        intersection = intersect(tensorModes[idx1], tensorModes[idx2])

        # candidates: each extent rounded up to a multiple of 1, 2, 4, ..., 512
        candidates = [unique([cld(padded_extents[i], 2^k) * 2^k for k in 0:9]) for i in intersection]
        best = argmin(prod, Iterators.filter(c -> prod(c) % 512 == 0, Iterators.product(candidates...)))
        padded_extents[intersection] .= best
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
    A = CUDA.zeros(dataType, padded_extents[tensorModes[2]])
    B = CUDA.zeros(dataType, padded_extents[tensorModes[3]])
    C = CUDA.zeros(dataType, padded_extents[tensorModes[1]])
    D = CUDA.zeros(dataType, padded_extents[tensorModes[1]])

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

    # Free the padded tensors right away: tests run in parallel on the same GPU.
    foreach(CUDA.unsafe_free!, (A, B, C, D))

    # ------------------------------
    # cuTENSOR implementation.
    # ------------------------------

    # cuTENSOR does not need padding, so we directly use the host tensors.
    # The elementwise operations are applied separately.
    A_unpadded = CuArray(A_host)
    B_unpadded = CuArray(B_host)
    A = opA.(A_unpadded)
    B = opB.(B_unpadded)
    C = CuArray(C_host)
    foreach(CUDA.unsafe_free!, (A_unpadded, B_unpadded))

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
    foreach(CUDA.unsafe_free!, (A, B, C))
    CUDA.reclaim()  # return the memory to the driver, for the other test workers

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

    @testcase "Unsupported descriptors are rejected, not computed wrongly" begin
        # Each case must either be rejected with an ArgumentError, or give the right result:
        # without the descriptor checks, they run and silently compute something else.
        # Explicit kernel parameters keep this test independent of the default heuristics.
        M = N = K = 512
        A_big = CUDA.rand(Float16, 2M, K)
        A = CUDA.rand(Float16, M, K)
        B = CUDA.rand(Float16, K, N)
        C = CUDA.rand(Float16, M, N)
        # C large enough that a mistake in how it is read changes D well beyond the tolerance
        C_transposed = CUDA.rand(Float16, N, M) .* Float16(100)
        C_float32 = CUDA.rand(Float32, M, N) .* 100f0

        function contract(a, c, modeC, β)
            d = CUDA.zeros(Float16, M, N)
            plan = Tensors.ContractionPlan(
                1,
                a, TensorDescriptor(a), [1, 3],
                B, TensorDescriptor(B), [3, 2],
                β,
                c, TensorDescriptor(c), modeC,
                d, TensorDescriptor(d), [1, 2];
                operator = Operator.WMMAOp{16, 16, 16, Float16, Float32},
                computeType = Float16, accumulateType = Float32,
                blockShape = (M = 128, N = 128, K = 32), warpsPerBlock = 8,
                computeWarp = (M = 32, N = 64, K = 16))
            Tensors.contraction!(plan, 1, a, B, β, c, d)
            return Array(d)
        end

        rejected_or_correct(f, reference) =
            try
                isapprox(f(), reference; rtol = 1e-2)
            catch err
                err isa ArgumentError || rethrow()
                true
            end

        host(x) = Array{Float32}(Array(x))
        AB(a) = host(a) * host(B)

        # Only dense column-major tensors are supported: the top half of a taller matrix.
        A_view = view(A_big, 1:M, :)    # strides (1, 2M)
        @test rejected_or_correct(() -> contract(A_view, C, [1, 2], 0), AB(A_view))

        # C must have the same modes as D (C is read with D's layout).
        @test rejected_or_correct(() -> contract(A, C_transposed, [2, 1], 1),
                                  AB(A) + permutedims(host(C_transposed)))

        # C must have the same data type as D.
        @test rejected_or_correct(() -> contract(A, C_float32, [1, 2], 1), AB(A) + host(C_float32))
    end
end
