# List of configurations to use for testing and benchmarking.

module Configs


## lazy module loading

using CUDA
using cuTENSOR
using GemmKernels
using LinearAlgebra
using Random

struct LazyModule
    pkg::Base.PkgId
    LazyModule(name, uuid) = new(Base.PkgId(uuid, name))
end
function Base.getproperty(lazy_mod::LazyModule, sym::Symbol)
    pkg = getfield(lazy_mod, :pkg)
    mod = get(Base.loaded_modules, pkg, nothing)
    if mod === nothing
        error("This functionality requires the $(pkg.name) package, which should be installed and loaded first.")
    end
    getfield(mod, sym)
end

const ForwardDiff = LazyModule("ForwardDiff", Base.UUID("f6369f11-7733-5829-9624-2563aa707210"))


## interface

export allocate_data, initialize_data,
       prepare, execute,
       prepare_baseline, execute_baseline,
       calculate_reference, verify,
       get_configs

abstract type AbstractProblem end

compare(x, y, T) = error("Unimplemented compare(x, y, T) function for type $T")
compare(x, y, T::Type{<:AbstractFloat}) = isapprox(x, y; rtol=sqrt(eps(T)))
compare(x, y, T::Type{<:Integer}) = (x == y)
compare(x, y, T::Type{Complex{U}}) where {U} = compare(x, y, U)

function verify(prob::AbstractProblem, A::AbstractArray{T}, B::AbstractArray{T}) where T
    all(compare.(A, B, T))
end

prepare_baseline(prob::AbstractProblem, args...) = error("No baseline function available for $gemm")
baseline(prob::AbstractProblem, args...) = error("No baseline function available for $gemm")


## matrix multiplication

export MatrixMultiplication, FPUMatrixMultiplication, TropicalMatrixMultiplication,
       WMMAMatrixMultiplication, BiasedWMMAMatrixMultiplication,
       DiagonalWMMAMatrixMultiplication, ComplexWMMAMatrixMultiplication, DualWMMAMatrixMultiplication

abstract type MatrixMultiplicationKind end

struct FPU <: MatrixMultiplicationKind end
struct TropicalFPU <: MatrixMultiplicationKind end

struct WMMA <: MatrixMultiplicationKind end
struct BiasedWMMA <: MatrixMultiplicationKind end
struct DiagonalWMMA <: MatrixMultiplicationKind end
struct ComplexWMMA <: MatrixMultiplicationKind end
struct DualWMMA <: MatrixMultiplicationKind end

struct MatrixMultiplication{K} <: AbstractProblem
    shape          # Named tuple containing M, N, K
    alpha          # Value of alpha
    beta           # Value of beta
    bias           # Bias vector: 1 element per column of the result matrix.
    a_type         # Type of the A matrix on host and in GMEM.
    b_type         # Type of the B matrix on host and in GMEM.
    c_type         # Type of the C matrix on host and in GMEM.
    d_type         # Type of the D matrix on host and in GMEM.
    transpose_a    # Whether or not A is transposed
    transpose_b    # Whether or not B is transposed
end

function Base.sizeof(gemm::MatrixMultiplication)
    M, N, K = gemm.shape

    return sizeof(gemm.a_type) * M * K +
           sizeof(gemm.b_type) * K * N +
           sizeof(gemm.c_type) * M * N +
           sizeof(gemm.d_type) * M * N
end

function allocate_data(gemm::MatrixMultiplication)
    M, N, K = gemm.shape

    a = CuArray{gemm.a_type}(undef, gemm.transpose_a ? (K, M) : (M, K))
    b = CuArray{gemm.b_type}(undef, gemm.transpose_b ? (N, K) : (K, N))
    c = CuArray{gemm.c_type}(undef, (M, N))
    d = CuArray{gemm.c_type}(undef, (M, N))

    return a, b, c, d
end

function initialize_data(gemm::MatrixMultiplication, a, b, c, d; seed=0)
    rng = MersenneTwister(seed)
    copy!(a, rand(rng, eltype(a), size(a)))
    copy!(b, rand(rng, eltype(b), size(b)))
    copy!(c, rand(rng, eltype(c), size(c)))
    fill!(d, zero(eltype(d)))
end

function execute(gemm::MatrixMultiplication, a, b, c, d; config, kernel)
    alpha = gemm.alpha
    beta = gemm.beta
    a_transf = (alpha == one(alpha)) ? Transform.Elementwise(identity) : Transform.Elementwise(x -> x * alpha)
    c_transf = (beta == one(beta)) ? Transform.Elementwise(identity) : Transform.Elementwise(x -> x * beta)

    epilogue = if gemm.bias === nothing
        Epilogue.Default()
    else
        Epilogue.Bias(pointer(gemm.bias))
    end

    GemmKernels.matmul(config, a, b, c, d;
                       transform_shared_to_regs_a = a_transf,
                       transform_shared_to_regs_c = c_transf,
                       epilogue, kernel)
    return d
end

function calculate_reference(gemm::MatrixMultiplication, a, b, c, d)
    copy!(d, c)
    LinearAlgebra.mul!(d,
                       gemm.transpose_a ? transpose(a) : a,
                       gemm.transpose_b ? transpose(b) : b,
                       gemm.alpha, gemm.beta)
    return d
end

function FPUMatrixMultiplication(; M, N, K, A_type, B_type, CD_type, transpose_a, transpose_b)
    compute_type = promote_type(A_type, B_type)

    MatrixMultiplication{FPU}(
        (; M, N, K),
        convert(compute_type, 2),
        convert(compute_type, 3),
        nothing,
        A_type,
        B_type,
        CD_type,
        CD_type,
        transpose_a,
        transpose_b)
end

Base.show(io::IO, gemm::MatrixMultiplication{FPU}) =
    print(io, "FPU GEMM $(gemm.a_type)*$(gemm.b_type)=$(gemm.c_type) ($(gemm.shape.M)×$(gemm.shape.K)) · ($(gemm.shape.K)×$(gemm.shape.N)) ($( !gemm.transpose_a ? 'N' : 'T' )$( !gemm.transpose_b ? 'N' : 'T' ))")

function prepare_baseline(gemm::MatrixMultiplication{FPU}, a, b, c, d)
    supported_types = Set([
            (Float16, Float16, Float32),
            (Float32, Float32, Float32),
            (Float32, Float32, Float64),
            (Float64, Float64, Float64),
            (Int16, Int16, Int16),
            (Int32, Int32, Int32),
            (Int64, Int64, Int64),
        ])
    if !haskey(supported_types, (gemm.a_type, gemm.b_type, gemm.c_type))
        error("No baseline function available for $gemm")
    end

    copyto!(d, c)

    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_DEFAULT_MATH)
    return ()
end

function execute_baseline(gemm::MatrixMultiplication{FPU}, a, b, c, d)
    CUBLAS.gemmEx!(gemm.transpose_a ? 'T' : 'N',
                   gemm.transpose_b ? 'T' : 'N',
                   gemm.alpha, a, b, gemm.beta, d)
end

function prepare(gemm::MatrixMultiplication{FPU}; BLOCK_M, BLOCK_N, BLOCK_K, OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB)
    compute_type = promote_type(gemm.a_type, gemm.b_type)
    cd_type = gemm.c_type

    config = GemmKernels.get_config(
        gemm_shape = gemm.shape,
        block_shape = (M = BLOCK_M, N = BLOCK_N, K = BLOCK_K),
        operator = Operator.FPUOp{OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB, compute_type, cd_type},
        global_a_layout = gemm.transpose_a ? Layout.UnsafeAlignedRowMajor{gemm.a_type} : Layout.UnsafeAlignedColMajor{gemm.a_type},
        global_b_layout = gemm.transpose_b ? Layout.UnsafeAlignedRowMajor{gemm.b_type} : Layout.UnsafeAlignedColMajor{gemm.b_type},

        global_c_layout = Layout.UnsafeAlignedColMajor{cd_type},
        global_d_layout = Layout.UnsafeAlignedColMajor{cd_type},

        is_a_col_major = !gemm.transpose_a,
        is_b_col_major = !gemm.transpose_b,
    )

    kernel = Kernel.matmul_pipelined

    (; config, kernel)
end

function TropicalMatrixMultiplication(; M, N, K, A_type, B_type, CD_type, transpose_a, transpose_b)
    compute_type = promote_type(A_type, B_type)

    MatrixMultiplication{TropicalFPU}(
        (; M, N, K),
        convert(compute_type, 1),
        convert(compute_type, 1),
        nothing,
        A_type,
        B_type,
        CD_type,
        CD_type,
        transpose_a,
        transpose_b,

        get_custom_mul!((a, b, c) -> max(a + b, c)))
end

function calculate_reference(gemm::MatrixMultiplication{TropicalFPU}, a, b, c, d)
    Ah = Array(a)
    Bh = Array(b)
    Ch = Array(c)

    if gemm.transpose_a
        Ah = transpose(Ah)
    end
    if gemm.transpose_b
        Bh = transpose(Bh)
    end

    for i in 1:gemm.shape.M
        for j in 1:gemm.shape.N
            res = gemm.beta * Ch[i, j]

            for k in 1:gemm.shape.K
                res = gemm.alpha * element_update(Ah[i, k], Bh[k, j], res)
            end

            Ch[i, j] = res
        end
    end

    copyto!(d, Ch)
    return d
end

Base.show(io::IO, gemm::MatrixMultiplication{TropicalFPU}) =
    print(io, "Tropical GEMM $(gemm.a_type)*$(gemm.b_type)=$(gemm.c_type) ($(gemm.shape.M)×$(gemm.shape.K)) · ($(gemm.shape.K)×$(gemm.shape.N)) ($( !gemm.transpose_a ? 'N' : 'T' )$( !gemm.transpose_b ? 'N' : 'T' ))")

function prepare(gemm::MatrixMultiplication{TropicalFPU}; OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB)
    compute_type = promote_type(gemm.a_type, gemm.b_type)
    cd_type = gemm.c_type

    config = GemmKernels.get_config(
        gemm_shape = gemm.shape,
        block_shape = (M = 64, N = 64, K = 32),
        operator = Operator.TropicalFPUOp{OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB, compute_type, cd_type},
        global_a_layout = gemm.transpose_a ? Layout.UnsafeAlignedRowMajor{gemm.a_type} : Layout.UnsafeAlignedColMajor{gemm.a_type},
        global_b_layout = gemm.transpose_b ? Layout.UnsafeAlignedRowMajor{gemm.b_type} : Layout.UnsafeAlignedColMajor{gemm.b_type},

        global_c_layout = Layout.UnsafeAlignedColMajor{cd_type},
        global_d_layout = Layout.UnsafeAlignedColMajor{cd_type},

        is_a_col_major = !gemm.transpose_a,
        is_b_col_major = !gemm.transpose_b,
    )

    kernel = Kernel.matmul_pipelined

    (; config, kernel)
end

function WMMAMatrixMultiplication(; M, N, K, AB_type, CD_type, transpose_a, transpose_b, zero_c)
    MatrixMultiplication{WMMA}(
        (; M, N, K),
        convert(AB_type, 2),
        convert(AB_type, zero_c ? 0 : 3),
        nothing,
        AB_type,
        AB_type,
        CD_type,
        CD_type,
        transpose_a,
        transpose_b)
end

Base.show(io::IO, gemm::MatrixMultiplication{WMMA}) =
    print(io, "WMMA GEMM $(gemm.a_type)*$(gemm.b_type)=$(gemm.c_type) ($(gemm.shape.M)×$(gemm.shape.K)) · ($(gemm.shape.K)×$(gemm.shape.N)) ($( !gemm.transpose_a ? 'N' : 'T' )$( !gemm.transpose_b ? 'N' : 'T' ))")

function prepare_baseline(gemm::MatrixMultiplication{WMMA}, a, b, c, d)
    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)
    copyto!(d, c)
    return ()
end

function execute_baseline(gemm::MatrixMultiplication{WMMA}, a, b, c, d)
    CUBLAS.gemmEx!(gemm.transpose_a ? 'T' : 'N',
                   gemm.transpose_b ? 'T' : 'N',
                   gemm.alpha, a, b, gemm.beta, d)
end

function prepare(gemm::MatrixMultiplication{WMMA}; BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N, OP_M, OP_N, OP_K, zero_c, kernel)
    @assert gemm.a_type == gemm.b_type
    ab_type = gemm.a_type
    @assert gemm.c_type == gemm.d_type
    cd_type = gemm.c_type

    config = GemmKernels.get_config(
        gemm_shape = gemm.shape,
        block_shape = (M = BLOCK_M, N = BLOCK_N, K = BLOCK_K),
        warps_per_block = WARPS_M * WARPS_N,

        compute_warp = (M = BLOCK_M ÷ WARPS_M, N = BLOCK_N ÷ WARPS_N, K = OP_K),

        global_a_layout = gemm.transpose_a ? Layout.UnsafeAlignedRowMajor{gemm.a_type} : Layout.UnsafeAlignedColMajor{gemm.a_type},
        global_b_layout = gemm.transpose_b ? Layout.UnsafeAlignedRowMajor{gemm.b_type} : Layout.UnsafeAlignedColMajor{gemm.b_type},
        global_c_layout = zero_c ? Layout.Zero{gemm.c_type} : Layout.UnsafeAlignedColMajor{gemm.c_type},
        global_d_layout = Layout.UnsafeAlignedColMajor{gemm.d_type},

        shared_a_layout = Layout.Padded{gemm.transpose_a ? Layout.UnsafeAlignedRowMajor{gemm.a_type} : Layout.UnsafeAlignedColMajor{gemm.a_type}, 16 ÷ sizeof(gemm.a_type)},
        shared_b_layout = Layout.Padded{gemm.transpose_b ? Layout.UnsafeAlignedRowMajor{gemm.b_type} : Layout.UnsafeAlignedColMajor{gemm.b_type}, 16 ÷ sizeof(gemm.b_type)},
        shared_c_layout = Layout.UnsafeAlignedColMajor{gemm.c_type},
        shared_d_layout = Layout.UnsafeAlignedColMajor{gemm.d_type},

        operator = Operator.WMMAOp{OP_M, OP_N, OP_K, ab_type, cd_type},

        is_a_col_major = !gemm.transpose_a,
        is_b_col_major = !gemm.transpose_b,
    )

    (; config, kernel)
end

function BiasedWMMAMatrixMultiplication(; M, N, K, AB_type, CD_type, transpose_a, transpose_b)
    # Bias vector: this vector contains 1 element per column of the result matrix.
    # This bias element is added to all elements in one column of the D matrix.
    # D is a M x N matrix, so the bias is an N-element vector.
    bias = CuArray(rand(CD_type, (1, N)))

    MatrixMultiplication{BiasedWMMA}(
        (; M, N, K),
        convert(AB_type, 1),
        convert(AB_type, 1),
        bias,
        AB_type,
        AB_type,
        CD_type,
        CD_type,
        transpose_a,
        transpose_b
    )
end

Base.show(io::IO, gemm::MatrixMultiplication{BiasedWMMA}) =
    print(io, "WMMA GEMM+bias $(gemm.a_type)*$(gemm.b_type)+$(gemm.c_type)=$(gemm.d_type) ($(gemm.shape.M)×$(gemm.shape.K)) · ($(gemm.shape.K)×$(gemm.shape.N)) ($( !gemm.transpose_a ? 'N' : 'T' )$( !gemm.transpose_b ? 'N' : 'T' ))")

function verify(gemm::MatrixMultiplication{BiasedWMMA}, A::AbstractArray{T}, dB::AbstractArray{T}) where T
    all(compare.(A .+ gemm.bias, B, T))
end

function prepare(gemm::MatrixMultiplication{BiasedWMMA}; OP_M, OP_N, OP_K)
    @assert gemm.a_type == gemm.b_type
    ab_type = gemm.a_type
    @assert gemm.c_type == gemm.d_type
    cd_type = gemm.c_type

    config = GemmKernels.get_config(
        gemm_shape = gemm.shape,
        operator = Operator.WMMAOp{OP_M, OP_N, OP_K, ab_type, cd_type},
        global_a_layout = gemm.transpose_a ? Layout.UnsafeAlignedRowMajor{gemm.a_type} : Layout.UnsafeAlignedColMajor{gemm.a_type},
        global_b_layout = gemm.transpose_b ? Layout.UnsafeAlignedRowMajor{gemm.b_type} : Layout.UnsafeAlignedColMajor{gemm.b_type},

        global_c_layout = Layout.UnsafeAlignedColMajor{gemm.c_type},
        global_d_layout = Layout.UnsafeAlignedColMajor{gemm.d_type},

        is_a_col_major = !gemm.transpose_a,
        is_b_col_major = !gemm.transpose_b,
    )

    kernel = Kernel.matmul_pipelined

    (; config, kernel)
end

function DiagonalWMMA(; M, N, K, AB_type, CD_type, transpose_a, transpose_b)
    @assert transpose_a == false
    @assert M == K "Diagonal only supports square A matrix (A == M)"

    MatrixMultiplication{DiagonalWMMA}(
        (; M, N, K),
        convert(AB_type, 1),
        convert(AB_type, 1),
        nothing,
        AB_type,
        AB_type,
        CD_type,
        CD_type,
        transpose_a,
        transpose_b)
end

function calculate_reference(gemm::MatrixMultiplication{DiagonalWMMA}, a, b, c, d)
    Ah = Array(a)
    Bh = Array(b)
    Ch = Array(c)

    if gemm.transpose_a
        Ah = transpose(Ah)
    end
    if gemm.transpose_b
        Bh = transpose(Bh)
    end

    # NOTE: the testing framework will generate an MxK A matrix,
    # but we only use the first M elements in the first column,
    # and interpret that as the diagonal elements.
    mul!(Ch, Diagonal(Ah[1:gemm.shape.M,1]), Bh, gemm.alpha, gemm.beta)

    copyto!(d, Ch)
    return d
end

Base.show(io::IO, gemm::MatrixMultiplication{DiagonalWMMA}) =
    print(io, "WMMA Diagonal GEMM diag($(gemm.a_type))*$(gemm.b_type)=$(gemm.c_type) ($(gemm.shape.M)×$(gemm.shape.K)) · ($(gemm.shape.K)×$(gemm.shape.N)) ($( !gemm.transpose_a ? 'N' : 'T' )$( !gemm.transpose_b ? 'N' : 'T' ))")

function prepare(gemm::MatrixMultiplication{DiagonalWMMA}; OP_M, OP_N, OP_K)
    @assert gemm.a_type == gemm.b_type
    ab_type = gemm.a_type
    @assert gemm.c_type == gemm.d_type
    cd_type = gemm.c_type

    config = GemmKernels.get_config(
        gemm_shape = gemm.shape,
        operator = Operator.WMMAOp{OP_M, OP_N, OP_K, ab_type, cd_type},
        global_a_layout = Layout.Diagonal{gemm.a_type},
        global_b_layout = gemm.transpose_b ? Layout.UnsafeAlignedRowMajor{gemm.b_type} : Layout.UnsafeAlignedColMajor{gemm.b_type},

        global_c_layout = Layout.UnsafeAlignedColMajor{gemm.c_type},
        global_d_layout = Layout.UnsafeAlignedColMajor{gemm.d_type},

        shared_a_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{Float16}, 8},

        is_a_col_major = !gemm.transpose_a,
        is_b_col_major = !gemm.transpose_b,
    )

    kernel = Kernel.matmul_singlestage

    (; config, kernel)
end

function ComplexWMMA(; M, N, K, AB_type, CD_type, transpose_a, transpose_b)
    MatrixMultiplication{ComplexWMMA}(
        (; M, N, K),
        convert(Complex{AB_type}, 1),
        convert(Complex{AB_type}, 1),
        nothing,
        Complex{AB_type},
        Complex{AB_type},
        Complex{CD_type},
        Complex{CD_type},
        transpose_a,
        transpose_b
    )
end

Base.show(io::IO, gemm::MatrixMultiplication{ComplexWMMA}) =
    print(io, "WMMA Complex GEMM $(gemm.a_type)*$(gemm.b_type)=$(gemm.c_type) ($(gemm.shape.M)×$(gemm.shape.K)) · ($(gemm.shape.K)×$(gemm.shape.N)) ($( !gemm.transpose_a ? 'N' : 'T' )$( !gemm.transpose_b ? 'N' : 'T' ))")

function prepare(gemm::MatrixMultiplication{ComplexWMMA}; OP_M, OP_N, OP_K)
    @assert gemm.a_type == gemm.b_type
    ab_type = gemm.a_type
    @assert gemm.c_type == gemm.d_type
    cd_type = gemm.c_type

    config = GemmKernels.get_config(
        gemm_shape = gemm.shape,
        operator = Operator.WMMAComplexOp{OP_M, OP_N, OP_K, ab_type, cd_type},

        global_a_layout = gemm.transpose_a ? Layout.InterleavedRowMajor{Float16} : Layout.InterleavedColMajor{Float16},
        global_b_layout = gemm.transpose_b ? Layout.InterleavedRowMajor{Float16} : Layout.InterleavedColMajor{Float16},
        global_c_layout = Layout.InterleavedColMajor{Float32},
        global_d_layout = Layout.InterleavedColMajor{Float32},

        shared_a_layout = gemm.transpose_a ? Layout.Padded{Layout.SplitRowMajor{Float16}, 8} : Layout.Padded{Layout.SplitColMajor{Float16}, 8},
        shared_b_layout = gemm.transpose_b ? Layout.Padded{Layout.SplitRowMajor{Float16}, 8} : Layout.Padded{Layout.SplitColMajor{Float16}, 8},
        shared_c_layout = Layout.SplitColMajor{Float32},
        shared_d_layout = Layout.SplitColMajor{Float32},

        warps_per_block = 8,

        compute_warp = (M = 16, N = 32, K = 16),

        block_shape = (M = 64, N = 64, K = 32),

        mem_a_warp = gemm.transpose_a ? (M = 4, K = 32) : (M = 64, K = 2),
        mem_b_warp = gemm.transpose_b ? (K = 2, N = 64) : (K = 32, N = 4),
        mem_cd_warp = (M = 64, N = 1),

        mem_a_thread = gemm.transpose_a ? (M = 1, K = 4) : (M = 4, K = 1),
        mem_b_thread = gemm.transpose_b ? (K = 1, N = 4) : (K = 4, N = 1),
        mem_cd_thread = (M = 2, N = 1),

        is_a_col_major = !gemm.transpose_a,
        is_b_col_major = !gemm.transpose_b
    )

    kernel = Kernel.matmul_pipelined

    (; config, kernel)
end

function DualWMMAMatrixMultiplication(; M, N, K, AB_type, CD_type, transpose_a, transpose_b)
    MatrixMultiplication{DualWMMA}(
        (; M, N, K),
        convert(Complex{AB_type}, 1),
        convert(Complex{AB_type}, 1),
        nothing,
        Complex{AB_type},
        Complex{AB_type},
        Complex{CD_type},
        Complex{CD_type},
        transpose_a,
        transpose_b,
    )
end

function calculate_reference(gemm::MatrixMultiplication{DualWMMA}, a, b, c, d)
    dual_conv(M) = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, M)

    copy!(d, c)

    mul!(dual_conv(d), dual_conv(Complex{Float32}.(a)), dual_conv(Complex{Float32}.(b)),
         gemm.alpha, gemm.beta)
    return d
end

Base.show(io::IO, gemm::MatrixMultiplication{DualWMMA}) =
    print(io, "WMMA Dual GEMM $(gemm.a_type)*$(gemm.b_type)=$(gemm.c_type) ($(gemm.shape.M)×$(gemm.shape.K)) · ($(gemm.shape.K)×$(gemm.shape.N)) ($( !gemm.transpose_a ? 'N' : 'T' )$( !gemm.transpose_b ? 'N' : 'T' ))")

function verify(gemm::MatrixMultiplication{DualWMMA}, A::AbstractArray{T}, B::AbstractArray{T}) where T
    A_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, A)
    B_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, B)
    all(compare.(A_dual, B_dual, T))
end

function prepare(gemm::MatrixMultiplication{DualWMMA}; OP_M, OP_N, OP_K)
    @assert gemm.a_type == gemm.b_type
    ab_type = gemm.a_type
    @assert gemm.c_type == gemm.d_type
    cd_type = gemm.c_type

    config = GemmKernels.get_config(
        gemm_shape = gemm.shape,
        operator = Operator.WMMADualOp{OP_M, OP_N, OP_K, ab_type, cd_type},

        global_a_layout = Layout.InterleavedColMajor{Float16},
        global_b_layout = Layout.InterleavedColMajor{Float16},
        global_c_layout = Layout.InterleavedColMajor{Float32},
        global_d_layout = Layout.InterleavedColMajor{Float32},

        shared_a_layout = Layout.Padded{Layout.SplitColMajor{Float16}, 8},
        shared_b_layout = Layout.Padded{Layout.SplitColMajor{Float16}, 8},
        shared_c_layout = Layout.SplitColMajor{Float32},
        shared_d_layout = Layout.SplitColMajor{Float32},

        warps_per_block = 8,

        compute_warp = (M = 16, N = 32, K = 16),

        block_shape = (M = 64, N = 64, K = 32),

        mem_a_warp = (M = 64, K = 2),
        mem_b_warp = (K = 32, N = 4),
        mem_cd_warp = (M = 64, N = 1),

        mem_a_thread = (M = 4, K = 1),
        mem_b_thread = (K = 4, N = 1),
        mem_cd_thread = (M = 2, N = 1)
    )

    kernel = Kernel.matmul_pipelined

    (; config, kernel)
end


## tensor contraction

export TensorContraction, WMMATensorContraction

struct TensorContraction{K} <: AbstractProblem
    name             # A parseable name for the configuration.
    modes            # The modes of the input tensors.
    extents          # The extents of the input tensors.

    alpha            # Value of alpha
    beta             # Value of beta
    a_type           # Type of the A tensor on host and in GMEM.
    b_type           # Type of the B tensor on host and in GMEM.
    c_type           # Type of the C tensor on host and in GMEM.
    d_type           # Type of the D tensor on host and in GMEM.
    compute_type     # The type to compute in.
    accumulate_type  # The type to accumulate into.
end

Base.show(io::IO, tc::TensorContraction) = print(io, "TC $(tc.name)")

function old_pad_extents(extents, modes, multiple=512)
    padded_extents = [extents...]
    for (idx1, idx2) in [(1, 2), (3, 2), (1, 3)]
        intersection = intersect(modes[idx1], modes[idx2])

        if prod(extents[intersection]) % multiple == 0
            continue
        end

        extent_to_pad = argmax(extents[intersection] .% multiple)

        padded_extents[intersection[extent_to_pad]] =
            Int64(ceil(extents[intersection[extent_to_pad]] / multiple) *
            multiple)
    end
    return Tuple(padded_extents)
end

# Exponent of 2 in the prime factorisation of n.
function number_of_two_factors_in_prime_fact(n::Integer)
    rv = 0

    while n % 2 == 0
        rv += 1
        n /= 2
    end

    rv
end

# Increase number to introduce an extra factor of 2 in the prime factorisation.
function increment_to_increase_power_of_two_in_fact(n::Integer)
    round_to_multiple_of = 2 ^ (number_of_two_factors_in_prime_fact(n) + 1)
    Int(ceil(n / round_to_multiple_of) * round_to_multiple_of)
end

# Sequence that increases the exponent of power of two starting from given number, and continuing until the point where we have at least a multiple of multiple_to_stop_at.
function sequence_increasing_power_two(start::Integer, multiple_to_stop_at::Int)
    sequence = [start]

    while sequence[end] % multiple_to_stop_at != 0
        push!(sequence, increment_to_increase_power_of_two_in_fact(sequence[end]))
    end

    sequence
end

# Pad (extents_1, extents_2, ...) to (extents_1 + padding_1, extents_2 + padding_2, ...)
# such that (extents_1 + padding_1) * (extents_2 + padding_2) * ... is a multiple of multiple.
function find_minimal_padding(extents, multiple)
    # For now, let's do a "smart" brute force search. I think we can do this
    # iteratively, by increasing a particular extent to the next multiple of
    # the smallest power of two that does not divide the extent. e.g. a
    # dimension 72 (which is a multiple of 8, but not 16), will be padded to
    # the next multiple of 16, i.e. 96. In each iteration, we choose the extent
    # that requires the least amount of padding. I'm not convinced yet of the
    # optimality of that approach, though, and since the search space is small,
    # anyway, we'll just exhaustively search all options, albeit in a smart
    # way.

    # It only makes sense to increase the extent to a value that increases the
    # power of 2 in the prime factorisation of the extent. We also only need to
    # do this up to the smallest power of two that ensures that the product is
    # a multiple regardless of the choice of the padding for the other
    # dimensions. E.g. to pad (72, 72) to a multiple of 512, we only need to
    # consider increasing the padding up to multiples of 2^6, since 72 is a
    # multiple of 2^3, so whatever we choose for the padding of the second
    # dimension, it will be a multiple of 2^3, so we need at most 6 other
    # 2-factors for the first dimension to get a multiple of 512=2^9.

    res = collect(Iterators.filter(x -> prod(x) % multiple == 0, Iterators.product(Configs.sequence_increasing_power_two.(extents, multiple)...)));
    _, idx = findmin(prod, res)
    res[idx]
end

function pad_extents(extents, modes, multiples=(M=512,N=512,K=512))
    padded_extents = [extents...]

    for (idx1, idx2, multiple) in [(1, 2, multiples.M), (3, 2, multiples.K), (1, 3, multiples.N)]
        # Set of dimensions that contribute to e.g. M
        intersection = intersect(modes[idx1], modes[idx2])

        # Pad those dimensions
        padded_extents[intersection] .= find_minimal_padding(padded_extents[intersection], multiple)
    end

    Tuple(padded_extents)
end

function padded_view(x::AbstractArray, dims)
    # first, get a contiguous view with the destination size
    y = view(x, 1:prod(dims))

    # then, reshape into ND
    return reshape(y, dims)
end

function Base.sizeof(tc::TensorContraction)
    padded_extents = pad_extents(tc.extents, tc.modes)
    return sizeof(tc.a_type) * prod(padded_extents[tc.modes[2]]) +
           sizeof(tc.b_type) * prod(padded_extents[tc.modes[3]]) +
           sizeof(tc.c_type) * prod(padded_extents[tc.modes[1]]) +
           sizeof(tc.d_type) * prod(padded_extents[tc.modes[1]])
end

function allocate_data(tc::TensorContraction)
    padded_extents = pad_extents(tc.extents, tc.modes)
    a = CuArray{tc.a_type}(undef, padded_extents[tc.modes[2]])
    b = CuArray{tc.b_type}(undef, padded_extents[tc.modes[3]])
    c = CuArray{tc.c_type}(undef, padded_extents[tc.modes[1]])
    d = CuArray{tc.d_type}(undef, padded_extents[tc.modes[1]])

    return a, b, c, d
end

# fill a tensor with a pattern
function pattern!(a::AbstractArray{T}, start=1, increment=1) where T
    function pattern_kernel(start, increment)
        i = threadIdx().x + (blockIdx().x - 1)*blockDim().x
        stride = blockDim().x * gridDim().x
        val = start + (i-1) * increment
        while i <= length(a)
            a[i] = val
            i += stride
            val += increment * stride
        end
        return
    end

    start = convert(T, start)
    increment = convert(T, increment)

    kernel = @cuda launch=false pattern_kernel(start, increment)
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, length(a))
    blocks = min(config.blocks, cld(length(a), threads))
    kernel(start, increment; threads, blocks)
end

function initialize_data(tc::TensorContraction, a, b, c, d; kwargs...)
    if isempty(kwargs)
        # initialize data
        rng = CUDA.RNG(0)
        rand!(rng, a)
        rand!(rng, b)
        rand!(rng, c)
        fill!(d, 0)
    else
        # use the params to get appropriately padded tensors
        padding_multiple = max(kwargs[:BLOCK_M], kwargs[:BLOCK_N], kwargs[:BLOCK_K])
        padded_extents = pad_extents(tc.extents, tc.modes, (M = kwargs[:BLOCK_M], N = kwargs[:BLOCK_N], K = kwargs[:BLOCK_K]))
        padded_a = padded_view(a, padded_extents[tc.modes[2]])
        padded_b = padded_view(b, padded_extents[tc.modes[3]])
        padded_c = padded_view(c, padded_extents[tc.modes[1]])
        padded_d = padded_view(d, padded_extents[tc.modes[1]])

        # set the padding to 0
        fill!(padded_a, 0)
        fill!(padded_b, 0)
        fill!(padded_c, 0)

        # initialize the actual data
        data_a = view(padded_a, ntuple(i->1:tc.extents[tc.modes[2]][i], ndims(a))...)
        data_b = view(padded_b, ntuple(i->1:tc.extents[tc.modes[3]][i], ndims(b))...)
        data_c = view(padded_c, ntuple(i->1:tc.extents[tc.modes[1]][i], ndims(c))...)
        data_d = view(padded_d, ntuple(i->1:tc.extents[tc.modes[1]][i], ndims(d))...)
        initialize_data(tc, data_a, data_b, data_c, data_d)
    end
end

function calculate_reference(tc::TensorContraction, a, b, c, d)
    # use unpadded buffers
    let a = padded_view(a, tc.extents[tc.modes[2]]),
        b = padded_view(b, tc.extents[tc.modes[3]]),
        c = padded_view(c, tc.extents[tc.modes[1]]),
        d = padded_view(d, tc.extents[tc.modes[1]])

        # re-initialize the data; this is needed as we don't use the full buffer
        initialize_data(tc, a, b, c, d)

        copy!(d, c)

        plan = cuTENSOR.plan_contraction(
            a, tc.modes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
            b, tc.modes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
            d, tc.modes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
            cuTENSOR.CUTENSOR_OP_IDENTITY,
            algo=cuTENSOR.CUTENSOR_ALGO_GETT,
            compute_type=tc.accumulate_type
        )

        cuTENSOR.contract!(
            tc.alpha,
            a, tc.modes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
            b, tc.modes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
            tc.beta,
            d, tc.modes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
            cuTENSOR.CUTENSOR_OP_IDENTITY;
            compute_type=tc.accumulate_type,
            plan
        )

        return d
    end
end

function prepare_baseline(tc::TensorContraction, a, b, c, d)
    # use unpadded buffers
    let a = padded_view(a, tc.extents[tc.modes[2]]),
        b = padded_view(b, tc.extents[tc.modes[3]]),
        c = padded_view(c, tc.extents[tc.modes[1]]),
        d = padded_view(d, tc.extents[tc.modes[1]])

        copy!(d, c)

        plan = cuTENSOR.plan_contraction(
            a, tc.modes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
            b, tc.modes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
            d, tc.modes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
            cuTENSOR.CUTENSOR_OP_IDENTITY,
            algo=cuTENSOR.CUTENSOR_ALGO_GETT,
            compute_type=tc.accumulate_type
        )

        (; plan)
    end
end

function execute_baseline(tc::TensorContraction, a, b, c, d; plan)
    # use unpadded buffers
    let a = padded_view(a, tc.extents[tc.modes[2]]),
        b = padded_view(b, tc.extents[tc.modes[3]]),
        c = padded_view(c, tc.extents[tc.modes[1]]),
        d = padded_view(d, tc.extents[tc.modes[1]])

        cuTENSOR.contract!(
            tc.alpha,
            a, tc.modes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
            b, tc.modes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
            tc.beta,
            d, tc.modes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
            cuTENSOR.CUTENSOR_OP_IDENTITY;
            compute_type=tc.accumulate_type,
            plan
        )
    end
end

function WMMATensorContraction(; name, extents, data_type, compute_type, accumulate_type, zero_c)
    # Parsing the name into a 3D vector of the modes of each tensor.
    modes = Vector{Vector{Int}}(undef, 0)
    for tensor in split(name, "-")
        tensorMode = Vector{Int}(undef, 0)

        for mode in split(tensor, ".")
            push!(tensorMode, parse(Int, mode))
        end

        push!(modes, tensorMode)
    end

    TensorContraction{WMMA}(
        name,
        modes,
        Tuple(extents),

        convert(compute_type, 2),
        convert(compute_type, zero_c ? 0 : 3),
        data_type,
        data_type,
        data_type,
        data_type,
        compute_type,
        accumulate_type
    )
end

function prepare(tc::TensorContraction, a, b, c, d;
                                        BLOCK_M, BLOCK_N, BLOCK_K,
                                        WARPS_M, WARPS_N,
                                        OP_M, OP_N, OP_K,
                                        kernel,
                                        is_A_col_major, is_B_col_major, is_D_col_major,
                                        PERM_M, PERM_N, PERM_K)
    @assert tc.a_type == tc.b_type == tc.c_type == tc.d_type
    data_type = tc.a_type

    # get padded tensors
    padded_extents = pad_extents(tc.extents, tc.modes, (M = BLOCK_M, N = BLOCK_N, K = BLOCK_K))
    padded_a = padded_view(a, padded_extents[tc.modes[2]])
    padded_b = padded_view(b, padded_extents[tc.modes[3]])
    padded_c = padded_view(c, padded_extents[tc.modes[1]])
    padded_d = padded_view(d, padded_extents[tc.modes[1]])

    # get underlying output data to return to the caller
    data_d = view(padded_d, ntuple(i->1:tc.extents[tc.modes[1]][i], ndims(d))...)

    a_extent = padded_extents[tc.modes[2]]
    a_desc = Tensors.TensorDescriptor(
        length(a_extent), collect(Int, a_extent), collect(Int, cumprod((1, a_extent...))[1:end-1]), data_type, identity
    )
    b_extent = padded_extents[tc.modes[3]]
    b_desc = Tensors.TensorDescriptor(
        length(b_extent), collect(Int, b_extent), collect(Int, cumprod((1, b_extent...))[1:end-1]), data_type, identity
    )
    c_extent = padded_extents[tc.modes[1]]
    c_desc = Tensors.TensorDescriptor(
        length(c_extent), collect(Int, c_extent), collect(Int, cumprod((1, c_extent...))[1:end-1]), data_type, identity
    )

    GemmKernels.Tensors.OVERRIDE_do_override = true
    GemmKernels.Tensors.OVERRIDE_is_A_col_major = is_A_col_major
    GemmKernels.Tensors.OVERRIDE_is_B_col_major = is_B_col_major
    GemmKernels.Tensors.OVERRIDE_is_D_col_major = is_D_col_major
    GemmKernels.Tensors.OVERRIDE_perm_M = PERM_M
    GemmKernels.Tensors.OVERRIDE_perm_N = PERM_N
    GemmKernels.Tensors.OVERRIDE_perm_K = PERM_K

    plan = Tensors.ContractionPlan(
        a_desc, tc.modes[2],
        b_desc, tc.modes[3],
        c_desc, tc.modes[1],
        c_desc, tc.modes[1];
        operator=Operator.WMMAOp{OP_M, OP_N, OP_K, tc.compute_type, tc.accumulate_type},
        computeType=tc.compute_type,
        accumulateType=tc.accumulate_type,
        blockShape=(M = BLOCK_M, N = BLOCK_N, K = BLOCK_K),
        warpsPerBlock=WARPS_M * WARPS_N,
        computeWarp=(M = BLOCK_M ÷ WARPS_M, N = BLOCK_N ÷ WARPS_N, K = OP_K),
    )

    (; plan, kernel, padded_a, padded_b, padded_c, padded_d, data_d)
end

function execute(tc::TensorContraction, a, b, c, d; plan, kernel,
                                        padded_a, padded_b, padded_c, padded_d, data_d)
    Tensors.contraction!(plan, tc.alpha, padded_a, padded_b, tc.beta, padded_c, padded_d;
                         kernel)
    return data_d
end


## iterate all configs

function get_configs()
    rv = []

    # FPU Op
    for (A_type, B_type, CD_type) in [
            (Float16, Float16, Float32),
            (Float32, Float32, Float32),
            (Float32, Float32, Float64),
            (Float64, Float64, Float64),
            (Int16, Int16, Int16),
            (Int32, Int32, Int32),
            (Int64, Int64, Int64)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        N in [128, 256, 2048]

        # XXX: Should we do non-square matrices as well?
        M = K = N

        gemm = FPUMatrixMultiplication(; M, N, K, A_type, B_type, CD_type, transpose_a, transpose_b)

        for (OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB) in [(8, 16, 2, 4, 8, 1)],
            (BLOCK_M, BLOCK_N, BLOCK_K) in [(64, 64, 32)]

            try
                params = prepare(gemm; BLOCK_M, BLOCK_N, BLOCK_K, OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB)
                push!(rv, (; gemm, params...))
            catch err
                isa(err, GemmKernels.ConfigError) || rethrow()
            end
        end
    end

    # FPU Op shapes
    for (A_type, B_type, CD_type) in [
            (Float32, Float32, Float32)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        N in [128]

        # We'll only test square matrices.
        M = K = N

        gemm = FPUMatrixMultiplication(; M, N, K, A_type, B_type, CD_type, transpose_a, transpose_b)

        for (OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB) in vcat(
                # First, test some shapes with the default base shape (4, 8, 1).
                map(tup -> (tup..., 4, 8, 1),
                [( 4, 8,  1),
                ( 8, 8,  1),
                ( 4, 16, 1),
                ( 4, 8,  2),
                ( 8, 16, 2)]),
                # Then, test some different combinations of op shape + base shape.
                [(4,  32, 1, 1,  32, 1),
                (4,  32, 1, 2,  16, 1),
                (16, 16, 1, 4,  8,  1),
                (16, 16, 1, 8,  4,  1),
                (32, 4,  1, 16, 2,  1),
                (32, 4,  1, 32, 1,  1)]),
            (BLOCK_M, BLOCK_N, BLOCK_K) in [(128, 64, 32)]

            try
                params = prepare(gemm; BLOCK_M, BLOCK_N, BLOCK_K, OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB)
                push!(rv, (; gemm, params...))
            catch err
                isa(err, GemmKernels.ConfigError) || rethrow()
            end
        end
    end

    # Tropical GEMM
    for (A_type, B_type, CD_type, min_dimension) in [
            (Float32, Float32, Float32, 128)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (M, N, K) in min_dimension .* [
            [1, 1, 1],
            [2, 2, 1],
            [1, 1, 2],
            [2, 2, 2]]

        gemm = TropicalMatrixMultiplication(; M, N, K, A_type, B_type, CD_type, transpose_a, transpose_b)

        for (OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB) in [(8, 16, 2, 4, 8, 1)]

            try
                params = prepare(gemm; OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB)
                push!(rv, (; gemm, params...))
            catch err
                isa(err, GemmKernels.ConfigError) || rethrow()
            end
        end
    end

    # WMMA GEMM
    for (AB_type, CD_type, min_dimension) in [
        (Float16, Float16, 256),
        (Float16, Float32, 128)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (M, N, K) in vcat(min_dimension .* [
            [1, 1, 1],
            [2, 2, 1],
            [1, 1, 2],
            [2, 2, 2]], [[2048, 2048, 2048]]),
        zero_c in [false]

        gemm = WMMAMatrixMultiplication(; M, N, K, AB_type, CD_type, transpose_a, transpose_b, zero_c)

        for (BLOCK_M, BLOCK_N, BLOCK_K) in [(128, 128, 64)],
            (WARPS_M, WARPS_N) in [(4, 2)],
            (OP_M, OP_N, OP_K) in [
                (16, 16, 16),
                (8, 32, 16),
                (32, 8, 16),
            ],
            kernel in [Kernel.matmul_pipelined]

            params = prepare(gemm; BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N, OP_M, OP_N, OP_K, zero_c, kernel)
            push!(rv, (; gemm, params...))
        end
    end

    # WMMA GEMM parameters
    for (M, N, K) in [(256, 256, 256)],
        (AB_type, CD_type) in [(Float16, Float32)],
        transpose_a in [false, true],
        transpose_b in [false, true],
        zero_c in [false, true]

        gemm = WMMAMatrixMultiplication(; M, N, K, AB_type, CD_type, transpose_a, transpose_b, zero_c)

        for (BLOCK_M, BLOCK_N, BLOCK_K) in filter(x -> prod(x[1:2]) <= 128*128, collect(Iterators.product([64, 128, 256], [64, 128, 256], [16, 32, 64]))[:]),
            (WARPS_M, WARPS_N) in filter(x -> prod(x) >= 4, collect(Iterators.product([1, 2, 4], [1, 2, 4]))[:]),
            (OP_M, OP_N, OP_K) in [(16, 16, 16)],
            kernel in [Kernel.matmul_singlestage, Kernel.matmul_pipelined]

            try
                params = prepare(gemm; BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N, OP_M, OP_N, OP_K, zero_c, kernel)
                push!(rv, (; gemm, params...))
            catch err
                isa(err, GemmKernels.ConfigError) || rethrow()
            end
        end
    end

    # WMMA GEMM + bias
    for (AB_type, CD_type, min_dimension) in [
        (Float16, Float32, 128)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (M, N, K) in vcat(min_dimension .* [
            [1, 1, 1],
            [2, 2, 2]], [[4096, 4096, 4096]])

        gemm = BiasedWMMAMatrixMultiplication(; M, N, K, AB_type, CD_type, transpose_a, transpose_b)

        for (OP_M, OP_N, OP_K) in [
                (16, 16, 16),
                (8, 32, 16),
                (32, 8, 16),
            ]

            try
                params = prepare(gemm; OP_M, OP_N, OP_K)
                push!(rv, (; gemm, params...))
            catch err
                isa(err, GemmKernels.ConfigError) || rethrow()
            end
        end
    end

    # WMMA Diagonal GEMM
    for (AB_type, CD_type, min_dimension) in [
        (Float16, Float32, 128)],
        transpose_a = [false],
        transpose_b = [false, true],
        (M, N, K) in vcat(min_dimension .* [
            [1, 1, 1],
            [2, 2, 2]], [[4096, 4096, 4096]])

        gemm = DiagonalWMMA(; M, N, K, AB_type, CD_type, transpose_a, transpose_b)

        for (OP_M, OP_N, OP_K) in [
                (16, 16, 16),
                (8, 32, 16),
                (32, 8, 16),
            ]

            try
                params = prepare(gemm; OP_M, OP_N, OP_K)
                push!(rv, (; gemm, params...))
            catch err
                isa(err, GemmKernels.ConfigError) || rethrow()
            end
        end
    end

    # WMMA Complex GEMM
    for (AB_type, CD_type) in [(Float16, Float32)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (M, N, K) in [
            (128, 128, 128),
            (256, 256, 256),
            (2048, 2048, 2048)]

        gemm = ComplexWMMA(; M, N, K, AB_type, CD_type, transpose_a, transpose_b)

        for (OP_M, OP_N, OP_K) in [
                (16, 16, 16),
                (8, 32, 16),
                (32, 8, 16),
            ]

            try
                params = prepare(gemm; OP_M, OP_N, OP_K)
                push!(rv, (; gemm, params...))
            catch err
                isa(err, GemmKernels.ConfigError) || rethrow()
            end
        end
    end

    # WMMA Dual GEMM
    for (AB_type, CD_type) in [(Float16, Float32)],
        transpose_a = [false],
        transpose_b = [false],
        (M, N, K) in [
            (128, 128, 128),
            (256, 256, 256),
            (2048, 2048, 2048)]
        gemm = DualWMMAMatrixMultiplication(; M, N, K, AB_type, CD_type, transpose_a, transpose_b)

        for (OP_M, OP_N, OP_K) in [
                (16, 16, 16),
                (8, 32, 16),
                (32, 8, 16),
            ]

            try
                params = prepare(gemm; OP_M, OP_N, OP_K)
                push!(rv, (; gemm, params...))
            catch err
                isa(err, GemmKernels.ConfigError) || rethrow()
            end
        end
    end

    rv
end

end
using .Configs
