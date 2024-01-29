# List of configurations to use for testing and benchmarking.

module Configs

export Configuration, get_configs, generate_inputs, run_gemm, run_baseline, verify,
       @get_fpu_config, @get_tropical_config, @get_wmma_config, @get_wmma_bias_config,
       @get_wmma_diagonal_config, @get_wmma_complex_config, @get_wmma_dual_config,
       get_configs

## lazy module loading

using CUDA
using cuTENSOR
using GemmKernels
using GemmKernels.Tensors
using LinearAlgebra

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

struct Configuration
    name           # Human-readable name of the configuration.
    config         # GemmKernels.Config instance to use.
    alpha          # Value of alpha
    beta           # Value of beta
    a_type         # Type of the A matrix on host and in GMEM.
    b_type         # Type of the B matrix on host and in GMEM.
    c_type         # Type of the C matrix on host and in GMEM.
    d_type         # Type of the D matrix on host and in GMEM.
    transpose_a    # Whether or not A is transposed
    transpose_b    # Whether or not B is transposed
    calc_reference # mul!-like function to calculate the reference
    epilogue       # The epilogue to use.
    verify         # Verify function to use.
    kernel         # The kernel function to use.
    baseline       # Baseline implementation to compare performance against
end

struct ContractionConfiguration
    name           # Human-readable name of the configuration.
    plan           # GemmKernels.Config instance to use.
    alpha          # Value of alpha
    beta           # Value of beta
    a_type         # Type of the A tensor on host and in GMEM.
    b_type         # Type of the B tensor on host and in GMEM.
    c_type         # Type of the C tensor on host and in GMEM.
    d_type         # Type of the D tensor on host and in GMEM.
    transpose_a    # Whether or not A is transposed
    transpose_b    # Whether or not B is transposed
    epilogue       # The epilogue to use.
    verify         # Verify function to use.
    kernel         # The kernel function to use.
    baseline       # Baseline implementation to compare performance against
    extents
    padded_extents
    tensorModes
    accumulate_type
end

function get_custom_mul!(element_update)
    (C, A, B, alpha, beta) -> begin
        M = size(C, 1)
        N = size(C, 2)
        K = size(A, 2)

        # XXX: this assumes CPU execution
        Ah = Array(A)
        Bh = Array(B)
        Ch = Array(C)

        for i in 1:M
            for j in 1:N
                res = beta * Ch[i, j]

                for k in 1:K
                    res = alpha * element_update(Ah[i, k], Bh[k, j], res)
                end

                Ch[i, j] = res
            end
        end

        copyto!(C, Ch)
    end
end

# Generate input matrices.
function generate_inputs(cf::Configuration)
    M = cf.config.matmul_shape.M
    N = cf.config.matmul_shape.N
    K = cf.config.matmul_shape.K

    a = CuArray{cf.a_type}(undef, cf.transpose_a ? (K, M) : (M, K))
    b = CuArray{cf.b_type}(undef, cf.transpose_b ? (N, K) : (K, N))
    c = CuArray{cf.c_type}(undef, (M, N))
    d = CuArray{cf.c_type}(undef, (M, N))

    function reference_mul!(c, a, b)
        # mul! determines transpose from the type of the matrix
        (cf.calc_reference)(c,
                            cf.transpose_a ? transpose(a) : a,
                            cf.transpose_b ? transpose(b) : b,
                            cf.alpha, cf.beta)
    end

    return reference_mul!, a, b, c, d
end

function generate_inputs_tc(cf::ContractionConfiguration)
    a_h = rand(cf.a_type, cf.extents[cf.tensorModes[2]])
    b_h = rand(cf.b_type, cf.extents[cf.tensorModes[3]])
    c_h = rand(cf.c_type, cf.extents[cf.tensorModes[1]])

    # Prep cuTENSOR ref solution.
    a = CuArray(a_h)
    b = CuArray(b_h)
    c = CuArray(c_h)

    plan = cuTENSOR.plan_contraction(
        a, cf.tensorModes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
        b, cf.tensorModes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
        c, cf.tensorModes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cuTENSOR.CUTENSOR_OP_IDENTITY,
        algo=cuTENSOR.CUTENSOR_ALGO_GETT,
        compute_type=cf.accumulate_type
    )

    cuTENSOR.contract!(
        cf.alpha,
        (a), cf.tensorModes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
        (b), cf.tensorModes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cf.beta,
        (c), cf.tensorModes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cuTENSOR.CUTENSOR_OP_IDENTITY,
        compute_type=cf.accumulate_type,
        plan=plan
    )

    c_ref = Array(c)
    CUDA.unsafe_free!(a)
    CUDA.unsafe_free!(b)
    CUDA.unsafe_free!(c)

    # Prep GemmKernels.jl padded inputs.
    a = CuArray(zeros(cf.a_type, cf.padded_extents[cf.tensorModes[2]]))
    b = CuArray(zeros(cf.b_type, cf.padded_extents[cf.tensorModes[3]]))
    c = CuArray(zeros(cf.c_type, cf.padded_extents[cf.tensorModes[1]]))
    d = CuArray(zeros(cf.d_type, cf.padded_extents[cf.tensorModes[1]]))

    a[(1:extent for extent in cf.extents[cf.tensorModes[2]])...] = a_h
    b[(1:extent for extent in cf.extents[cf.tensorModes[3]])...] = b_h
    c[(1:extent for extent in cf.extents[cf.tensorModes[1]])...] = c_h

    c_ref, a, b, c, d
end

# Run the GEMM.
function run_gemm(cf::Configuration, a, b, c, d)
    alpha = cf.alpha
    beta = cf.beta
    a_transf = (alpha == one(alpha)) ? Transform.Elementwise(identity) : Transform.Elementwise(x -> x * alpha)
    c_transf = (beta == one(beta)) ? Transform.Elementwise(identity) : Transform.Elementwise(x -> x * beta)

    GemmKernels.matmul(cf.config, a, b, c, d;
                       transform_shared_to_regs_a = a_transf,
                       transform_shared_to_regs_c = c_transf,
                       epilogue = cf.epilogue,
                       kernel = cf.kernel)
end

function run_tc(cf::ContractionConfiguration, a, b, c, d)
    Tensors.contraction!(cf.plan, cf.alpha, a, b, cf.beta, c, d)
end

# Run the baseline.
function run_baseline(cf::Configuration, a, b, c, d)
    @assert !isnothing(cf.baseline)
    cf.baseline(a, b, c, d, cf.alpha, cf.beta, cf.transpose_a, cf.transpose_b)
end

function run_baseline(cf::ContractionConfiguration, a, b, c, d)
    @assert !isnothing(cf.baseline)
    cf.baseline(cf, a, b, c, d, cf.alpha, cf.beta, cf.transpose_a, cf.transpose_b)
end

# Verify results.
function verify(cf::Configuration, c_ref, d)
    cf.verify(c_ref, d, cf.a_type)
end

compare(x, y, T) = error("Unimplemented compare(x, y, T) function for type $T")
compare(x, y, T::Type{<:AbstractFloat}) = isapprox(x, y; rtol=sqrt(eps(T)))
compare(x, y, T::Type{<:Integer}) = (x == y)
compare(x, y, T::Type{Complex{U}}) where {U} = compare(x, y, U)

function verify_default(c_ref, d, T)
    all(compare.(c_ref, d, T))
end

function verify(cf::ContractionConfiguration, c_ref, d)
    d_h = Array(d[(1:extent for extent in cf.extents[cf.tensorModes[1]])...])
    cf.verify(c_ref, d_h)
end

function verify_bias(c_ref, d, bias, T)
    all(compare.(c_ref .+ bias, d, T))
end

function verify_dual(c_ref, d, T)
    c_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, c_ref)
    d_dual = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, d)
    all(compare.(c_dual, d_dual, T))
end

function fpu_baseline(a, b, c, d, alpha, beta, transpose_a, transpose_b)
    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_DEFAULT_MATH)
    CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, a, b, beta, c)
end

function wmma_baseline(a, b, c, d, alpha, beta, transpose_a, transpose_b)
    CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)
    CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, a, b, beta, c)
end

function cublas_mul!(c, a, b, alpha, beta)
    # minimalistic version of mul!, without ever falling back to GPUCompiler.jl
    transpose_a = a isa Adjoint || a isa Transpose
    transpose_b = b isa Adjoint || b isa Transpose
    CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T',
                    alpha, parent(a), parent(b), beta, c)
    c
end

function tc_baseline(cf, a, b, c, d, alpha, beta, transpose_a, transpose_b)
    plan = cuTENSOR.plan_contraction(
        a, cf.tensorModes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
        b, cf.tensorModes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
        c, cf.tensorModes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cuTENSOR.CUTENSOR_OP_IDENTITY,
        algo=cuTENSOR.CUTENSOR_ALGO_GETT,
        compute_type=cf.accumulate_type
    )

    cuTENSOR.contract!(
        alpha,
        (a), cf.tensorModes[2], cuTENSOR.CUTENSOR_OP_IDENTITY,
        (b), cf.tensorModes[3], cuTENSOR.CUTENSOR_OP_IDENTITY,
        beta,
        (c), cf.tensorModes[1], cuTENSOR.CUTENSOR_OP_IDENTITY,
        cuTENSOR.CUTENSOR_OP_IDENTITY,
        compute_type=cf.accumulate_type,
        plan=plan
    )
end

macro get_fpu_config()
    esc(quote let
        baseline_func = Dict(
                (Float16, Float16, Float32) => $fpu_baseline,
                (Float32, Float32, Float32) => $fpu_baseline,
                (Float32, Float32, Float64) => nothing,
                (Float64, Float64, Float64) => $fpu_baseline,
                (Int16, Int16, Int16) => nothing,
                (Int32, Int32, Int32) => nothing,
                (Int64, Int64, Int64) => nothing,
            )[(A_type, B_type, CD_type)]

        compute_type = promote_type(A_type, B_type)

        conf = GemmKernels.get_config(
                gemm_shape = (M = M, N = N, K = K),
                block_shape = (M = BLOCK_M, N = BLOCK_N, K = BLOCK_K),
                operator = Operator.FPUOp{OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB, compute_type, CD_type},
                global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{A_type} : Layout.UnsafeAlignedColMajor{A_type},
                global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{B_type} : Layout.UnsafeAlignedColMajor{B_type},

                global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                is_a_col_major = !transpose_a,
                is_b_col_major = !transpose_b,
        )

        name = "FPU GEMM $(A_type)*$(B_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K)), base shape ($(OP_MB), $(OP_NB), $(OP_KB))"

        Configuration(name,
                      conf,
                      convert(compute_type, 2),
                      convert(compute_type, 3),
                      A_type,
                      B_type,
                      CD_type,
                      CD_type,
                      transpose_a,
                      transpose_b,
                      $LinearAlgebra.mul!,
                      Epilogue.Default(),
                      $verify_default,
                      Kernel.matmul_pipelined,
                      baseline_func)
    end end)
end

macro get_tropical_config()
    esc(quote let
        name = "Tropical GEMM $(A_type)*$(B_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K)), base shape ($(OP_MB), $(OP_NB), $(OP_KB))"
        compute_type = promote_type(A_type, B_type)

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        block_shape = (M = 64, N = 64, K = 32),
                                        operator = Operator.TropicalFPUOp{OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB, compute_type, CD_type},
                                        global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{A_type} : Layout.UnsafeAlignedColMajor{A_type},
                                        global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{B_type} : Layout.UnsafeAlignedColMajor{B_type},

                                        global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                        global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        Configuration(name,
                      conf,
                      convert(compute_type, 1),
                      convert(compute_type, 1),
                      A_type,
                      B_type,
                      CD_type,
                      CD_type,
                      transpose_a,
                      transpose_b,
                      get_custom_mul!((a, b, c) -> max(a + b, c)),
                      Epilogue.Default(),
                      $verify_default,
                      Kernel.matmul_pipelined,
                      nothing)
    end end)
end

macro get_wmma_config()
    esc(quote let
                name = "WMMA GEMM $(AB_type)*$(AB_type)$(zero_c ? "" : "+$(CD_type)")=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) Block ($BLOCK_M, $BLOCK_N, $BLOCK_K) Warps ($WARPS_M, $WARPS_N) OP ($(OP_M), $(OP_N), $(OP_K))"

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        block_shape = (M = BLOCK_M, N = BLOCK_N, K = BLOCK_K),
                                        warps_per_block = WARPS_M * WARPS_N,

                                        compute_warp = (M = BLOCK_M ÷ WARPS_M, N = BLOCK_N ÷ WARPS_N, K = OP_K),

                                        global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},
                                        global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},
                                        global_c_layout = zero_c ? Layout.Zero{CD_type} : Layout.UnsafeAlignedColMajor{CD_type},
                                        global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        shared_a_layout = Layout.Padded{transpose_a ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type}, 16 ÷ sizeof(AB_type)},
                                        shared_b_layout = Layout.Padded{transpose_b ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type}, 16 ÷ sizeof(AB_type)},
                                        shared_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                        shared_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        operator = Operator.WMMAOp{OP_M, OP_N, OP_K, AB_type, CD_type},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        Configuration(name,
                      conf,
                      convert(AB_type, 2),
                      convert(AB_type, zero_c ? 0 : 3),
                      AB_type,
                      AB_type,
                      CD_type,
                      CD_type,
                      transpose_a,
                      transpose_b,
                      $LinearAlgebra.mul!,
                      Epilogue.Default(),
                      $verify_default,
                      kernel,
                      $wmma_baseline)
    end end)
end

macro get_wmma_bias_config()
    esc(quote let
        name = "WMMA GEMM+bias $(AB_type)*$(AB_type)+$(CD_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"

        # Bias vector: this vector contains 1 element per column of the result matrix.
        # This bias element is added to all elements in one column of the D matrix.
        # D is a M x N matrix, so the bias is an N-element vector.
        bias = CuArray(rand(CD_type, (1, N)))

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        operator = Operator.WMMAOp{OP_M, OP_N, OP_K, AB_type, CD_type},
                                        global_a_layout = transpose_a ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},
                                        global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},

                                        global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                        global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        Configuration(name,
                      conf,
                      convert(AB_type, 1),
                      convert(AB_type, 1),
                      AB_type,
                      AB_type,
                      CD_type,
                      CD_type,
                      transpose_a,
                      transpose_b,
                      mul!,
                      Epilogue.Bias(pointer(bias)),
                      (c_h, d, T) -> $verify_bias(c_h, d, bias, T),
                      Kernel.matmul_pipelined,
                      nothing)
    end end)
end

macro get_wmma_diagonal_config()
    esc(quote let
        @assert M == K "Diagonal only supports square A matrix (A == M)"

        transpose_a = false
        name = "WMMA diagonal GEMM diag($(AB_type))*$(AB_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"

        # NOTE: the testing framework will generate an MxK A matrix,
        # but we only use the first M elements in the first column,
        # and interpret that as the diagonal elements.

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        operator = Operator.WMMAOp{OP_M, OP_N, OP_K, AB_type, CD_type},
                                        global_a_layout = Layout.Diagonal{AB_type},
                                        global_b_layout = transpose_b ? Layout.UnsafeAlignedRowMajor{AB_type} : Layout.UnsafeAlignedColMajor{AB_type},

                                        global_c_layout = Layout.UnsafeAlignedColMajor{CD_type},
                                        global_d_layout = Layout.UnsafeAlignedColMajor{CD_type},

                                        shared_a_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{Float16}, 8},

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b,
                                        )

        # XXX: perform this on the GPU
        function reference_mul!(C, A, B, alpha, beta)
            # XXX: alpha/beta ignored, and not used by other configs?
            Ah = Array(A)
            Bh = Array(B)
            Ch = Array(C)
            mul!(Ch, Diagonal(Ah[1:M,1]), Bh, true, true)
            copyto!(C, Ch)
        end

        Configuration(name,
                      conf,
                      convert(AB_type, 1),
                      convert(AB_type, 1),
                      AB_type,
                      AB_type,
                      CD_type,
                      CD_type,
                      transpose_a,
                      transpose_b,
                      reference_mul!,
                      Epilogue.Default(),
                      $verify_default,
                      Kernel.matmul_singlestage,
                      nothing)
    end end)
end

macro get_wmma_complex_config()
    esc(quote let
        name = "WMMA Complex GEMM $(AB_type)*$(AB_type)=$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        operator = Operator.WMMAComplexOp{OP_M, OP_N, OP_K, AB_type, CD_type},

                                        global_a_layout = transpose_a ? Layout.InterleavedRowMajor{Float16} : Layout.InterleavedColMajor{Float16},
                                        global_b_layout = transpose_b ? Layout.InterleavedRowMajor{Float16} : Layout.InterleavedColMajor{Float16},
                                        global_c_layout = Layout.InterleavedColMajor{Float32},
                                        global_d_layout = Layout.InterleavedColMajor{Float32},

                                        shared_a_layout = transpose_a ? Layout.Padded{Layout.SplitRowMajor{Float16}, 8} : Layout.Padded{Layout.SplitColMajor{Float16}, 8},
                                        shared_b_layout = transpose_b ? Layout.Padded{Layout.SplitRowMajor{Float16}, 8} : Layout.Padded{Layout.SplitColMajor{Float16}, 8},
                                        shared_c_layout = Layout.SplitColMajor{Float32},
                                        shared_d_layout = Layout.SplitColMajor{Float32},

                                        warps_per_block = 8,

                                        compute_warp = (M = 16, N = 32, K = 16),

                                        block_shape = (M = 64, N = 64, K = 32),

                                        mem_a_warp = transpose_a ? (M = 4, K = 32) : (M = 64, K = 2),
                                        mem_b_warp = transpose_b ? (K = 2, N = 64) : (K = 32, N = 4),
                                        mem_cd_warp = (M = 64, N = 1),

                                        mem_a_thread = transpose_a ? (M = 1, K = 4) : (M = 4, K = 1),
                                        mem_b_thread = transpose_b ? (K = 1, N = 4) : (K = 4, N = 1),
                                        mem_cd_thread = (M = 2, N = 1),

                                        is_a_col_major = !transpose_a,
                                        is_b_col_major = !transpose_b
                                        )

        Configuration(name,
                      conf,
                      convert(Complex{AB_type}, 1),
                      convert(Complex{AB_type}, 1),
                      Complex{AB_type},
                      Complex{AB_type},
                      Complex{CD_type},
                      Complex{CD_type},
                      transpose_a,
                      transpose_b,
                      mul!,
                      Epilogue.Default(),
                      $verify_default,
                      Kernel.matmul_pipelined,
                      nothing)
    end end)
end

macro get_wmma_dual_config()
    esc(quote let
        name = "WMMA Dual GEMM d$(AB_type)*d$(AB_type)=d$(CD_type) ($(M)×$(K)) · ($(K)×$(N)) ($( !transpose_a ? 'N' : 'T' )$( !transpose_b ? 'N' : 'T' )) OP ($(OP_M), $(OP_N), $(OP_K))"

        conf = GemmKernels.get_config(
                                        gemm_shape = (M = M, N = N, K = K),
                                        operator = Operator.WMMADualOp{OP_M, OP_N, OP_K, AB_type, CD_type},

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

        dual_conv(M) = reinterpret(ForwardDiff.Dual{Float32,Float32,1}, M)

        Configuration(name,
                      conf,
                      convert(Complex{AB_type}, 1),
                      convert(Complex{AB_type}, 1),
                      Complex{AB_type},
                      Complex{AB_type},
                      Complex{CD_type},
                      Complex{CD_type},
                      transpose_a,
                      transpose_b,
                      (C, A, B, alpha, beta) -> mul!(dual_conv(C), dual_conv(Complex{Float32}.(A)), dual_conv(Complex{Float32}.(B)), true, true),
                      Epilogue.Default(),
                      $verify_dual,
                      Kernel.matmul_pipelined,
                      nothing)
    end end)
end

macro get_tc_wmma_config()
    esc(quote let
                name = "TC $(parseable_name) Block ($BLOCK_M, $BLOCK_N, $BLOCK_K) Warps ($WARPS_M, $WARPS_N) OP ($(OP_M), $(OP_N), $(OP_K))"

        # Parsing the name into a threedimensional vector of the modes of each tensor.
        tensorModes = Vector{Vector{Int}}(undef, 0)
        for tensor in split(parseable_name, "-")
            tensorMode = Vector{Int}(undef, 0)

            for mode in split(tensor, ".")
                push!(tensorMode, parse(Int, mode))
            end

            push!(tensorModes, tensorMode)
        end

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

        a_extent = padded_extents[tensorModes[2]]
        a_desc = TensorDescriptor(
            length(a_extent), collect(Int, a_extent), collect(Int, cumprod((1, a_extent...))[1:end-1]), data_type, identity
        )
        b_extent = padded_extents[tensorModes[3]]
        b_desc = TensorDescriptor(
            length(b_extent), collect(Int, b_extent), collect(Int, cumprod((1, b_extent...))[1:end-1]), data_type, identity
        )
        c_extent = padded_extents[tensorModes[1]]
        c_desc = TensorDescriptor(
            length(c_extent), collect(Int, c_extent), collect(Int, cumprod((1, c_extent...))[1:end-1]), data_type, identity
        )

        plan = Tensors.ContractionPlan(
            a_desc, tensorModes[2],
            b_desc, tensorModes[3],
            c_desc, tensorModes[1],
            c_desc, tensorModes[1];
            operator = Operator.WMMAOp{OP_M, OP_N, OP_K, compute_type, accumulate_type},
            computeType=compute_type,
            accumulateType=accumulate_type,
            blockShape=(M = BLOCK_M, N = BLOCK_N, K = BLOCK_K),
            warpsPerBlock = WARPS_M * WARPS_N,
            computeWarp = (M = BLOCK_M ÷ WARPS_M, N = BLOCK_N ÷ WARPS_N, K = OP_K),
        )

        conf = plan.algorithmPlan.gemmConf

        ContractionConfiguration(
            name,
            plan,
            convert(compute_type, 2),
            convert(compute_type, zero_c ? 0 : 3),
            data_type,
            data_type,
            data_type,
            data_type,
            !conf.is_a_col_major,
            !conf.is_b_col_major,
            Epilogue.Default(),
            verify_default, # TODO
            kernel,
            tc_baseline, 
            extents,
            padded_extents,
            tensorModes,
            accumulate_type
        )
    end end)
end

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
        (OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB) in [(8, 16, 2, 4, 8, 1)],
        (BLOCK_M, BLOCK_N, BLOCK_K) in [(64, 64, 32)],
        N in [128, 256, 2048]

        # XXX: Should we do non-square matrices as well?
        M = K = N

        try
            push!(rv, @get_fpu_config)
        catch err
            isa(err, GemmKernels.ConfigError) || rethrow()
        end
    end

    # FPU Op shapes
    for (A_type, B_type, CD_type) in [
            (Float32, Float32, Float32)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB) in vcat(
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
        (BLOCK_M, BLOCK_N, BLOCK_K) in [(128, 64, 32)],
        N in [128]

        # We'll only test square matrices.
        M = K = N

        try
            push!(rv, @get_fpu_config)
        catch err
            isa(err, GemmKernels.ConfigError) || rethrow()
        end
    end

    # Tropical GEMM
    for (A_type, B_type, CD_type, min_dimension) in [
            (Float32, Float32, Float32, 128)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB) in [(8, 16, 2, 4, 8, 1)],
        (M, N, K) in min_dimension .* [
            [1, 1, 1],
            [2, 2, 1],
            [1, 1, 2],
            [2, 2, 2]]

        try
            push!(rv, @get_tropical_config)
        catch err
            isa(err, GemmKernels.ConfigError) || rethrow()
        end
    end

    # WMMA GEMM
    for (AB_type, CD_type, min_dimension) in [
        (Float16, Float16, 256),
        (Float16, Float32, 128)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (BLOCK_M, BLOCK_N, BLOCK_K) in [(128, 128, 64)],
        (WARPS_M, WARPS_N) in [(4, 2)],
        (OP_M, OP_N, OP_K) in [
            (16, 16, 16),
            (8, 32, 16),
            (32, 8, 16),
        ],
        (M, N, K) in vcat(min_dimension .* [
            [1, 1, 1],
            [2, 2, 1],
            [1, 1, 2],
            [2, 2, 2]], [[2048, 2048, 2048]]),
        zero_c in [false],
        kernel in [Kernel.matmul_pipelined]

        push!(rv, @get_wmma_config)
    end

    # WMMA GEMM parameters
    for (M, N, K) in [(256, 256, 256)],
        (AB_type, CD_type) in [(Float16, Float32)],
        transpose_a in [false, true],
        transpose_b in [false, true],
        (BLOCK_M, BLOCK_N, BLOCK_K) in filter(x -> prod(x[1:2]) <= 128*128, collect(Iterators.product([64, 128, 256], [64, 128, 256], [16, 32, 64]))[:]),
        (WARPS_M, WARPS_N) in filter(x -> prod(x) >= 4, collect(Iterators.product([1, 2, 4], [1, 2, 4]))[:]),
        zero_c in [false, true],
        (OP_M, OP_N, OP_K) in [(16, 16, 16)],
        kernel in [Kernel.matmul_singlestage, Kernel.matmul_pipelined]

        try
            push!(rv, @get_wmma_config)
        catch err
            isa(err, GemmKernels.ConfigError) || rethrow()
        end
    end

    # WMMA GEMM + bias
    for (AB_type, CD_type, min_dimension) in [
        (Float16, Float32, 128)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [
            (16, 16, 16),
            (8, 32, 16),
            (32, 8, 16),
        ],
        (M, N, K) in vcat(min_dimension .* [
            [1, 1, 1],
            [2, 2, 2]], [[4096, 4096, 4096]])

        try
            push!(rv, @get_wmma_bias_config)
        catch err
            isa(err, GemmKernels.ConfigError) || rethrow()
        end
    end

    # WMMA Diagonal GEMM
    for (AB_type, CD_type, min_dimension) in [
        (Float16, Float32, 128)],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [
            (16, 16, 16),
            (8, 32, 16),
            (32, 8, 16),
        ],
        (M, N, K) in vcat(min_dimension .* [
            [1, 1, 1],
            [2, 2, 2]], [[4096, 4096, 4096]])

        try
            push!(rv, @get_wmma_diagonal_config)
        catch err
            isa(err, GemmKernels.ConfigError) || rethrow()
        end
    end

    # WMMA Complex GEMM
    for (AB_type, CD_type) in [(Float16, Float32)],
        transpose_a = [false, true],
        transpose_b = [false, true],
        (OP_M, OP_N, OP_K) in [
            (16, 16, 16),
            (8, 32, 16),
            (32, 8, 16),
        ],
        (M, N, K) in [
            (128, 128, 128),
            (256, 256, 256),
            (2048, 2048, 2048)]

        try
            push!(rv, @get_wmma_complex_config)
        catch err
            isa(err, GemmKernels.ConfigError) || rethrow()
        end
    end

    # WMMA Dual GEMM
    for (AB_type, CD_type) in [(Float16, Float32)],
        transpose_a = [false],
        transpose_b = [false],
        (OP_M, OP_N, OP_K) in [
            (16, 16, 16),
            (8, 32, 16),
            (32, 8, 16),
        ],
        (M, N, K) in [
            (128, 128, 128),
            (256, 256, 256),
            (2048, 2048, 2048)]

        try
            push!(rv, @get_wmma_dual_config)
        catch err
            isa(err, GemmKernels.ConfigError) || rethrow()
        end
    end

    rv
end

end
using .Configs
