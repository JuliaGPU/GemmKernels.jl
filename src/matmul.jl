#
# low-level
#

function matmul(conf::Config, a, b, c, d;
                transform_global_to_shared_a = Transform.Elementwise(),
                transform_global_to_shared_b = Transform.Elementwise(),
                transform_global_to_shared_c = Transform.Elementwise(),
                transform_shared_to_global_d = Transform.Elementwise(),
                transform_shared_to_regs_a = Transform.Elementwise(),
                transform_shared_to_regs_b = Transform.Elementwise(),
                transform_shared_to_regs_c = Transform.Elementwise(),
                transform_regs_to_shared_d = Transform.Elementwise(),
                epilogue = Epilogue.Default(),
                kernel = Kernel.matmul_singlestage)

    args = [conf, a, b, c, d,
            transform_global_to_shared_a, transform_global_to_shared_b, transform_global_to_shared_c, transform_shared_to_global_d,
            transform_shared_to_regs_a, transform_shared_to_regs_b, transform_shared_to_regs_c, transform_regs_to_shared_d,
            epilogue]

    threads = conf.warps_per_block * 32
    blocks = (cld(conf.matmul_shape.M, conf.block_shape.M),
              cld(conf.matmul_shape.N, conf.block_shape.N))

    shmem = Kernel.shmem_size(conf, kernel)
    max_shmem = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    if shmem > max_shmem
        error("Requested too much shared memory: The current GPU can use at most $(Base.format_bytes(max_shmem)), while this configuration required $(Base.format_bytes(shmem))")
    end

    hostkernel = @cuda launch=false kernel(args...)
    attributes(hostkernel.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    hostkernel(args...; threads, blocks, shmem)
end


#
# BLAS-like
#

# Select the best kernel
kernel(layout_a, layout_b) = Kernel.matmul_singlestage
kernel(::Type{Layout.UnsafeAlignedColMajor{T}}, ::Type{Layout.UnsafeAlignedColMajor{T}}) where {T} = Kernel.matmul_pipelined
kernel(::Type{Layout.UnsafeAlignedColMajor{T}}, ::Type{Layout.UnsafeAlignedRowMajor{T}}) where {T} = Kernel.matmul_pipelined
kernel(::Type{Layout.UnsafeAlignedRowMajor{T}}, ::Type{Layout.UnsafeAlignedColMajor{T}}) where {T} = Kernel.matmul_pipelined
kernel(::Type{Layout.UnsafeAlignedRowMajor{T}}, ::Type{Layout.UnsafeAlignedRowMajor{T}}) where {T} = Kernel.matmul_pipelined

const configs = Dict{}()
@inline function get_config(args...)
    val = get(configs, args, nothing)
    if val === nothing
        val = configs[args] = create_config(args...)
    end
    return val
end
@noinline function create_config(A::Type, sizeA::Dims, stridesA::Dims, transA::Bool,
                                 B::Type, sizeB::Dims, stridesB::Dims, transB::Bool,
                                 C::Type, sizeC::Dims, stridesC::Dims,
                                 alpha::Type, zeroAlpha::Bool,
                                 beta::Type, zeroBeta::Bool,
                                 wmma::Union{Bool,Nothing})
    m = sizeA[transA ? 2 : 1]
    k = sizeA[transA ? 1 : 2]
    n = sizeB[transB ? 1 : 2]
    if m != sizeC[1] || n != sizeC[2] || k != sizeB[transB ? 2 : 1]
        throw(DimensionMismatch("Dimensions do not match"))
    end

    a_layout_base = transA ? Layout.RowMajor : Layout.ColMajor
    b_layout_base = transB ? Layout.RowMajor : Layout.ColMajor
    a_aligned_layout_base = transA ? Layout.UnsafeAlignedRowMajor : Layout.UnsafeAlignedColMajor
    b_aligned_layout_base = transB ? Layout.UnsafeAlignedRowMajor : Layout.UnsafeAlignedColMajor

    # determine operator to use
    wmma_types = [
        (Float16, Float16, Float16),
        (Float16, Float16, Float32),
        # TODO: more, and device-capability dependent
    ]
    compute_type = promote_type(eltype(A), eltype(B))
    use_wmma = something(wmma, (compute_type, compute_type, eltype(C)) in wmma_types)

    # determine shared memory layouts
    ## padded to avoid bank conflicts
    if use_wmma
        # in the case of WMMA, the shared memory needs to have the correct type already,
        # as we'll use WMMA intrinsics to load from it.
        shared_a_layout = Layout.Padded{a_aligned_layout_base{compute_type}, 8}
        shared_b_layout = Layout.Padded{b_aligned_layout_base{compute_type}, 8}
    else
        shared_a_layout = Layout.Padded{a_aligned_layout_base{eltype(A)}, 8}
        shared_b_layout = Layout.Padded{b_aligned_layout_base{eltype(B)}, 8}
    end
    ## outputs are never transposed, and padding them doesn't seem worth it
    shared_c_layout = shared_d_layout = Layout.UnsafeAlignedColMajor{eltype(C)}

    # determine block shape
    # XXX: heuristic should take much more into account (GEMM size, at least)
    block_shape = if use_wmma
        heuristic_block_shape(shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout)
    else
        # XXX: heuristic for FPU
        (M = 128, N = 128, K = 32)
    end

    # determine global memory layouts
    ## check if columns begin at aligned addresses, allowing use of vectorized loads & stores
    a_aligned = (stridesA[2] * sizeof(eltype(A))) % 16 == 0
    b_aligned = (stridesB[2] * sizeof(eltype(B))) % 16 == 0
    c_aligned = (stridesC[2] * sizeof(eltype(C))) % 16 == 0
    ## if alpha is zero, we don't need to load A or B
    if zeroAlpha
        global_a_layout = Layout.Zero{eltype(A)}
        global_b_layout = Layout.Zero{eltype(B)}
    else
        global_a_layout = if a_aligned && m%block_shape.M == 0 &&  k%block_shape.K == 0
            a_aligned_layout_base{eltype(A)}
        else
            a_layout_base{eltype(A)}
        end
        global_b_layout = if b_aligned && k%block_shape.K == 0 && n%block_shape.N == 0
            b_aligned_layout_base{eltype(B)}
        else
            b_layout_base{eltype(B)}
        end
    end
    ## if beta is zero, we don't need to load C
    global_c_layout = if zeroBeta
        Layout.Zero{eltype(C)}
    else
        if c_aligned && m%block_shape.M == 0 && n%block_shape.N == 0
            Layout.UnsafeAlignedColMajor{eltype(C)}
        else
            Layout.ColMajor{eltype(C)}
        end
    end
    global_d_layout = if c_aligned && m%block_shape.M == 0 && n%block_shape.N == 0
        Layout.UnsafeAlignedColMajor{eltype(C)}
    else
        Layout.ColMajor{eltype(C)}
    end

    conf = if use_wmma
        get_config(;
            gemm_shape = (M = m, N = n, K = k), block_shape,
            operator = Operator.WMMAOp{16, 16, 16, compute_type, eltype(C)},

            global_a_layout, global_b_layout, global_c_layout, global_d_layout,
            shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout,

            is_a_col_major = !transA,
            is_b_col_major = !transB
        )
    else
        get_config(;
            gemm_shape = (M = m, N = n, K = k), block_shape,
            operator = Operator.FPUOp{8, 8, 1, 4, 8, 1, compute_type, eltype(C)},

            global_a_layout, global_b_layout, global_c_layout, global_d_layout,
            shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout,

            is_a_col_major = !transA,
            is_b_col_major = !transB
        )
    end

    return conf, compute_type, kernel(global_a_layout, global_b_layout)
end

function matmatmul!(C::CuArray, transA::Char, transB::Char, A::CuArray, B::CuArray,
                    alpha::Number, beta::Number; wmma::Union{Bool,Nothing}=nothing)
    conf, compute_type, kernel = get_config(
        typeof(A), size(A), strides(A), transA=='T',
        typeof(B), size(B), strides(B), transB=='T',
        typeof(C), size(C), strides(C),
        typeof(alpha), iszero(alpha),
        typeof(beta), iszero(beta),
        wmma
    )

    alpha = convert(compute_type, alpha)
    beta = convert(eltype(C), beta)
    matmul(conf, A, B, C, C;
           transform_shared_to_regs_a = Transform.Elementwise(x -> x * alpha),
           transform_shared_to_regs_c = Transform.Elementwise(x -> x * beta),
           kernel
          )
    C
end

# convenience function
function mul!(C::CuArray,
              A::Union{CuArray, Adjoint{<:Any,<:CuArray}, Transpose{<:Any,<:CuArray}},
              B::Union{CuArray, Adjoint{<:Any,<:CuArray}, Transpose{<:Any,<:CuArray}},
              alpha=true, beta=false)
    transA = A isa Adjoint || A isa Transpose
    transB = B isa Adjoint || B isa Transpose
    matmatmul!(C, transA ? 'T' : 'N', transB ? 'T' : 'N',
               parent(A), parent(B), alpha, beta; info)
end
