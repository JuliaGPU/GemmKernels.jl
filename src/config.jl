@staticdef struct Config
    #= Params =#
    matmul_shape                # MNK, overall shape of the MATMUL operation
    block_shape                 # MNK, shape of each CTA tile
    warps_per_block             # scalar, number of warps per CTA

    mem_a_warp                  # MK, shape of each warp tile during memory operations involving matrix A
    mem_a_thread                # MK, shape of each thread tile during memory operations involving matrix A

    mem_b_warp                  # KN, shape of each warp tile during memory operations involving matrix B
    mem_b_thread                # KN, shape of each thread tile during memory operations involving matrix B

    mem_cd_warp                 # MN, shape of each warp tile during memory operations involving matrix C or D
    mem_cd_thread               # MN, shape of each thread tile during memory operations involving matrix C or D

    compute_warp                # MNK, shape of each warp tile during the inner loop computations
    compute_op_shape            # MNK, shape of the operation used in the inner loop

    #= Layouts =#
    global_a_layout             # layout of the A matrix in global memory
    global_b_layout             # layout of the B matrix in global memory
    global_c_layout             # layout of the C matrix in global memory
    global_d_layout             # layout of the D matrix in global memory

    shared_a_layout             # layout of the A matrix in shared memory
    shared_b_layout             # layout of the B matrix in shared memory
    shared_c_layout             # layout of the C matrix in shared memory
    shared_d_layout             # layout of the D matrix in shared memory

    #= Operator =#
    operator                    # which operator to use in the inner loop

    #= Is A & B stored in Column major order? This determines the iteration order of the parallelisation =#
    is_a_col_major
    is_b_col_major
end

function Base.show(io::IO, config::Config)
    println(io, "matmul_shape:     $(config.matmul_shape)")
    println(io, "block_shape:      $(config.block_shape)")
    println(io, "warps_per_block:  $(config.warps_per_block)")

    println(io, "mem_a_warp:       $(config.mem_a_warp)")
    println(io, "mem_a_thread:     $(config.mem_a_thread)")

    println(io, "mem_b_warp:       $(config.mem_b_warp)")
    println(io, "mem_b_thread:     $(config.mem_b_thread)")

    println(io, "mem_cd_warp:      $(config.mem_cd_warp)")
    println(io, "mem_cd_thread:    $(config.mem_cd_thread)")

    println(io, "compute_warp:     $(config.compute_warp)")
    println(io, "compute_op_shape: $(config.compute_op_shape)")

    println(io, "global_a_layout:  $(config.global_a_layout)")
    println(io, "global_b_layout:  $(config.global_b_layout)")
    println(io, "global_c_layout:  $(config.global_c_layout)")
    println(io, "global_d_layout:  $(config.global_d_layout)")

    println(io, "shared_a_layout:  $(config.shared_a_layout)")
    println(io, "shared_b_layout:  $(config.shared_b_layout)")
    println(io, "shared_c_layout:  $(config.shared_c_layout)")
    println(io, "shared_d_layout:  $(config.shared_d_layout)")

    println(io, "operator:         $(config.operator)")

    println(io, "is_a_col_major:   $(config.is_a_col_major)")
    println(io, "is_b_col_major:   $(config.is_b_col_major)")
end

struct ConfigError <: Exception
    message::String
end

Base.showerror(io::IO, e::ConfigError) = print(io, "ConfigError: ", e.message)

function heuristic_block_shape(shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout)
    # Determining the tile size of each block is a little trickier.
    # We apply the following heuristics:
    # 1) Ideally, the block shape in the M and N dimensions should be square or
    #    nearly-square to maximise data reuse. More specifically, tiling sizes
    #    of the form (M = k, N = k) and (M = 2k, N = k) work well.
    # 2) The tile size should be as large as possible.
    # 3) The size in the M and N dimension is limited by the fact that a tile of
    #    the C (and D) matrix of that size must fit in shared memory.
    # 4) The size in the K dimension is limited by the fact that both a M x K tile
    #    of A and a K x N tile of B must fit in shared memory, at the same time.

    num_bytes_A(M, N, K) = prod(Layout.physical_size(shared_a_layout, (M = M, K = K))) * sizeof(Layout.eltype(shared_a_layout))
    num_bytes_B(M, N, K) = prod(Layout.physical_size(shared_b_layout, (K = K, N = N))) * sizeof(Layout.eltype(shared_b_layout))
    num_bytes_C(M, N, K) = prod(Layout.physical_size(shared_c_layout, (M = M, N = N))) * sizeof(Layout.eltype(shared_c_layout))
    num_bytes_D(M, N, K) = prod(Layout.physical_size(shared_d_layout, (M = M, N = N))) * sizeof(Layout.eltype(shared_d_layout))

    next_MN(M, N, K) = M == N ? (2 * M, N, K) : (M, 2 * N, K)
    next_K(M, N, K) = (M, N, 2 * K)

    cur = 1, 1, 1     # M, N, K
    nxt = next_MN(cur...)

    while (max(num_bytes_C(nxt...), num_bytes_D(nxt...)) <= 64 * 1024)
        cur = nxt
        nxt = next_MN(cur...)
    end

    nxt = next_K(cur...)

    while (num_bytes_A(nxt...) + num_bytes_B(nxt...) <= 64 * 1024)
        cur = nxt
        nxt = next_K(cur...)
    end

    return (M = cur[1], N = cur[2], K = cur[3])
end

# Helper function that returns the logical size of a set of adjacent elements, taking care not
# to make the size larger than the parent tile
function adjacent_elements(num, parent_size, is_col_major)
    p = Tuple(parent_size)

    if is_col_major
        t = (min(num, p[1]), num ÷ min(num, p[1]))
    else
        t = (num ÷ min(num, p[2]), min(num, p[2]))
    end

    return typeof(parent_size)(t)
end

check_operator_config(::Type{T}) where {T} = nothing

function check_operator_config(operator::Type{<:Operator.GeneralFPUOp})
    op_shape = Operator.base_shape(operator)

    # The 32 threads in a warp must at least handle one element of the operator.
    if op_shape.M * op_shape.N < 32
        throw(ConfigError("The operator shape is too small. The dimensions of the operator shape must adhere to the following constraint: OPERATOR_M * OPERATOR_N ≥ 32."))
    end

    if op_shape.mb * op_shape.nb != 32
        throw(ConfigError("The base FPU operator shape should adhere to the following constraint: OPERATOR_M_BASE * OPERATOR_N_BASE = 32."))
    end

    if op_shape.kb != 1
        throw(ConfigError("The base FPU operator shape should adhere to the following constraint: OPERATOR_K_BASE = 1."))
    end

    if any((op_shape.M, op_shape.N, op_shape.K) .% (op_shape.mb, op_shape.nb, op_shape.kb) .!= 0)
        throw(ConfigError("The operator shape should adhere to the following constraint: OPERATOR_M, OPERATOR_N, OPERATOR_K are multiples of OPERATOR_M_BASE, OPERATOR_N_BASE, OPERATOR_K_BASE, respectively."))
    end
end

function check_wmma_shape(operator::Type)
    op_shape = Operator.shape(operator)

    if op_shape ∉ [
        (M=16, N=16, K=16),
        (M=8, N=32, K=16),
        (M=32, N=8, K=16),
    ]
        throw(ConfigError("Unsupported WMMA Operator shape $(op_shape)!"))
    end
end

check_operator_config(operator::Type{<:Operator.WMMAOp}) = check_wmma_shape(operator)
check_operator_config(operator::Type{<:Operator.WMMAComplexOp}) = check_wmma_shape(operator)
check_operator_config(operator::Type{<:Operator.WMMADualOp}) = check_wmma_shape(operator)

require_tile_sized_global(layout) = true
require_tile_sized_global(::Type{<:Layout.Zero{T}}) where {T} = false
require_tile_sized_global(::Type{<:Layout.ColMajor{T}}) where {T} = false
require_tile_sized_global(::Type{<:Layout.RowMajor{T}}) where {T} = false

function get_config(; gemm_shape, operator, global_a_layout, global_c_layout, kwargs...)
    params = Dict(kwargs)

    # Use some simple heuristics to get sensible defaults for parameters the user does not specify.

    # Get the global layouts for B & D.
    # Fallback to the layouts of A & C, respectively.
    global_b_layout = get(params, :global_b_layout, global_a_layout)
    global_d_layout = get(params, :global_d_layout, global_c_layout)

    # Get the shared layouts for A, B, C, D.
    # For A & B, add padding to reduce bank conflicts, but preserve 128-bit (16 byte) alignment.
    shared_a_layout = get(params, :shared_a_layout,
                          Layout.Padded{global_a_layout, 16 ÷ sizeof(Layout.eltype(global_a_layout))})
    shared_b_layout = get(params, :shared_b_layout,
                          Layout.Padded{global_b_layout, 16 ÷ sizeof(Layout.eltype(global_b_layout))})
    shared_c_layout = get(params, :shared_c_layout, global_c_layout)
    shared_d_layout = get(params, :shared_d_layout, global_d_layout)

    # Apply heuristic for block shape
    block_shape = get(params, :block_shape,
        heuristic_block_shape(shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout))

    if block_shape.M * block_shape.K < 128 || block_shape.K * block_shape.N < 128 || block_shape.K < 8
        throw(ConfigError("The block shape is too small. The dimensions of the block shape must adhere to the following constraints: BLOCK_M * BLOCK_K ≥ 128, BLOCK_K * BLOCK_N ≥ 128, BLOCK_K ≥ 8."))
    end

    op_shape = Operator.shape(operator)

    if block_shape.M < 2 * op_shape.M || block_shape.N < 2 * op_shape.N || block_shape.K < op_shape.K
        # TODO: Find out why this is.
        throw(ConfigError("There is a mismatch between the block shape and the operator shape. Their dimensions must adhere to the following constraints: BLOCK_M ≥ 2 * OPERATOR_M, BLOCK_N ≥ 2 * OPERATOR_N, BLOCK_K ≥ OPERATOR_K."))
    end

    check_operator_config(operator)

    # 8 warps in a 4 x 2 arrangement usually works well
    warps_per_block_default = 8
    compute_warp_default = (M = block_shape.M ÷ 4, N = block_shape.N ÷ 2, K = op_shape.K)

    # Best effort to make sure that the compute warp shape is not smaller than the operator shape.
    if (block_shape.M ÷ op_shape.M) < 4 || (block_shape.N ÷ op_shape.N) < 2
        compute_warp_default = (
            M = max(op_shape.M, block_shape.M ÷ 4),
            N = max(op_shape.N, block_shape.N ÷ 2),
            K = op_shape.K
        )
        warps_per_block_default = min(block_shape.M ÷ op_shape.M, 4) * min(block_shape.N ÷ op_shape.N, 2)
    end

    warps_per_block = get(params, :warps_per_block, warps_per_block_default)
    compute_warp = get(params, :compute_warp, compute_warp_default)


    # Is the layout col-major or not? This is needed to find good values for mem_a_warp, mem_b_warp, etc.
    # TODO: Let the layouts handle this?
    is_a_col_major = get(params, :is_a_col_major, true)
    is_b_col_major = get(params, :is_b_col_major, true)
    is_cd_col_major = get(params, :is_cd_col_major, true)

    # Heuristics for memory tiling sizes:
    # 1) The tiles should encompass 128 bits (16 bytes) to enable vectorisation.
    # 2) The tiles should be as small as possible (i.e. each thread exactly 128 bits) to enable coalescing.

    num_elems_per_thread_a = min(16 ÷ sizeof(Layout.eltype(global_a_layout)), (block_shape.M * block_shape.K) ÷ (32 * warps_per_block))
    num_elems_per_thread_b = min(16 ÷ sizeof(Layout.eltype(global_b_layout)), (block_shape.K * block_shape.N) ÷ (32 * warps_per_block))
    num_elems_per_thread_c = min(16 ÷ sizeof(Layout.eltype(global_c_layout)), (block_shape.M * block_shape.N) ÷ (32 * warps_per_block))

    mem_a_warp = get(params, :mem_a_warp,
        adjacent_elements(32 * num_elems_per_thread_a, (M = block_shape.M, K = block_shape.K), is_a_col_major))
    mem_b_warp = get(params, :mem_b_warp,
        adjacent_elements(32 * num_elems_per_thread_b, (K = block_shape.K, N = block_shape.N), is_b_col_major))
    mem_cd_warp = get(params, :mem_cd_warp,
        adjacent_elements(32 * num_elems_per_thread_c, (M = block_shape.M, N = block_shape.N), is_cd_col_major))

    mem_a_thread = get(params, :mem_a_thread,
        adjacent_elements(num_elems_per_thread_a, (M = block_shape.M, K = block_shape.K), is_a_col_major))
    mem_b_thread = get(params, :mem_b_thread,
        adjacent_elements(num_elems_per_thread_b, (K = block_shape.K, N = block_shape.N), is_b_col_major))
    mem_cd_thread = get(params, :mem_cd_thread,
        adjacent_elements(num_elems_per_thread_c, (M = block_shape.M, N = block_shape.N), is_cd_col_major))

    # Make sure that we have at least one iteration in the memory copy loops.
    prod(mem_a_warp) * warps_per_block ≤ block_shape.M * block_shape.K || throw(ConfigError("mem_a_warp is too big for the selected block shape: need at least one iteration in the memory copy loop!"))
    prod(mem_b_warp) * warps_per_block ≤ block_shape.K * block_shape.N || throw(ConfigError("mem_b_warp is too big for the selected block shape: need at least one iteration in the memory copy loop!"))
    prod(mem_cd_warp) * warps_per_block ≤ block_shape.M * block_shape.N || throw(ConfigError("mem_cd_warp is too big for the selected block shape: need at least one iteration in the memory copy loop!"))

    # Check sizes of tiles
    check_tile_multiple(num, den, dims, msg) = all([num[dim] % den[dim] == 0 for dim in dims]) || throw(ConfigError(msg))

    check_tile_multiple(block_shape, compute_warp, [:M, :N, :K], "block_shape must be a multiple of compute_warp!")
    check_tile_multiple(compute_warp, op_shape, [:M, :N, :K], "compute_warp must be a multiple of op_shape!")
    require_tile_sized_global(global_a_layout) && check_tile_multiple(gemm_shape, block_shape, [:M, :K], "gemm_shape.MK must be a multiple of block_shape.MK!")
    require_tile_sized_global(global_b_layout) && check_tile_multiple(gemm_shape, block_shape, [:K, :N], "gemm_shape.KN must be a multiple of block_shape.KN!")
    require_tile_sized_global(global_c_layout) && check_tile_multiple(gemm_shape, block_shape, [:M, :N], "gemm_shape.MN must be a multiple of block_shape.MN!")
    require_tile_sized_global(global_d_layout) && check_tile_multiple(gemm_shape, block_shape, [:M, :N], "gemm_shape.MN must be a multiple of block_shape.MN!")

    return Config(
        #= Params =#
        gemm_shape,
        block_shape,
        warps_per_block,
        mem_a_warp,
        mem_a_thread,
        mem_b_warp,
        mem_b_thread,
        mem_cd_warp,
        mem_cd_thread,
        compute_warp,
        op_shape,

        #= Layouts =#
        global_a_layout,
        global_b_layout,
        global_c_layout,
        global_d_layout,

        shared_a_layout,
        shared_b_layout,
        shared_c_layout,
        shared_d_layout,

        #= Operators =#
        operator,

        #= Is A & B Col Major? =#
        is_a_col_major,
        is_b_col_major,
    )
end
