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

    #= Is A & B stored in Column major order? This determines the iteration order of the parallellisation =#
    is_a_col_major
    is_b_col_major
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

function handle_operator_config(operator)
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

    if block_shape.M < 2 * op_shape.M || block_shape.N < 2 * op_shape.N
        # TODO: Find out why this is.
        throw(ConfigError("There is a mismatch between the block shape and the operator shape. Their dimensions must adhere to the following constraints: BLOCK_M ≥ 2 * OPERATOR_M, BLOCK_N ≥ 2 * OPERATOR_N."))
    end

    if operator <: Operator.GeneralFPUOp
        handle_operator_config(operator)
    end

    # 8 warps in a 4 x 2 arrangement usually works well
    warps_per_block_default = 8
    compute_warp_default = (M = block_shape.M ÷ 4, N = block_shape.N ÷ 2, K = op_shape.K)

    # Best effort to make sure that the compute warp shape is not smaller than the operator shape.
    if (block_shape.M ÷ op_shape.M) < 4 || (block_shape.N ÷ op_shape.N) < 2
        compute_warp_default = (
            M = block_shape.M ÷ min(block_shape.M ÷ op_shape.M, 4),
            N = block_shape.N ÷ min(block_shape.N ÷ op_shape.N, 2),
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

    mem_a_warp = get(params, :mem_a_warp,
        adjacent_elements(32 * 16 ÷ sizeof(Layout.eltype(global_a_layout)), (M = block_shape.M, K = block_shape.K), is_a_col_major))
    mem_b_warp = get(params, :mem_b_warp,
        adjacent_elements(32 * 16 ÷ sizeof(Layout.eltype(global_b_layout)), (K = block_shape.K, N = block_shape.N), is_b_col_major))
    mem_cd_warp = get(params, :mem_cd_warp,
        adjacent_elements(32 * 16 ÷ sizeof(Layout.eltype(global_c_layout)), (M = block_shape.M, N = block_shape.N), is_cd_col_major))

    mem_a_thread = get(params, :mem_a_thread,
        adjacent_elements(16 ÷ sizeof(Layout.eltype(global_a_layout)), (M = block_shape.M, K = block_shape.K), is_a_col_major))
    mem_b_thread = get(params, :mem_b_thread,
        adjacent_elements(16 ÷ sizeof(Layout.eltype(global_b_layout)), (K = block_shape.K, N = block_shape.N), is_b_col_major))
    mem_cd_thread = get(params, :mem_cd_thread,
        adjacent_elements(16 ÷ sizeof(Layout.eltype(global_c_layout)), (M = block_shape.M, N = block_shape.N), is_cd_col_major))

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
