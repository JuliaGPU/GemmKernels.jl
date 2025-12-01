using CUDA, GemmKernels
using DataFrames
using Serialization

isinteractive() || include("tuning/wmma-contraction.jl")

function main()
    PROBLEM_ID = 1

    problems = generate_problems()

    # Select problem.
    problem = problems[PROBLEM_ID]

    # Configuration.
    block = 16, 256, 16
    warp = 2, 8
    op = 8, 32, 16
    layout = [1, 2], [3], [4]
    col_major = ["B", "D"]
    swizzle = "vertical-4"
    kernel = "pipelined"

    config = (;
        # name
        name = problem.name,
        extents = problem.extents,

        # block
        BLOCK_M = block[1]::Int,
        BLOCK_N = block[2]::Int,
        BLOCK_K = block[3]::Int,

        # warp
        WARPS_M = warp[1]::Int,
        WARPS_N = warp[2]::Int,

        # operator
        OP_M = op[1]::Int,
        OP_N = op[2]::Int,
        OP_K = op[3]::Int,

        # layout
        PERM_M = layout[1]::Array,
        PERM_N = layout[2]::Array,
        PERM_K = layout[3]::Array,

        # default: false, unless otherwise specified
        is_A_col_major = ("A" in col_major),
        is_B_col_major = ("B" in col_major),
        is_D_col_major = ("D" in col_major),

        # swizzle
        cta_swizzle_str = swizzle::String,

        # kernel
        kernel_str = kernel::String,
    )

    # Run.
    data = allocate_data(problem)
    params = create_params(config)
    args = prepare(problem, data...; params...)
    execute(problem, data...; args...)
end

isinteractive() || main()
