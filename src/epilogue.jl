export Epilogue
module Epilogue

using CUDA
using GemmKernels
using GemmKernels.Tiling
using GPUifyLoops: @unroll

# ----------------
# Default epilogue
# ----------------

struct Default end

@inline function (ep::Default)(d, shmem_d, transform, conf::Type{GemmKernels.Config{MATMUL_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR, IS_A_COL_MAJOR, IS_B_COL_MAJOR}}) where {MATMUL_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR, IS_A_COL_MAJOR, IS_B_COL_MAJOR}
    # Constants
    block_i = (blockIdx().x - 1) * BLOCK_SHAPE.M
    block_j = (blockIdx().y - 1) * BLOCK_SHAPE.N

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    laneId = (threadIdx().x - 1) % 32 + 1

    gemm_sz = Tile(MATMUL_SHAPE)
    block_tile = Tile(BLOCK_SHAPE)

    # Cooperatively store a BLOCK_SHAPE.M x BLOCK_SHAPE.N tile of D from shared to global memory within one threadblock
    @unroll for warp_tile = parallellise(block_tile.MN, Tile(MEM_CD_WARP), warpId, WARPS_PER_BLOCK)
        @unroll for thread_tile = parallellise(warp_tile, Tile(MEM_CD_THREAD), laneId, 32)
            x = Layout.load(SHARED_D_LAYOUT, shmem_d, thread_tile)
            x = transform(x, thread_tile)
            Layout.store!(GLOBAL_D_LAYOUT, d, x, translate_base(thread_tile, (M = block_i, N = block_j)))
        end
    end
end

# -------------
# Bias epilogue
# -------------

struct Bias{B}
    bias_pointer::B
end

@inline function apply_bias!(x, bias_pointer::CuPtr{Float32}, thread_tile)
    dev_ptr = reinterpret(Core.LLVMPtr{Float32, AS.Global}, bias_pointer)

    @unroll for i = 1 : size(x, 1)
        @unroll for j = 1 : size(x, 2)
            # Load bias value for this column
            col = thread_tile.index.N + j
            b = unsafe_load(dev_ptr, col)

            @inbounds x[i, j] = ntuple(k -> VecElement{Float32}(x[i, j][k].value + b), Val(4))
        end
    end
end

@inline function (ep::Bias{B})(d, shmem_d, transform, conf::Type{GemmKernels.Config{MATMUL_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR, IS_A_COL_MAJOR, IS_B_COL_MAJOR}}) where {MATMUL_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR, IS_A_COL_MAJOR, IS_B_COL_MAJOR, B}
    # Constants
    block_i = (blockIdx().x - 1) * BLOCK_SHAPE.M
    block_j = (blockIdx().y - 1) * BLOCK_SHAPE.N

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    laneId = (threadIdx().x - 1) % 32 + 1

    gemm_sz = Tile(MATMUL_SHAPE)
    block_tile = Tile(BLOCK_SHAPE)

    # Cooperatively store a BLOCK_SHAPE.M x BLOCK_SHAPE.N tile of D from shared to global memory within one threadblock
    @unroll for warp_tile = parallellise(block_tile.MN, Tile(MEM_CD_WARP), warpId, WARPS_PER_BLOCK)
        @unroll for thread_tile = parallellise(warp_tile, Tile(MEM_CD_THREAD), laneId, 32)
            x = Layout.load(SHARED_D_LAYOUT, shmem_d, thread_tile)
            apply_bias!(x, ep.bias_pointer, translate_base(thread_tile, (M = block_i, N = block_j)))
            x = transform(x, thread_tile)
            Layout.store!(GLOBAL_D_LAYOUT, d, x, translate_base(thread_tile, (M = block_i, N = block_j)))
        end
    end
end

end
