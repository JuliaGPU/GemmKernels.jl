export Epilogue
module Epilogue

using CUDA
using GemmKernels
using GemmKernels: CTASwizzle
using GemmKernels.Tiling
using LLVMLoopInfo: @loopinfo

# ----------------
# Default epilogue
# ----------------

struct Default end

@inline function (ep::Default)(conf::GemmKernels.Config, d, shmem_d, transform)
    # Constants
    block_i, block_j = CTASwizzle.cta_swizzle(conf.cta_swizzle, blockIdx(), conf.block_shape, conf.matmul_shape)

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    laneId = (threadIdx().x - 1) % 32 + 1

    gemm_sz = Tile(conf.matmul_shape)
    block_tile = Tile(conf.block_shape)

    # Cooperatively store a block_shape.M x block_shape.N tile of D from shared to global memory within one threadblock
    @loopinfo unroll for warp_tile = parallelise(block_tile.MN, Tile(conf.mem_cd_warp), warpId, conf.warps_per_block, conf.is_cd_col_major)
        @loopinfo unroll for thread_tile = parallelise(warp_tile, Tile(conf.mem_cd_thread), laneId, 32, conf.is_cd_col_major)
            x = @inbounds Layout.load(conf.shared_d_layout, shmem_d, thread_tile)
            x = transform(x, thread_tile)
            @inbounds Layout.store!(conf.global_d_layout, d, x, translate_base(thread_tile, (M = block_i, N = block_j)))
        end
    end
end

# -------------
# Bias epilogue
# -------------

struct Bias{B}
    bias_pointer::B
end

@inline function apply_bias(x, bias_pointer::CuPtr{Float32}, thread_tile)
    dev_ptr = reinterpret(Core.LLVMPtr{Float32, AS.Global}, bias_pointer)

    # Load bias value for this column
    col = thread_tile.index.N + 1
    b = unsafe_load(dev_ptr, col)

    return ntuple(k -> VecElement{Float32}(x[k].value + b), Val(4))
end

@inline function (ep::Bias{B})(conf::GemmKernels.Config, d, shmem_d, transform) where {B}
    # Constants
    block_i, block_j = CTASwizzle.cta_swizzle(conf.cta_swizzle, blockIdx(), conf.block_shape, conf.matmul_shape)

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    laneId = (threadIdx().x - 1) % 32 + 1

    gemm_sz = Tile(conf.matmul_shape)
    block_tile = Tile(conf.block_shape)

    # Cooperatively store a block_shape.M x block_shape.N tile of D from shared to global memory within one threadblock
    @loopinfo unroll for warp_tile = parallelise(block_tile.MN, Tile(conf.mem_cd_warp), warpId, conf.warps_per_block)
        @loopinfo unroll for thread_tile = parallelise(warp_tile, Tile(conf.mem_cd_thread), laneId, 32)
            x = @inbounds Layout.load(conf.shared_d_layout, shmem_d, thread_tile)
            x = apply_bias(x, ep.bias_pointer, translate_base(thread_tile, (M = block_i, N = block_j)))
            x = transform(x, thread_tile)
            @inbounds Layout.store!(conf.global_d_layout, d, x, translate_base(thread_tile, (M = block_i, N = block_j)))
        end
    end
end


# ------------------
# Split-K reduction
# ------------------

# Split-K: the grid has `S` thread blocks per output tile (blockIdx().z), each of which
# computes the product over its own part of the K dimension. Each block writes its partial
# tile, as it is laid out in shared memory, to a workspace in global memory, and counts the
# finished parts of its tile with an atomic counter. The block that finishes last adds the S
# partial tiles, always in the order of their part (so the result does not depend on the
# order in which blocks finish), resets the counter for the next launch, and runs the
# `inner` epilogue on the sum.
#
# Use with `plan_matmul(...; split_k = S)`, which wraps the epilogue and allocates the
# workspace (S partial tiles per output tile) and the counters (one per output tile) when
# the plan is first executed. Here, the pointers are still null.
struct SplitK{S, T, E}
    workspace::CuPtr{T}
    counters::CuPtr{Int32}
    inner::E
end

SplitK{S, T}(inner) where {S, T} = SplitK{S, T, typeof(inner)}(CuPtr{T}(0), CuPtr{Int32}(0), inner)

# The number of parts the K dimension is split into (1 for any other epilogue).
split_count(::SplitK{S}) where {S} = S
split_count(_) = 1

@inline function (ep::SplitK{S, T})(conf::GemmKernels.Config, d, shmem_d, transform) where {S, T}
    block_i, block_j = CTASwizzle.cta_swizzle(conf.cta_swizzle, blockIdx(), conf.block_shape, conf.matmul_shape)
    tiles_m = cld(conf.matmul_shape.M, conf.block_shape.M)
    tile = (block_j ÷ conf.block_shape.N) * tiles_m + block_i ÷ conf.block_shape.M   # 0-based
    part = blockIdx().z - 1                                                         # 0-based

    n = length(shmem_d)
    base = tile * S * n
    workspace = reinterpret(Core.LLVMPtr{T, AS.Global}, ep.workspace)
    counter = reinterpret(Core.LLVMPtr{Int32, AS.Global}, ep.counters) + tile * sizeof(Int32)

    # store this part of the tile
    i = Int(threadIdx().x)
    while i <= n
        @inbounds unsafe_store!(workspace, shmem_d[i], base + part * n + i)
        i += Int(blockDim().x)
    end
    threadfence()
    sync_threads()

    # count the finished parts; the block that finishes the last one does the reduction
    is_last = false
    if threadIdx().x == 1
        is_last = CUDA.atomic_add!(counter, Int32(1)) == Int32(S - 1)
    end
    sync_threads_or(is_last) || return
    threadfence()

    i = Int(threadIdx().x)
    while i <= n
        acc = zero(T)
        for p in 0:S-1
            acc += unsafe_load(workspace, base + p * n + i)
        end
        @inbounds shmem_d[i] = acc
        i += Int(blockDim().x)
    end
    threadIdx().x == 1 && unsafe_store!(counter, Int32(0))
    sync_threads()

    ep.inner(conf, d, shmem_d, transform)
end

end
