export Kernel
module Kernel

using CUDA
using GemmKernels
using GemmKernels.Tiling
using GemmKernels: LocalArray, @immutable, @unrolled, @not_unrolled, constant, variadic, tid, bid_x, bid_y, warpid, b, vstorea!, vloada, Vec
using Base.Cartesian: @ntuple
using LLVMLoopInfo: @loopinfo

function matmul_singlestage(conf::GemmKernels.Config, a, b, c, d,
                            transf_gl2sh_a, transf_gl2sh_b, transf_gl2sh_c, transf_sh2gl_d,
                            transf_sh2rf_a, transf_sh2rf_b, transf_sh2rf_c, transf_rf2sh_d,
                            epilogue)
    # Calculate the number of fragments needed to fully cover a warp tile
    num_fragments_m = conf.compute_warp.M ÷ conf.compute_op_shape.M
    num_fragments_n = conf.compute_warp.N ÷ conf.compute_op_shape.N

    # Constants
    block_i = (blockIdx().x - 1) * conf.block_shape.M
    block_j = (blockIdx().y - 1) * conf.block_shape.N

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    laneId = (threadIdx().x - 1) % 32 + 1

    gemm_sz = Tile(conf.matmul_shape)
    block_tile = Tile(conf.block_shape)

    # (1) Cooperatively load a block_shape.M x block_shape.N tile of C from global to shared memory within one threadblock
    shmem_c = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_c_layout), Layout.physical_size(conf.shared_c_layout, block_tile.MN.size))

    @loopinfo unroll for warp_tile = parallelise(block_tile.MN, Tile(conf.mem_cd_warp), warpId, conf.warps_per_block)
        @loopinfo unroll for thread_tile = parallelise(warp_tile, Tile(conf.mem_cd_thread), laneId, 32)
            x = @inbounds Layout.load(conf.global_c_layout, c, translate(thread_tile, (M = variadic(block_i), N = variadic(block_j))))
            x = transf_gl2sh_c(x, thread_tile)
            @inbounds Layout.store!(conf.shared_c_layout, shmem_c, x, thread_tile)
        end
    end

    sync_threads()

    # (2) Load a compute_warp.M x compute_warp.N tile of C from shared memory into registers
    warp_tile = @inbounds subdivide(block_tile.MN, Tile(conf.compute_warp).MN, warpId, conf.warps_per_block)

    c_frags = LocalArray{Tuple{num_fragments_m, num_fragments_n}, Operator.fragtype_accum(conf.operator, conf.shared_c_layout)}(undef)

    @loopinfo unroll for i = 1 : num_fragments_m
        @loopinfo unroll for j = 1 : num_fragments_n
            tile = translate(warp_tile, (M = constant((i-1)*conf.compute_op_shape.M), N = constant((j-1)*conf.compute_op_shape.N)))
            @inbounds @immutable c_frags[i, j] = transf_sh2rf_c(Operator.load_c(conf.operator, conf.shared_c_layout, shmem_c, tile), tile)
        end
    end

    sync_threads()

    # (3) Compute a block_shape.M x block_shape.N x block_shape.K matrix product within one threadblock
    shmem_a = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_a_layout), Layout.physical_size(conf.shared_a_layout, block_tile.MK.size))
    shmem_b = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_b_layout), Layout.physical_size(conf.shared_b_layout, block_tile.KN.size),
                                  length(shmem_a) * sizeof(Layout.eltype(conf.shared_a_layout)))

    @loopinfo unroll for block_k = 0 : block_tile.size.K : gemm_sz.size.K - 1
        if Layout.threadblock_condition(conf.global_a_layout, conf.global_b_layout, block_i, block_j, block_k, block_tile)
            # (3.1) Cooperatively load a block_shape.M x block_shape.K tile of A from global to shared memory within one threadblock
            @loopinfo unroll for warp_tile = parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major)
                @loopinfo unroll for thread_tile = parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major)
                    x = @inbounds Layout.load(conf.global_a_layout, a, translate(thread_tile, (M = variadic(block_i), K = variadic(block_k))))
                    x = transf_gl2sh_a(x, thread_tile)
                    @inbounds Layout.store!(conf.shared_a_layout, shmem_a, x, thread_tile)
                end
            end

            # (3.2) Cooperatively load a block_shape.K x block_shape.N tile of B from global to shared memory within one threadblock
            @loopinfo unroll for warp_tile = parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major)
                @loopinfo unroll for thread_tile = parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major)
                    x = @inbounds Layout.load(conf.global_b_layout, b, translate(thread_tile, (K = variadic(block_k), N = variadic(block_j))))
                    x = transf_gl2sh_b(x, thread_tile)
                    @inbounds Layout.store!(conf.shared_b_layout, shmem_b, x, thread_tile)
                end
            end

            sync_threads()

            # (3.3) Calculate a compute_warp.M x compute_warp.N tile of D, using a compute_warp.M x compute_warp.N x compute_warp.K operation
            @loopinfo unroll for warp_tile = parallelise(block_tile, Tile(conf.compute_warp), warpId, conf.warps_per_block)
                # (3.3.1) Load a compute_warp.M x compute_warp.K tile of A from shared memory into registers
                a_frags = LocalArray{Tuple{num_fragments_m}, Operator.fragtype_a(conf.operator, conf.shared_a_layout)}(undef)

                @loopinfo unroll for i = 1 : num_fragments_m
                    a_tile = translate(warp_tile.MK, (M = constant((i-1)*conf.compute_op_shape.M), K = constant(0)))
                    @inbounds @immutable a_frags[i] = transf_sh2rf_a(Operator.load_a(conf.operator, conf.shared_a_layout, shmem_a, a_tile), a_tile)
                end

                # (3.3.2) Load a compute_warp.K x compute_warp.N tile of B from shared memory into registers
                b_frags = LocalArray{Tuple{num_fragments_n}, Operator.fragtype_b(conf.operator, conf.shared_b_layout)}(undef)

                @loopinfo unroll for j = 1 : num_fragments_n
                    b_tile = translate(warp_tile.KN, (K = constant(0), N = constant((j-1)*conf.compute_op_shape.N)))
                    @inbounds @immutable b_frags[j] = transf_sh2rf_b(Operator.load_b(conf.operator, conf.shared_b_layout, shmem_b, b_tile), b_tile)
                end

                # (3.3.3) Compute a compute_warp.M x compute_warp.N x compute_warp.K matrix product within one warp
                @loopinfo unroll for i = 1 : num_fragments_m
                    @loopinfo unroll for j = 1 : num_fragments_n
                        @inbounds @immutable c_frags[i, j] = Operator.mma(conf.operator, a_frags[i], b_frags[j], c_frags[i, j])
                    end
                end
            end
        end

        sync_threads()
    end

    # (4) Store the compute_warp.M x compute_warp.N tile of D from registers to shared memory
    shmem_d = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_d_layout), Layout.physical_size(conf.shared_d_layout, block_tile.MN.size))

    warp_tile = @inbounds subdivide(block_tile.MN, Tile(conf.compute_warp).MN, warpId, conf.warps_per_block)

    @loopinfo unroll for i = 1 : num_fragments_m
        @loopinfo unroll for j = 1 : num_fragments_n
            tile = translate(warp_tile, (M = constant((i-1)*conf.compute_op_shape.M), N = constant((j-1)*conf.compute_op_shape.N)))
            @inbounds Operator.store_d(conf.operator, conf.shared_d_layout, shmem_d, transf_rf2sh_d(c_frags[i, j], tile), tile)
        end
    end

    sync_threads()

    # (5) Run the epilogue
    epilogue(conf, d, shmem_d, transf_sh2gl_d)

    return
end

function shmem_size(conf::GemmKernels.Config, ::typeof(matmul_singlestage))
    size_a = sizeof(Layout.eltype(conf.shared_a_layout)) *
             prod(Layout.physical_size(conf.shared_a_layout,
                  (; conf.block_shape.M, conf.block_shape.K)))
    size_b = sizeof(Layout.eltype(conf.shared_b_layout)) *
             prod(Layout.physical_size(conf.shared_b_layout,
                  (; conf.block_shape.K, conf.block_shape.N)))
    size_c = sizeof(Layout.eltype(conf.shared_c_layout)) *
             prod(Layout.physical_size(conf.shared_c_layout,
                  (; conf.block_shape.M, conf.block_shape.N)))
    size_d = sizeof(Layout.eltype(conf.shared_d_layout)) *
             prod(Layout.physical_size(conf.shared_d_layout,
                  (; conf.block_shape.M, conf.block_shape.N)))
    max(size_c, size_a + size_b, size_d)
end

function matmul_pipelined(conf::GemmKernels.Config, a, b, c, d,
                          transf_gl2sh_a, transf_gl2sh_b, transf_gl2sh_c, transf_sh2gl_d,
                          transf_sh2rf_a, transf_sh2rf_b, transf_sh2rf_c, transf_rf2sh_d,
                          epilogue)
    # Calculate the number of fragments needed to fully cover a warp tile
    num_fragments_m = conf.compute_warp.M ÷ conf.compute_op_shape.M
    num_fragments_n = conf.compute_warp.N ÷ conf.compute_op_shape.N

    # Constants
    block_i = (blockIdx().x - 1) * conf.block_shape.M
    block_j = (blockIdx().y - 1) * conf.block_shape.N

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    laneId = (threadIdx().x - 1) % 32 + 1

    gemm_sz = Tile(conf.matmul_shape)
    block_tile = Tile(conf.block_shape)

    # (1) Cooperatively load a block_shape.M x block_shape.N tile of C from global to shared memory within one threadblock
    shmem_c = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_c_layout), Layout.physical_size(conf.shared_c_layout, block_tile.MN.size))

    @loopinfo unroll for warp_tile = parallelise(block_tile.MN, Tile(conf.mem_cd_warp), warpId, conf.warps_per_block)
        @loopinfo unroll for thread_tile = parallelise(warp_tile, Tile(conf.mem_cd_thread), laneId, 32)
            x = @inbounds Layout.load(conf.global_c_layout, c, translate(thread_tile, (M = variadic(block_i), N = variadic(block_j))))
            x = transf_gl2sh_c(x, thread_tile)
            @inbounds Layout.store!(conf.shared_c_layout, shmem_c, x, thread_tile)
        end
    end

    sync_threads()

    # (2) Load a compute_warp.M x compute_warp.N tile of C from shared memory into registers
    warp_tile = subdivide(block_tile.MN, Tile(conf.compute_warp).MN, warpId, conf.warps_per_block)

    c_frags = LocalArray{Tuple{num_fragments_m, num_fragments_n}, Operator.fragtype_accum(conf.operator, conf.shared_c_layout)}(undef)

    @loopinfo unroll for i = 1 : num_fragments_m
        @loopinfo unroll for j = 1 : num_fragments_n
            tile = translate(warp_tile, (M = constant((i-1)*conf.compute_op_shape.M), N = constant((j-1)*conf.compute_op_shape.N)))
            @inbounds @immutable c_frags[i, j] = transf_sh2rf_c(Operator.load_c(conf.operator, conf.shared_c_layout, shmem_c, tile), tile)
        end
    end

    sync_threads()

    # (3) Compute a block_shape.M x block_shape.N x block_shape.K matrix product within one threadblock
    shmem_a = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_a_layout), Layout.physical_size(conf.shared_a_layout, block_tile.MK.size))
    shmem_b = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_b_layout), Layout.physical_size(conf.shared_b_layout, block_tile.KN.size),
                                  length(shmem_a) * sizeof(Layout.eltype(conf.shared_a_layout)))

    # Sizes of a_fragment and b_fragment
    a_frag_i = (block_tile.size.M * block_tile.size.K) ÷ (conf.mem_a_warp.M * conf.mem_a_warp.K * conf.warps_per_block)
    a_frag_j = (conf.mem_a_warp.M * conf.mem_a_warp.K) ÷ (conf.mem_a_thread.M * conf.mem_a_thread.K * 32)
    b_frag_i = (block_tile.size.K * block_tile.size.N) ÷ (conf.mem_b_warp.K * conf.mem_b_warp.N * conf.warps_per_block)
    b_frag_j = (conf.mem_b_warp.K * conf.mem_b_warp.N) ÷ (conf.mem_b_thread.K * conf.mem_b_thread.N * 32)

    a_fragment = LocalArray{Tuple{a_frag_i, a_frag_j}, Layout.fragtype(conf.global_a_layout, conf.mem_a_thread)}(undef)
    b_fragment = LocalArray{Tuple{b_frag_i, b_frag_j}, Layout.fragtype(conf.global_b_layout, conf.mem_b_thread)}(undef)

    a_frags = LocalArray{Tuple{2, num_fragments_m}, Operator.fragtype_a(conf.operator, conf.shared_a_layout)}(undef)
    b_frags = LocalArray{Tuple{2, num_fragments_n}, Operator.fragtype_b(conf.operator, conf.shared_b_layout)}(undef)

    warp_tile_mn = subdivide(block_tile, Tile(conf.compute_warp), warpId, conf.warps_per_block)

    # ld.global(0 : block_shape.K)
    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major))
            @inbounds @immutable a_fragment[i,j] = Layout.load(conf.global_a_layout, a, translate(thread_tile, (M = variadic(block_i), K = constant(0))))
        end
    end

    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
            @inbounds @immutable b_fragment[i,j] = Layout.load(conf.global_b_layout, b, translate(thread_tile, (K = constant(0), N = variadic(block_j))))
        end
    end

    # st.shared()
    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major))
            x = transf_gl2sh_a(@inbounds(a_fragment[i, j]), thread_tile)
            @inbounds Layout.store!(conf.shared_a_layout, shmem_a, x, thread_tile)
        end
    end

    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
            x = transf_gl2sh_b(@inbounds(b_fragment[i, j]), thread_tile)
            @inbounds Layout.store!(conf.shared_b_layout, shmem_b, x, thread_tile)
        end
    end

    sync_threads()

    # ld.shared(0 : compute_op_shape.K, stage = 1)
    warp_tile = translate(warp_tile_mn, (M = constant(0), N = constant(0), K = constant(0)))

    @loopinfo unroll for i = 1 : num_fragments_m
        a_tile = translate(warp_tile.MK, (M = constant((i-1)*conf.compute_op_shape.M), K = constant(0)))
        @inbounds @immutable a_frags[1, i] = transf_sh2rf_a(Operator.load_a(conf.operator, conf.shared_a_layout, shmem_a, a_tile), a_tile)
    end

    @loopinfo unroll for j = 1 : num_fragments_n
        b_tile = translate(warp_tile.KN, (K = constant(0), N = constant((j-1)*conf.compute_op_shape.N)))
        @inbounds @immutable b_frags[1, j] = transf_sh2rf_b(Operator.load_b(conf.operator, conf.shared_b_layout, shmem_b, b_tile), b_tile)
    end

    # ld.global(block_shape.K : 2 * block_shape.K)
    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major))
            @inbounds @immutable a_fragment[i, j] = Layout.load(conf.global_a_layout, a, translate(thread_tile, (M = variadic(block_i), K = constant(block_tile.size.K))))
        end
    end

    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
            @inbounds @immutable b_fragment[i, j] = Layout.load(conf.global_b_layout, b, translate(thread_tile, (K = constant(block_tile.size.K), N = variadic(block_j))))
        end
    end

    @loopinfo unrollcount=1 for block_k = 0 : block_tile.size.K : gemm_sz.size.K - 1
        @loopinfo unroll for (i, warp_k) = enumerate(0 : conf.compute_op_shape.K : block_tile.size.K - 1)
            cur_stage = mod1(i, 2)
            nxt_stage = mod1(i + 1, 2)

            if i == block_tile.size.K ÷ conf.compute_op_shape.K # last iteration
                sync_threads()

                # st.shared()
                @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major))
                    @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major))
                        x = transf_gl2sh_a(@inbounds(a_fragment[i, j]), thread_tile)
                        @inbounds Layout.store!(conf.shared_a_layout, shmem_a, x, thread_tile)
                    end
                end

                @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
                    @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
                        x = transf_gl2sh_b(@inbounds(b_fragment[i, j]), thread_tile)
                        @inbounds Layout.store!(conf.shared_b_layout, shmem_b, x, thread_tile)
                    end
                end

                sync_threads()

                # avoid out of bounds access for global memory
                if block_k < (gemm_sz.size.K - 2 * block_tile.size.K)
                    # ld.global(block_k + 2 * block_shape.K : block_k + 3 * block_shape.K)
                    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major))
                        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major))
                            @inbounds @immutable a_fragment[i, j] = Layout.load(conf.global_a_layout, a, translate(thread_tile, (M = variadic(block_i), K = variadic(block_k + 2 * block_tile.size.K))))
                        end
                    end

                    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
                        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
                            @inbounds @immutable b_fragment[i, j] = Layout.load(conf.global_b_layout, b, translate(thread_tile, (K = variadic(block_k + 2 * block_tile.size.K), N = variadic(block_j))))
                        end
                    end
                end
            end

            # ld.shared((warp_k + compute_op_shape.K) % block_shape.K, stage = nxt_stage)
            warp_tile = translate(warp_tile_mn, (M = constant(0), N = constant(0), K = constant((warp_k + conf.compute_op_shape.K) % block_tile.size.K)))

            @loopinfo unroll for i = 1 : num_fragments_m
                a_tile = translate(warp_tile.MK, (M = constant((i-1)*conf.compute_op_shape.M), K = constant(0)))
                @inbounds @immutable a_frags[nxt_stage, i] = transf_sh2rf_a(Operator.load_a(conf.operator, conf.shared_a_layout, shmem_a, a_tile), a_tile)
            end

            @loopinfo unroll for j = 1 : num_fragments_n
                b_tile = translate(warp_tile.KN, (K = constant(0), N = constant((j-1)*conf.compute_op_shape.N)))
                @inbounds @immutable b_frags[nxt_stage, j] = transf_sh2rf_b(Operator.load_b(conf.operator, conf.shared_b_layout, shmem_b, b_tile), b_tile)
            end

            # mma(cur_stage)
            @loopinfo unroll for i = 1 : num_fragments_m
                @loopinfo unroll for j = 1 : num_fragments_n
                    @inbounds @immutable c_frags[i, j] = Operator.mma(conf.operator, a_frags[cur_stage, i], b_frags[cur_stage, j], c_frags[i, j])
                end
            end
        end

        sync_threads()
    end

    # (4) Store the compute_warp.M x compute_warp.N tile of D from registers to shared memory
    shmem_d = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_d_layout), Layout.physical_size(conf.shared_d_layout, block_tile.MN.size))

    warp_tile = subdivide(block_tile.MN, Tile(conf.compute_warp).MN, warpId, conf.warps_per_block)

    @loopinfo unroll for i = 1 : num_fragments_m
        @loopinfo unroll for j = 1 : num_fragments_n
            tile = translate(warp_tile, (M = constant((i-1)*conf.compute_op_shape.M), N = constant((j-1)*conf.compute_op_shape.N)))
            @inbounds Operator.store_d(conf.operator, conf.shared_d_layout, shmem_d, transf_rf2sh_d(c_frags[i, j], tile), tile)
        end
    end

    sync_threads()

    # (5) Run the epilogue
    epilogue(conf, d, shmem_d, transf_sh2gl_d)

    return
end

function shmem_size(conf::GemmKernels.Config, ::typeof(matmul_pipelined))
    size_a = sizeof(Layout.eltype(conf.shared_a_layout)) *
             prod(Layout.physical_size(conf.shared_a_layout,
                  (; conf.block_shape.M, conf.block_shape.K)))
    size_b = sizeof(Layout.eltype(conf.shared_b_layout)) *
             prod(Layout.physical_size(conf.shared_b_layout,
                  (; conf.block_shape.K, conf.block_shape.N)))
    size_c = sizeof(Layout.eltype(conf.shared_c_layout)) *
             prod(Layout.physical_size(conf.shared_c_layout,
                  (; conf.block_shape.M, conf.block_shape.N)))
    size_d = sizeof(Layout.eltype(conf.shared_d_layout)) *
             prod(Layout.physical_size(conf.shared_d_layout,
                  (; conf.block_shape.M, conf.block_shape.N)))
    max(size_c, size_a + size_b, size_d)
end

# Volta Kernel {{{
# TODO: Introduce necessary abstractions so we do not need a separate kernel!

# ld global {{{
# Load from global memory
@inline function volta_ld_global(A, B, cta_m, cta_n, cta_k, conf)
    # Fragments for the data from the global loads (and hence shared stores).
    # index: (m3|k2|k1|k0)
    a_frag = LocalArray{Tuple{16}, Float16}(undef)

    # index: (n7|n6|n2|n1|n0)
    b_frag = LocalArray{Tuple{32}, Float16}(undef)

    # Load A from Global Memory.
    @unrolled for ins_1 = 0:1
        m = b(tid(), 2, 0) +
            b(tid(), 3, 1) +
            b(tid(), 4, 2) +
            b(ins_1, 0, 3) +
            b(tid(), 5, 4) +
            b(tid(), 6, 5) +
            b(tid(), 7, 6)

        k = b(tid(), 0, 3) +
            b(tid(), 1, 4)

        tile = Tile(M = 1, K = 8)
        tile = translate(tile, (M = cta_m + m, K = cta_k + k))

        @inbounds val = Layout.load(conf.global_a_layout, A, tile)

        @unrolled for offset = 0:7
            frag_offset = b(offset, 0, 0) +  # k0
                          b(offset, 1, 1) +  # k1
                          b(offset, 2, 2) +  # k2
                          b(ins_1, 0, 3)     # m3

            @inbounds @immutable a_frag[frag_offset] = val[offset].value
        end
    end

    # Load B from Global Memory.
    @unrolled for ins = 0:3
        n = b(tid(), 0, 3) +
            b(tid(), 1, 4) +
            b(tid(), 2, 5) +
            b(ins, 0, 6) +
            b(ins, 1, 7)

        k = b(tid(), 3, 0) +
            b(tid(), 4, 1) +
            b(tid(), 5, 2) +
            b(tid(), 6, 3) +
            b(tid(), 7, 4)

        tile = Tile(K = 1, N = 8)
        tile = translate(tile, (K = cta_k + k, N = cta_n + n))

        @inbounds val = Layout.load(conf.global_b_layout, B, tile)

        @unrolled for offset = 0:7
            frag_offset = b(offset, 0, 0) + # n0
                          b(offset, 1, 1) + # n1
                          b(offset, 2, 2) + # n2
                          b(ins, 0, 3) +    # n6
                          b(ins, 1, 4)      # n7
            @inbounds @immutable b_frag[frag_offset] = val[offset].value
        end
    end

    a_frag, b_frag
end
# }}}

# st shared {{{
@inline function volta_st_shared(shmem_a, shmem_b, a_frag, b_frag, conf)
    # Store A to Shared Memory.
    block_tile = Tile(M = 128, N = 256, K = 32)

    let
        # TODO: Introduce abstraction for this: CopyStrategy or CopyThreadMapping.
        # And multiple dispatch function: preferred_copy_strategy(global_layout, shared_layout).

        # Thread mapping for global -> shared copy for A.
        # TODO: Do not iterate over ins here, but instead iterate over individual elements,
        # and let the layout handle vectorisation for us... Maybe by letting the layout emit
        # stores with alignment, and letting LSV handle the rest?
        @unrolled for ins = 0 : 3
            m = b(tid(), 2, 0) +
                b(tid(), 3, 1) +
                b(tid(), 4, 2) +
                b(ins, 1, 3) +
                b(tid(), 5, 4) +
                b(tid(), 6, 5) +
                b(tid(), 7, 6)

            k = b(ins, 0, 2) +
                b(tid(), 0, 3) +
                b(tid(), 1, 4)

            @inbounds val = @ntuple 4 i -> begin
                offset = constant(i-1)
                frag_offset = b(offset, 0, 0) + # k0
                              b(offset, 1, 1) + # k1
                              b(ins, 0, 2) +    # k2
                              b(ins, 1, 3)      # m3
                VecElement{Float16}(a_frag[frag_offset])
            end

            tile = Tile(M = 1, K = 4)
            tile = translate(tile, (M = m, K = k))

            Layout.store!(conf.shared_a_layout, shmem_a, val, tile)
        end
    end

    let
        @unrolled for ins = 0:3
            n = b(tid(), 0, 3) +
                b(tid(), 1, 4) +
                b(tid(), 2, 5) +
                b(ins, 0, 6) +
                b(ins, 1, 7)

            k = b(tid(), 3, 0) +
                b(tid(), 4, 1) +
                b(tid(), 5, 2) +
                b(tid(), 6, 3) +
                b(tid(), 7, 4)

            @inbounds val = @ntuple 8 i -> begin
                offset = constant(i-1)
                frag_offset = b(offset, 0, 0) + # n0
                            b(offset, 1, 1) + # n1
                            b(offset, 2, 2) + # n2
                            b(ins, 0, 3) +    # n6
                            b(ins, 1, 4)      # n7
                VecElement{Float16}(b_frag[frag_offset])
            end

            tile = Tile(K = 1, N = 8)
            tile = translate(tile, (K = k, N = n))

            Layout.store!(conf.shared_b_layout, shmem_b, val, tile)
        end
    end
end
# }}}

# ld shared {{{
@inline function volta_ld_shared(shmem_a, shmem_b, warp_m, warp_n, warp_k, conf)
    warp_mma_k = warp_k ÷ conf.compute_warp.K

    b_frag = LocalArray{Tuple{16}, Float16}(undef)

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    block_tile = Tile(M = 128, N = 256, K = 32)
    warp_tile_mn = subdivide(block_tile, Tile(M = 64, N = 64, K = 32), warpId, 8)

    tile = translate(warp_tile_mn, (M = constant(0), N = constant(0), K = warp_k))

    a_frag = Operator.load_a(conf.operator, conf.shared_a_layout, shmem_a, tile)
    b_frag = Operator.load_b(conf.operator, conf.shared_b_layout, shmem_b, tile)

    a_frag, b_frag
end
# }}}

# epilogue {{{
@inline function volta_epilogue_st_shared(epilogue_it, shmem_d, warp_m, warp_n, acc_frag, conf)
    # index: (m5|m2|m1|n5|n4|n2|n0)

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    block_tile = Tile(M = 128, N = 256, K = 32)
    warp_tile = subdivide(block_tile.MN, Tile(M = 64, N = 64), warpId, 8)
    m_offset = b(epilogue_it, 0, 2) + b(epilogue_it, 1, 5)
    tile = translate(warp_tile, (M = m_offset, N = constant(0)))

    frag_base = b(epilogue_it, 0, 5) + # m2
                b(epilogue_it, 1, 6)   # m5

    frag = view(acc_frag, (convert(Int, frag_base) + 1):length(acc_frag))

    Operator.store_d(conf.operator, Layout.Padded{Layout.UnsafeAlignedRowMajor{Float16}, 2}, shmem_d, frag, tile)
end

@inline function volta_epilogue_ld_shared(epilogue_it, shmem_d)
    # index: (m1|n7|n6|n1|n0)
    frag = LocalArray{Tuple{32}, Float32}(undef)

    @unrolled for ins = 0:7
        # TODO: vectorise
        @unrolled for offset = 0:3
            m = b(tid(), 4, 0) +
                b(ins, 2, 1) +
                b(epilogue_it, 0, 2) +
                b(tid(), 5, 3) +
                b(tid(), 6, 4) +
                b(epilogue_it, 1, 5) +
                b(tid(), 7, 6)

            n = b(offset, 0, 0) +
                b(offset, 1, 1) +
                b(tid(), 0, 2) +
                b(tid(), 1, 3) +
                b(tid(), 2, 4) +
                b(tid(), 3, 5) +
                b(ins, 0, 6) +
                b(ins, 1, 7)

            offset_M = b(tid(), 4, 0) +     # m0
                       b(ins, 2, 1) +       # m1
                       b(tid(), 5, 2) +     # m3
                       b(tid(), 6, 3) +     # m4
                       b(tid(), 7, 4)       # m6

            offset_N = n

            shmem_offset = convert(Int, offset_N) + 258 * convert(Int, offset_M)

            frag_index = b(offset, 0, 0) + # n0
                         b(offset, 1, 1) + # n1
                         b(ins, 0, 2) +    # n6
                         b(ins, 1, 3) +    # n7
                         b(ins, 2, 4)      # m1

            @inbounds @immutable frag[frag_index] = shmem_d[1 + shmem_offset]
        end
    end

    frag
end

@inline function volta_epilogue_st_global(epilogue_it, D, shmem_d, cta_m, cta_n, frag, conf)
    # TODO: EXTRACT
    @unrolled for ins = 0:7
        m = b(tid(), 4, 0) +
            b(ins, 2, 1) +
            b(epilogue_it, 0, 2) +
            b(tid(), 5, 3) +
            b(tid(), 6, 4) +
            b(epilogue_it, 1, 5) +
            b(tid(), 7, 6)

        n = b(tid(), 0, 2) +
            b(tid(), 1, 3) +
            b(tid(), 2, 4) +
            b(tid(), 3, 5) +
            b(ins, 0, 6) +
            b(ins, 1, 7)


        @inbounds val = @ntuple 4 i -> begin
            offset = constant(i-1)
            frag_index = b(offset, 0, 0) + # n0
                         b(offset, 1, 1) + # n1
                         b(ins, 0, 2) +    # n6
                         b(ins, 1, 3) +    # n7
                         b(ins, 2, 4)      # m1
            VecElement{Float32}(frag[frag_index])
        end

        @inbounds vstorea!(Vec{4, Float32}, D, cta_n + n + conf.matmul_shape.N * cta_m + conf.matmul_shape.N * m, val)
    end
end

@inline function volta_epilogue(D, shmem_d, acc_frag, cta_m, cta_n, warp_m, warp_n, conf)
    # TODO: EXTRACT
    @unrolled for epilogue_it = 0:3
        # TODO: Can we not remove this one?
        sync_threads()

        # Store tile of D to shared memory.
        volta_epilogue_st_shared(epilogue_it, shmem_d, warp_m, warp_n, acc_frag, conf)

        sync_threads()

        # Load tile of D from shared memory.
        frag = volta_epilogue_ld_shared(epilogue_it, shmem_d)

        # Store tile of D to global memory.
        volta_epilogue_st_global(epilogue_it, D, shmem_d, cta_m, cta_n, frag, conf)
    end
end
# }}}

# kernel {{{
# row-major A x row-major B = row-major D
function volta_kernel(conf::GemmKernels.Config, A, B, C, D,
        transf_gl2sh_a, transf_gl2sh_b, transf_gl2sh_c, transf_sh2gl_d,
        transf_sh2rf_a, transf_sh2rf_b, transf_sh2rf_c, transf_rf2sh_d,
        epilogue)
    # The modulo is so that the BitArrayIndex knows which bits are 0.
    num_warps_m = conf.block_shape.M ÷ conf.compute_warp.M
    num_warps_n = conf.block_shape.N ÷ conf.compute_warp.N

    num_blocks_m = conf.matmul_shape.M ÷ conf.block_shape.M
    num_blocks_n = conf.matmul_shape.N ÷ conf.block_shape.N

    num_warps = num_warps_m * num_warps_n
    warpid_m = (warpid() % num_warps) % num_warps_m
    warpid_n = (warpid() % num_warps) ÷ num_warps_n

    cta_m = (bid_x() % num_blocks_m) * conf.block_shape.M
    cta_n = (bid_y() % num_blocks_n) * conf.block_shape.N

    warp_m = warpid_m * conf.compute_warp.M
    warp_n = warpid_n * conf.compute_warp.N

    shmem_a = CuDynamicSharedArray(Float16, (conf.block_shape.M * conf.block_shape.K, 2))
    shmem_b = CuDynamicSharedArray(Float16, (conf.block_shape.K * conf.block_shape.N, 2),
                                   length(shmem_a) * sizeof(Float16))

    block_tile = Tile(conf.block_shape)

    shmem_d = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_d_layout), Layout.physical_size(conf.shared_d_layout, block_tile.MN.size))

    # index: (m5|m2|m1|n5|n4|n2|n0)
    acc_frag = LocalArray{Tuple{128}, Float32}(@ntuple 128 i -> zero(Float32))

    # Two pairs of fragments for the shared -> register file pipeline.
    shared_a_frags = LocalArray{Tuple{2}, NTuple{16, Float16}}(undef)
    shared_b_frags = LocalArray{Tuple{2}, NTuple{16, Float16}}(undef)

    # Prologue.
    # ld_global(main_loop_it=0)
    cta_k = constant(0)
    global_a_frag, global_b_frag = volta_ld_global(A, B, cta_m, cta_n, cta_k, conf)

    # st_shared(main_loop_it=0)
    main_loop_it = constant(0)
    volta_st_shared(view(shmem_a, :, convert(Int, main_loop_it % 2) + 1),
                    view(shmem_b, :, convert(Int, main_loop_it % 2) + 1),
                    global_a_frag, global_b_frag, conf)
    sync_threads()

    # ld_shared(main_loop_it=0, warp_mma_k=0)
    main_loop_it = constant(0)
    warp_k = constant(0)
    warp_mma_k = constant(0)
    shared_a_frag, shared_b_frag = volta_ld_shared(view(shmem_a, :, convert(Int, main_loop_it % 2) + 1),
                                                   view(shmem_b, :, convert(Int, main_loop_it % 2) + 1),
                                                   warp_m, warp_n, warp_k, conf)

    @inbounds @immutable shared_a_frags[convert(Int, warp_mma_k % 2) + 1] = shared_a_frag
    @inbounds @immutable shared_b_frags[convert(Int, warp_mma_k % 2) + 1] = shared_b_frag

    NUM_MAIN_LOOP_ITERS = conf.matmul_shape.K ÷ conf.block_shape.K
    @not_unrolled for main_loop_it = 0 : NUM_MAIN_LOOP_ITERS - 1
        # The modulo is so that the BitArrayIndex knowns which bits are 0.
        # TODO: Do this automatically in the @not_unrolled macro?
        # TODO: Generate _next variables automatically.
        cta_k = (main_loop_it % NUM_MAIN_LOOP_ITERS) * conf.block_shape.K

        main_loop_it_next = variadic((main_loop_it_orig + 1)) % NUM_MAIN_LOOP_ITERS
        cta_k_next = main_loop_it_next * conf.block_shape.K

        # CTA_M x CTA_N x CTA_K GEMM per CTA
        NUM_WARP_MMA_K_ITERS = conf.block_shape.K ÷ conf.compute_warp.K
        @unrolled for warp_mma_k = 0 : NUM_WARP_MMA_K_ITERS - 1
            warp_k = warp_mma_k * conf.compute_warp.K

            # TODO: Do this in macro.
            warp_mma_k_next = constant(warp_mma_k_orig + 1) % NUM_WARP_MMA_K_ITERS
            warp_k_next = warp_mma_k_next * conf.compute_warp.K

            if warp_mma_k == NUM_WARP_MMA_K_ITERS-1
                # st_shared(main_loop_it+1)
                volta_st_shared(view(shmem_a, :, convert(Int, main_loop_it_next % 2) + 1),
                                view(shmem_b, :, convert(Int, main_loop_it_next % 2) + 1),
                                global_a_frag, global_b_frag, conf)
                sync_threads()
            end

            # ld_shared(main_loop_it, warp_mma_k + 1)
            shared_a_frag, shared_b_frag = volta_ld_shared(view(shmem_a, :, convert(Int, main_loop_it % 2) + 1),
                                                           view(shmem_b, :, convert(Int, main_loop_it % 2) + 1),
                                                           warp_m, warp_n, warp_k_next, conf)

            @inbounds @immutable shared_a_frags[convert(Int, warp_mma_k_next % 2) + 1] = shared_a_frag
            @inbounds @immutable shared_b_frags[convert(Int, warp_mma_k_next % 2) + 1] = shared_b_frag

            # TODO: Predicate the load?
            if warp_mma_k == 0
                # ld_global(main_loop_it + 1)
                # Copy the data for a CTA_M x CTA_N x CTA_K GEMM from GMEM to SHMEM, cooperatively in a CTA.
                global_a_frag, global_b_frag = volta_ld_global(A, B, cta_m, cta_n, cta_k_next, conf)
            end

            # WARP_M x WARP_N x WARP_K = 64 x 64 x 4 GEMM per warp
            # mma(main_loop_it, warp_mma_k)
            @inbounds shared_a_frag = shared_a_frags[convert(Int, warp_mma_k % 2) + 1]
            @inbounds shared_b_frag = shared_b_frags[convert(Int, warp_mma_k % 2) + 1]

            acc_frag = Operator.mma(conf.operator, shared_a_frag, shared_b_frag, acc_frag)
        end
    end

    # epilogue: store matrix from registers to global memory
    volta_epilogue(D, shmem_d, acc_frag, cta_m, cta_n, warp_m, warp_n, conf)

    nothing
end
# }}}

function shmem_size(conf::GemmKernels.Config, ::typeof(volta_kernel))
    # TODO: Do not hardcode this.
    return 48*1024
end


end
