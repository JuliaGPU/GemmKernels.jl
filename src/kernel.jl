export Kernel
module Kernel

using CUDA
using GemmKernels
using GemmKernels.Tiling
using GemmKernels: LocalArray, @immutable, CTASwizzle
using LLVMLoopInfo: @loopinfo

function matmul_singlestage(conf::GemmKernels.Config, a, b, c, d,
                            transf_gl2sh_a, transf_gl2sh_b, transf_gl2sh_c, transf_sh2gl_d,
                            transf_sh2rf_a, transf_sh2rf_b, transf_sh2rf_c, transf_rf2sh_d,
                            epilogue)
    # Calculate the number of fragments needed to fully cover a warp tile
    num_fragments_m = conf.compute_warp.M ÷ conf.compute_op_shape.M
    num_fragments_n = conf.compute_warp.N ÷ conf.compute_op_shape.N

    # Constants
    block_i, block_j = CTASwizzle.cta_swizzle(conf.cta_swizzle, blockIdx(), conf.block_shape)

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    laneId = (threadIdx().x - 1) % 32 + 1

    gemm_sz = Tile(conf.matmul_shape)
    block_tile = Tile(conf.block_shape)

    # (1) Cooperatively load a block_shape.M x block_shape.N tile of C from global to shared memory within one threadblock
    shmem_c = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_c_layout), Layout.physical_size(conf.shared_c_layout, block_tile.MN.size))

    @loopinfo unroll for warp_tile = parallelise(block_tile.MN, Tile(conf.mem_cd_warp), warpId, conf.warps_per_block)
        @loopinfo unroll for thread_tile = parallelise(warp_tile, Tile(conf.mem_cd_thread), laneId, 32)
            x = @inbounds Layout.load(conf.global_c_layout, c, translate_base(thread_tile, (M = block_i, N = block_j)))
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
            tile = translate_offset(warp_tile, (M = (i-1)*conf.compute_op_shape.M, N = (j-1)*conf.compute_op_shape.N))
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
                    x = @inbounds Layout.load(conf.global_a_layout, a, translate_base(thread_tile, (M = block_i, K = block_k)))
                    x = transf_gl2sh_a(x, thread_tile)
                    @inbounds Layout.store!(conf.shared_a_layout, shmem_a, x, thread_tile)
                end
            end

            # (3.2) Cooperatively load a block_shape.K x block_shape.N tile of B from global to shared memory within one threadblock
            @loopinfo unroll for warp_tile = parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major)
                @loopinfo unroll for thread_tile = parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major)
                    x = @inbounds Layout.load(conf.global_b_layout, b, translate_base(thread_tile, (K = block_k, N = block_j)))
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
                    a_tile = translate_offset(warp_tile.MK, (M = (i-1)*conf.compute_op_shape.M, K = 0))
                    @inbounds @immutable a_frags[i] = transf_sh2rf_a(Operator.load_a(conf.operator, conf.shared_a_layout, shmem_a, a_tile), a_tile)
                end

                # (3.3.2) Load a compute_warp.K x compute_warp.N tile of B from shared memory into registers
                b_frags = LocalArray{Tuple{num_fragments_n}, Operator.fragtype_b(conf.operator, conf.shared_b_layout)}(undef)

                @loopinfo unroll for j = 1 : num_fragments_n
                    b_tile = translate_offset(warp_tile.KN, (K = 0, N = (j-1)*conf.compute_op_shape.N))
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
            tile = translate_offset(warp_tile, (M = (i-1)*conf.compute_op_shape.M, N = (j-1)*conf.compute_op_shape.N))
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
    block_i, block_j = CTASwizzle.cta_swizzle(conf.cta_swizzle, blockIdx(), conf.block_shape)

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    laneId = (threadIdx().x - 1) % 32 + 1

    gemm_sz = Tile(conf.matmul_shape)
    block_tile = Tile(conf.block_shape)

    # (1) Cooperatively load a block_shape.M x block_shape.N tile of C from global to shared memory within one threadblock
    shmem_c = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_c_layout), Layout.physical_size(conf.shared_c_layout, block_tile.MN.size))

    @loopinfo unroll for warp_tile = parallelise(block_tile.MN, Tile(conf.mem_cd_warp), warpId, conf.warps_per_block)
        @loopinfo unroll for thread_tile = parallelise(warp_tile, Tile(conf.mem_cd_thread), laneId, 32)
            x = @inbounds Layout.load(conf.global_c_layout, c, translate_base(thread_tile, (M = block_i, N = block_j)))
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
            tile = translate_offset(warp_tile, (M = (i-1)*conf.compute_op_shape.M, N = (j-1)*conf.compute_op_shape.N))
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
            @inbounds @immutable a_fragment[i,j] = Layout.load(conf.global_a_layout, a, translate_base(thread_tile, (M = block_i, K = 0)))
        end
    end

    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
            @inbounds @immutable b_fragment[i,j] = Layout.load(conf.global_b_layout, b, translate_base(thread_tile, (K = 0, N = block_j)))
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
    warp_tile = translate_offset(warp_tile_mn, (M = 0, N = 0, K = 0))

    @loopinfo unroll for i = 1 : num_fragments_m
        a_tile = translate_offset(warp_tile.MK, (M = (i-1)*conf.compute_op_shape.M, K = 0))
        @inbounds @immutable a_frags[1, i] = transf_sh2rf_a(Operator.load_a(conf.operator, conf.shared_a_layout, shmem_a, a_tile), a_tile)
    end

    @loopinfo unroll for j = 1 : num_fragments_n
        b_tile = translate_offset(warp_tile.KN, (K = 0, N = (j-1)*conf.compute_op_shape.N))
        @inbounds @immutable b_frags[1, j] = transf_sh2rf_b(Operator.load_b(conf.operator, conf.shared_b_layout, shmem_b, b_tile), b_tile)
    end

    # ld.global(block_shape.K : 2 * block_shape.K)
    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major))
            @inbounds @immutable a_fragment[i, j] = Layout.load(conf.global_a_layout, a, translate_base(thread_tile, (M = block_i, K = block_tile.size.K)))
        end
    end

    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
            @inbounds @immutable b_fragment[i, j] = Layout.load(conf.global_b_layout, b, translate_base(thread_tile, (K = block_tile.size.K, N = block_j)))
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
                            @inbounds @immutable a_fragment[i, j] = Layout.load(conf.global_a_layout, a, translate_base(thread_tile, (M = block_i, K = block_k + 2 * block_tile.size.K)))
                        end
                    end

                    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
                        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
                            @inbounds @immutable b_fragment[i, j] = Layout.load(conf.global_b_layout, b, translate_base(thread_tile, (K = block_k + 2 * block_tile.size.K, N = block_j)))
                        end
                    end
                end
            end

            # ld.shared((warp_k + compute_op_shape.K) % block_shape.K, stage = nxt_stage)
            warp_tile = translate_offset(warp_tile_mn, (M = 0, N = 0, K = (warp_k + conf.compute_op_shape.K) % block_tile.size.K))

            @loopinfo unroll for i = 1 : num_fragments_m
                a_tile = translate_offset(warp_tile.MK, (M = (i-1)*conf.compute_op_shape.M, K = 0))
                @inbounds @immutable a_frags[nxt_stage, i] = transf_sh2rf_a(Operator.load_a(conf.operator, conf.shared_a_layout, shmem_a, a_tile), a_tile)
            end

            @loopinfo unroll for j = 1 : num_fragments_n
                b_tile = translate_offset(warp_tile.KN, (K = 0, N = (j-1)*conf.compute_op_shape.N))
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
            tile = translate_offset(warp_tile, (M = (i-1)*conf.compute_op_shape.M, N = (j-1)*conf.compute_op_shape.N))
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

function matmul_pipelined_ng(conf::GemmKernels.Config, a, b, c, d,
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
            x = @inbounds Layout.load(conf.global_c_layout, c, translate_base(thread_tile, (M = block_i, N = block_j)))
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
            tile = translate_offset(warp_tile, (M = (i-1)*conf.compute_op_shape.M, N = (j-1)*conf.compute_op_shape.N))
            @inbounds @immutable c_frags[i, j] = transf_sh2rf_c(Operator.load_c(conf.operator, conf.shared_c_layout, shmem_c, tile), tile)
        end
    end

    sync_threads()

    # (3) Compute a block_shape.M x block_shape.N x block_shape.K matrix product within one threadblock
    shmem_a = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_a_layout), (Layout.physical_size(conf.shared_a_layout, block_tile.MK.size)..., 2))
    shmem_b = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_b_layout), (Layout.physical_size(conf.shared_b_layout, block_tile.KN.size)..., 2),
                                  length(shmem_a) * sizeof(Layout.eltype(conf.shared_a_layout)))

    # Sizes of a_fragment and b_fragment
    a_frag_i = (block_tile.size.M * block_tile.size.K) ÷ (conf.mem_a_warp.M * conf.mem_a_warp.K * conf.warps_per_block)
    a_frag_j = (conf.mem_a_warp.M * conf.mem_a_warp.K) ÷ (conf.mem_a_thread.M * conf.mem_a_thread.K * 32)
    b_frag_i = (block_tile.size.K * block_tile.size.N) ÷ (conf.mem_b_warp.K * conf.mem_b_warp.N * conf.warps_per_block)
    b_frag_j = (conf.mem_b_warp.K * conf.mem_b_warp.N) ÷ (conf.mem_b_thread.K * conf.mem_b_thread.N * 32)

    # Fragments to buffer the loads from global memory for A and B.
    a_fragment = LocalArray{Tuple{a_frag_i, a_frag_j}, Layout.fragtype(conf.global_a_layout, conf.mem_a_thread)}(undef)
    b_fragment = LocalArray{Tuple{b_frag_i, b_frag_j}, Layout.fragtype(conf.global_b_layout, conf.mem_b_thread)}(undef)

    # Fragments to buffer the loads from shared memory for A and B.
    a_frags = LocalArray{Tuple{2, num_fragments_m}, Operator.fragtype_a(conf.operator, conf.shared_a_layout)}(undef)
    b_frags = LocalArray{Tuple{2, num_fragments_n}, Operator.fragtype_b(conf.operator, conf.shared_b_layout)}(undef)

    warp_tile_mn = subdivide(block_tile, Tile(conf.compute_warp), warpId, conf.warps_per_block)

    # Prologue.

    # ld_global(main_loop_it = 0)
    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major))
            @inbounds @immutable a_fragment[i, j] = Layout.load(conf.global_a_layout, a, translate_base(thread_tile, (M = block_i, K = 0)))
        end
    end

    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
            @inbounds @immutable b_fragment[i, j] = Layout.load(conf.global_b_layout, b, translate_base(thread_tile, (K = 0, N = block_j)))
        end
    end

    # st_shared(main_loop_it = 0)
    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major))
            x = transf_gl2sh_a(@inbounds(a_fragment[i, j]), thread_tile)
            @inbounds Layout.store!(conf.shared_a_layout, view(shmem_a, :, :, 1), x, thread_tile)
        end
    end

    @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
        @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
            x = transf_gl2sh_b(@inbounds(b_fragment[i, j]), thread_tile)
            @inbounds Layout.store!(conf.shared_b_layout, view(shmem_b, :, :, 1), x, thread_tile)
        end
    end

    sync_threads()

    # ld_shared(main_loop_it = 0, warp_mma_k = 0)
    warp_mma_k = 0
    warp_k = warp_mma_k * conf.compute_op_shape.K
    warp_tile = translate_offset(warp_tile_mn, (M = 0, N = 0, K = warp_k))
    @loopinfo unroll for i = 1 : num_fragments_m
        a_tile = translate_offset(warp_tile.MK, (M = (i-1)*conf.compute_op_shape.M, K = 0))
        @inbounds @immutable a_frags[warp_mma_k % 2 + 1, i] = transf_sh2rf_a(Operator.load_a(conf.operator, conf.shared_a_layout, view(shmem_a, :, :, 1), a_tile), a_tile)
    end

    @loopinfo unroll for j = 1 : num_fragments_n
        b_tile = translate_offset(warp_tile.KN, (K = 0, N = (j-1)*conf.compute_op_shape.N))
        @inbounds @immutable b_frags[warp_mma_k % 2 + 1, j] = transf_sh2rf_b(Operator.load_b(conf.operator, conf.shared_b_layout, view(shmem_b, :, :, 1), b_tile), b_tile)
    end

    NUM_MAIN_LOOP_ITERS = gemm_sz.size.K ÷ block_tile.size.K
    @loopinfo unrollcount=2 for main_loop_it = 0 : NUM_MAIN_LOOP_ITERS - 1
        block_k = main_loop_it * block_tile.size.K

        main_loop_it_next = (main_loop_it + 1) % NUM_MAIN_LOOP_ITERS
        block_k_next = main_loop_it_next * block_tile.size.K

        NUM_WARP_MMA_K_ITERS = conf.block_shape.K ÷ conf.compute_op_shape.K
        @loopinfo unroll for warp_mma_k = 0 : NUM_WARP_MMA_K_ITERS - 1
            warp_k = warp_mma_k * conf.compute_op_shape.K
            warp_tile = translate_offset(warp_tile_mn, (M = 0, N = 0, K = warp_k))

            warp_mma_k_next = (warp_mma_k + 1) % NUM_WARP_MMA_K_ITERS
            warp_k_next = warp_mma_k_next * conf.compute_op_shape.K
            warp_tile_next = translate_offset(warp_tile_mn, (M = 0, N = 0, K = warp_k_next))

            main_loop_it_next_warp_k = if warp_mma_k_next == 0
                main_loop_it_next
            else
                main_loop_it
            end

            if warp_mma_k == conf.block_shape.K ÷ conf.compute_op_shape.K - 1 # last iteration of inner warp loop.
                # st.shared(main_loop_it_next)
                @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major))
                    @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major))
                        x = transf_gl2sh_a(@inbounds(a_fragment[i, j]), thread_tile)
                        @inbounds Layout.store!(conf.shared_a_layout, view(shmem_a, :, :, main_loop_it_next % 2 + 1), x, thread_tile)
                    end
                end

                @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
                    @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
                        x = transf_gl2sh_b(@inbounds(b_fragment[i, j]), thread_tile)
                        @inbounds Layout.store!(conf.shared_b_layout, view(shmem_b, :, :, main_loop_it_next % 2 + 1), x, thread_tile)
                    end
                end

                sync_threads()
            end

            # ld.shared(main_loop_it, warp_mma_k_next)
            @loopinfo unroll for i = 1 : num_fragments_m
                a_tile = translate_offset(warp_tile_next.MK, (M = (i-1)*conf.compute_op_shape.M, K = 0))
                @inbounds @immutable a_frags[warp_mma_k_next % 2 + 1, i] = transf_sh2rf_a(Operator.load_a(conf.operator, conf.shared_a_layout, view(shmem_a, :, :, main_loop_it_next_warp_k % 2 + 1), a_tile), a_tile)
            end

            @loopinfo unroll for j = 1 : num_fragments_n
                b_tile = translate_offset(warp_tile_next.KN, (K = 0, N = (j-1)*conf.compute_op_shape.N))
                @inbounds @immutable b_frags[warp_mma_k_next % 2 + 1, j] = transf_sh2rf_b(Operator.load_b(conf.operator, conf.shared_b_layout, view(shmem_b, :, :, main_loop_it_next_warp_k % 2 + 1), b_tile), b_tile)
            end

            if warp_mma_k == 0 # first iteration of inner warp loop.
                # ld.global(main_loop_it_next)
                @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.MK, Tile(conf.mem_a_warp), warpId, conf.warps_per_block, conf.is_a_col_major))
                    @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_a_thread), laneId, 32, conf.is_a_col_major))
                        @inbounds @immutable a_fragment[i, j] = Layout.load(conf.global_a_layout, a, translate_base(thread_tile, (M = block_i, K = block_k_next)))
                    end
                end

                @loopinfo unroll for (i, warp_tile) = enumerate(parallelise(block_tile.KN, Tile(conf.mem_b_warp), warpId, conf.warps_per_block, conf.is_b_col_major))
                    @loopinfo unroll for (j, thread_tile) = enumerate(parallelise(warp_tile, Tile(conf.mem_b_thread), laneId, 32, conf.is_b_col_major))
                        @inbounds @immutable b_fragment[i, j] = Layout.load(conf.global_b_layout, b, translate_base(thread_tile, (K = block_k_next, N = block_j)))
                    end
                end
            end

            # mma(main_loop_it, warp_mma_k)
            @loopinfo unroll for i = 1 : num_fragments_m
                @loopinfo unroll for j = 1 : num_fragments_n
                    @inbounds @immutable c_frags[i, j] = Operator.mma(conf.operator, a_frags[warp_mma_k % 2 + 1, i], b_frags[warp_mma_k % 2 + 1, j], c_frags[i, j])
                end
            end
        end
    end

    # (4) Store the compute_warp.M x compute_warp.N tile of D from registers to shared memory
    shmem_d = @inbounds CuDynamicSharedArray(Layout.eltype(conf.shared_d_layout), Layout.physical_size(conf.shared_d_layout, block_tile.MN.size))

    warp_tile = @inbounds subdivide(block_tile.MN, Tile(conf.compute_warp).MN, warpId, conf.warps_per_block)

    @loopinfo unroll for i = 1 : num_fragments_m
        @loopinfo unroll for j = 1 : num_fragments_n
            tile = translate_offset(warp_tile, (M = (i-1)*conf.compute_op_shape.M, N = (j-1)*conf.compute_op_shape.N))
            @inbounds Operator.store_d(conf.operator, conf.shared_d_layout, shmem_d, transf_rf2sh_d(c_frags[i, j], tile), tile)
        end
    end

    sync_threads()

    # (5) Run the epilogue
    epilogue(conf, d, shmem_d, transf_sh2gl_d)

    return
end

function shmem_size(conf::GemmKernels.Config, ::typeof(matmul_pipelined_ng))
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
    max(size_c, 2 * (size_a + size_b), size_d)
end

end
