# vim: fdm=marker

using GemmKernels
using GemmKernels: Config, LocalArray, @immutable, mma884_row_row, @staticdef, BitArrayIndex, @unrolled, @not_unrolled, constant, variadic, tid, bid_x, bid_y, warpid, vloada, vstorea!, Vec, b, Layout
using Test
using CUDA
using LLVMLoopInfo: @loopinfo
using Base.Cartesian: @ntuple
using Base
using LinearAlgebra
using GemmKernels.Operator: VoltaMmaSyncOp, mma, load_a, load_b, store_d
using GemmKernels.Layout: VoltaSwizzledOperandA, VoltaSwizzledOperandB
using GemmKernels.Tiling

# globals {{{
conf = GemmKernels.get_config(
    gemm_shape = (M = 2048, N = 2048, K = 2048),
    block_shape = (M = 128, N = 256, K = 32),
    warps_per_block = 8,

    compute_warp = (M = 64, N = 64, K = 4),

    global_a_layout = Layout.UnsafeAlignedRowMajor{Float16},
    global_b_layout = Layout.UnsafeAlignedRowMajor{Float16},
    global_c_layout = Layout.Zero{Float32},
    global_d_layout = Layout.UnsafeAlignedRowMajor{Float32},

    shared_a_layout = Layout.VoltaSwizzledOperandA{Float16},
    shared_b_layout = Layout.VoltaSwizzledOperandB{Float16},
    shared_c_layout = Layout.Zero{Float32},
    shared_d_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{Float32}, 2},

    operator = VoltaMmaSyncOp,

    is_a_col_major = false,
    is_b_col_major = false
   )

# The kernel calculates A * B = D (in row-major), as this is CUTLASS's
# convention.
# To calculate A * B = D in col-major, just flip the A and B operands
# and transpose: A * B = D <=> B^T * A^T = D^T.
A = CUDA.rand(Float16, (conf.matmul_shape.N, conf.matmul_shape.K))
B = CUDA.rand(Float16, (conf.matmul_shape.K, conf.matmul_shape.M))
D = CUDA.zeros(Float32, (conf.matmul_shape.N, conf.matmul_shape.M))
# }}}

# ld global {{{
# Load from global memory
@inline function ld_global(A, B, cta_m, cta_n, cta_k, conf)
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
        tile = translate_base(tile, (M = convert(Int, cta_m), K = convert(Int, cta_k)))
        tile = translate_base(tile, (M = convert(Int, m.variadic_part), K = convert(Int, k.variadic_part)))
        tile = translate_offset(tile, (M = convert(Int, m.known_one), K = convert(Int, k.known_one)))

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
        tile = translate_base(tile, (K = convert(Int, cta_k), N = convert(Int, cta_n)))
        tile = translate_base(tile, (K = convert(Int, k.variadic_part), N = convert(Int, n.variadic_part)))
        tile = translate_offset(tile, (K = convert(Int, k.known_one), N = convert(Int, n.known_one)))

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
@inline function st_shared(shmem_a, shmem_b, a_frag, b_frag)
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

            tile = translate_base(tile, (M = convert(Int, m.variadic_part), K = convert(Int, k.variadic_part)))
            tile = translate_offset(tile, (M = convert(Int, m.known_one), K = convert(Int, k.known_one)))

            Layout.store!(VoltaSwizzledOperandA{Float16}, shmem_a, val, tile)
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
            tile = translate_base(tile, (K = convert(Int, k.variadic_part), N = convert(Int, n.variadic_part)))
            tile = translate_offset(tile, (K = convert(Int, k.known_one), N = convert(Int, n.known_one)))

            Layout.store!(VoltaSwizzledOperandB{Float16}, shmem_b, val, tile)
        end
    end
end
# }}}

# ld shared {{{
@inline function ld_shared(shmem_a, shmem_b, warp_m, warp_n, warp_k, conf)
    warp_mma_k = warp_k ÷ conf.compute_warp.K

    b_frag = LocalArray{Tuple{16}, Float16}(undef)

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    block_tile = Tile(M = 128, N = 256, K = 32)
    warp_tile_mn = subdivide(block_tile, Tile(M = 64, N = 64, K = 32), warpId, 8)

    tile = translate_offset(warp_tile_mn, (M = 0, N = 0, K = convert(Int, warp_k)))

    a_frag = load_a(VoltaMmaSyncOp, VoltaSwizzledOperandA{Float16}, shmem_a, tile)
    b_frag = load_b(VoltaMmaSyncOp, VoltaSwizzledOperandB{Float16}, shmem_b, tile)

    a_frag, b_frag
end
# }}}

# epilogue {{{
@inline function epilogue_st_shared(epilogue_it, shmem_d, warp_m, warp_n, acc_frag)
    # index: (m5|m2|m1|n5|n4|n2|n0)

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    block_tile = Tile(M = 128, N = 256, K = 32)
    warp_tile = subdivide(block_tile.MN, Tile(M = 64, N = 64), warpId, 8)
    m_offset = convert(Int, b(epilogue_it, 0, 2) + b(epilogue_it, 1, 5))
    tile = translate_offset(warp_tile, (M = m_offset, N = 0))

    frag_base = b(epilogue_it, 0, 5) + # m2
                b(epilogue_it, 1, 6)   # m5

    frag = view(acc_frag, (convert(Int, frag_base) + 1):length(acc_frag))

    store_d(VoltaMmaSyncOp, Layout.Padded{Layout.UnsafeAlignedRowMajor{Float16}, 2}, shmem_d, frag, tile)
end

@inline function epilogue_ld_shared(epilogue_it, shmem_d)
    # TODO: EXTRACT
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

@inline function epilogue_st_global(epilogue_it, D, shmem_d, cta_m, cta_n, frag, conf)
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

@inline function epilogue(D, shmem_d, acc_frag, cta_m, cta_n, warp_m, warp_n, conf)
    # TODO: EXTRACT
    @unrolled for epilogue_it = 0:3
        # TODO: Can we not remove this one?
        sync_threads()

        # Store tile of D to shared memory.
        epilogue_st_shared(epilogue_it, shmem_d, warp_m, warp_n, acc_frag)

        sync_threads()

        # Load tile of D from shared memory.
        frag = epilogue_ld_shared(epilogue_it, shmem_d)

        # Store tile of D to global memory.
        epilogue_st_global(epilogue_it, D, shmem_d, cta_m, cta_n, frag, conf)
    end
end
# }}}

# kernel {{{
# row-major A x row-major B = row-major D
function kernel(A, B, D, conf::Config)
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

    shmem_d = CuDynamicSharedArray(Float32, 32 * (256 + 2))

    # index: (m5|m2|m1|n5|n4|n2|n0)
    acc_frag = LocalArray{Tuple{128}, Float32}(@ntuple 128 i -> zero(Float32))

    # Two pairs of fragments for the shared -> register file pipeline.
    shared_a_frags = LocalArray{Tuple{2}, NTuple{16, Float16}}(undef)
    shared_b_frags = LocalArray{Tuple{2}, NTuple{16, Float16}}(undef)

    # Prologue.
    # ld_global(main_loop_it=0)
    cta_k = constant(0)
    global_a_frag, global_b_frag = ld_global(A, B, cta_m, cta_n, cta_k, conf)

    # st_shared(main_loop_it=0)
    main_loop_it = constant(0)
    st_shared(view(shmem_a, :, convert(Int, main_loop_it % 2) + 1),
                view(shmem_b, :, convert(Int, main_loop_it % 2) + 1),
                global_a_frag, global_b_frag)
    sync_threads()

    # ld_shared(main_loop_it=0, warp_mma_k=0)
    main_loop_it = constant(0)
    warp_k = constant(0)
    warp_mma_k = constant(0)
    shared_a_frag, shared_b_frag = ld_shared(view(shmem_a, :, convert(Int, main_loop_it % 2) + 1),
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
                st_shared(view(shmem_a, :, convert(Int, main_loop_it_next % 2) + 1),
                          view(shmem_b, :, convert(Int, main_loop_it_next % 2) + 1),
                          global_a_frag, global_b_frag)
                sync_threads()
            end

            # ld_shared(main_loop_it, warp_mma_k + 1)
            shared_a_frag, shared_b_frag = ld_shared(view(shmem_a, :, convert(Int, main_loop_it % 2) + 1),
                                                     view(shmem_b, :, convert(Int, main_loop_it % 2) + 1),
                                                     warp_m, warp_n, warp_k_next, conf)

            @inbounds @immutable shared_a_frags[convert(Int, warp_mma_k_next % 2) + 1] = shared_a_frag
            @inbounds @immutable shared_b_frags[convert(Int, warp_mma_k_next % 2) + 1] = shared_b_frag

            # TODO: Predicate the load?
            if warp_mma_k == 0
                # ld_global(main_loop_it + 1)
                # Copy the data for a CTA_M x CTA_N x CTA_K GEMM from GMEM to SHMEM, cooperatively in a CTA.
                global_a_frag, global_b_frag = ld_global(A, B, cta_m, cta_n, cta_k_next, conf)
            end

            # WARP_M x WARP_N x WARP_K = 64 x 64 x 4 GEMM per warp
            # mma(main_loop_it, warp_mma_k)
            @inbounds shared_a_frag = shared_a_frags[convert(Int, warp_mma_k % 2) + 1]
            @inbounds shared_b_frag = shared_b_frags[convert(Int, warp_mma_k % 2) + 1]

            # acc_frag = warp_mma(shared_a_frag, shared_b_frag, acc_frag)
            acc_frag = mma(VoltaMmaSyncOp, shared_a_frag, shared_b_frag, acc_frag)
        end
    end

    # epilogue: store matrix from registers to global memory
    epilogue(D, shmem_d, acc_frag, cta_m, cta_n, warp_m, warp_n, conf)

    nothing
end
# }}}

# driver {{{
function test(; dump_code=false, debug=false)
    blocks = (cld(conf.matmul_shape.M, conf.block_shape.M),
              cld(conf.matmul_shape.N, conf.block_shape.N))

    if debug
        @device_code_warntype interactive=true @cuda threads=(conf.warps_per_block * 32) blocks=blocks shmem=48*1024 kernel(B, A, D, conf)
        return
    end

    if dump_code
        @device_code dir="gemm-output" @cuda threads=(conf.warps_per_block * 32) blocks=blocks shmem=48*1024 kernel(B, A, D, conf)
    end

    D_ref = similar(D)

    CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)
    CUDA.CUBLAS.gemmEx!('N', 'N', Float32(1), A, B, Float32(0), D_ref)

    # TODO: do not hardcode shared memory size
    @cuda threads=(conf.warps_per_block * 32) blocks=blocks shmem=48*1024 kernel(B, A, D, conf)

    compare(x, y) = isapprox(x, y; rtol=sqrt(eps(Float16)))

    @test isapprox(D_ref, D; rtol=sqrt(eps(Float16)))
    @test isapprox(D_ref, D; rtol=sqrt(eps(Float16)), norm=M -> LinearAlgebra.norm(M, Inf))
    @test all(compare.(D, D_ref))
end

function cublas()
    CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)
    CUDA.CUBLAS.gemmEx!('N', 'N', Float32(1), A, B, Float32(0), D)
end

isinteractive() || test()
# }}}
