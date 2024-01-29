# vim: fdm=marker

using GemmKernels: LocalArray, @immutable, mma884_row_row, @staticdef, BitArrayIndex, @unrolled, @not_unrolled, constant, variadic, tid, bid_x, bid_y, warpid, vloada, vstorea!, Vec, b
using Test
using CUDA
using LLVMLoopInfo: @loopinfo
using Base.Cartesian: @ntuple
using Base
using LinearAlgebra

# configuration {{{
@staticdef struct Config
    # Size of the matrices in global memory.
    GLOBAL_M
    GLOBAL_N
    GLOBAL_K

    # Tile size at the CTA level.
    CTA_M
    CTA_N
    CTA_K

    # Tile size at the warp level.
    WARP_M
    WARP_N
    WARP_K

    # Number of stages in the global -> shared copy pipeline.
    GLOBAL_TO_SHARED_STAGES

    # Number of stages in the shared -> register file copy pipeline.
    SHARED_TO_REGS_STAGES
end

NUM_WARPS_M(c::Config) = c.CTA_M ÷ c.WARP_M
NUM_WARPS_N(c::Config) = c.CTA_N ÷ c.WARP_N
NUM_THREADS(c::Config) = NUM_WARPS_M(c) * NUM_WARPS_N(c) * 32
NUM_BLOCKS_M(c::Config) = c.GLOBAL_M ÷ c.CTA_M
NUM_BLOCKS_N(c::Config) = c.GLOBAL_N ÷ c.CTA_N

# }}}

# globals {{{
conf = Config(
    2048, 2048, 2048, # GLOBAL_MNK
    128, 256, 32,     # CTA_MNK
    64, 64, 4,        # WARP_MNK
    2,                # GLOBAL_TO_SHARED_STAGES
    2,                # SHARED_TO_REGS_STAGES
)

# The kernel calculates A * B = D (in row-major), as this is CUTLASS's
# convention.
# To calculate A * B = D in col-major, just flip the A and B operands
# and transpose: A * B = D <=> B^T * A^T = D^T.
A = CUDA.rand(Float16, (conf.GLOBAL_N, conf.GLOBAL_K))
B = CUDA.rand(Float16, (conf.GLOBAL_K, conf.GLOBAL_M))
D = CUDA.zeros(Float32, (conf.GLOBAL_N, conf.GLOBAL_M))
# }}}

# warp mma {{{
@inline function warp_mma(a_frags, b_frags, acc_frags)
    # WARP_M x WARP_N x WARP_K = 64 x 64 x 4 GEMM per warp using 16 mma.syncs
    @unrolled for instruction = 0:15
        inner_row = b(instruction, 0)
        outer_row = b(instruction, 1)
        inner_col = b(instruction, 2)
        outer_col = b(instruction, 3)

        # 16 x 16 x 4 GEMM per warp using 1 mma.sync:
        # one mma.sync is an 8 x 8 x 4 GEMM per QP,
        # or a 16 x 16 x 4 GEMM per warp.

        # zig-zag sequence to increase reuse of the A operand
        # This doesn't seem to influence performance (at least not significantly),
        # but CUTLASS does this, so we might as well too.
        zz_inner_row = inner_row ⊻ (inner_col % 2)
        zz_outer_row = outer_row ⊻ (inner_col % 2)

        # a, b, and c fragments for this particular mma.sync.
        a_frag = LocalArray{Tuple{4}, Float16}(undef)
        b_frag = LocalArray{Tuple{4}, Float16}(undef)
        c_frag = LocalArray{Tuple{8}, Float32}(undef)

        # Get the A fragment for this mma.sync
        @unrolled for i = 0:3
            offset = b(i, 0, 0) +            # k0
                     b(i, 1, 1) +            # k1
                     b(zz_inner_row, 0, 2) + # m2+k3
                     b(zz_outer_row, 0, 3)   # m5
            @inbounds @immutable a_frag[i] = a_frags[offset]
        end

        # Get the B fragment for this mma.sync
        @unrolled for i = 0:3
            offset = b(i, 0, 0) +           # n0
                     b(i, 1, 1) +           # n1
                     b(inner_col, 0, 2) +   # n2
                     b(outer_col, 0, 3)     # n5

            @inbounds @immutable b_frag[i] = b_frags[offset]
        end

        # Get the C fragment for this mma.sync
        @unrolled for i = 0:7
            # index: (m5|m2|m1|n5|n4|n2|n0)
            offset = b(i, 0, 0) +            # n0
                     b(inner_col, 0, 1) +    # n2
                     b(i, 2, 2) +            # n4
                     b(outer_col, 0, 3) +    # n5
                     b(i, 1, 4) +            # m1
                     b(zz_inner_row, 0, 5) + # m2
                     b(zz_outer_row, 0, 6)   # m5
            @inbounds @immutable c_frag[i] = acc_frags[offset]
        end

        # offset in unit of 4x4x4 tiles
        inst_m = b(zz_inner_row, 0, 2) + b(tid(), 2, 3) + b(zz_outer_row, 0, 5)
        inst_n = b(inner_col, 0, 2) + b(tid(), 3, 3) + b(outer_col, 0, 5)
        inst_k = constant(0)

        d_frag = mma884_row_row(a_frag.data, b_frag.data, c_frag.data)

        # Store D fragment for this mma.sync
        @unrolled for i = 0:7
            # index: (m5|m2|m1|n5|n4|n2|n0)
            offset = b(i, 0, 0) +            # n0
                     b(inner_col, 0, 1) +    # n2
                     b(i, 2, 2) +            # n4
                     b(outer_col, 0, 3) +    # n5
                     b(i, 1, 4) +            # m1
                     b(zz_inner_row, 0, 5) + # m2
                     b(zz_outer_row, 0, 6)   # m5
            @inbounds @immutable acc_frags[offset] = d_frag[i]
        end
    end

    acc_frags
end
# }}}

# swizzling {{{
# swizzling function for the shared memory layout for A
@inline function swizzle_a(m, k, conf)
    # m : 7 bits
    # k : 5 bits
    offset = b(k, 0, 0) +
             b(k, 1, 1) +
             (b(k, 3, 2) ⊻ b(m, 2, 2)) +
             (b(k, 4, 3) ⊻ b(m, 3, 3) ⊻ b(m, 4, 3)) +
             b(m, 0, 4) +
             b(m, 1, 5) +
             b(m, 4, 6) +
             b(m, 5, 7) +
             b(m, 6, 8) +
             b(k, 2, 9) +
             b(k, 3, 10) +
             b(k, 4, 11) +
             b(k, 5, 12)

    return offset
end

# swizzling function for the shared memory layout for B
@inline function swizzle_b(k, n, conf)
    # k: 5 bits
    # n: 8 bits
    offset = b(n, 0, 0) +
             b(n, 1, 1) +
             b(n, 2, 2) +
             (b(n, 3, 3) ⊻ b(k, 0, 3)) +
             (b(n, 4, 4) ⊻ b(k, 1, 4)) +
             b(n, 5, 5) +
             b(n, 6, 6) +
             b(n, 7, 7) +
             b(n, 3, 8) +
             b(n, 4, 9) +
             b(k, 2, 10) +
             b(k, 3, 11) +
             b(k, 4, 12) +
             b(k, 5, 13)

    return offset
end
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

        @inbounds val = vloada(Vec{8, Float16}, A, cta_k + k + conf.GLOBAL_K * (cta_m + m))

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

        @inbounds val = vloada(Vec{8, Float16}, B, cta_n + n + conf.GLOBAL_N * (cta_k + k))


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
@inline function st_shared(stage, shmem_a, shmem_b, a_frag, b_frag, conf)
    # Store A to Shared Memory.
    @unrolled for ins = 0:3
            m = b(tid(), 2, 0) +
                b(tid(), 3, 1) +
                b(tid(), 4, 2) +
                b(ins, 1, 3) +
                b(tid(), 5, 4) +
                b(tid(), 6, 5) +
                b(tid(), 7, 6)

            k = b(ins, 0, 2) +
                b(tid(), 0, 3) +
                b(tid(), 1, 4) +
                b(stage, 0, 5)

        @inbounds val = @ntuple 4 i -> begin
            offset = constant(i-1)
            frag_offset = b(offset, 0, 0) + # k0
                          b(offset, 1, 1) + # k1
                          b(ins, 0, 2) +  # k2
                          b(ins, 1, 3)    # m3
            VecElement{Float16}(a_frag[frag_offset])
        end
        @inbounds vstorea!(Vec{4, Float16}, shmem_a, swizzle_a(m, k, conf), val)
    end

    # Store B to Shared Memory.
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
            b(tid(), 7, 4) +
            b(stage, 0, 5)

        @inbounds val = @ntuple 8 i -> begin
            offset = constant(i-1)
            frag_offset = b(offset, 0, 0) + # n0
                          b(offset, 1, 1) + # n1
                          b(offset, 2, 2) + # n2
                          b(ins, 0, 3) +    # n6
                          b(ins, 1, 4)      # n7
            VecElement{Float16}(b_frag[frag_offset])
        end
        @inbounds vstorea!(Vec{8, Float16}, shmem_b, swizzle_b(k, n, conf), val)
    end
end
# }}}

# ld shared {{{
@inline function ld_shared(stage, shmem_a, shmem_b, warp_m, warp_n, warp_k, conf)
    warp_mma_k = warp_k ÷ conf.WARP_K

    # Fragments for the data from the shared loads (and hence the MMAs).
    # index: (m5|m2|k1|k0)
    a_frag = LocalArray{Tuple{16}, Float16}(undef)

    # index: (n5|n2|n1|n0)
    b_frag = LocalArray{Tuple{16}, Float16}(undef)

    # Load A from Shared Memory.
    @unrolled for ins = 0:1
        m = b(tid(), 0, 0) +
            b(tid(), 1, 1) +
            b(warp_mma_k, 1, 2) +
            b(tid(), 2, 3) +
            b(tid(), 4, 4) +
            b(ins, 0, 5)

        k = b(stage, 0, 5)

        @inbounds val = vloada(Vec{8, Float16}, shmem_a, swizzle_a(warp_m+m, warp_k+k, conf))

        @unrolled for offset = 0:7
            frag_offset = b(offset, 0, 0) +                         # k0
                          b(offset, 1, 1) +                         # k1
                          (b(offset, 2, 2) ⊻ b(warp_mma_k, 1, 2)) + # m2 = (m2+k3) + k3
                          b(ins, 0, 3)                              # m5

            @inbounds @immutable a_frag[frag_offset] = val[offset].value
        end
    end

    # Load B from Shared Memory.
    @unrolled for ins = 0:1
        k = b(tid(), 0, 0) +
            b(tid(), 1, 1) +
            b(stage, 0, 5)

        n = b(tid(), 3, 3) +
            b(tid(), 4, 4) +
            b(ins, 0, 5)

        @inbounds val = vloada(Vec{8, Float16}, shmem_b, swizzle_b(warp_k+k, warp_n+n, conf))

        @unrolled for offset = 0:7
            frag_offset = b(offset, 0, 0) +    # n0
                          b(offset, 1, 1) +    # n1
                          b(offset, 2, 2) +    # n2
                          b(ins, 0, 3)         # n5

            @inbounds @immutable b_frag[frag_offset] = val[offset].value
        end
    end

    a_frag.data, b_frag.data
end
# }}}

# epilogue {{{
@inline function epilogue_st_shared(epilogue_it, shmem_d, warp_m, warp_n, acc_frag)
    # index: (m5|m2|m1|n5|n4|n2|n0)

    @unrolled for ins = 0:15
        # TODO: vectorise
        @unrolled for offset = 0:1
            m = b(tid(), 0, 0) +
                b(ins, 3, 1) +
                b(epilogue_it, 0, 2) +
                b(tid(), 2, 3) +
                b(tid(), 4, 4) +
                b(epilogue_it, 1, 5) +
                warp_m

            n = b(offset, 0, 0) +
                b(tid(), 1, 1) +
                b(ins, 0, 2) +
                b(tid(), 3, 3) +
                b(ins, 1, 4) +
                b(ins, 2, 5) +
                warp_n

            frag_index = b(offset, 0, 0) +      # n0
                         b(ins, 0, 1) +         # n2
                         b(ins, 1, 2) +         # n4
                         b(ins, 2, 3) +         # n5
                         b(ins, 3, 4) +         # m1
                         b(epilogue_it, 0, 5) + # m2
                         b(epilogue_it, 1, 6)   # m5

            offset_M = b(tid(), 0, 0) +         # m0
                       b(ins, 3, 1) +           # m1
                       b(tid(), 2, 2) +         # m3
                       b(tid(), 4, 3) +         # m4
                       b(warpid(), 0, 4)        # m6

            offset_N = n

            offset = convert(Int, offset_N) + 258 * convert(Int, offset_M)

            @inbounds shmem_d[1 + offset] = acc_frag[frag_index]
        end
    end
end

@inline function epilogue_ld_shared(epilogue_it, shmem_d)
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

        @inbounds vstorea!(Vec{4, Float32}, D, cta_n + n + conf.GLOBAL_N * cta_m + conf.GLOBAL_N * m, val)
    end
end

@inline function epilogue(D, shmem_d, acc_frag, cta_m, cta_n, warp_m, warp_n, conf)
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
    num_warps = NUM_WARPS_M(conf) * NUM_WARPS_N(conf)
    warpid_m = (warpid() % num_warps) % NUM_WARPS_M(conf)
    warpid_n = (warpid() % num_warps) ÷ NUM_WARPS_M(conf)

    cta_m = (bid_x() % NUM_BLOCKS_M(conf)) * conf.CTA_M
    cta_n = (bid_y() % NUM_BLOCKS_N(conf)) * conf.CTA_N

    warp_m = warpid_m * conf.WARP_M
    warp_n = warpid_n * conf.WARP_N

    SHMEM_A_SIZE = (conf.CTA_M * conf.CTA_K) * conf.SHARED_TO_REGS_STAGES
    SHMEM_B_SIZE = (conf.CTA_K * conf.CTA_N) * conf.SHARED_TO_REGS_STAGES

    shmem_ab = CuDynamicSharedArray(Float16, SHMEM_A_SIZE + SHMEM_B_SIZE)

    shmem_a = view(shmem_ab, 1:SHMEM_A_SIZE)
    shmem_b = view(shmem_ab, 1+SHMEM_A_SIZE:SHMEM_A_SIZE+SHMEM_B_SIZE)

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
    stage = constant(0)
    st_shared(stage, shmem_a, shmem_b, global_a_frag, global_b_frag, conf)
    sync_threads()

    # ld_shared(main_loop_it=0, warp_mma_k=0)
    main_loop_it = constant(0)
    warp_k = constant(0)
    warp_mma_k = constant(0)
    stage = constant(0)
    shared_a_frag, shared_b_frag = ld_shared(stage, shmem_a, shmem_b, warp_m, warp_n, warp_k, conf)

    @inbounds @immutable shared_a_frags[convert(Int, warp_mma_k % 2) + 1] = shared_a_frag
    @inbounds @immutable shared_b_frags[convert(Int, warp_mma_k % 2) + 1] = shared_b_frag

    NUM_MAIN_LOOP_ITERS = conf.GLOBAL_K ÷ conf.CTA_K
    @not_unrolled for main_loop_it = 0 : NUM_MAIN_LOOP_ITERS - 1
        # The modulo is so that the BitArrayIndex knowns which bits are 0.
        # TODO: Do this automatically in the @not_unrolled macro?
        # TODO: Generate _next variables automatically.
        cta_k = (main_loop_it % NUM_MAIN_LOOP_ITERS) * conf.CTA_K

        main_loop_it_next = variadic((main_loop_it_orig + 1)) % NUM_MAIN_LOOP_ITERS
        cta_k_next = main_loop_it_next * conf.CTA_K

        stage = variadic(main_loop_it_orig) % 2
        stage_next = variadic((main_loop_it_orig + 1)) % 2

        # CTA_M x CTA_N x CTA_K GEMM per CTA
        NUM_WARP_MMA_K_ITERS = conf.CTA_K ÷ conf.WARP_K
        @unrolled for warp_mma_k = 0 : NUM_WARP_MMA_K_ITERS - 1
            warp_k = warp_mma_k * conf.WARP_K

            # TODO: Do this in macro.
            warp_mma_k_next = constant(warp_mma_k_orig + 1) % NUM_WARP_MMA_K_ITERS
            warp_k_next = warp_mma_k_next * conf.WARP_K

            if warp_mma_k == NUM_WARP_MMA_K_ITERS-1
                # st_shared(main_loop_it+1)
                st_shared(stage_next, shmem_a, shmem_b, global_a_frag, global_b_frag, conf)
                sync_threads()
            end

            # ld_shared(main_loop_it, warp_mma_k + 1)
            shared_a_frag, shared_b_frag = ld_shared(stage, shmem_a, shmem_b, warp_m, warp_n, warp_k_next, conf)

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
            acc_frag = warp_mma(shared_a_frag, shared_b_frag, acc_frag)
        end
    end

    # epilogue: store matrix from registers to global memory
    epilogue(D, shmem_d, acc_frag, cta_m, cta_n, warp_m, warp_n, conf)

    nothing
end
# }}}

# driver {{{
function test(; dump_code=false, debug=false)
    if debug
        @device_code_warntype interactive=true @cuda threads=NUM_THREADS(conf) blocks=(NUM_BLOCKS_M(conf), NUM_BLOCKS_N(conf)) shmem=48*1024 kernel(B, A, D, conf)
        return
    end

    if dump_code
        @device_code dir="gemm-output" @cuda threads=NUM_THREADS(conf) blocks=(NUM_BLOCKS_M(conf), NUM_BLOCKS_N(conf)) shmem=48*1024 kernel(B, A, D, conf)
    end

    D_ref = similar(D)

    CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)
    CUDA.CUBLAS.gemmEx!('N', 'N', Float32(1), A, B, Float32(0), D_ref)

    # TODO: do not hardcode shared memory size
    @cuda threads=NUM_THREADS(conf) blocks=(NUM_BLOCKS_M(conf), NUM_BLOCKS_N(conf)) shmem=48*1024 kernel(B, A, D, conf)

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
