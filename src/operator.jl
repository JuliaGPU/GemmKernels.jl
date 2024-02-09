export Operator
module Operator

using CUDA
using GemmKernels: vloada, vstorea!, Layout, LocalArray, @immutable, b, @unrolled, constant, mma884_row_row, tid, constant, variadic, Vec
using GemmKernels.Tiling
using LLVMLoopInfo: @loopinfo

# -------------------------------------
# Default definition for padded layouts
# -------------------------------------

for f in (:fragtype_a, :fragtype_b, :fragtype_accum, :load_a, :load_b, :load_c, :store_d)
    @eval @inline $f(op, ::Type{Layout.Padded{L, P}}, args...) where {L, P} = $f(op, L, args...)
end

# ---
# FPU
# ---

# CT is the compute type used to perform scalar operations in.
# AT is the accumulator type used to accumulate partial results.
# mb, nb, kb are the base operator shapes (kb must be equal to 1 for now).
# M, N, K must be multiples of mb, nb, and kb respectively.
abstract type GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT} end

@inline shape(::Type{<:GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT}}) where {M, N, K, mb, nb, kb, CT, AT} = (M = M, N = N, K = K)
@inline base_shape(::Type{<:GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT}}) where {M, N, K, mb, nb, kb, CT, AT} = (M = M, N = N, K = K, mb = mb, nb = nb, kb = kb)

for (layout_type, convert_index_func) in [
                                        (Layout.ColMajor, identity),
                                        (Layout.UnsafeAlignedColMajor, identity),
                                        (Layout.RowMajor, x -> reverse(Tuple(x))),
                                        (Layout.UnsafeAlignedRowMajor, x -> reverse(Tuple(x))),
                                       ]
    @eval begin
        @inline function fragtype_a(::Type{<:GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT}}, ::Type{$layout_type{DT}}) where {M, N, K, mb, nb, kb, CT, AT, DT}
            return NTuple{M * K ÷ mb, CT}
        end
        @inline function fragtype_b(::Type{<:GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT}}, ::Type{$layout_type{DT}}) where {M, N, K, mb, nb, kb, CT, AT, DT}
            return NTuple{K * N ÷ nb, CT}
        end

        @inline function fragtype_accum(::Type{<:GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT}}, ::Type{$layout_type{DT}}) where {M, N, K, mb, nb, kb, CT, AT, DT}
            return NTuple{M * N ÷ 32, AT}
        end

        @inline function load_a(::Type{<:GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT}}, ::Type{$layout_type{DT}}, workspace, tile::Tile) where {M, N, K, mb, nb, kb, CT, AT, DT}
            laneId = (threadIdx().x - 1) % 32 + 1

            op_y = (laneId - 1) % mb + 1
            y, x = (tile.base.M + tile.offset.M + op_y, tile.base.K + tile.offset.K + 1)

            frag = LocalArray{Tuple{M ÷ mb, K}, CT}(undef)
            @loopinfo unroll for m = 1 : M ÷ mb
                @loopinfo unroll for k = 1 : K
                    y_layout, x_layout = $convert_index_func((y + mb * (m - 1), x + (k - 1)))
                    @inbounds @immutable frag[m,k] = workspace[y_layout, x_layout]
                end
            end

            return NTuple{M * K ÷ mb, CT}(frag)
        end

        @inline function load_b(::Type{<:GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT}}, ::Type{$layout_type{DT}}, workspace, tile::Tile) where {M, N, K, mb, nb, kb, CT, AT, DT}
            laneId = (threadIdx().x - 1) % 32 + 1

            op_x = (laneId - 1) ÷ mb + 1
            y, x = (tile.base.K + tile.offset.K + 1, tile.base.N + tile.offset.N + op_x)

            frag = LocalArray{Tuple{K, N ÷ nb}, CT}(undef)
            @loopinfo unroll for n = 1 : N ÷ nb
                @loopinfo unroll for k = 1 : K
                    y_layout, x_layout = $convert_index_func((y + (k - 1), x + nb * (n - 1)))
                    @inbounds @immutable frag[k,n] = workspace[y_layout, x_layout]
                end
            end

            return NTuple{K * N ÷ nb, CT}(frag)
        end

        @inline function load_c(::Type{<:GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT}}, ::Type{$layout_type{DT}}, workspace, tile::Tile) where {M, N, K, mb, nb, kb, CT, AT, DT}
            laneId = (threadIdx().x - 1) % 32 + 1

            op_y = (laneId - 1) % mb + 1
            op_x = (laneId - 1) ÷ mb + 1

            y, x = (tile.base.M + tile.offset.M + op_y, tile.base.N + tile.offset.N + op_x)

            frag = LocalArray{Tuple{M ÷ mb, N ÷ nb}, AT}(undef)
            @loopinfo unroll for m = 1 : M ÷ mb
                @loopinfo unroll for n = 1 : N ÷ nb
                    @inbounds @immutable frag[m,n] = workspace[y + mb * (m - 1), x + nb * (n - 1)]
                end
            end

            return NTuple{M * N ÷ 32, AT}(frag)
        end

        @inline function store_d(::Type{<:GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT}}, ::Type{$layout_type{DT}}, workspace, frag, tile::Tile) where {M, N, K, mb, nb, kb, CT, AT, DT}
            laneId = (threadIdx().x - 1) % 32 + 1

            op_y = (laneId - 1) % mb + 1
            op_x = (laneId - 1) ÷ mb + 1

            y, x = (tile.base.M + tile.offset.M + op_y, tile.base.N + tile.offset.N + op_x)

            frag = LocalArray{Tuple{M ÷ mb, N ÷ nb}, AT}(frag)
            @loopinfo unroll for m = 1 : M ÷ mb
                @loopinfo unroll for n = 1 : N ÷ nb
                    @inbounds workspace[y + mb * (m - 1), x + nb * (n - 1)] = frag[m,n]
                end
            end
        end
    end
end

abstract type FPUOp{M, N, K, mb, nb, kb, CT, AT} <: GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT} end
@inline function operator_fma(::Type{FPUOp{M, N, K, mb, nb, kb, CT, AT}}, a::CT, b::CT, c::AT) where {M, N, K, mb, nb, kb, CT, AT}
    return fma(a, b, c)
end

abstract type TropicalFPUOp{M, N, K, mb, nb, kb, CT, AT} <: GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT} end
@inline function operator_fma(::Type{TropicalFPUOp{M, N, K, mb, nb, kb, CT, AT}}, a::CT, b::CT, c::AT) where {M, N, K, mb, nb, kb, CT, AT}
    return max(a + b, c)
end

@inline function mma(operator_type::Type{<:GeneralFPUOp{M, N, K, mb, nb, kb, CT, AT}}, a_frag, b_frag, c_frag) where {M, N, K, mb, nb, kb, CT, AT}
    a_frag = LocalArray{Tuple{M ÷ mb, K}, CT}(a_frag)
    b_frag = LocalArray{Tuple{K, N ÷ nb}, CT}(b_frag)
    c_frag = LocalArray{Tuple{M ÷ mb, N ÷ nb}, AT}(c_frag)

    @loopinfo unroll for m = 1 : M ÷ mb
        @loopinfo unroll for n = 1 : N ÷ nb
            @loopinfo unroll for k = 1 : K
                @inbounds @immutable c_frag[m,n] = operator_fma(operator_type, a_frag[m, k], b_frag[k, n], c_frag[m, n])
            end
        end
    end

    return NTuple{M * N ÷ 32, AT}(c_frag)
end

# ----
# WMMA
# ----

# WMMAOp's register types cannot be configured, and CT/AT should be identical to their
# respective shared memory layouts eltypes. this is because WMMA intrinsics are used
# to load/store shared memory, so we cannot perform any conversions on the fly.
# note that there still can be a conversion between global and shared memory.
struct WMMAOp{M, N, K, CT, AT} end

@inline shape(::Type{WMMAOp{M, N, K, CT, AT}}) where {M, N, K, CT, AT} = (M = M, N = N, K = K)

for (M, N, K) in [
        (16, 16, 16),
        (8, 32, 16),
        (32, 8, 16)
    ],
    (layout_type, wmma_layout_type) in [
        (Layout.ColMajor, WMMA.ColMajor),
        (Layout.UnsafeAlignedColMajor, WMMA.ColMajor),
        (Layout.RowMajor, WMMA.RowMajor),
        (Layout.UnsafeAlignedRowMajor, WMMA.RowMajor),
    ]
    @eval begin
        # TODO: Have accessors in CUDA.jl to get the fragment sizes?
        # FP16 (16, 16, 16), (8, 32, 16), and (32, 8, 16)
        @inline fragtype_a(::Type{WMMAOp{$M, $N, $K, CT, AT}}, ::Type{$layout_type{CT}}) where {CT, AT} = WMMA.Fragment{$M, $N, $K, 16, CT, $wmma_layout_type, WMMA.MatrixA}
        @inline fragtype_b(::Type{WMMAOp{$M, $N, $K, CT, AT}}, ::Type{$layout_type{CT}}) where {CT, AT} = WMMA.Fragment{$M, $N, $K, 16, CT, $wmma_layout_type, WMMA.MatrixB}
        @inline fragtype_accum(::Type{WMMAOp{$M, $N, $K, CT, AT}}, ::Type{$layout_type{AT}}) where {CT, AT} = WMMA.Fragment{$M, $N, $K, 8, AT, WMMA.Unspecified, WMMA.Accumulator}
    end
end

# convert_index_func: function used to transpose the index in case of a row-major layout
for (layout_type, wmma_layout_type, convert_index_func) in [
                                        (Layout.ColMajor, WMMA.ColMajor, identity),
                                        (Layout.UnsafeAlignedColMajor, WMMA.ColMajor, identity),
                                        (Layout.RowMajor, WMMA.RowMajor, transpose),
                                        (Layout.UnsafeAlignedRowMajor, WMMA.RowMajor, transpose),
                                       ]
    @eval begin
        @inline function load_a(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{CT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_index = linearise($convert_index_func(tile), size(workspace))

            ptr = pointer(workspace, linear_index)
            return WMMA.load_a(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function load_b(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{CT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_index = linearise($convert_index_func(tile), size(workspace))

            ptr = pointer(workspace, linear_index)
            return WMMA.load_b(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function load_c(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{AT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_index = linearise($convert_index_func(tile), size(workspace))

            ptr = pointer(workspace, linear_index)
            return WMMA.load_c(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function store_d(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{AT}}, workspace, frag, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_index = linearise($convert_index_func(tile), size(workspace))

            ptr = pointer(workspace, linear_index)
            WMMA.store_d(ptr, frag, size(workspace, 1), $wmma_layout_type, conf)
        end
    end
end

function mma(::Type{WMMAOp{M, N, K, CT, AT}}, a_frag, b_frag, c_frag) where {M, N, K, CT, AT}
    conf = WMMA.Config{M, N, K, AT}
    return WMMA.mma(a_frag, b_frag, c_frag, conf)
end

# -----------
# WMMAComplex
# -----------

struct WMMAComplexOp{M, N, K, CT, AT} end

@inline shape(::Type{WMMAComplexOp{M, N, K, CT, AT}}) where {M, N, K, CT, AT} = (M = M, N = N, K = K)

# convert_index_func: function used to transpose the index in case of a row-major layout
for (layout_type, base_layout, wmma_layout_type, convert_index_func) in [
                                        (Layout.SplitColMajor, Layout.UnsafeAlignedColMajor, WMMA.ColMajor, identity),
                                        (Layout.SplitRowMajor, Layout.UnsafeAlignedRowMajor, WMMA.RowMajor, transpose),
                                       ]
    @eval begin
        @inline fragtype_a(::Type{WMMAComplexOp{M, N, K, CT, AT}}, ::Type{$layout_type{CT}}) where {M, N, K, CT, AT} = NTuple{2, fragtype_a(WMMAOp{M, N, K, CT, AT}, $base_layout{CT})}
        @inline fragtype_b(::Type{WMMAComplexOp{M, N, K, CT, AT}}, ::Type{$layout_type{CT}}) where {M, N, K, CT, AT} = NTuple{2, fragtype_b(WMMAOp{M, N, K, CT, AT}, $base_layout{CT})}
        @inline fragtype_accum(::Type{WMMAComplexOp{M, N, K, CT, AT}}, ::Type{$layout_type{AT}}) where {M, N, K, CT, AT} = NTuple{2, fragtype_accum(WMMAOp{M, N, K, CT, AT}, $base_layout{AT})}

        @inline function load_a(::Type{WMMAComplexOp{M, N, K, CT, AT}}, ::Type{$layout_type{CT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}
            ind = linearise($convert_index_func(tile), (size(workspace)[1], size(workspace)[2]))

            return (WMMA.load_a(pointer(workspace, ind), size(workspace)[1], $wmma_layout_type, conf),
                    WMMA.load_a(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], $wmma_layout_type, conf))
        end

        @inline function load_b(::Type{WMMAComplexOp{M, N, K, CT, AT}}, ::Type{$layout_type{CT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}
            ind = linearise($convert_index_func(tile), (size(workspace)[1], size(workspace)[2]))

            return (WMMA.load_b(pointer(workspace, ind), size(workspace)[1], $wmma_layout_type, conf),
                    WMMA.load_b(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], $wmma_layout_type, conf))
        end

        @inline function load_c(::Type{WMMAComplexOp{M, N, K, CT, AT}}, ::Type{$layout_type{AT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}
            ind = linearise($convert_index_func(tile), (size(workspace)[1], size(workspace)[2]))

            return (WMMA.load_c(pointer(workspace, ind), size(workspace)[1], $wmma_layout_type, conf),
                    WMMA.load_c(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], $wmma_layout_type, conf))
        end

        @inline function store_d(::Type{WMMAComplexOp{M, N, K, CT, AT}}, ::Type{$layout_type{AT}}, workspace, frag, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}
            ind = linearise($convert_index_func(tile), (size(workspace)[1], size(workspace)[2]))

            WMMA.store_d(pointer(workspace, ind), frag[1], size(workspace)[1], $wmma_layout_type, conf)
            WMMA.store_d(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), frag[2], size(workspace)[1], $wmma_layout_type, conf)
        end
    end
end

using LLVM

@inline function mma(::Type{WMMAComplexOp{M, N, K, CT, AT}}, a_frag, b_frag, c_frag) where {M, N, K, CT, AT}
    conf = WMMA.Config{M, N, K, AT}

    c_re = c_frag[1]
    c_im = c_frag[2]

    c_re = WMMA.mma(a_frag[1],  b_frag[1], c_re, conf)
    c_re = WMMA.mma(.-(a_frag[2]), b_frag[2], c_re, conf)

    c_im = WMMA.mma(a_frag[1], b_frag[2], c_im, conf)
    c_im = WMMA.mma(a_frag[2], b_frag[1], c_im, conf)

    return (c_re, c_im)
end

# --------
# WMMADual
# --------

struct WMMADualOp{M, N, K, CT, AT} end

@inline shape(::Type{WMMADualOp{M, N, K, CT, AT}}) where {M, N, K, CT, AT} = (M = M, N = N, K = K)

@inline fragtype_a(::Type{WMMADualOp{M, N, K, CT, AT}}, ::Type{Layout.SplitColMajor{CT}}) where {M, N, K, CT, AT} = NTuple{2, fragtype_a(WMMAOp{M, N, K, CT, AT}, Layout.UnsafeAlignedColMajor{CT})}
@inline fragtype_b(::Type{WMMADualOp{M, N, K, CT, AT}}, ::Type{Layout.SplitColMajor{CT}}) where {M, N, K, CT, AT} = NTuple{2, fragtype_b(WMMAOp{M, N, K, CT, AT}, Layout.UnsafeAlignedColMajor{CT})}
@inline fragtype_accum(::Type{WMMADualOp{M, N, K, CT, AT}}, ::Type{Layout.SplitColMajor{AT}}) where {M, N, K, CT, AT} = NTuple{2, fragtype_accum(WMMAOp{M, N, K, CT, AT}, Layout.UnsafeAlignedColMajor{AT})}

@inline function load_a(::Type{WMMADualOp{M, N, K, CT, AT}}, ::Type{Layout.SplitColMajor{CT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
    conf = WMMA.Config{M, N, K, AT}
    ind = linearise(tile, (size(workspace)[1], size(workspace)[2]))

    return (WMMA.load_a(pointer(workspace, ind), size(workspace)[1], WMMA.ColMajor, conf),
            WMMA.load_a(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], WMMA.ColMajor, conf))
end

@inline function load_b(::Type{WMMADualOp{M, N, K, CT, AT}}, ::Type{Layout.SplitColMajor{CT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
    conf = WMMA.Config{M, N, K, AT}
    ind = linearise(tile, (size(workspace)[1], size(workspace)[2]))

    return (WMMA.load_b(pointer(workspace, ind), size(workspace)[1], WMMA.ColMajor, conf),
            WMMA.load_b(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], WMMA.ColMajor, conf))
end

@inline function load_c(::Type{WMMADualOp{M, N, K, CT, AT}}, ::Type{Layout.SplitColMajor{AT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
    conf = WMMA.Config{M, N, K, AT}
    ind = linearise(tile, (size(workspace)[1], size(workspace)[2]))

    return (WMMA.load_c(pointer(workspace, ind), size(workspace)[1], WMMA.ColMajor, conf),
            WMMA.load_c(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], WMMA.ColMajor, conf))
end

@inline function store_d(::Type{WMMADualOp{M, N, K, CT, AT}}, ::Type{Layout.SplitColMajor{AT}}, workspace, frag, tile::Tile) where {M, N, K, CT, AT}
    conf = WMMA.Config{M, N, K, AT}
    ind = linearise(tile, (size(workspace)[1], size(workspace)[2]))

    WMMA.store_d(pointer(workspace, ind), frag[1], size(workspace)[1], WMMA.ColMajor, conf)
    WMMA.store_d(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), frag[2], size(workspace)[1], WMMA.ColMajor, conf)
end

@inline function mma(::Type{WMMADualOp{M, N, K, CT, AT}}, a_frag, b_frag, c_frag) where {M, N, K, CT, AT}
    conf = WMMA.Config{M, N, K, AT}

    c_re = c_frag[1]
    c_du = c_frag[2]

    c_re = WMMA.mma(a_frag[1],  b_frag[1], c_re, conf)

    c_du = WMMA.mma(a_frag[1], b_frag[2], c_du, conf)
    c_du = WMMA.mma(a_frag[2], b_frag[1], c_du, conf)

    return (c_re, c_du)
end

# --------------
# Volta mma.sync
# --------------

# TODO: Generalise this to:
# - Other configurations than NN
# - Other architectures
# - Other shapes?
struct VoltaMmaSyncOp end

@inline shape(::Type{VoltaMmaSyncOp}) = (M = 64, N = 64, K = 4)

@inline function load_a(::Type{VoltaMmaSyncOp}, L::Type{Layout.VoltaSwizzledOperandA{Float16}}, workspace, tile::Tile)
    # Index: (m5|m2|k1|k0)
    a_frag = LocalArray{Tuple{16}, Float16}(undef)

    @unrolled for ins = 0:1
        m = b(tid(), 0, 0)  +
            b(tid(), 1, 1)  +
            b(tile.index.K, 3, 2) +
            b(tid(), 2, 3)  +
            b(tid(), 4, 4)  +
            b(ins, 0, 5)

        k = constant(0)

        @inbounds val = vloada(Vec{8, Float16}, workspace, Layout.swizzle(L, tile.index.M+m, tile.index.K+k))

        @unrolled for offset = 0:7
            frag_offset = b(offset, 0, 0) +                           # k0
                          b(offset, 1, 1) +                           # k1
                          (b(offset, 2, 2) ⊻ b(tile.index.K, 3, 2)) + # m2 = (m2+k3) + k3
                          b(ins, 0, 3)                                # m5

            @inbounds @immutable a_frag[frag_offset] = val[offset].value
        end
    end

    a_frag.data
end

@inline function load_b(::Type{VoltaMmaSyncOp}, L::Type{Layout.VoltaSwizzledOperandB{Float16}}, workspace, tile::Tile)
    # index: (n5|n2|n1|n0)
    b_frag = LocalArray{Tuple{16}, Float16}(undef)

    @unrolled for ins = 0:1
        k = b(tid(), 0, 0) +
            b(tid(), 1, 1)

        n = b(tid(), 3, 3) +
            b(tid(), 4, 4) +
            b(ins, 0, 5)

        @inbounds val = vloada(Vec{8, Float16}, workspace, Layout.swizzle(L, tile.index.K+k, tile.index.N+n))

        @unrolled for offset = 0:7
            frag_offset = b(offset, 0, 0) +    # n0
                          b(offset, 1, 1) +    # n1
                          b(offset, 2, 2) +    # n2
                          b(ins, 0, 3)         # n5

            @inbounds @immutable b_frag[frag_offset] = val[offset].value
        end
    end

    b_frag.data
end

@inline function store_d(::Type{VoltaMmaSyncOp}, ::Type{Layout.Padded{Layout.UnsafeAlignedRowMajor{Float16}, P}}, workspace, frag, tile::Tile) where {P}
    # index: (m5|m2|m1|n5|n4|n2|n0)

    @unrolled for ins = 0:15
        # TODO: vectorise
        @unrolled for offset = 0:1
            m = b(tid(), 0, 0) +
                b(ins, 3, 1) +
                b(tid(), 2, 3) +
                b(tid(), 4, 4) +
                tile.index.M

            n = b(offset, 0, 0) +
                b(tid(), 1, 1) +
                b(ins, 0, 2) +
                b(tid(), 3, 3) +
                b(ins, 1, 4) +
                b(ins, 2, 5) +
                tile.index.N

            frag_index = b(offset, 0, 0) +   # n0
                         b(ins, 0, 1) +      # n2
                         b(ins, 1, 2) +      # n4
                         b(ins, 2, 3) +      # n5
                         b(ins, 3, 4)        # m1

            offset_M = b(tid(), 0, 0) +      # m0
                       b(ins, 3, 1) +        # m1
                       b(tid(), 2, 2) +      # m3
                       b(tid(), 4, 3) +      # m4
                       b(tid(), 5, 4)        # m6

            offset_N = n

            offset = convert(Int, offset_N) + 258 * convert(Int, offset_M)

            @inbounds workspace[1 + offset] = frag[frag_index]
        end
    end
end

@inline function mma(::Type{VoltaMmaSyncOp}, a_frag, b_frag, acc_frag)
    # The optimal Volta mma.sync macro-mma is a 64 x 64 x 4 matrix
    # multiply-accumulate per warp, using 16 mma.sync instructions.
    @unrolled for instruction = 0:15
        inner_row = b(instruction, 0)
        outer_row = b(instruction, 1)
        inner_col = b(instruction, 2)
        outer_col = b(instruction, 3)

        # Each mma.sync instruction will perform a 16 x 16 x 4 matrix
        # multiply-accumulate per warp.
        # Behind the scenes, each of those mma.syncs are actually four
        # 8 x 8 x 4 multiply-accumulates: one for each quad-pair (QP).

        # Use a zig-zag sequence to increase reuse of the A operand.
        # NOTE: This does not seem to influence performance (at least not
        # significantly), but CUTLASS does this, so we might as well too...
        #
        # Basically, this changes the order of instructions from e.g. this:
        #
        # 0     4       8       12
        # 1     5       9       13
        # 2     6      10       14
        # 3     7      11       15
        #
        # to this:
        #
        # 0     7       8       15
        # 1     6       9       14
        # 2     5      10       13
        # 3     4      11       12
        #
        # by flipping the order we iterate through the rows every 2nd column.
        zz_inner_row = inner_row ⊻ (inner_col % 2)
        zz_outer_row = outer_row ⊻ (inner_col % 2)

        # Extract the a, b, and c fragments for this particular mma.sync
        # instruction.
        ins_a_frag = LocalArray{Tuple{4}, Float16}(undef)
        ins_b_frag = LocalArray{Tuple{4}, Float16}(undef)
        ins_c_frag = LocalArray{Tuple{8}, Float32}(undef)

        # Get the A fragment for this mma.sync
        @unrolled for i = 0:3
            offset = b(i, 0, 0) +            # k0
                     b(i, 1, 1) +            # k1
                     b(zz_inner_row, 0, 2) + # m2
                     b(zz_outer_row, 0, 3)   # m5
            @inbounds @immutable ins_a_frag[i] = a_frag[offset]
        end

        # Get the B fragment for this mma.sync
        @unrolled for i = 0:3
            offset = b(i, 0, 0) +           # n0
                     b(i, 1, 1) +           # n1
                     b(inner_col, 0, 2) +   # n2
                     b(outer_col, 0, 3)     # n5

            @inbounds @immutable ins_b_frag[i] = b_frag[offset]
        end

        # Get the C fragment for this mma.sync
        @unrolled for i = 0:7
            offset = b(i, 0, 0) +            # n0
                     b(inner_col, 0, 1) +    # n2
                     b(i, 2, 2) +            # n4
                     b(outer_col, 0, 3) +    # n5
                     b(i, 1, 4) +            # m1
                     b(zz_inner_row, 0, 5) + # m2
                     b(zz_outer_row, 0, 6)   # m5

            @inbounds @immutable ins_c_frag[i] = acc_frag[offset]
        end

        # Perform mma.sync
        ins_d_frag = mma884_row_row(ins_a_frag.data, ins_b_frag.data, ins_c_frag.data)

        # Store D fragment for this mma.sync
        @unrolled for i = 0:7
            offset = b(i, 0, 0) +            # n0
                     b(inner_col, 0, 1) +    # n2
                     b(i, 2, 2) +            # n4
                     b(outer_col, 0, 3) +    # n5
                     b(i, 1, 4) +            # m1
                     b(zz_inner_row, 0, 5) + # m2
                     b(zz_outer_row, 0, 6)   # m5

            @inbounds @immutable acc_frag[offset] = ins_d_frag[i]
        end
    end

    acc_frag
end

end
