export Operator
module Operator

using CUDA
using GemmKernels
using GemmKernels.Tiling
using GemmKernels: LocalArray, @immutable
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
function operator_fma(::Type{FPUOp{M, N, K, mb, nb, kb, CT, AT}}, a::CT, b::CT, c::AT) where {M, N, K, mb, nb, kb, CT, AT}
    return fma(a, b, c)
end

abstract type TropicalFPUOp{M, N, K, CT, AT} <: GeneralFPUOp{M, N, K, 4, 8, 1, CT, AT} end
function operator_fma(::Type{TropicalFPUOp{M, N, K, CT, AT}}, a::CT, b::CT, c::AT) where {M, N, K, CT, AT}
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

# convert_index_func: function used to transpose the index in case of a row-major layout
for (layout_type, wmma_layout_type, convert_index_func) in [
                                        (Layout.ColMajor, WMMA.ColMajor, identity),
                                        (Layout.UnsafeAlignedColMajor, WMMA.ColMajor, identity),
                                        (Layout.RowMajor, WMMA.RowMajor, x -> reverse(Tuple(x))),
                                        (Layout.UnsafeAlignedRowMajor, WMMA.RowMajor, x -> reverse(Tuple(x))),
                                       ]
    @eval begin
        @inline fragtype_a(::Type{WMMAOp{16, 16, 16, CT, AT}}, ::Type{$layout_type{CT}}) where {CT, AT} = WMMA.Fragment{16, 16, 16, 16, CT, $wmma_layout_type, WMMA.MatrixA}
        @inline fragtype_b(::Type{WMMAOp{16, 16, 16, CT, AT}}, ::Type{$layout_type{CT}}) where {CT, AT} = WMMA.Fragment{16, 16, 16, 16, CT, $wmma_layout_type, WMMA.MatrixB}
        @inline fragtype_accum(::Type{WMMAOp{16, 16, 16, CT, AT}}, ::Type{$layout_type{AT}}) where {CT, AT} = WMMA.Fragment{16, 16, 16, 8, AT, WMMA.Unspecified, WMMA.Accumulator}

        @inline function load_a(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{CT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(CT)
            return WMMA.load_a(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function load_b(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{CT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(CT)
            return WMMA.load_b(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function load_c(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{AT}}, workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(AT)
            return WMMA.load_c(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function store_d(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{AT}}, workspace, frag, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(AT)
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

struct WMMAComplexOp{M, N, K} end

@inline shape(::Type{WMMAComplexOp{M, N, K}}) where {M, N, K} = (M = M, N = N, K = K)

# convert_index_func: function used to transpose the index in case of a row-major layout
for (layout_type, wmma_layout_type, convert_index_func) in [
                                        (Layout.SplitColMajor, WMMA.ColMajor, identity),
                                        (Layout.SplitRowMajor, WMMA.RowMajor, x -> reverse(Tuple(x))),
                                       ]
    @eval begin
        @inline fragtype_a(::Type{WMMAComplexOp{16, 16, 16}}, ::Type{$layout_type{Float16}}) = NTuple{2, WMMA.Fragment{16, 16, 16, 16, Float16, $wmma_layout_type, WMMA.MatrixA}}
        @inline fragtype_b(::Type{WMMAComplexOp{16, 16, 16}}, ::Type{$layout_type{Float16}}) = NTuple{2, WMMA.Fragment{16, 16, 16, 16, Float16, $wmma_layout_type, WMMA.MatrixB}}
        @inline fragtype_accum(::Type{WMMAComplexOp{16, 16, 16}}, ::Type{$layout_type{Float32}}) = NTuple{2, WMMA.Fragment{16, 16, 16, 8, Float32, WMMA.Unspecified, WMMA.Accumulator}}

        @inline function load_a(::Type{WMMAComplexOp{M, N, K}}, ::Type{$layout_type{Float16}}, workspace, tile::Tile) where {M, N, K}
            conf = WMMA.Config{16, 16, 16, Float32}
            ind = linearise($convert_index_func(tile.index), (size(workspace)[1], size(workspace)[2]))

            return (WMMA.load_a(pointer(workspace, ind), size(workspace)[1], $wmma_layout_type, conf),
                    WMMA.load_a(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], $wmma_layout_type, conf))
        end

        @inline function load_b(::Type{WMMAComplexOp{M, N, K}}, ::Type{$layout_type{Float16}}, workspace, tile::Tile) where {M, N, K}
            conf = WMMA.Config{16, 16, 16, Float32}
            ind = linearise($convert_index_func(tile.index), (size(workspace)[1], size(workspace)[2]))

            return (WMMA.load_b(pointer(workspace, ind), size(workspace)[1], $wmma_layout_type, conf),
                    WMMA.load_b(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], $wmma_layout_type, conf))
        end

        @inline function load_c(::Type{WMMAComplexOp{M, N, K}}, ::Type{$layout_type{Float32}}, workspace, tile::Tile) where {M, N, K}
            conf = WMMA.Config{M, N, K, Float32}
            ind = linearise($convert_index_func(tile.index), (size(workspace)[1], size(workspace)[2]))

            return (WMMA.load_c(pointer(workspace, ind), size(workspace)[1], $wmma_layout_type, conf),
                    WMMA.load_c(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], $wmma_layout_type, conf))
        end

        @inline function store_d(::Type{WMMAComplexOp{M, N, K}}, ::Type{$layout_type{Float32}}, workspace, frag, tile::Tile) where {M, N, K}
            conf = WMMA.Config{M, N, K, Float32}
            ind = linearise($convert_index_func(tile.index), (size(workspace)[1], size(workspace)[2]))

            WMMA.store_d(pointer(workspace, ind), frag[1], size(workspace)[1], $wmma_layout_type, conf)
            WMMA.store_d(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), frag[2], size(workspace)[1], $wmma_layout_type, conf)
        end
    end
end

using LLVM

@inline function mma(::Type{WMMAComplexOp{M, N, K}}, a_frag, b_frag, c_frag) where {M, N, K}
    conf = WMMA.Config{16, 16, 16, Float32}

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

struct WMMADualOp{M, N, K} end

@inline shape(::Type{WMMADualOp{M, N, K}}) where {M, N, K} = (M = M, N = N, K = K)

@inline fragtype_a(::Type{WMMADualOp{16, 16, 16}}, ::Type{Layout.SplitColMajor{Float16}}) = NTuple{2, WMMA.Fragment{16, 16, 16, 16, Float16, WMMA.ColMajor, WMMA.MatrixA}}
@inline fragtype_b(::Type{WMMADualOp{16, 16, 16}}, ::Type{Layout.SplitColMajor{Float16}}) = NTuple{2, WMMA.Fragment{16, 16, 16, 16, Float16, WMMA.ColMajor, WMMA.MatrixB}}
@inline fragtype_accum(::Type{WMMADualOp{16, 16, 16}}, ::Type{Layout.SplitColMajor{Float32}}) = NTuple{2, WMMA.Fragment{16, 16, 16, 8, Float32, WMMA.Unspecified, WMMA.Accumulator}}

@inline function load_a(::Type{WMMADualOp{M, N, K}}, ::Type{Layout.SplitColMajor{Float16}}, workspace, tile::Tile) where {M, N, K}
    conf = WMMA.Config{16, 16, 16, Float32}
    ind = linearise(tile.index, (size(workspace)[1], size(workspace)[2]))

    return (WMMA.load_a(pointer(workspace, ind), size(workspace)[1], WMMA.ColMajor, conf),
            WMMA.load_a(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], WMMA.ColMajor, conf))
end

@inline function load_b(::Type{WMMADualOp{M, N, K}}, ::Type{Layout.SplitColMajor{Float16}}, workspace, tile::Tile) where {M, N, K}
    conf = WMMA.Config{16, 16, 16, Float32}
    ind = linearise(tile.index, (size(workspace)[1], size(workspace)[2]))

    return (WMMA.load_b(pointer(workspace, ind), size(workspace)[1], WMMA.ColMajor, conf),
            WMMA.load_b(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], WMMA.ColMajor, conf))
end

@inline function load_c(::Type{WMMADualOp{M, N, K}}, ::Type{Layout.SplitColMajor{Float32}}, workspace, tile::Tile) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ind = linearise(tile.index, (size(workspace)[1], size(workspace)[2]))

    return (WMMA.load_c(pointer(workspace, ind), size(workspace)[1], WMMA.ColMajor, conf),
            WMMA.load_c(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), size(workspace)[1], WMMA.ColMajor, conf))
end

@inline function store_d(::Type{WMMADualOp{M, N, K}}, ::Type{Layout.SplitColMajor{Float32}}, workspace, frag, tile::Tile) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ind = linearise(tile.index, (size(workspace)[1], size(workspace)[2]))

    WMMA.store_d(pointer(workspace, ind), frag[1], size(workspace)[1], WMMA.ColMajor, conf)
    WMMA.store_d(pointer(workspace, ind + size(workspace)[1] * size(workspace)[2]), frag[2], size(workspace)[1], WMMA.ColMajor, conf)
end

@inline function mma(::Type{WMMADualOp{M, N, K}}, a_frag, b_frag, c_frag) where {M, N, K}
    conf = WMMA.Config{16, 16, 16, Float32}

    c_re = c_frag[1]
    c_du = c_frag[2]

    c_re = WMMA.mma(a_frag[1],  b_frag[1], c_re, conf)

    c_du = WMMA.mma(a_frag[1], b_frag[2], c_du, conf)
    c_du = WMMA.mma(a_frag[2], b_frag[1], c_du, conf)

    return (c_re, c_du)
end

end
