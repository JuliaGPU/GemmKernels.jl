export Operator
module Operator

using CUDA
using GemmKernels
using GemmKernels.Tiling
using GemmKernels: LocalArray
using KernelAbstractions.Extras: @unroll
using Base: setindex

# -------------------------------------
# Default definition for padded layouts
# -------------------------------------

for f in (:fragtype_a, :fragtype_b, :fragtype_accum, :load_a, :load_b, :load_c, :store_d)
    @eval @inline $f(op, ::Type{Layout.Padded{L, P}}, args...) where {L, P} = $f(op, L, args...)
end


# ---
# FPU
# ---

struct FPUOp{M, N, K, T} end

@inline shape(::Type{FPUOp{M, N, K, T}}) where {M, N, K, T} = (M = M, N = N, K = K)

for (layout_type, convert_index_func) in [
                                        (Layout.AlignedColMajor, identity),
                                        (Layout.AlignedRowMajor, x -> reverse(Tuple(x)))
                                       ]
    @eval begin
        # ? Maybe the other way around is more efficient ? 
        # ! There is a difference between datatype and computetype needed.
        @inline fragtype_a(::Type{FPUOp{M, N, K, T}}, ::Type{$layout_type{T}}) where {M, N, K, T} = NTuple{M * K ÷ 4, T}
        @inline fragtype_b(::Type{FPUOp{M, N, K, T}}, ::Type{$layout_type{T}}) where {M, N, K, T} = NTuple{K * N ÷ 8, T}

        @inline function fragtype_accum(::Type{FPUOp{M, N, K, T}}, ::Type{$layout_type{DT}}) where {M, N, K, T, DT}
            return NTuple{M * N ÷ 32, DT}
        end

        @inline function load_a(::Type{FPUOp{M, N, K, T}}, ::Type{$layout_type{T}}, workspace, tile::Tile) where {M, N, K, T}
            laneId = (threadIdx().x - 1) % 32 + 1

            op_y = (laneId - 1) ÷ 8 + 1
            y, x = $convert_index_func((tile.base.M + tile.offset.M + op_y, tile.base.K + tile.offset.K + 1))

            frag = LocalArray{Tuple{M * K ÷ 4}, T}(undef)
            @unroll for m = 1 : M ÷ 4
                @unroll for k = 1 : K
                    @inbounds frag = setindex(frag, T(workspace[y + 4 * (m - 1), x + (k - 1)]), m + (M ÷ 4) * (k - 1))
                end
            end
            
            return NTuple{M * K ÷ 4, T}(frag)
        end

        @inline function load_b(::Type{FPUOp{M, N, K, T}}, ::Type{$layout_type{T}}, workspace, tile::Tile) where {M, N, K, T}
            laneId = (threadIdx().x - 1) % 32 + 1

            op_x = (laneId - 1) % 8 + 1
            y, x = $convert_index_func((tile.base.K + tile.offset.K + 1, tile.base.N + tile.offset.N + op_x))

            frag = LocalArray{Tuple{K * N ÷ 8}, T}(undef)
            @unroll for n = 1 : N ÷ 8
                @unroll for k = 1 : K
                    @inbounds frag = setindex(frag, T(workspace[y + (k - 1), x + 8 * (n - 1)]), n + (N ÷ 8) * (k - 1))
                end
            end

            return NTuple{K * N ÷ 8, T}(frag)
        end

        @inline function load_c(::Type{FPUOp{M, N, K, T}}, ::Type{$layout_type{DT}}, workspace, tile::Tile) where {M, N, K, T, DT}
            laneId = (threadIdx().x - 1) % 32 + 1

            op_y = (laneId - 1) ÷ 8 + 1
            op_x = (laneId - 1) % 8 + 1

            y, x = $convert_index_func((tile.base.M + tile.offset.M + op_y, tile.base.N + tile.offset.N + op_x))

            frag = LocalArray{Tuple{M * N ÷ 32}, DT}(undef)
            @unroll for m = 1 : M ÷ 4
                @unroll for n = 1 : N ÷ 8 
                    @inbounds frag = setindex(frag, DT(workspace[y + 4 * (m - 1), x + 8 * (n - 1)]), m + (M ÷ 4) * (n - 1))
                end
            end

            return NTuple{M * N ÷ 32, DT}(frag)
        end

        @inline function store_d(::Type{FPUOp{M, N, K, T}}, ::Type{$layout_type{DT}}, workspace, frag, tile::Tile) where {M, N, K, T, DT}
            laneId = (threadIdx().x - 1) % 32 + 1

            op_y = (laneId - 1) ÷ 8 + 1
            op_x = (laneId - 1) % 8 + 1

            y, x = $convert_index_func((tile.base.M + tile.offset.M + op_y, tile.base.N + tile.offset.N + op_x))

            @unroll for m = 1 : M ÷ 4
                @unroll for n = 1 : N ÷ 8 
                    @inbounds workspace[y + 4 * (m - 1), x + 8 * (n - 1)] = frag[m + (M ÷ 4) * (n - 1)]
                end
            end
        end
    end
end

function mma(::Type{FPUOp{M, N, K, T}}, a_frag, b_frag, c_frag::NTuple{C, DT}) where {M, N, K, T, C, DT}
    @unroll for k = 1 : K
        @unroll for m = 1 : M ÷ 4
            @unroll for n = 1 : N ÷ 8 
                @inbounds c_frag = setindex(
                    c_frag,
                    DT(fma(a_frag[m + (M ÷ 4) * (k - 1)], b_frag[n + (N ÷ 8) * (k - 1)], c_frag[m + (M ÷ 4) * (n - 1)])),
                    m + (M ÷ 4) * (n - 1)
                )
            end
        end
    end
    return c_frag
end

# ----
# WMMA
# ----

struct WMMAOp{M, N, K, T} end

@inline shape(::Type{WMMAOp{M, N, K, T}}) where {M, N, K, T} = (M = M, N = N, K = K)

# convert_index_func: function used to transpose the index in case of a row-major layout
for (layout_type, wmma_layout_type, convert_index_func) in [
                                        (Layout.AlignedColMajor, WMMA.ColMajor, identity),
                                        (Layout.AlignedRowMajor, WMMA.RowMajor, x -> reverse(Tuple(x)))
                                       ]
    @eval begin
        @inline fragtype_a(::Type{WMMAOp{16, 16, 16, T}}, ::Type{$layout_type{Float16}}) where {T} = WMMA.Fragment{16, 16, 16, 16, Float16, $wmma_layout_type, WMMA.MatrixA}
        @inline fragtype_b(::Type{WMMAOp{16, 16, 16, T}}, ::Type{$layout_type{Float16}}) where {T} = WMMA.Fragment{16, 16, 16, 16, Float16, $wmma_layout_type, WMMA.MatrixB}
        @inline fragtype_accum(::Type{WMMAOp{16, 16, 16, T}}, ::Type{$layout_type{T}}) where {T} = WMMA.Fragment{16, 16, 16, 8, T, WMMA.Unspecified, WMMA.Accumulator}

        @inline function load_a(::Type{WMMAOp{M, N, K, T}}, ::Type{$layout_type{Float16}}, workspace, tile::Tile) where {M, N, K, T}
            conf = WMMA.Config{M, N, K, T}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(Float16)
            return WMMA.load_a(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function load_b(::Type{WMMAOp{M, N, K, T}}, ::Type{$layout_type{Float16}}, workspace, tile::Tile) where {M, N, K, T}
            conf = WMMA.Config{M, N, K, T}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(Float16)
            return WMMA.load_b(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function load_c(::Type{WMMAOp{M, N, K, T}}, ::Type{$layout_type{T}}, workspace, tile::Tile) where {M, N, K, T}
            conf = WMMA.Config{M, N, K, T}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(T)
            return WMMA.load_c(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function store_d(::Type{WMMAOp{M, N, K, T}}, ::Type{$layout_type{T}}, workspace, frag, tile::Tile) where {M, N, K, T}
            conf = WMMA.Config{M, N, K, T}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(T)
            WMMA.store_d(ptr, frag, size(workspace, 1), $wmma_layout_type, conf)
        end
    end
end

function mma(::Type{WMMAOp{M, N, K, T}}, a_frag, b_frag, c_frag) where {M, N, K, T}
    conf = WMMA.Config{M, N, K, T}
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
