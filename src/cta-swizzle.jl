export CTASwizzle
module CTASwizzle

# CTA Swizzling functions to improve L2 hit rate.

# Debugging
function visualise_cta_swizzle(swizzle_type, block_shape = (M = 128, N = 128), matmul_shape = (M = 1024, N = 1024))
    res = fill("", (cld(matmul_shape.M, block_shape.M),
                    cld(matmul_shape.N, block_shape.N)))

    BX, BY = number_of_blocks(swizzle_type, block_shape, matmul_shape)

    for bx = 1:BX, by = 1:BY
        block_i, block_j = cta_swizzle(swizzle_type, (x = bx, y = by), block_shape)
        res[block_i ÷ block_shape.M + 1, block_j ÷ block_shape.N + 1] = "($bx, $by)"
    end

    res
end

# ----------------------------
# Identity swizzling operation
# ----------------------------
export Identity

struct Identity end

@inline function number_of_blocks(::Type{Identity}, block_shape, matmul_shape)
    (cld(matmul_shape.M, block_shape.M),
     cld(matmul_shape.N, block_shape.N))
end

@inline function cta_swizzle(::Type{Identity}, blockIdx, block_shape)
    bx = blockIdx.x - 1
    by = blockIdx.y - 1

    block_i = bx
    block_j = by

    block_i * block_shape.M, block_j * block_shape.N
end

# ----------------------------
# Horizontally tiled swizzling
# ----------------------------

# Example: TileSize = 4:
#
# "(1, 1)"   "(2, 1)"   "(3, 1)"   "(4, 1)"  || "(1, 2)"   "(2, 2)"   "(3, 2)"   "(4, 2)"
# =======================================================================================
# "(5, 1)"   "(6, 1)"   "(7, 1)"   "(8, 1)"  || "(5, 2)"   "(6, 2)"   "(7, 2)"   "(8, 2)"
# =======================================================================================
# "(9, 1)"   "(10, 1)"  "(11, 1)"  "(12, 1)" || "(9, 2)"   "(10, 2)"  "(11, 2)"  "(12, 2)"
# =======================================================================================
# "(13, 1)"  "(14, 1)"  "(15, 1)"  "(16, 1)" || "(13, 2)"  "(14, 2)"  "(15, 2)"  "(16, 2)"
# =======================================================================================
# "(17, 1)"  "(18, 1)"  "(19, 1)"  "(20, 1)" || "(17, 2)"  "(18, 2)"  "(19, 2)"  "(20, 2)"
# =======================================================================================
# "(21, 1)"  "(22, 1)"  "(23, 1)"  "(24, 1)" || "(21, 2)"  "(22, 2)"  "(23, 2)"  "(24, 2)"
# =======================================================================================
# "(25, 1)"  "(26, 1)"  "(27, 1)"  "(28, 1)" || "(25, 2)"  "(26, 2)"  "(27, 2)"  "(28, 2)"
# =======================================================================================
# "(29, 1)"  "(30, 1)"  "(31, 1)"  "(32, 1)" || "(29, 2)"  "(30, 2)"  "(31, 2)"  "(32, 2)"

export HorizontallyTiled

struct HorizontallyTiled{TileSize} end

@inline function number_of_blocks(::Type{HorizontallyTiled{TileSize}}, block_shape, matmul_shape) where {TileSize}
    (cld(matmul_shape.M * TileSize, block_shape.M),
     cld(matmul_shape.N, block_shape.N * TileSize))
end

@inline function cta_swizzle(::Type{HorizontallyTiled{TileSize}}, blockIdx, block_shape) where {TileSize}
    bx = blockIdx.x - 1
    by = blockIdx.y - 1

    block_i = bx ÷ TileSize
    block_j = (by * TileSize) + (bx % TileSize)

    block_i * block_shape.M, block_j * block_shape.N
end

# --------------------------
# Vertically tiled swizzling
# --------------------------

# Example: TileSize = 4:
#
# "(1, 1)" || "(1, 5)" || "(1, 9)"  || "(1, 13)" || "(1, 17)" || "(1, 21)" || "(1, 25)" || "(1, 29)"
# "(1, 2)" || "(1, 6)" || "(1, 10)" || "(1, 14)" || "(1, 18)" || "(1, 22)" || "(1, 26)" || "(1, 30)"
# "(1, 3)" || "(1, 7)" || "(1, 11)" || "(1, 15)" || "(1, 19)" || "(1, 23)" || "(1, 27)" || "(1, 31)"
# "(1, 4)" || "(1, 8)" || "(1, 12)" || "(1, 16)" || "(1, 20)" || "(1, 24)" || "(1, 28)" || "(1, 32)"
# ====================================================================================================
# "(2, 1)" || "(2, 5)" || "(2, 9)"  || "(2, 13)" || "(2, 17)" || "(2, 21)" || "(2, 25)" || "(2, 29)"
# "(2, 2)" || "(2, 6)" || "(2, 10)" || "(2, 14)" || "(2, 18)" || "(2, 22)" || "(2, 26)" || "(2, 30)"
# "(2, 3)" || "(2, 7)" || "(2, 11)" || "(2, 15)" || "(2, 19)" || "(2, 23)" || "(2, 27)" || "(2, 31)"
# "(2, 4)" || "(2, 8)" || "(2, 12)" || "(2, 16)" || "(2, 20)" || "(2, 24)" || "(2, 28)" || "(2, 32)"

export VerticallyTiled

struct VerticallyTiled{TileSize} end

@inline function number_of_blocks(::Type{VerticallyTiled{TileSize}}, block_shape, matmul_shape) where {TileSize}
    (cld(matmul_shape.M, block_shape.M * TileSize),
     cld(matmul_shape.N * TileSize, block_shape.N))
end

@inline function cta_swizzle(::Type{VerticallyTiled{TileSize}}, blockIdx, block_shape) where {TileSize}
    bx = blockIdx.x - 1
    by = blockIdx.y - 1

    block_i = (bx * TileSize) + (by % TileSize)
    block_j = by ÷ TileSize

    block_i * block_shape.M, block_j * block_shape.N
end

# ----------------------------
# Lebesgue space-filling curve
# ----------------------------

# Example:
# "(1, 1)"   "(3, 1)"   "(9, 1)"   "(11, 1)"  "(33, 1)"  "(35, 1)"  "(41, 1)"  "(43, 1)"
# "(2, 1)"   "(4, 1)"   "(10, 1)"  "(12, 1)"  "(34, 1)"  "(36, 1)"  "(42, 1)"  "(44, 1)"
# "(5, 1)"   "(7, 1)"   "(13, 1)"  "(15, 1)"  "(37, 1)"  "(39, 1)"  "(45, 1)"  "(47, 1)"
# "(6, 1)"   "(8, 1)"   "(14, 1)"  "(16, 1)"  "(38, 1)"  "(40, 1)"  "(46, 1)"  "(48, 1)"
# "(17, 1)"  "(19, 1)"  "(25, 1)"  "(27, 1)"  "(49, 1)"  "(51, 1)"  "(57, 1)"  "(59, 1)"
# "(18, 1)"  "(20, 1)"  "(26, 1)"  "(28, 1)"  "(50, 1)"  "(52, 1)"  "(58, 1)"  "(60, 1)"
# "(21, 1)"  "(23, 1)"  "(29, 1)"  "(31, 1)"  "(53, 1)"  "(55, 1)"  "(61, 1)"  "(63, 1)"
# "(22, 1)"  "(24, 1)"  "(30, 1)"  "(32, 1)"  "(54, 1)"  "(56, 1)"  "(62, 1)"  "(64, 1)"

@inline function extract_even_bits(x)
    x = x & 0x55555555
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF

    x
end

@inline extract_odd_bits(x) = extract_even_bits(x >> 1)

export LebesgueCurve

struct LebesgueCurve end

@inline function number_of_blocks(::Type{LebesgueCurve}, block_shape, matmul_shape)
    (cld(matmul_shape.M, block_shape.M) * cld(matmul_shape.N, block_shape.N), 1)
end

@inline function cta_swizzle(::Type{LebesgueCurve}, blockIdx, block_shape)
    bx = blockIdx.x - 1

    block_i = extract_even_bits(bx)
    block_j = extract_odd_bits(bx)

    block_i * block_shape.M, block_j * block_shape.N
end

end
