export CTASwizzle
module CTASwizzle

# CTA Swizzling functions to improve L2 hit rate.
#
# Every swizzling type `S` implements:
#
#  - `number_of_blocks(S, block_shape, matmul_shape)`: the grid to launch. This must be
#    exactly one thread block per output tile: blocks that fall outside of the output
#    would either access memory out of bounds, or (for layouts that wrap coordinates
#    around, such as the GETT layouts) silently recompute and re-store another tile.
#  - `cta_swizzle(S, blockIdx, block_shape, matmul_shape)`: the (M, N) coordinate of the
#    first element of the output tile that the thread block with index `blockIdx` computes.
#    This must be a bijection between the blocks in the grid and the output tiles.
#  - optionally, `is_supported(S, block_shape, matmul_shape)`, if the swizzling function
#    only works for some problem shapes.
#
# Note that CUDA dispatches thread blocks in order of increasing linear block index,
# with blockIdx().x varying fastest. The tiled swizzles below hence use a 1D grid, so
# that the traversal order is fully determined by `cta_swizzle`.

@inline num_tiles(block_shape, matmul_shape) =
    (cld(matmul_shape.M, block_shape.M), cld(matmul_shape.N, block_shape.N))

is_supported(swizzle_type, block_shape, matmul_shape) = true

# Debugging
function visualise_cta_swizzle(swizzle_type, block_shape = (M = 128, N = 128), matmul_shape = (M = 1024, N = 1024))
    res = fill("", num_tiles(block_shape, matmul_shape))

    BX, BY = number_of_blocks(swizzle_type, block_shape, matmul_shape)

    for by = 1:BY, bx = 1:BX
        block_i, block_j = cta_swizzle(swizzle_type, (x = bx, y = by), block_shape, matmul_shape)
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
    num_tiles(block_shape, matmul_shape)
end

@inline function cta_swizzle(::Type{Identity}, blockIdx, block_shape, matmul_shape)
    bx = blockIdx.x - 1
    by = blockIdx.y - 1

    block_i = bx
    block_j = by

    block_i * block_shape.M, block_j * block_shape.N
end

# ----------------------------
# Horizontally tiled swizzling
# ----------------------------

# The output tiles are divided into vertical strips that are TileSize tiles wide. The
# strips are processed from left to right, and every strip is traversed row by row.
# If the number of tile columns is not a multiple of TileSize, the last strip is narrower.
#
# Example: TileSize = 4, 6 x 6 tiles (numbers indicate the 1D block index):
#
#   1   2   3   4 || 25  26
#   5   6   7   8 || 27  28
#   9  10  11  12 || 29  30
#  13  14  15  16 || 31  32
#  17  18  19  20 || 33  34
#  21  22  23  24 || 35  36

export HorizontallyTiled

struct HorizontallyTiled{TileSize} end

@inline function number_of_blocks(::Type{HorizontallyTiled{TileSize}}, block_shape, matmul_shape) where {TileSize}
    rows, cols = num_tiles(block_shape, matmul_shape)
    (rows * cols, 1)
end

@inline function cta_swizzle(::Type{HorizontallyTiled{TileSize}}, blockIdx, block_shape, matmul_shape) where {TileSize}
    rows, cols = num_tiles(block_shape, matmul_shape)
    idx = blockIdx.x - 1

    # Divide only by compile-time constants (rows and cols are part of the Config): this
    # avoids run-time divisions and the divide-by-zero checks that come with them.
    full_strips = cols ÷ TileSize
    last_width = max(1, cols - full_strips * TileSize)

    strip_idx = idx ÷ (rows * TileSize)
    idx_in_strip = idx - strip_idx * rows * TileSize

    if strip_idx < full_strips
        block_i = idx_in_strip ÷ TileSize
        block_j = strip_idx * TileSize + idx_in_strip % TileSize
    else
        block_i = idx_in_strip ÷ last_width
        block_j = strip_idx * TileSize + idx_in_strip % last_width
    end

    block_i * block_shape.M, block_j * block_shape.N
end

# --------------------------
# Vertically tiled swizzling
# --------------------------

# The transpose of HorizontallyTiled: the output tiles are divided into horizontal strips
# that are TileSize tiles high. The strips are processed from top to bottom, and every
# strip is traversed column by column. If the number of tile rows is not a multiple of
# TileSize, the last strip is lower.
#
# Example: TileSize = 4, 6 x 6 tiles (numbers indicate the 1D block index):
#
#   1   5   9  13  17  21
#   2   6  10  14  18  22
#   3   7  11  15  19  23
#   4   8  12  16  20  24
#  =====================
#  25  27  29  31  33  35
#  26  28  30  32  34  36

export VerticallyTiled

struct VerticallyTiled{TileSize} end

@inline function number_of_blocks(::Type{VerticallyTiled{TileSize}}, block_shape, matmul_shape) where {TileSize}
    rows, cols = num_tiles(block_shape, matmul_shape)
    (rows * cols, 1)
end

@inline function cta_swizzle(::Type{VerticallyTiled{TileSize}}, blockIdx, block_shape, matmul_shape) where {TileSize}
    rows, cols = num_tiles(block_shape, matmul_shape)
    idx = blockIdx.x - 1

    # See HorizontallyTiled: only divide by compile-time constants.
    full_strips = rows ÷ TileSize
    last_height = max(1, rows - full_strips * TileSize)

    strip_idx = idx ÷ (cols * TileSize)
    idx_in_strip = idx - strip_idx * cols * TileSize

    if strip_idx < full_strips
        block_i = strip_idx * TileSize + idx_in_strip % TileSize
        block_j = idx_in_strip ÷ TileSize
    else
        block_i = strip_idx * TileSize + idx_in_strip % last_height
        block_j = idx_in_strip ÷ last_height
    end

    block_i * block_shape.M, block_j * block_shape.N
end

# ----------------------------
# Lebesgue space-filling curve
# ----------------------------

# Only a bijection if the output consists of a square, power-of-two number of tiles.
#
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

function is_supported(::Type{LebesgueCurve}, block_shape, matmul_shape)
    rows, cols = num_tiles(block_shape, matmul_shape)
    rows == cols && ispow2(rows)
end

@inline function number_of_blocks(::Type{LebesgueCurve}, block_shape, matmul_shape)
    rows, cols = num_tiles(block_shape, matmul_shape)
    (rows * cols, 1)
end

@inline function cta_swizzle(::Type{LebesgueCurve}, blockIdx, block_shape, matmul_shape)
    bx = blockIdx.x - 1

    block_i = extract_even_bits(bx)
    block_j = extract_odd_bits(bx)

    block_i * block_shape.M, block_j * block_shape.N
end

end
