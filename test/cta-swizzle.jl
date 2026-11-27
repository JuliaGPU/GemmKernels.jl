using GemmKernels
using GemmKernels: CTASwizzle

################################################################################

# Simulate the launch of a grid on the host, and check that every output tile is
# computed by exactly one thread block, and that no block falls outside of the output.
function check_cta_swizzle(swizzle, block_shape, matmul_shape)
    rows = cld(matmul_shape.M, block_shape.M)
    cols = cld(matmul_shape.N, block_shape.N)
    count = zeros(Int, rows, cols)

    BX, BY = CTASwizzle.number_of_blocks(swizzle, block_shape, matmul_shape)
    for by in 1:BY, bx in 1:BX
        block_i, block_j = CTASwizzle.cta_swizzle(swizzle, (x = bx, y = by), block_shape, matmul_shape)
        i = block_i ÷ block_shape.M + 1
        j = block_j ÷ block_shape.N + 1
        (1 ≤ i ≤ rows && 1 ≤ j ≤ cols) || return false
        count[i, j] += 1
    end

    return all(==(1), count)
end

@testset "CTA swizzling" begin
    block_shape = (M = 16, N = 32, K = 8)

    @testcase "$(swizzle)" for swizzle in [CTASwizzle.Identity,
                                           CTASwizzle.HorizontallyTiled{1}, CTASwizzle.HorizontallyTiled{2},
                                           CTASwizzle.HorizontallyTiled{4}, CTASwizzle.HorizontallyTiled{16},
                                           CTASwizzle.VerticallyTiled{1}, CTASwizzle.VerticallyTiled{2},
                                           CTASwizzle.VerticallyTiled{4}, CTASwizzle.VerticallyTiled{16}]
        for rows in 1:37, cols in 1:37
            matmul_shape = (M = rows * block_shape.M, N = cols * block_shape.N, K = 64)
            @test CTASwizzle.is_supported(swizzle, block_shape, matmul_shape)
            @test check_cta_swizzle(swizzle, block_shape, matmul_shape)
        end
    end

    @testcase "LebesgueCurve" begin
        for rows in 1:33, cols in 1:33
            matmul_shape = (M = rows * block_shape.M, N = cols * block_shape.N, K = 64)
            if CTASwizzle.is_supported(CTASwizzle.LebesgueCurve, block_shape, matmul_shape)
                @test rows == cols && ispow2(rows)
                @test check_cta_swizzle(CTASwizzle.LebesgueCurve, block_shape, matmul_shape)
            end
        end
    end

    @testcase "Traversal order" begin
        # HorizontallyTiled{4} on 6 x 6 tiles: strips of 4 columns, traversed row by row.
        block_shape = (M = 1, N = 1, K = 1)
        matmul_shape = (M = 6, N = 6, K = 1)
        tile(swizzle, idx) = CTASwizzle.cta_swizzle(swizzle, (x = idx, y = 1), block_shape, matmul_shape)
        @test [tile(CTASwizzle.HorizontallyTiled{4}, idx) for idx in 1:6] ==
              [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
        @test [tile(CTASwizzle.HorizontallyTiled{4}, idx) for idx in 25:28] ==
              [(0, 4), (0, 5), (1, 4), (1, 5)]

        # VerticallyTiled{4} is its transpose: strips of 4 rows, traversed column by column.
        @test [tile(CTASwizzle.VerticallyTiled{4}, idx) for idx in 1:6] ==
              [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1)]
        @test [tile(CTASwizzle.VerticallyTiled{4}, idx) for idx in 25:28] ==
              [(4, 0), (5, 0), (4, 1), (5, 1)]
    end
end
