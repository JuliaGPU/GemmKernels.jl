using GemmKernels.Tiling

################################################################################

@testset "Tiling API" begin
    @testset "Tiles" begin
        @testcase "Index" begin
            @test Tile(M = 4, N = 4, K = 4).index == (M = 0, N = 0, K = 0)
        end

        @testcase "Projection" begin
            @test Tile(M = 1, N = 2, K = 3).MN  == Tile(M = 1, N = 2)
            @test Tile(M = 1, N = 2, K = 3).NM  == Tile(N = 2, M = 1)
            @test Tile(M = 1, N = 2, K = 3).M   == Tile(M = 1)
            @test Tile(M = 1, N = 2, K = 3).KMN == Tile(K = 3, M = 1, N = 2)
        end

        @testcase "Transposition" begin
            @test transpose(Tile(M = 1, N = 2)) == Tile(N = 2, M = 1)
            @test transpose(Tile(M = 1, N = 2, K = 3)) == Tile(K = 3, N = 2, M = 1)
        end

        @testcase "Translate base" begin
            tile = translate_base(Tile(M = 10, N = 20), (M = 1, N = 2))
            @test tile.size == (M = 10, N = 20)
            @test tile.base == (M = 1, N = 2)
            @test tile.offset == (M = 0, N = 0)
        end

        @testcase "Translate offset" begin
            tile = translate_offset(Tile(M = 10, N = 20), (M = 1, N = 2))
            @test tile.size == (M = 10, N = 20)
            @test tile.base == (M = 0, N = 0)
            @test tile.offset == (M = 1, N = 2)
        end

        @testcase "Linearise" begin
            tile = Tile(M = 3, N = 5)
            for i = 0 : 2, j = 0 : 4
                tile_t = translate_offset(tile, (M = i, N = j))
                @test linearise(tile_t.index, (M = 100, N = 200)) == j * 100 + i + 1
                @test linearise(tile_t.NM.index, (N = 200, M = 100)) == i * 200 + j + 1
            end
        end
    end

    @testset "Tile iteration" begin
        @testcase "Subdivide" begin
            tile_size = (M = 8, N = 4)
            num_tiles = (M = 2, N = 4)
            tile = Tile(M = num_tiles.M * tile_size.M, N = num_tiles.N * tile_size.N)

            for i = 1 : num_tiles.M * num_tiles.N
                t = subdivide(tile, Tile(tile_size), i, num_tiles.M * num_tiles.N)

                @test t.offset == (M = 0, N = 0)
                @test t.base   == (M = tile_size.M * mod(i - 1, num_tiles.M), N = tile_size.N * fld(i - 1, num_tiles.M))
                @test t.size   == tile_size
            end
        end

        @testcase "Parallelise" begin
            tile_size = (M = 8, N = 4)
            num_tiles = (M = 2, N = 8)
            tile = Tile(M = num_tiles.M * tile_size.M, N = num_tiles.N * tile_size.N)

            for i = 1 : (num_tiles.M * num_tiles.N) ÷ 2
                t1, t2 = parallelise(tile, Tile(tile_size), i, (num_tiles.M * num_tiles.N) ÷ 2)

                @test t1.offset == (M = 0, N = 0)
                @test t2.offset == (M = 0, N = 4 * tile_size.N)

                @test t1.base   == (M = tile_size.M * mod(i - 1, num_tiles.M), N = tile_size.N * fld(i - 1, num_tiles.M))
                @test t2.base   == (M = tile_size.M * mod(i - 1, num_tiles.M), N = tile_size.N * fld(i - 1, num_tiles.M))

                @test t1.size == tile_size
                @test t2.size == tile_size
            end
        end
    end
end

################################################################################
