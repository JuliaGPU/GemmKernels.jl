using GemmKernels: constant, variadic
using GemmKernels.Tiling

################################################################################

@testset "Tiling API" begin
    @testset "Tiles" begin
        @testcase "Construction" begin
            t1 = Tile(M = 4, N = 8)
            t2 = Tile((M = 4, N = 8))

            @test t1 == t2
        end

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

        @testcase "Translation" begin
            @testcase "Translate base (variadic delta)" begin
                tile = translate(Tile(M = 10, N = 20), (M = variadic(1), N = variadic(2)))
                @test tile.size == (M = 10, N = 20)
                @test tile.index == (M = variadic(1), N = variadic(2))
            end

            @testcase "Translate offset (constant delta)" begin
                tile = translate(Tile(M = 10, N = 20), (M = constant(1), N = constant(2)))
                @test tile.size == (M = 10, N = 20)
                @test tile.index == (M = constant(1), N = constant(2))
            end

            @testcase "Translate base + offset (variadic + constant delta)" begin
                tile = Tile(M = 10, N = 20)

                tile = translate(tile, (M = variadic(1), N = variadic(2)))
                tile = translate(tile, (M = constant(4), N = constant(8)))

                tile = translate(tile, (M = variadic(16), N = variadic(32)))
                tile = translate(tile, (M = constant(64), N = constant(128)))

                @test tile.size == (M = 10, N = 20)
                @test tile.index == (M = variadic(1+16) + constant(4+64), N = variadic(2+32) + constant(8+128))
            end
        end

        @testcase "Linearise" begin
            tile = Tile(M = 3, N = 5)

            for i = 0 : 2, j = 0 : 4
                tile_t = translate(tile, (M = constant(i), N = constant(j)))

                @test linearise(tile_t, (M = 100, N = 200)) == j * 100 + i + 1
                @test linearise(tile_t.NM, (N = 200, M = 100)) == i * 200 + j + 1
            end
        end
    end

    @testset "Tile iteration" begin
        function test_parallelisation(tile_size, num_tiles, num_entities, test_subdivision=false)
            tile_size = (M = tile_size[1], N = tile_size[2])
            num_tiles = (M = num_tiles[1], N = num_tiles[2])
            parent_tile = Tile(M = num_tiles.M * tile_size.M, N = num_tiles.N * tile_size.N)
            num_iterations = prod(parent_tile.size) ÷ prod(tile_size) ÷ num_entities

            if test_subdivision
                @test num_iterations == 1
            end

            results = Dict() # key: (entity, iteration)

            for entity = 1 : num_entities
                # XXX: Set size of iterator so we can use collect()
                rv = []

                if !test_subdivision
                    for tile in parallelise(parent_tile, Tile(tile_size), entity, num_entities)
                        push!(rv, tile)
                    end
                else
                    tile = subdivide(parent_tile, Tile(tile_size), entity, num_entities)
                    push!(rv, tile)
                end

                # (1) Test number of results.
                @test length(rv) == num_iterations

                for (i, tile) in enumerate(rv)
                    results[(entity, i)] = tile
                end
            end

            # (2) Test that all parts of the parent tile are covered exactly once.
            arr = zeros(Int, (parent_tile.size.M, parent_tile.size.N))

            for tile in values(results)
                ind_m = convert(Int, tile.index.M)
                ind_n = convert(Int, tile.index.N)

                for m = ind_m : (ind_m + tile.size.M - 1),
                    n = ind_n : (ind_n + tile.size.N - 1)

                    arr[1 + m, 1 + n] += 1
                end
            end

            @test all(arr .== 1)

            allsame(A) = !isempty(A) && all(x -> x == A[1], A)

            # (3) Test that the variadic part only depends on the entity.
            for entity = 1 : num_entities
                M_variadic_parts_for_entity = [results[entity, i].index.M.variadic_part for i in 1:num_iterations]
                N_variadic_parts_for_entity = [results[entity, i].index.N.variadic_part for i in 1:num_iterations]

                @test allsame(M_variadic_parts_for_entity)
                @test allsame(N_variadic_parts_for_entity)
            end

            # (4) Test that the constant part only depends on the iteration.
            for i = 1 : num_iterations
                M_constant_parts_for_iteration = [results[entity, i].index.M.known_one for entity in 1:num_entities]
                N_constant_parts_for_iteration = [results[entity, i].index.N.known_one for entity in 1:num_entities]

                @test allsame(M_constant_parts_for_iteration)
                @test allsame(N_constant_parts_for_iteration)
            end
        end

        test_subdivision(tile_size, num_tiles) = test_parallelisation(tile_size, num_tiles, prod(num_tiles), true)

        @testcase "Parallelise" begin
            for tile_size in [(8, 4), (2, 4)],
                num_tiles in [(1, 8), (2, 8), (4, 4), (8, 2), (8, 1)],
                num_entities in [1, 2, 4, 8]

                @testcase "tile_size = $tile_size, num_tiles = $num_tiles, num_entities = $num_entities" begin
                    test_parallelisation(tile_size, num_tiles, num_entities)
                end
            end
        end

        @testcase "Subdivide" begin
            for tile_size in [(8, 4), (2, 4)],
                num_tiles in [(1, 8), (2, 8), (4, 4), (8, 2), (8, 1)]

                @testcase "tile_size = $tile_size, num_tiles = $num_tiles" begin
                    test_subdivision(tile_size, num_tiles)
                end
            end
        end
    end
end

################################################################################
