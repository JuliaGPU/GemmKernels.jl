using GemmKernels

@testset "BitArrayIndex" begin
    @testset "Conversions to/from integer" begin
        for n = 0:15
            @test convert(UInt, constant(n)) == n
            @test convert(UInt, variadic(n)) == n
        end
    end

    @testset "Addition" begin
        for m = 0:15, n = 0:15
            @test_throws "Can only add BitArrayIndices whose bits do not overlap!" variadic(m) + variadic(n)
        end

        for m = 1:15
            @test_throws "Can only add BitArrayIndices whose bits do not overlap!" constant(m) + constant(m)
        end
    end

    @testset "Shifts" begin
        # Test some combinations of constant and variadic parts.
        INPUTS = vcat(
            # all constants
            [(i, constant(i)) for i in 0:15],

            # all variadic
            [(i, variadic(i)) for i in 0:15],

            # variadic|constant
            [(i, BitArrayIndex((~i) & 0b0011, i & 0b0011, i & 0b1100)) for i = 0:15],

            # constant|variadic
            [(i, BitArrayIndex((~i) & 0b1100, i & 0b1100, i & 0b0011)) for i = 0:15]
        )

        @testset "Left shift" begin
            for (i, I) = INPUTS, N = 0:4
                @test (i << N) == convert(UInt, I << N)
            end
        end

        @testset "Right shift" begin
            for (i, I) = INPUTS, N = 0:4
                @test (i >> N) == convert(UInt, I >> N)
            end
        end
    end

    @testset "Elementwise Bit Operations" begin
        # We should only need to test 1 bit, as everything is broadcast over
        # all bits anyway.
        ALL_POSSIBILITIES = [
            (0, constant(0)),
            (1, constant(1)),
            (0, variadic(0)),
            (1, variadic(1))
        ]
        @testset "OR" begin
            for (i, I) in ALL_POSSIBILITIES
                for (j, J) in ALL_POSSIBILITIES
                    @test (i | j) == convert(UInt, I | J)
                end
            end
        end

        @testset "AND" begin
            for (i, I) in ALL_POSSIBILITIES
                for (j, J) in ALL_POSSIBILITIES
                    @test (i & j) == convert(UInt, I & J)
                end
            end
        end

        @testset "XOR" begin
            for (i, I) in ALL_POSSIBILITIES
                for (j, J) in ALL_POSSIBILITIES
                    @test (i ⊻ j) == convert(UInt, I ⊻ J)
                end
            end
        end
    end
end
