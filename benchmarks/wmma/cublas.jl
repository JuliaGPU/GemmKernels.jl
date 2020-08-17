using CUDA

M = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
K = parse(Int, ARGS[3])

transpose_a = (ARGS[4] == "n" ? false : ARGS[4] == "t" ? true : error("Invalid memory layout for A: $(ARGS[4])"))
transpose_b = (ARGS[5] == "n" ? false : ARGS[5] == "t" ? true : error("Invalid memory layout for B: $(ARGS[5])"))

function benchmark_matmul(a, b, c)
    alpha = rand(Float32)
    beta = rand(Float32)

    CUDA.@sync begin
        CUDA.CUBLAS.gemmEx!(!transpose_a ? 'N' : 'T', !transpose_b ? 'N' : 'T', alpha, a, b, beta, c)
    end
end

a_h = rand(Float16, (M, K)) / sqrt(Float16(K))
b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
c_h = rand(Float32, (M, N))

# Transpose input if necessary
a_h = transpose_a ? transpose(a_h) : a_h
b_h = transpose_b ? transpose(b_h) : b_h

a   = CuArray(a_h)
b   = CuArray(b_h)
c   = CuArray(c_h)

# warmup
benchmark_matmul(a, b, c)

# profile
for i = 1 : 10
    CUDA.@profile benchmark_matmul(a, b, c)
end
