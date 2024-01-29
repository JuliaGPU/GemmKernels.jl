# vim: fdm=marker

using Test
using CUDA
using Base: llvmcall
using LLVMLoopInfo: @loopinfo
using Base.Cartesian: @ntuple
import Base.+, Base.|, Base.*, Base.÷, Base.%, Base.&, Base.⊻, Base.<<, Base.>>, Base.==, Base.<
import Base.getindex, Base.setindex, Base.setindex!
using LinearAlgebra

# indexing utils {{{

# Struct that represent an index into an array where each bit can be known at
# compile-time to be 0 or 1, or where the bit is variadic (i.e. not known at
# compile time).
# The invariants maintained by instances of this structure are:
# 1) A bit can either be known zero, known one, or variadic, i.e. there is
#    no overlap between known_zero, known_one, and variadic.
# 2) The underlying integer represented by this index is variadic_part + known_one,
#    or, because of the no-bit-overlap assumption, variadic_part | known_one.
# 3) The known_zero and known_one fields are always compile-time constants.
#    (Potentially after full loop unrolling by the compiler.)
#
# For any operation supported by this structure, the known_zero and known_one
# of the output are also a pure function of the known_zero and known_one of the
# input(s), and are hence also constant (Potentially after constant folding by
# the compiler.)
#
# This allows us to use register+constant addressing mode even in the case of
# complex swizzling operations, which SeparateConstOffsetFromGEP does not
# handle so well.
#
# NOTE: This index is zero-based when converted to an integer!
struct BitArrayIndex
    known_zero::UInt    # bitmask representing the bits known to be 0
    known_one::UInt     # bitmask representing the bits known to be 1
    variadic_part::UInt # the variadic part of the index
end

constant(i::Int) = BitArrayIndex(~reinterpret(UInt, i), reinterpret(UInt, i), reinterpret(UInt, 0))
variadic(i::Int) = BitArrayIndex(reinterpret(UInt, 0), reinterpret(UInt, 0), reinterpret(UInt, i))
Base.convert(UInt, I::BitArrayIndex) = I.variadic_part + I.known_one
Base.convert(Int, I::BitArrayIndex) = reinterpret(Int, I.variadic_part + I.known_one)

==(I::BitArrayIndex, j::Integer) = convert(Int, I) == j
<(I::BitArrayIndex, j::Integer) = convert(Int, I) < j

function +(I::BitArrayIndex, J::BitArrayIndex)
    # We only allow additions where no bits overlap, i.e. where I + J == I | J.
    I_variadic = ~(I.known_zero | I.known_one)
    J_variadic = ~(J.known_zero | J.known_one)

    # Could the addition of the indices result in a carry?
    could_carry = ((I_variadic | I.known_one) & (J_variadic | J.known_one)) != 0
    # could_carry && (@cuprintln("Can only add BitArrayIndices whose bits do not overlap!"); error())

    # Now that we have verified no bits overlap, we can dispatch to |.
    I | J
end

function |(I::BitArrayIndex, J::BitArrayIndex)
    known_zero = I.known_zero & J.known_zero
    known_one = I.known_one | J.known_one
    variadic_part = (I.variadic_part + J.variadic_part) & ~known_zero & ~known_one
    BitArrayIndex(known_zero, known_one, variadic_part)
end

function (&)(I::BitArrayIndex, J::BitArrayIndex)
    known_zero = I.known_zero | J.known_zero
    known_one = I.known_one & J.known_one

    # NOTE: We can't just do variadic = (I.variadic & J.variadic) & ~known_zero & ~known_one,
    # since e.g. in the case where I is all variadic, and J is all known-ones (hence J.variadic is 0),
    # the output would be 0, rather than I.variadic.
    # Instead, we set a bit b to 1 in the variadic part iff:
    # 1) There is a known 1-bit in that input, and
    # 2) This bit is variadic in the output, which only happens if that bit is
    #    variadic in the other input (i.e. not known 0 AND not known 1).
    new_I_var = I.variadic_part | (I.known_one & ~J.known_zero & ~J.known_one)
    new_J_var = J.variadic_part | (J.known_one & ~I.known_zero & ~I.known_one)
    variadic_part = (new_I_var & new_J_var) & ~known_zero & ~known_one
    BitArrayIndex(known_zero, known_one, variadic_part)
end

function ⊻(I::BitArrayIndex, J::BitArrayIndex)
    known_zero = (I.known_zero & J.known_zero) | (I.known_one & J.known_one)
    known_one = (I.known_zero & J.known_one) | (I.known_one & J.known_zero)

    # NOTE: We can't just do variadic = (I.variadic ⊻ J.variadic) & ~known_zero & ~known_one,
    # since e.g. in the case where I is all variadic, and J is all known-ones (hence J.variadic is 0),
    # the output would be I.variadic, rather than ~I.variadic.
    # Instead, we set a bit b to 1 in the variadic part iff:
    # 1) There is a known 1-bit in that input, and
    # 2) This bit is variadic in the output, which only happens if that bit is
    #    variadic in the other input (i.e. not known 0 AND not known 1).
    new_I_var = I.variadic_part | (I.known_one & ~J.known_zero & ~J.known_one)
    new_J_var = J.variadic_part | (J.known_one & ~I.known_zero & ~I.known_one)
    variadic_part = (new_I_var ⊻ new_J_var) & ~known_zero & ~known_one
    BitArrayIndex(known_zero, known_one, variadic_part)
end

function <<(I::BitArrayIndex, c::Integer)
    # When shifting left by N bits, we know that the last N bits are zero.
    mask = (1 << c) - 1
    known_zero = (I.known_zero << c) | UInt(mask)
    known_one = I.known_one << c
    variadic_part = I.variadic_part << c
    BitArrayIndex(known_zero, known_one, variadic_part)
end

function >>(I::BitArrayIndex, c::Integer)
    # When shifting right by N bits, we know that the first N bits are zero.
    mask = ((1 << c) - 1) << (8 * sizeof(I.known_zero) - c - 1)
    known_zero = (I.known_zero >> c) | UInt(mask)
    known_one = I.known_one >> c
    variadic_part = I.variadic_part >> c
    BitArrayIndex(known_zero, known_one, variadic_part)
end

@inline _IsPowerOfTwo(c) = (c != 0) && ((c & (c - 1)) == 0)

@inline *(I::BitArrayIndex, c::Integer) = I * Val(c)
@inline *(c::Integer, I::BitArrayIndex) = I * Val(c)

@generated function *(I::BitArrayIndex, ::Val{c}) where {c}
    # We only allow multiplications with powers of two.
    # _IsPowerOfTwo(c) || (@cuprintln("Can only multiply by powers of two"); error())
    N = convert(Int, log2(c))

    # Dispatch to <<
    quote
        I << $N
    end
end

@inline ÷(I::BitArrayIndex, c::Integer) = I ÷ Val(c)

@generated function ÷(I::BitArrayIndex, ::Val{c}) where {c}
    # We only allow division by powers of two.
    # _IsPowerOfTwo(c) || (@cuprintln("Can only divide by powers of two"); error())
    N = convert(Int, log2(c))

    # Dispatch to >>
    quote
        I >> $N
    end
end

@inline %(I::BitArrayIndex, c::Integer) = I % Val(c)

@generated function %(I::BitArrayIndex, ::Val{c}) where {c}
    # We only allow modulo by powers of two.
    # _IsPowerOfTwo(c) || (@cuprintln("Can only modulo with powers of two"); error())
    N = c-1

    # Dispatch to &
    quote
        I & constant($N)
    end
end

# Convenience methods to automatically convert to 1-based index.
Base.@propagate_inbounds Base.getindex(A::AbstractArray, I::BitArrayIndex) = Base.getindex(A, 1 + convert(Int, I))
Base.@propagate_inbounds Base.getindex(T::Tuple, I::BitArrayIndex) = Base.getindex(T, 1 + convert(Int, I))
Base.@propagate_inbounds Base.setindex(A::AbstractArray, v, I::BitArrayIndex) = Base.setindex(A, v, 1 + convert(Int, I))
Base.@propagate_inbounds Base.setindex!(A::AbstractArray, v, I::BitArrayIndex) = Base.setindex!(A, v, 1 + convert(Int, I))

Base.@propagate_inbounds vloada(v, A::AbstractArray, I::BitArrayIndex) = vloada(v, pointer(A, 1 + convert(Int, I)))
Base.@propagate_inbounds vstorea!(v, A::AbstractArray, I::BitArrayIndex, x) = vstorea!(v, pointer(A, 1 + convert(Int, I)), x)

# CUDA-specific helpers.
@inline tid() = variadic(threadIdx().x - 1)
@inline bid_x() = variadic(blockIdx().x - 1)
@inline bid_y() = variadic(blockIdx().y - 1)
@inline warpid() = tid() ÷ 32
@inline laneid() = tid() % 32

# Loop-specific helpers.
macro unrolled(ex)
    Meta.isexpr(ex, :for) || error("@unrolled: expected for loop")

    for_init = ex.args[1]
    for_body = ex.args[2]

    (for_init.head == :(=)) || error("@unrolled: expected assignment in for loop init")

    # add '_orig' suffix to iteration variable.
    unchanged_var_name = for_init.args[1]
    changed_var_name = Symbol(string(for_init.args[1]) * "_orig")

    for_init.args[1] = changed_var_name

    # change the body to set the iteration variable to const
    ex.args[2] = quote
        $unchanged_var_name = constant($changed_var_name)
        $for_body
        $(Expr(:loopinfo, (Symbol("llvm.loop.unroll.full"), 1)))
    end

    quote
        $(esc(ex))
    end
end

macro not_unrolled(ex)
    Meta.isexpr(ex, :for) || error("@not_unrolled: expected for loop")

    for_init = ex.args[1]
    for_body = ex.args[2]

    (for_init.head == :(=)) || error("@not_unrolled: expected assignment in for loop init")

    # add '_orig' suffix to iteration variable.
    unchanged_var_name = for_init.args[1]
    changed_var_name = Symbol(string(for_init.args[1]) * "_orig")

    # add '_next' suffix to variable containing next iteration.
    next_var_name = Symbol(string(for_init.args[1]) * "_next")

    for_init.args[1] = changed_var_name

    # change the body to set the iteration variable to variadic
    ex.args[2] = quote
        $unchanged_var_name = variadic($changed_var_name)
        $for_body
        $(Expr(:loopinfo, (Symbol("llvm.loop.unroll.disable"), 1)))
    end

    quote
        $(esc(ex))
    end
end

function test_bitarray()
    @testset "BitArrayIndex" begin
        @testset "Conversions to/from integer" begin
            for n = 0:15
                @test convert(UInt, constant(n)) == n
                @test convert(UInt, variadic(n)) == n
            end
        end

        # @testset "Addition" begin
        #     for m = 0:15, n = 0:15
        #         @test_throws "Can only add BitArrayIndices whose bits do not overlap!" variadic(m) + variadic(n)
        #     end

        #     for m = 1:15
        #         @test_throws "Can only add BitArrayIndices whose bits do not overlap!" constant(m) + constant(m)
        #     end
        # end

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
end
# }}}

# debugging {{{
# mark regions in source code, e.g. debug_mark(Val(0))
@generated function debug_mark(::Val{number}) where {number}
    ir = """
    call void asm sideeffect "pmevent $number;", "~{memory}"()
    ret void
    """

    quote
        llvmcall($ir, Nothing, Tuple{})
    end
end
# }}}

# mma.sync implementation {{{
mma884_row_row(a, b, c) = llvmcall("""
                 %ah0i = insertelement <2 x half> undef, half %0, i32 0
                 %ah0f = insertelement <2 x half> %ah0i, half %1, i32 1
                 %ah1i = insertelement <2 x half> undef, half %2, i32 0
                 %ah1f = insertelement <2 x half> %ah1i, half %3, i32 1

                 %bh0i = insertelement <2 x half> undef, half %4, i32 0
                 %bh0f = insertelement <2 x half> %bh0i, half %5, i32 1
                 %bh1i = insertelement <2 x half> undef, half %6, i32 0
                 %bh1f = insertelement <2 x half> %bh1i, half %7, i32 1

                 %a0 = bitcast <2 x half> %ah0f to i32
                 %a1 = bitcast <2 x half> %ah1f to i32

                 %b0 = bitcast <2 x half> %bh0f to i32
                 %b1 = bitcast <2 x half> %bh1f to i32

                 %mma = call { float, float, float, float, float, float, float, float } asm sideeffect "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {\$0,\$1,\$2,\$3,\$4,\$5,\$6,\$7}, {\$8, \$9}, {\$10,\$11}, {\$12,\$13,\$14,\$15,\$16,\$17,\$18,\$19};", "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,f,f,f,f,f,f,f,f"(i32 %a0, i32 %a1, i32 %b0, i32 %b1, float %8, float %9, float %10, float %11, float %12, float %13, float %14, float %15)

                 %mma0 = extractvalue { float, float, float, float, float, float, float, float } %mma, 0
                 %mma1 = extractvalue { float, float, float, float, float, float, float, float } %mma, 1
                 %mma2 = extractvalue { float, float, float, float, float, float, float, float } %mma, 2
                 %mma3 = extractvalue { float, float, float, float, float, float, float, float } %mma, 3
                 %mma4 = extractvalue { float, float, float, float, float, float, float, float } %mma, 4
                 %mma5 = extractvalue { float, float, float, float, float, float, float, float } %mma, 5
                 %mma6 = extractvalue { float, float, float, float, float, float, float, float } %mma, 6
                 %mma7 = extractvalue { float, float, float, float, float, float, float, float } %mma, 7

                 %rv0 = insertvalue [8 x float] undef, float %mma0, 0
                 %rv1 = insertvalue [8 x float] %rv0, float %mma1, 1
                 %rv2 = insertvalue [8 x float] %rv1, float %mma2, 2
                 %rv3 = insertvalue [8 x float] %rv2, float %mma3, 3
                 %rv4 = insertvalue [8 x float] %rv3, float %mma4, 4
                 %rv5 = insertvalue [8 x float] %rv4, float %mma5, 5
                 %rv6 = insertvalue [8 x float] %rv5, float %mma6, 6
                 %rv7 = insertvalue [8 x float] %rv6, float %mma7, 7

                 ret [8 x float] %rv7
                 """,
                 NTuple{8, Float32},
                 Tuple{
                    Float16, Float16, Float16, Float16,
                    Float16, Float16, Float16, Float16,
                    Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32
                 },
                 Float16.(a)...,
                 Float16.(b)...,
                 Float32.(c)...)
# }}}

# utils {{{
"""
    @staticdef struct Struct
        foo
        bar::Typ
        ...
    end

Generate a 'static' version of a struct where each field is encoded as a type parameter.
This is useful when the field values should be specialized on, e.g., in the context of GPU
kernels.

The macro will generate a struct definition, including a constructor that takes each field
as an argument (in the same order, and using the same types as the original definition).
In addition, a `getproperty` accessor is defined such that the fields can be accessed
using convenient dot syntax (i.e., `obj.field`).
"""
macro staticdef(ex)
    # decode struct definition
    Meta.isexpr(ex, :struct) || error("@staticdef: expected struct definition")
    is_mutable, struct_name, struct_body = ex.args
    is_mutable && error("@staticdef: struct definition must be immutable")
    ## decode fields
    @assert Meta.isexpr(struct_body, :block)
    fields = struct_body.args
    filter!(field -> !isa(field, LineNumberNode), fields)
    field_names = Symbol[]
    field_types = Dict{Symbol, Any}()
    for field in fields
        if Meta.isexpr(field, :(::))
            name, typ = field.args

        else
            name = field
            typ = Any
        end
        push!(field_names, name)
        field_types[name] = typ
    end

    # generate new struct definition, forwarding args to typevars
    typevars = Symbol.(uppercase.(String.(field_names)))
    struct_ex = quote
        struct $(esc(struct_name)){$(typevars...)}
            function $(esc(struct_name))($(fields...))
                new{$(field_names...)}()
            end
        end
    end

    # generate a getproperty accessor
    getproperties_ex = quote end
    if !isempty(field_names)
      current = nothing
      for field_name in field_names
          typevar = Symbol(uppercase(String(field_name)))
          test = :(field === $(QuoteNode(field_name)))
          if current === nothing
              current = Expr(:if, test, typevar)
              getproperties_ex = current
          else
              new = Expr(:elseif, test, typevar)
              push!(current.args, new)
              current = new
          end
      end
      ## finally, call `getfield` to emit an error
      push!(current.args, :(getfield(conf, field)))
      getproperties_ex = quote
          function Base.getproperty(conf::$(esc(struct_name)){$(typevars...)},
                                    field::Symbol) where {$(typevars...)}
            $getproperties_ex
          end
      end
    end

    quote
        $struct_ex
        $getproperties_ex
    end
end
# }}}

# local array {{{
# a simple immutable array type backed by stack memory
#
# similar to StaticArrays, but immutable to prevent optimization bugs (JuliaLang/julia#41800)

struct LocalArray{S <: Tuple, T, N, L} <: AbstractArray{T,N}
    data::NTuple{L,T}

    LocalArray{S,T,N,L}(::UndefInitializer) where {S,T,N,L} = new{S,T,N,L}()
    LocalArray{S,T,N,L}(data::NTuple{L,T}) where {S,T,N,L} = new{S,T,N,L}(data)
end

@inline @generated function LocalArray{S,T}(args...) where {S,T}
    dims = (S.parameters...,)
    N = length(dims)
    L = prod(dims)
    @assert isbitstype(T)
    quote
        LocalArray{S, T, $N, $L}(args...)
    end
end

# array interface
Base.IndexStyle(::Type{<:LocalArray}) = IndexLinear()
Base.size(x::LocalArray{S}) where {S} = (S.parameters...,)

# indexing
Base.@propagate_inbounds function Base.getindex(v::LocalArray, i::Int)
    @boundscheck checkbounds(v,i)
    @inbounds v.data[i]
end
Base.@propagate_inbounds function Base.setindex(v::LocalArray{S,T,N,L}, val, i::Integer) where {S,T,N,L}
    @boundscheck checkbounds(v,i)
    new_data = Base.setindex(v.data, convert(T, val), i)
    LocalArray{S,T,N,L}(new_data)
end
## XXX: Base's setindex doesn't have a ND version
Base.@propagate_inbounds function Base.setindex(v::LocalArray{S,T,N,L}, val, is::Int...) where {S,T,N,L}
    @boundscheck checkbounds(v,is...)
    I = CartesianIndex(is...)
    i = LinearIndices(v)[I]
    new_data = Base.setindex(v.data, convert(T, val), i)
    LocalArray{S,T,N,L}(new_data)
end

# helper macro
"""
    @immutable local_array[...]
    @immutable local_array[...] = ...

Helper macro that rewrites array indexing operations on a `LocalArray` to call non-mutating
versions of the array indexing functions. Instead, the macro will generate an expression
that overwrites the array with the new value.
"""
macro immutable(ex)
    # is this an assignment?
    if Meta.isexpr(ex, :(=))
        ex, val = ex.args
    else
        val = nothing
    end
    Meta.isexpr(ex, :ref) || error("@immutable only works with indexing operations")

    # getindex does not need to be rewritten
    val === nothing && return ex

    # rewrite setindex to overwrite the array
    arr, idxs... = ex.args
    quote
        $(esc(arr)) = Base.setindex($(esc(arr)), $(esc(val)), $(map(esc, idxs)...))
    end
end
# }}}

# configuration {{{
@staticdef struct Config
    # Size of the matrices in global memory.
    GLOBAL_M
    GLOBAL_N
    GLOBAL_K

    # Tile size at the CTA level.
    CTA_M
    CTA_N
    CTA_K

    # Tile size at the warp level.
    WARP_M
    WARP_N
    WARP_K

    # Number of stages in the global -> shared copy pipeline.
    GLOBAL_TO_SHARED_STAGES

    # Number of stages in the shared -> register file copy pipeline.
    SHARED_TO_REGS_STAGES
end

NUM_WARPS_M(c::Config) = c.CTA_M ÷ c.WARP_M
NUM_WARPS_N(c::Config) = c.CTA_N ÷ c.WARP_N
NUM_THREADS(c::Config) = NUM_WARPS_M(c) * NUM_WARPS_N(c) * 32
NUM_BLOCKS_M(c::Config) = c.GLOBAL_M ÷ c.CTA_M
NUM_BLOCKS_N(c::Config) = c.GLOBAL_N ÷ c.CTA_N

# }}}

# globals {{{
conf = Config(
    2048, 2048, 2048, # GLOBAL_MNK
    128, 256, 32,     # CTA_MNK
    64, 64, 4,        # WARP_MNK
    2,                # GLOBAL_TO_SHARED_STAGES
    2,                # SHARED_TO_REGS_STAGES
)

# The kernel calculates A * B = D (in row-major), as this is CUTLASS's
# convention.
# To calculate A * B = D in col-major, just flip the A and B operands
# and transpose: A * B = D <=> B^T * A^T = D^T.
A = CUDA.rand(Float16, (conf.GLOBAL_N, conf.GLOBAL_K))
B = CUDA.rand(Float16, (conf.GLOBAL_K, conf.GLOBAL_M))
D = CUDA.zeros(Float32, (conf.GLOBAL_N, conf.GLOBAL_M))
# }}}

# vectorisation {{{

struct Vec{N, T} end

@noinline function throw_alignmenterror(ptr)
    @cuprintln "ERROR: AlignmentError: Pointer $ptr is not properly aligned for vectorized load/store"
    error()
end

@inline function checkalignment(ptr, alignment)
    checkalignment(Bool, ptr, alignment) || throw_alignmenterror(ptr)
    nothing
end

@inline function checkalignment(::Type{Bool}, ptr, alignment)
    Int(ptr) % alignment == 0
end

@inline @generated function vloada(::Type{Vec{N, T}}, ptr::Core.LLVMPtr{T, AS}) where {N, T, AS}
    alignment = sizeof(T) * N

    return quote
        vec_ptr = Base.bitcast(Core.LLVMPtr{NTuple{N, VecElement{T}}, AS}, ptr)
        @boundscheck checkalignment(vec_ptr, $alignment)
        return unsafe_load(vec_ptr, 1, Val($alignment))
    end
end

@inline @generated function vstorea!(::Type{Vec{N, T}}, ptr::Core.LLVMPtr{T, AS},
                                     x::NTuple{M,<:Any}) where {N, T, AS, M}
    alignment = sizeof(T) * N

    ex = quote end

    # we may be storing more values than we can using a single vectorized operation
    # (e.g., when types mismatch, storing 8 Float16s in a Float32 shared memory layout)
    for offset = 0:N:M-1
        append!(ex.args, (quote
            y = @ntuple $N j -> VecElement{T}(x[j+$offset].value)
            vec_ptr = Base.bitcast(Core.LLVMPtr{NTuple{N, VecElement{T}}, AS}, ptr)
            @boundscheck checkalignment(vec_ptr, $alignment)
            unsafe_store!(vec_ptr, y, $offset ÷ N + 1, Val($alignment))
        end).args)
    end

    return ex
end
# }}}

# bit manipulation utils {{{
# extract bit
@inline b(val, i) = b(val, i, 0)

# move bit from position i to j
@inline b(val::Integer, i, j) = ((val >> i) & 1) << j
@inline b(val::BitArrayIndex, i, j) = ((val >> i) & constant(1)) << j
# }}}

# warp mma {{{
@inline function warp_mma(a_frags, b_frags, acc_frags, warp_mma_k, conf::Config)
    # WARP_M x WARP_N x WARP_K = 64 x 64 x 4 GEMM per warp using 16 mma.syncs
    @unrolled for instruction = 0:15
        inner_row = b(instruction, 0)
        outer_row = b(instruction, 1)
        inner_col = b(instruction, 2)
        outer_col = b(instruction, 3)

        # 16 x 16 x 4 GEMM per warp using 1 mma.sync:
        # one mma.sync is an 8 x 8 x 4 GEMM per QP,
        # or a 16 x 16 x 4 GEMM per warp.

        # zig-zag sequence to increase reuse of the A operand
        # This doesn't seem to influence performance (at least not significantly),
        # but CUTLASS does this, so we might as well too.
        zz_inner_row = inner_row ⊻ (inner_col % 2)
        zz_outer_row = outer_row ⊻ (inner_col % 2)

        # a, b, and c fragments for this particular mma.sync.
        a_frag = LocalArray{Tuple{4}, Float16}(undef)
        b_frag = LocalArray{Tuple{4}, Float16}(undef)
        c_frag = LocalArray{Tuple{8}, Float32}(undef)

        # Get the A fragment for this mma.sync
        @unrolled for i = 0:3
            offset = b(i, 0, 0) +                                    # k0
                        b(i, 1, 1) +                                    # k1
                        (b(zz_inner_row, 0, 2) ⊻ b(warp_mma_k, 1, 2)) + # m2+k3
                        b(zz_outer_row, 0, 3)                           # m5
            @inbounds @immutable a_frag[i] = a_frags[offset]
        end

        # Get the B fragment for this mma.sync
        @unrolled for i = 0:3
            offset = b(i, 0, 0) +           # n0
                        b(i, 1, 1) +           # n1
                        b(inner_col, 0, 2) +   # n2
                        b(outer_col, 0, 3)     # n5

            @inbounds @immutable b_frag[i] = b_frags[offset]
        end

        # Get the C fragment for this mma.sync
        @unrolled for i = 0:7
            # index: (m5|m2|m1|n5|n4|n2|n0)
            offset = b(i, 0, 0) +            # n0
                        b(inner_col, 0, 1) +    # n2
                        b(i, 2, 2) +            # n4
                        b(outer_col, 0, 3) +    # n5
                        b(i, 1, 4) +            # m1
                        b(zz_inner_row, 0, 5) + # m2
                        b(zz_outer_row, 0, 6)   # m5
            @inbounds @immutable c_frag[i] = acc_frags[offset]
        end

        # offset in unit of 4x4x4 tiles
        inst_m = b(zz_inner_row, 0, 2) + b(tid(), 2, 3) + b(zz_outer_row, 0, 5)
        inst_n = b(inner_col, 0, 2) + b(tid(), 3, 3) + b(outer_col, 0, 5)
        inst_k = constant(0)

        d_frag = mma884_row_row(a_frag.data, b_frag.data, c_frag.data)

        # Store D fragment for this mma.sync
        @unrolled for i = 0:7
            # index: (m5|m2|m1|n5|n4|n2|n0)
            offset = b(i, 0, 0) +            # n0
                        b(inner_col, 0, 1) +    # n2
                        b(i, 2, 2) +            # n4
                        b(outer_col, 0, 3) +    # n5
                        b(i, 1, 4) +            # m1
                        b(zz_inner_row, 0, 5) + # m2
                        b(zz_outer_row, 0, 6)   # m5
            @inbounds @immutable acc_frags[offset] = d_frag[i]
        end
    end

    acc_frags
end
# }}}

# swizzling {{{
# swizzling function for the shared memory layout for A
@inline function swizzle_a(m, k, conf)
    # m : 7 bits
    # k : 5 bits
    offset = b(k, 0, 0) +
             b(k, 1, 1) +
             (b(k, 3, 2) ⊻ b(m, 2, 2)) +
             (b(k, 4, 3) ⊻ b(m, 3, 3) ⊻ b(m, 4, 3)) +
             b(m, 0, 4) +
             b(m, 1, 5) +
             b(m, 4, 6) +
             b(m, 5, 7) +
             b(m, 6, 8) +
             b(k, 2, 9) +
             b(k, 3, 10) +
             b(k, 4, 11) +
             b(k, 5, 12)

    return offset
end

# swizzling function for the shared memory layout for B
@inline function swizzle_b(k, n, conf)
    # k: 5 bits
    # n: 8 bits
    offset = b(n, 0, 0) +
             b(n, 1, 1) +
             b(n, 2, 2) +
             (b(n, 3, 3) ⊻ b(k, 0, 3)) +
             (b(n, 4, 4) ⊻ b(k, 1, 4)) +
             b(n, 5, 5) +
             b(n, 6, 6) +
             b(n, 7, 7) +
             b(n, 3, 8) +
             b(n, 4, 9) +
             b(k, 2, 10) +
             b(k, 3, 11) +
             b(k, 4, 12) +
             b(k, 5, 13)

    return offset
end
# }}}

# ld global {{{
# Load from global memory
@inline function ld_global(A, B, cta_m, cta_n, cta_k, conf)
    # Fragments for the data from the global loads (and hence shared stores).
    # index: (m3|k2|k1|k0)
    a_frag = LocalArray{Tuple{16}, Float16}(undef)

    # index: (n7|n6|n2|n1|n0)
    b_frag = LocalArray{Tuple{32}, Float16}(undef)

    # Load A from Global Memory.
    @unrolled for ins_1 = 0:1
        m = b(tid(), 2, 0) +
            b(tid(), 3, 1) +
            b(tid(), 4, 2) +
            b(ins_1, 0, 3) +
            b(tid(), 5, 4) +
            b(tid(), 6, 5) +
            b(tid(), 7, 6)

        k = b(tid(), 0, 3) +
            b(tid(), 1, 4)

        @inbounds val = vloada(Vec{8, Float16}, A, cta_k + k + conf.GLOBAL_K * (cta_m + m))

        @unrolled for offset = 0:7
            frag_offset = b(offset, 0, 0) +  # k0
                          b(offset, 1, 1) +  # k1
                          b(offset, 2, 2) +  # k2
                          b(ins_1, 0, 3)     # m3

            @inbounds @immutable a_frag[frag_offset] = val[offset].value
        end
    end

    # Load B from Global Memory.
    @unrolled for ins = 0:3
        n = b(tid(), 0, 3) +
            b(tid(), 1, 4) +
            b(tid(), 2, 5) +
            b(ins, 0, 6) +
            b(ins, 1, 7)

        k = b(tid(), 3, 0) +
            b(tid(), 4, 1) +
            b(tid(), 5, 2) +
            b(tid(), 6, 3) +
            b(tid(), 7, 4)

        @inbounds val = vloada(Vec{8, Float16}, B, cta_n + n + conf.GLOBAL_N * (cta_k + k))


        @unrolled for offset = 0:7
            frag_offset = b(offset, 0, 0) + # n0
                          b(offset, 1, 1) + # n1
                          b(offset, 2, 2) + # n2
                          b(ins, 0, 3) +    # n6
                          b(ins, 1, 4)      # n7
            @inbounds @immutable b_frag[frag_offset] = val[offset].value
        end
    end

    a_frag, b_frag
end
# }}}

# st shared {{{
@inline function st_shared(stage, shmem_a, shmem_b, a_frag, b_frag, conf)
    # Store A to Shared Memory.
    @unrolled for ins = 0:3
            m = b(tid(), 2, 0) +
                b(tid(), 3, 1) +
                b(tid(), 4, 2) +
                b(ins, 1, 3) +
                b(tid(), 5, 4) +
                b(tid(), 6, 5) +
                b(tid(), 7, 6)

            k = b(ins, 0, 2) +
                b(tid(), 0, 3) +
                b(tid(), 1, 4) +
                b(stage, 0, 5)

        @inbounds val = @ntuple 4 i -> begin
            offset = constant(i-1)
            frag_offset = b(offset, 0, 0) + # k0
                          b(offset, 1, 1) + # k1
                          b(ins, 0, 2) +  # k2
                          b(ins, 1, 3)    # m3
            VecElement{Float16}(a_frag[frag_offset])
        end
        @inbounds vstorea!(Vec{4, Float16}, shmem_a, swizzle_a(m, k, conf), val)
    end

    # Store B to Shared Memory.
    @unrolled for ins = 0:3
        n = b(tid(), 0, 3) +
            b(tid(), 1, 4) +
            b(tid(), 2, 5) +
            b(ins, 0, 6) +
            b(ins, 1, 7)

        k = b(tid(), 3, 0) +
            b(tid(), 4, 1) +
            b(tid(), 5, 2) +
            b(tid(), 6, 3) +
            b(tid(), 7, 4) +
            b(stage, 0, 5)

        @inbounds val = @ntuple 8 i -> begin
            offset = constant(i-1)
            frag_offset = b(offset, 0, 0) + # n0
                          b(offset, 1, 1) + # n1
                          b(offset, 2, 2) + # n2
                          b(ins, 0, 3) +    # n6
                          b(ins, 1, 4)      # n7
            VecElement{Float16}(b_frag[frag_offset])
        end
        @inbounds vstorea!(Vec{8, Float16}, shmem_b, swizzle_b(k, n, conf), val)
    end
end
# }}}

# ld shared {{{
@inline function ld_shared(stage, shmem_a, shmem_b, warp_m, warp_n, warp_k, conf)
    warp_mma_k = warp_k ÷ conf.WARP_K

    # Fragments for the data from the shared loads (and hence the MMAs).
    # index: (m5|m2+k3|k1|k0)
    a_frag = LocalArray{Tuple{16}, Float16}(undef)

    # index: (n5|n2|n1|n0)
    b_frag = LocalArray{Tuple{16}, Float16}(undef)

    # Load A from Shared Memory.
    @unrolled for ins = 0:1
        m = b(tid(), 0, 0) +
            b(tid(), 1, 1) +
            b(warp_mma_k, 1, 2) +
            b(tid(), 2, 3) +
            b(tid(), 4, 4) +
            b(ins, 0, 5)

        k = b(stage, 0, 5)

        @inbounds val = vloada(Vec{8, Float16}, shmem_a, swizzle_a(warp_m+m, warp_k+k, conf))

        @unrolled for offset = 0:7
            frag_offset = b(offset, 0, 0) +    # k0
                          b(offset, 1, 1) +    # k1
                          b(offset, 2, 2) +    # m2+k3
                          b(ins, 0, 3)         # m5

            @inbounds @immutable a_frag[frag_offset] = val[offset].value
        end
    end

    # Load B from Shared Memory.
    @unrolled for ins = 0:1
        k = b(tid(), 0, 0) +
            b(tid(), 1, 1) +
            b(stage, 0, 5)

        n = b(tid(), 3, 3) +
            b(tid(), 4, 4) +
            b(ins, 0, 5)

        @inbounds val = vloada(Vec{8, Float16}, shmem_b, swizzle_b(warp_k+k, warp_n+n, conf))

        @unrolled for offset = 0:7
            frag_offset = b(offset, 0, 0) +    # n0
                          b(offset, 1, 1) +    # n1
                          b(offset, 2, 2) +    # n2
                          b(ins, 0, 3)         # n5

            @inbounds @immutable b_frag[frag_offset] = val[offset].value
        end
    end

    a_frag.data, b_frag.data
end
# }}}

# epilogue {{{
@inline function epilogue_st_shared(epilogue_it, shmem_d, warp_m, warp_n, acc_frag)
    # index: (m5|m2|m1|n5|n4|n2|n0)

    @unrolled for ins = 0:15
        # TODO: vectorise
        @unrolled for offset = 0:1
            m = b(tid(), 0, 0) +
                b(ins, 3, 1) +
                b(epilogue_it, 0, 2) +
                b(tid(), 2, 3) +
                b(tid(), 4, 4) +
                b(epilogue_it, 1, 5) +
                warp_m

            n = b(offset, 0, 0) +
                b(tid(), 1, 1) +
                b(ins, 0, 2) +
                b(tid(), 3, 3) +
                b(ins, 1, 4) +
                b(ins, 2, 5) +
                warp_n

            frag_index = b(offset, 0, 0) +      # n0
                         b(ins, 0, 1) +         # n2
                         b(ins, 1, 2) +         # n4
                         b(ins, 2, 3) +         # n5
                         b(ins, 3, 4) +         # m1
                         b(epilogue_it, 0, 5) + # m2
                         b(epilogue_it, 1, 6)   # m5

            offset_M = b(tid(), 0, 0) +         # m0
                       b(ins, 3, 1) +           # m1
                       b(tid(), 2, 2) +         # m3
                       b(tid(), 4, 3) +         # m4
                       b(warpid(), 0, 4)        # m6

            offset_N = n

            offset = convert(Int, offset_N) + 258 * convert(Int, offset_M)

            @inbounds shmem_d[1 + offset] = acc_frag[frag_index]
        end
    end
end

@inline function epilogue_ld_shared(epilogue_it, shmem_d)
    # index: (m1|n7|n6|n1|n0)
    frag = LocalArray{Tuple{32}, Float32}(undef)

    @unrolled for ins = 0:7
        # TODO: vectorise
        @unrolled for offset = 0:3
            m = b(tid(), 4, 0) +
                b(ins, 2, 1) +
                b(epilogue_it, 0, 2) +
                b(tid(), 5, 3) +
                b(tid(), 6, 4) +
                b(epilogue_it, 1, 5) +
                b(tid(), 7, 6)

            n = b(offset, 0, 0) +
                b(offset, 1, 1) +
                b(tid(), 0, 2) +
                b(tid(), 1, 3) +
                b(tid(), 2, 4) +
                b(tid(), 3, 5) +
                b(ins, 0, 6) +
                b(ins, 1, 7)

            offset_M = b(tid(), 4, 0) +     # m0
                       b(ins, 2, 1) +       # m1
                       b(tid(), 5, 2) +     # m3
                       b(tid(), 6, 3) +     # m4
                       b(tid(), 7, 4)       # m6

            offset_N = n

            shmem_offset = convert(Int, offset_N) + 258 * convert(Int, offset_M)

            frag_index = b(offset, 0, 0) + # n0
                         b(offset, 1, 1) + # n1
                         b(ins, 0, 2) +    # n6
                         b(ins, 1, 3) +    # n7
                         b(ins, 2, 4)      # m1

            @inbounds @immutable frag[frag_index] = shmem_d[1 + shmem_offset]
        end
    end

    frag
end

@inline function epilogue_st_global(epilogue_it, D, shmem_d, cta_m, cta_n, frag, conf)
    @unrolled for ins = 0:7
        m = b(tid(), 4, 0) +
            b(ins, 2, 1) +
            b(epilogue_it, 0, 2) +
            b(tid(), 5, 3) +
            b(tid(), 6, 4) +
            b(epilogue_it, 1, 5) +
            b(tid(), 7, 6)

        n = b(tid(), 0, 2) +
            b(tid(), 1, 3) +
            b(tid(), 2, 4) +
            b(tid(), 3, 5) +
            b(ins, 0, 6) +
            b(ins, 1, 7)


        @inbounds val = @ntuple 4 i -> begin
            offset = constant(i-1)
            frag_index = b(offset, 0, 0) + # n0
                         b(offset, 1, 1) + # n1
                         b(ins, 0, 2) +    # n6
                         b(ins, 1, 3) +    # n7
                         b(ins, 2, 4)      # m1
            VecElement{Float32}(frag[frag_index])
        end

        @inbounds vstorea!(Vec{4, Float32}, D, cta_n + n + conf.GLOBAL_N * cta_m + conf.GLOBAL_N * m, val)
    end
end

@inline function epilogue(D, shmem_d, acc_frag, cta_m, cta_n, warp_m, warp_n, conf)
    @unrolled for epilogue_it = 0:3
        # TODO: Can we not remove this one?
        sync_threads()

        # Store tile of D to shared memory.
        epilogue_st_shared(epilogue_it, shmem_d, warp_m, warp_n, acc_frag)

        sync_threads()

        # Load tile of D from shared memory.
        frag = epilogue_ld_shared(epilogue_it, shmem_d)

        # Store tile of D to global memory.
        epilogue_st_global(epilogue_it, D, shmem_d, cta_m, cta_n, frag, conf)
    end
end
# }}}

# kernel {{{
# row-major A x row-major B = row-major D
function kernel(A, B, D, conf::Config)
    # The modulo is so that the BitArrayIndex knows which bits are 0.
    num_warps = NUM_WARPS_M(conf) * NUM_WARPS_N(conf)
    warpid_m = (warpid() % num_warps) % NUM_WARPS_M(conf)
    warpid_n = (warpid() % num_warps) ÷ NUM_WARPS_M(conf)

    cta_m = (bid_x() % NUM_BLOCKS_M(conf)) * conf.CTA_M
    cta_n = (bid_y() % NUM_BLOCKS_N(conf)) * conf.CTA_N

    warp_m = warpid_m * conf.WARP_M
    warp_n = warpid_n * conf.WARP_N

    SHMEM_A_SIZE = (conf.CTA_M * conf.CTA_K) * conf.SHARED_TO_REGS_STAGES
    SHMEM_B_SIZE = (conf.CTA_K * conf.CTA_N) * conf.SHARED_TO_REGS_STAGES

    shmem_ab = CuDynamicSharedArray(Float16, SHMEM_A_SIZE + SHMEM_B_SIZE)

    shmem_a = view(shmem_ab, 1:SHMEM_A_SIZE)
    shmem_b = view(shmem_ab, 1+SHMEM_A_SIZE:SHMEM_A_SIZE+SHMEM_B_SIZE)

    shmem_d = CuDynamicSharedArray(Float32, 32 * (256 + 2))

    # index: (m5|m2|m1|n5|n4|n2|n0)
    acc_frag = LocalArray{Tuple{128}, Float32}(@ntuple 128 i -> zero(Float32))

    # Two pairs of fragments for the shared -> register file pipeline.
    shared_a_frags = LocalArray{Tuple{2}, NTuple{16, Float16}}(undef)
    shared_b_frags = LocalArray{Tuple{2}, NTuple{16, Float16}}(undef)

    # Prologue.
    # ld_global(main_loop_it=0)
    cta_k = constant(0)
    global_a_frag, global_b_frag = ld_global(A, B, cta_m, cta_n, cta_k, conf)

    # st_shared(main_loop_it=0)
    main_loop_it = constant(0)
    stage = constant(0)
    st_shared(stage, shmem_a, shmem_b, global_a_frag, global_b_frag, conf)
    sync_threads()

    # ld_shared(main_loop_it=0, warp_mma_k=0)
    main_loop_it = constant(0)
    warp_k = constant(0)
    warp_mma_k = constant(0)
    stage = constant(0)
    shared_a_frag, shared_b_frag = ld_shared(stage, shmem_a, shmem_b, warp_m, warp_n, warp_k, conf)

    @inbounds @immutable shared_a_frags[convert(Int, warp_mma_k % 2) + 1] = shared_a_frag
    @inbounds @immutable shared_b_frags[convert(Int, warp_mma_k % 2) + 1] = shared_b_frag

    NUM_MAIN_LOOP_ITERS = conf.GLOBAL_K ÷ conf.CTA_K
    @not_unrolled for main_loop_it = 0 : NUM_MAIN_LOOP_ITERS - 1
        # The modulo is so that the BitArrayIndex knowns which bits are 0.
        # TODO: Do this automatically in the @not_unrolled macro?
        # TODO: Generate _next variables automatically.
        cta_k = (main_loop_it % NUM_MAIN_LOOP_ITERS) * conf.CTA_K

        main_loop_it_next = variadic((main_loop_it_orig + 1)) % NUM_MAIN_LOOP_ITERS
        cta_k_next = main_loop_it_next * conf.CTA_K

        stage = variadic(main_loop_it_orig) % 2
        stage_next = variadic((main_loop_it_orig + 1)) % 2

        # CTA_M x CTA_N x CTA_K GEMM per CTA
        NUM_WARP_MMA_K_ITERS = conf.CTA_K ÷ conf.WARP_K
        @unrolled for warp_mma_k = 0 : NUM_WARP_MMA_K_ITERS - 1
            warp_k = warp_mma_k * conf.WARP_K

            # TODO: Do this in macro.
            warp_mma_k_next = constant(warp_mma_k_orig + 1) % NUM_WARP_MMA_K_ITERS
            warp_k_next = warp_mma_k_next * conf.WARP_K

            if warp_mma_k == NUM_WARP_MMA_K_ITERS-1
                # st_shared(main_loop_it+1)
                st_shared(stage_next, shmem_a, shmem_b, global_a_frag, global_b_frag, conf)
                sync_threads()
            end

            # ld_shared(main_loop_it, warp_mma_k + 1)
            shared_a_frag, shared_b_frag = ld_shared(stage, shmem_a, shmem_b, warp_m, warp_n, warp_k_next, conf)

            @inbounds @immutable shared_a_frags[convert(Int, warp_mma_k_next % 2) + 1] = shared_a_frag
            @inbounds @immutable shared_b_frags[convert(Int, warp_mma_k_next % 2) + 1] = shared_b_frag

            # TODO: Predicate the load?
            if warp_mma_k == 0
                # ld_global(main_loop_it + 1)
                # Copy the data for a CTA_M x CTA_N x CTA_K GEMM from GMEM to SHMEM, cooperatively in a CTA.
                global_a_frag, global_b_frag = ld_global(A, B, cta_m, cta_n, cta_k_next, conf)
            end

            # WARP_M x WARP_N x WARP_K = 64 x 64 x 4 GEMM per warp
            # mma(main_loop_it, warp_mma_k)
            @inbounds shared_a_frag = shared_a_frags[convert(Int, warp_mma_k % 2) + 1]
            @inbounds shared_b_frag = shared_b_frags[convert(Int, warp_mma_k % 2) + 1]
            acc_frag = warp_mma(shared_a_frag, shared_b_frag, acc_frag, warp_mma_k, conf)
        end
    end

    # epilogue: store matrix from registers to global memory
    epilogue(D, shmem_d, acc_frag, cta_m, cta_n, warp_m, warp_n, conf)

    nothing
end
# }}}

# driver {{{
function test(; dump_code=false, debug=false)
    if debug
        @device_code_warntype interactive=true @cuda threads=NUM_THREADS(conf) blocks=(NUM_BLOCKS_M(conf), NUM_BLOCKS_N(conf)) shmem=48*1024 kernel(B, A, D, conf)
        return
    end

    if dump_code
        @device_code dir="gemm-output" @cuda threads=NUM_THREADS(conf) blocks=(NUM_BLOCKS_M(conf), NUM_BLOCKS_N(conf)) shmem=48*1024 kernel(B, A, D, conf)
    end

    D_ref = similar(D)

    CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)
    CUDA.CUBLAS.gemmEx!('N', 'N', Float32(1), A, B, Float32(0), D_ref)

    # TODO: do not hardcode shared memory size
    @cuda threads=NUM_THREADS(conf) blocks=(NUM_BLOCKS_M(conf), NUM_BLOCKS_N(conf)) shmem=48*1024 kernel(B, A, D, conf)

    compare(x, y) = isapprox(x, y; rtol=sqrt(eps(Float16)))

    @test isapprox(D_ref, D; rtol=sqrt(eps(Float16)))
    @test isapprox(D_ref, D; rtol=sqrt(eps(Float16)), norm=M -> LinearAlgebra.norm(M, Inf))
    @test all(compare.(D, D_ref))
end

function cublas()
    CUDA.CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)
    CUDA.CUBLAS.gemmEx!('N', 'N', Float32(1), A, B, Float32(0), D)
end

isinteractive() || test()
# }}}
