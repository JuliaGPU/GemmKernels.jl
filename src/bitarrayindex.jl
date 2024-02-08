using GemmKernels: vloada, vstorea!, Vec

import Base.+, Base.|, Base.*, Base.÷, Base.%, Base.&, Base.⊻, Base.<<, Base.>>, Base.==, Base.<
import Base.getindex, Base.setindex, Base.setindex!

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

function Base.show(io::IO, I::BitArrayIndex)
    print(io, convert(Int, I))
    print(io, " (")

    for i in 8 * sizeof(UInt) - 1 : -1 : 0
        mask = 1 << i

        if (I.known_zero & mask) != 0
            print(io, "0")
        elseif (I.known_one & mask) != 0
            print(io, "1")
        else
            print(io, "V")
        end
    end

    print(io, ")")
end

constant(i::Int) = BitArrayIndex(~reinterpret(UInt, i), reinterpret(UInt, i), reinterpret(UInt, 0))
variadic(i::Int) = BitArrayIndex(reinterpret(UInt, 0), reinterpret(UInt, 0), reinterpret(UInt, i))
Base.convert(::Type{UInt}, I::BitArrayIndex) = I.variadic_part + I.known_one
Base.convert(::Type{Int}, I::BitArrayIndex) = reinterpret(Int, I.variadic_part + I.known_one)

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

# Loop-specific helpers
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

# Bit manipulation utils.

# TODO: Use more descriptive name.
# extract bit
@inline b(val, i) = b(val, i, 0)

# move bit from position i to j
@inline b(val::Integer, i, j) = ((val >> i) & 1) << j
@inline b(val::BitArrayIndex, i, j) = ((val >> i) & constant(1)) << j
