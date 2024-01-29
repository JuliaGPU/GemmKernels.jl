using Base.Cartesian: @ntuple

# ----------------------
# Explicit vectorisation
# ----------------------

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
