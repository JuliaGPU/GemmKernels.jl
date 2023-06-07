using GemmKernels

ModeType = AbstractVector{<:Union{Char,Integer}}

export AbstractAlgorithmPlan
abstract type AbstractAlgorithmPlan end

export ALGO
@enum ALGO::Int begin
    # Future work: explore search space for the best algorithm and the best GEMM config.
    ALGO_DEFAULT_PATIENT = -6

    ALGO_GETT = -4
    # Future work: implement transposition kernel. 
    ALGO_TGETT = -3
    # Future work: implement transposition kernel. 
    ALGO_TTGT = -2

    ALGO_DEFAULT = -1
end

export TensorDescriptor
mutable struct TensorDescriptor
    numModes::Int
    extent::Vector{Int}
    stride::Vector{Int}
    dataType::DataType
    unaryOp

    function TensorDescriptor(
        a; numModes=length(size(a)), extent=size(a), stride=strides(a), dataType=eltype(a), unaryOp=identity)
        return new(
            numModes,
            collect(Int, extent), collect(Int, stride),
            dataType, unaryOp
        )
    end
end

export ContractionDescriptor
mutable struct ContractionDescriptor
    descA::TensorDescriptor
    modeA::ModeType

    descB::TensorDescriptor
    modeB::ModeType

    descC::TensorDescriptor
    modeC::ModeType

    descD::TensorDescriptor
    modeD::ModeType

    computeType::DataType
    dataType::DataType

    function ContractionDescriptor(
        a, modeA::ModeType,
        b, modeB::ModeType,
        c, modeC::ModeType,
        d, modeD::ModeType;
        computeType=eltype(a),
        dataType=eltype(c)
    )
        return new(
            TensorDescriptor(a), modeA,
            TensorDescriptor(b), modeB,
            TensorDescriptor(c), modeC,
            TensorDescriptor(d), modeD,
            computeType, dataType
        )
    end
end