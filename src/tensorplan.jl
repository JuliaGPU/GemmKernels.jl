module TensorPlan

using GemmKernels
using GemmKernels.TensorLayout

ModeType = AbstractVector{<:Union{Char,Integer}}

export ALGO

@enum ALGO::Int begin
    ALGO_DEFAULT_PATIENT = -6
    ALGO_GETT = -4
    ALGO_TGETT = -3
    ALGO_TTGT = -2
    ALGO_DEFAULT = -1
end

@enum UNARY_OPERATOR::Int begin
    UOP_ELEMENTWISE = 0
end

export PLAN

Base.@kwdef mutable struct PLAN
    algo::ALGO

    M::Int
    N::Int
    K::Int

    # A tensor plan variables
    a_MK_strides_sizes::Vector{Int}
    a_MK_strides::Tuple{Vector{Int}, Vector{Int}}
    is_a_load_strided::Bool
    a_strided_over::Union{Vector{Int}, Nothing} = nothing

    # B tensor plan variables
    b_KN_strides_sizes::Vector{Int}
    b_KN_strides::Tuple{Vector{Int}, Vector{Int}}
    is_b_load_strided::Bool
    b_strided_over::Union{Vector{Int}, Nothing} = nothing

    # D tensor plan variables
    d_MN_strides_sizes::Vector{Int}
    d_MN_strides::Tuple{Vector{Int}, Vector{Int}}
    is_d_store_strided::Bool

    TensorLayoutA = nothing
    TensorLayoutB = nothing
    TensorLayoutC = nothing
    TensorLayoutD = nothing
    gemm_conf = nothing
end



export TensorDescriptor

Base.@kwdef mutable struct TensorDescriptor
    numModes::Int
    extent::Vector{Int}
    stride::Vector{Int}
    dataType::DataType
    unaryOp::UNARY_OPERATOR

    function TensorDescriptor(a; numModes=length(size(a)), extent=size(a), stride=strides(a), dataType=eltype(a), unaryOp=UOP_ELEMENTWISE)
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

    function ContractionDescriptor(
        a, modeA::ModeType,
        b, modeB::ModeType,
        c, modeC::ModeType,
        d, modeD::ModeType;
        # ? Think about this for the FPUOp
        computeType=eltype(a)
    )
        return new(
            TensorDescriptor(a), modeA,
            TensorDescriptor(b), modeB,
            TensorDescriptor(c), modeC,
            TensorDescriptor(d), modeD,
            computeType
        )
    end
end

export ContractionPlan

mutable struct ContractionPlan
    desc::ContractionDescriptor
    algo::ALGO

    TensorLayoutA
    TensorLayoutB
    TensorLayoutC
    TensorLayoutD

    gemmConf

    # For now, default algo is GETT
    function ContractionPlan(desc::ContractionDescriptor, algo::ALGO=ALGO_GETT)
        (
            gemmShape,
            TensorLayoutA, isAColMajor, 
            TensorLayoutB, isBColMajor,
            TensorLayoutC,
            TensorLayoutD,
        ) = createGETTContractionPlan(desc)

        if (isAColMajor)
            ASharedLayout = Layout.Padded{Layout.AlignedColMajor{desc.descA.dataType}, 8}
        else
            ASharedLayout = Layout.Padded{Layout.AlignedRowMajor{desc.descA.dataType}, 8}
        end

        if (isBColMajor)
            BSharedLayout = Layout.Padded{Layout.AlignedColMajor{desc.descB.dataType}, 8}
        else
            BSharedLayout = Layout.Padded{Layout.AlignedRowMajor{desc.descB.dataType}, 8}
        end

        gemmConf = GemmKernels.get_config(
            gemm_shape = gemmShape,
            operator = Operator.WMMAOp{16, 16, 16, desc.computeType},

            global_a_layout = TensorLayoutA,
            global_b_layout = TensorLayoutB,
            global_c_layout = TensorLayoutC,
            global_d_layout = TensorLayoutD,

            shared_a_layout = ASharedLayout,
            shared_b_layout = BSharedLayout,
            shared_c_layout = Layout.AlignedColMajor{desc.descC.dataType},
            shared_d_layout = Layout.AlignedColMajor{desc.descD.dataType},

            is_a_col_major = isAColMajor,
            is_b_col_major = isBColMajor,
        )

        return new(
            desc,
            algo,
            TensorLayoutA,
            TensorLayoutB,
            TensorLayoutC,
            TensorLayoutD,
            gemmConf
        )
    end

    # No need for creating a ContractionDescriptor.
    function ContractionPlan(
        a, modeA::ModeType,
        b, modeB::ModeType,
        c, modeC::ModeType,
        d, modeD::ModeType;
        algo::ALGO=ALGO_GETT,
        computeType=eltype(a)
    )
        desc = ContractionDescriptor(
            a, modeA,
            b, modeB,
            c, modeC,
            d, modeD,
            computeType=computeType
        )
        return ContractionPlan(desc, algo)
    end
end

export contraction!

function contraction!(plan::ContractionPlan, α, a, b, β, c, d)
    if plan.algo == ALGO_GETT
        GemmKernels.matmul(
            a, b, c, d, plan.gemmConf;
            transform_shared_to_regs_a = Transform.Elementwise(x -> α * x),
            transform_shared_to_regs_c = Transform.Elementwise(x -> β * x),
            kernel = Kernel.matmul_pipelined,
        )
    else 
        throw(ArgumentError("unsupported algo"))
    end
end

function createGETTContractionPlan(desc::ContractionDescriptor)
    modeA, modeB, modeC, modeD = desc.modeA, desc.modeB, desc.modeC, desc.modeD

    if modeC != modeD
        throw(ArgumentError("modeC and modeD must be the same"))
    end

    stridesToContract = intersect(modeB, modeA)

    AMStrides = setdiff(modeA, stridesToContract)
    BNStrides = setdiff(modeB, stridesToContract)

    AMStridesIndices = Vector{Int}(undef, 0)
    for stride in AMStrides
        append!(AMStridesIndices, findall(x -> x == stride, modeA))
    end

    BNStridesIndices = Vector{Int}(undef, 0)
    for stride in BNStrides
        append!(BNStridesIndices, findall(x -> x == stride, modeB))
    end

    AKStridesIndices = Vector{Int}(undef, 0)
    BKStridesIndices = Vector{Int}(undef, 0)
    for stride in stridesToContract
        append!(AKStridesIndices, findall(x -> x == stride, modeA))
        append!(BKStridesIndices, findall(x -> x == stride, modeB))
    end

    isALoadStrided = false
    AStridedOver = Vector{Int}(undef, 0)

    isBLoadStrided = false
    BStridedOver = Vector{Int}(undef, 0)

    if (1 in AMStridesIndices)
        isAColMajor = true
    else
        isAColMajor = false
    end

    if (1 in BNStridesIndices)
        isBColMajor = false
    else
        isBColMajor = true
    end

    if (1 in AKStridesIndices && isBColMajor == false)
        newPerm = sortperm(AKStridesIndices)
        AKStridesIndices = AKStridesIndices[newPerm]
        BKStridesIndices = BKStridesIndices[newPerm]
    end

    if (1 in BKStridesIndices && isAColMajor == true)
        newPerm = sortperm(BKStridesIndices)
        AKStridesIndices = AKStridesIndices[newPerm]
        BKStridesIndices = BKStridesIndices[newPerm]
    end

    if (1 in AKStridesIndices && 1 in BKStridesIndices && length(AKStridesIndices) > 1)
        if (desc.descA.extent[1] > desc.descB.extent[1])
            newPerm = sortperm(AKStridesIndices)
            AKStridesIndices = AKStridesIndices[newPerm]
            BKStridesIndices = BKStridesIndices[newPerm]

            isBLoadStrided = true
            isBColMajor = false
            append!(BStridedOver, 1 : BKStridesIndices[1] - 1)
        else
            newPerm = sortperm(BKStridesIndices)
            AKStridesIndices = AKStridesIndices[newPerm]
            BKStridesIndices = BKStridesIndices[newPerm]

            isALoadStrided = true
            isAColMajor = true
            append!(AStridedOver, 1 : AMStridesIndices[1] - 1)
        end
    end

    DMStridesIndices = Vector{Int}(undef, 0)
    for strideIndex in AMStridesIndices
        append!(DMStridesIndices, findall(x -> x == modeA[strideIndex], modeD))
    end

    DNStridesIndices = Vector{Int}(undef, 0)
    for strideIndex in BNStridesIndices
        append!(DNStridesIndices, findall(x -> x == modeB[strideIndex], modeD))
    end

    isDStoreStrided = (vcat(DMStridesIndices, DNStridesIndices) != 1:length(modeD))
    DStridedOver = Vector{Int}(undef, 0)

    if (isDStoreStrided == true)
        append!(DStridedOver, 1 : DMStridesIndices[1] - 1)
    end

    gemmShape = (
        M = prod(desc.descA.extent[AMStridesIndices]),
        N = prod(desc.descB.extent[BNStridesIndices]),
        K = prod(desc.descA.extent[AKStridesIndices]),
    )

    TensorLayoutA = TensorLayout.createALayout(desc.descA.dataType, desc.descA.extent, (AMStridesIndices, AKStridesIndices), isAColMajor, isALoadStrided, AStridedOver)
    TensorLayoutB = TensorLayout.createBLayout(desc.descB.dataType, desc.descB.extent, (BKStridesIndices, BNStridesIndices), isBColMajor, isBLoadStrided, BStridedOver)
    TensorLayoutC = TensorLayout.createCLayout(desc.descC.dataType, desc.descD.extent, (DMStridesIndices, DNStridesIndices), isDStoreStrided, DStridedOver)
    TensorLayoutD = TensorLayout.createDLayout(desc.descD.dataType, desc.descD.extent, (DMStridesIndices, DNStridesIndices), isDStoreStrided, DStridedOver)

    return (
        gemmShape,
        TensorLayoutA, isAColMajor, 
        TensorLayoutB, isBColMajor,
        TensorLayoutC,
        TensorLayoutD,
    )
end


# # TODO: Add workspaceSize
# function initContractionPlan(desc::contractionDescriptor, algo::ALGO=ALGO_GETT)

#     return 1
# end

# # TODO: Add unaryOp
# function initTensorDescriptor(numModes::Int, extent::Vector{Int}, stride::Vector{Int}, ::T) where T

#     return 1
# end


end