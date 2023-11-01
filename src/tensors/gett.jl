using GemmKernels
using GemmKernels.Layout
using GemmKernels.Tensors.GETTLayout

struct GETTPlan <: AbstractAlgorithmPlan
    gemmConf
end

export createGETTContractionPlan
function createGETTContractionPlan(desc::ContractionDescriptor)
    # Get the modes from the contraction descriptor.
    modeA, modeB, modeC, modeD = desc.modeA, desc.modeB, desc.modeC, desc.modeD

    # These are the modes over which the contraction will take place.
    # They will take up the K dimension of the GEMM.
    modesToContract = intersect(modeB, modeA)

    # The modes that are not shared between A and B will be the M and N dimensions of the GEMM, respectively.
    modesAM = setdiff(modeA, modesToContract)
    modesBN = setdiff(modeB, modesToContract)

    modesDM = setdiff(modeD, modesBN)
    modesDN = setdiff(modeD, modesAM)

    # If the first M mode of A is equal to the first M mode of D,
    # or if the first mode of A is not part of the M modes, 
    # we want to inherit the order of modes of D for optimal unit-stride
    # loads ands stores of D.
    if (modesAM[1] == modesDM[1] || findall(x -> x == modesAM[1], modeA)[1] != 1)
        modesAM = modesDM
    end
    
    # The same goes for B.
    if (modesBN[1] == modesDN[1] || findall(x -> x == modesBN[1], modeB)[1] != 1)
        modesBN = modesDN
    end

    # The mode identifiers are either Char or Int. They do not correspond to the actual dimensions of the tensors.
    # We now find the number of the dimension in the tensor that corresponds to each mode identifier.

    dimensionsAM = Vector{Int}(undef, 0)
    for mode in modesAM
        append!(dimensionsAM, findall(x -> x == mode, modeA))
    end

    dimensionsBN = Vector{Int}(undef, 0)
    for mode in modesBN
        append!(dimensionsBN, findall(x -> x == mode, modeB))
    end

    dimensionsAK = Vector{Int}(undef, 0)
    dimensionsBK = Vector{Int}(undef, 0)
    for mode in modesToContract
        append!(dimensionsAK, findall(x -> x == mode, modeA))
        append!(dimensionsBK, findall(x -> x == mode, modeB))
    end

    # In what follows, we deterimine whether the loading of the A and B matrices will be strided.
    # If the loading is strided, we also determine over which dimensions of the tensors the loading will be strided.

    isLoadStridedA = false
    strideOverA = Vector{Int}(undef, 0)

    isLoadStridedB = false
    strideOverB = Vector{Int}(undef, 0)

    # If the first dimension of tensor A does not belong to the M modes, then it belongs to the K modes.
    # This means we can transpose the matrix to maximize the number of vectorised loads.
    if (1 in dimensionsAM)
        isColMajorA = true
    else
        isColMajorA = false
    end

    if (1 in dimensionsBN)
        isColMajorB = false
    else
        isColMajorB = true
    end

    # Here the order of loading the K dimensions is determined. This must be consistent across
    # the A and B tensors.

    # If the load is vectorised over K in A and not in B, we use the order of the K dimensions in A.
    if (isColMajorA == false && isColMajorB == false)
        newPerm = sortperm(dimensionsAK)
        dimensionsAK = dimensionsAK[newPerm]
        dimensionsBK = dimensionsBK[newPerm]
    end

    # If the load is vectorised over K in B and not in A, we use the order of the K dimensions in B.
    if (isColMajorA == true && isColMajorB == true)
        newPerm = sortperm(dimensionsBK)
        dimensionsAK = dimensionsAK[newPerm]
        dimensionsBK = dimensionsBK[newPerm]
    end

    # If both loads over K could be vectorised, we choose the order of dimensions of the tensor
    # with the largest extent of the first dimension.
    # Unless the dimensions orders are identical, in which case the orders should not change and
    # both loads can be vectorised.
    if (isColMajorA == false && isColMajorB == true)
        if (sortperm(dimensionsAK) == sortperm(dimensionsBK))
            nothing
        elseif (desc.descA.extent[1] > desc.descB.extent[1])
            newPerm = sortperm(dimensionsAK)
            dimensionsAK = dimensionsAK[newPerm]
            dimensionsBK = dimensionsBK[newPerm]

            isLoadStridedB = true
            isColMajorB = false
            append!(strideOverB, 1 : dimensionsBK[1] - 1)
        else
            newPerm = sortperm(dimensionsBK)
            dimensionsAK = dimensionsAK[newPerm]
            dimensionsBK = dimensionsBK[newPerm]

            isLoadStridedA = true
            isColMajorA = true
            append!(strideOverA, 1 : dimensionsAM[1] - 1)
        end
    end    
    
    # We find the dimensions of the C and D tensors that correspond to the M and N modes.
    dimensionsDM = Vector{Int}(undef, 0)
    for dimension in dimensionsAM
        append!(dimensionsDM, findall(x -> x == modeA[dimension], modeD))
    end

    dimensionsDN = Vector{Int}(undef, 0)
    for dimension in dimensionsBN
        append!(dimensionsDN, findall(x -> x == modeB[dimension], modeD))
    end

    # The C load and D store are vectorised if the order of the M dimensions in A and C are 
    # identical and the order of the N dimensions in B and D are identical.
    # This only works if the first dimension of C is part of A.
    isColMajorD = true

    if (dimensionsDM == dimensionsDM[sortperm(dimensionsDM)] && dimensionsDN == dimensionsDN[sortperm(dimensionsDN)] && modeD[1] in modesAM)
        isStoreStridedD = false
    else
        isStoreStridedD = true
    end

    strideOverD = Vector{Int}(undef, 0)
    if (isStoreStridedD == true)
        append!(strideOverD, 1 : dimensionsDM[1] - 1)
    end

    # Finally, the GEMM shape is determined.
    gemmShape = (
        M = prod(desc.descA.extent[dimensionsAM]),
        N = prod(desc.descB.extent[dimensionsBN]),
        K = prod(desc.descA.extent[dimensionsAK]),
    )

    TensorLayoutA = GETTLayout.createGETTLayout(desc.descA.dataType, desc.descA.extent, (dimensionsAM, dimensionsAK), isColMajorA, isLoadStridedA, strideOverA)
    TensorLayoutB = GETTLayout.createGETTLayout(desc.descB.dataType, desc.descB.extent, (dimensionsBK, dimensionsBN), isColMajorB, isLoadStridedB, strideOverB)
    TensorLayoutD = GETTLayout.createGETTLayout(desc.descD.dataType, desc.descD.extent, (dimensionsDM, dimensionsDN), isColMajorD, isStoreStridedD, strideOverD)

    return (
        gemmShape,
        TensorLayoutA, isColMajorA,
        TensorLayoutB, isColMajorB,
        TensorLayoutD,
        TensorLayoutD,
    )
end

export setUpGETTKernel
function setUpGETTKernel(desc::ContractionDescriptor, operator)
    (
        gemmShape,
        TensorLayoutA, isColMajorA, 
        TensorLayoutB, isColMajorB,
        TensorLayoutC,
        TensorLayoutD,
    ) = createGETTContractionPlan(desc)

    if (isColMajorA)
        SharedLayoutA = Layout.Padded{Layout.UnsafeAlignedColMajor{desc.descA.dataType}, 16 ÷ sizeof(desc.descA.dataType)}
    else
        SharedLayoutA = Layout.Padded{Layout.UnsafeAlignedRowMajor{desc.descA.dataType}, 16 ÷ sizeof(desc.descA.dataType)}
    end

    if (isColMajorB)
        SharedLayoutB = Layout.Padded{Layout.UnsafeAlignedColMajor{desc.descB.dataType}, 16 ÷ sizeof(desc.descB.dataType)}
    else
        SharedLayoutB = Layout.Padded{Layout.UnsafeAlignedRowMajor{desc.descB.dataType}, 16 ÷ sizeof(desc.descB.dataType)}
    end

    if (operator == Operator.WMMAOp)
        operator = Operator.WMMAOp{16, 16, 16, desc.dataType, desc.computeType}
    elseif (operator <: Operator.GeneralFPUOp)
        operator = operator{8, 8, 1, 4, 8, 1, desc.dataType, desc.computeType}
    end

    gemmConf = GemmKernels.get_config(
        gemm_shape = gemmShape,
        operator = operator,

        global_a_layout = TensorLayoutA,
        global_b_layout = TensorLayoutB,
        global_c_layout = TensorLayoutC,
        global_d_layout = TensorLayoutD,

        shared_a_layout = SharedLayoutA,
        shared_b_layout = SharedLayoutB,
        shared_c_layout = Layout.UnsafeAlignedColMajor{desc.descC.dataType},
        shared_d_layout = Layout.UnsafeAlignedColMajor{desc.descD.dataType},

        is_a_col_major = isColMajorA,
        is_b_col_major = isColMajorB,
    )

    return GETTPlan(gemmConf)
end