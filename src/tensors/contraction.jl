using GemmKernels.Tensors

export ContractionPlan
mutable struct ContractionPlan
    desc::ContractionDescriptor
    algo::ALGO
    algorithmPlan::AbstractAlgorithmPlan

    function ContractionPlan(desc::ContractionDescriptor, algo::ALGO=ALGO_GETT)
        if (algo == ALGO_GETT)
            algorithmPlan = setUpGETTKernel(desc)
        end

        return new(desc, algo, algorithmPlan)
    end

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
    unaryOpA = plan.desc.descA.unaryOp
    unaryOpB = plan.desc.descB.unaryOp
    unaryOpC = plan.desc.descC.unaryOp

    α = plan.desc.computeType(α)
    β = plan.desc.computeType(β)

    if plan.algo == ALGO_GETT
        GemmKernels.matmul(
            a, b, c, d, plan.algorithmPlan.gemmConf,
            transform_shared_to_regs_a = Transform.Elementwise(x -> α * unaryOpA(x)),
            transform_shared_to_regs_b = Transform.Elementwise(x -> unaryOpB(x)),
            transform_shared_to_regs_c = Transform.Elementwise(x -> β * unaryOpC(x)),
            kernel = Kernel.matmul_pipelined,
        )
    else 
        throw(ArgumentError("Unsupported algorithm!"))
    end
end