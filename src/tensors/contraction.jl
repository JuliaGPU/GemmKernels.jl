using GemmKernels.Tensors
using GemmKernels.Operator

export ContractionPlan
mutable struct ContractionPlan
    desc::ContractionDescriptor
    algo::ALGO
    algorithmPlan::AbstractAlgorithmPlan
    operator

    function ContractionPlan(desc::ContractionDescriptor; algo::ALGO=ALGO_GETT, operator=Operator.WMMAOp)
        if (algo == ALGO_GETT)
            algorithmPlan = setUpGETTKernel(desc, operator)
        end

        return new(desc, algo, algorithmPlan, operator)
    end

    function ContractionPlan(
        a, modeA::ModeType,
        b, modeB::ModeType,
        c, modeC::ModeType,
        d, modeD::ModeType;
        algo::ALGO=ALGO_GETT,
        computeType=eltype(a),
        operator=Operator.WMMAOp
    )
        desc = ContractionDescriptor(
            a, modeA,
            b, modeB,
            c, modeC,
            d, modeD,
            computeType=computeType
        )
        return ContractionPlan(desc; algo=algo, operator=operator)
    end

    function ContractionPlan(plan::ContractionPlan, operator)
        return ContractionPlan(plan.desc; algo=plan.algo, operator=operator)
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

export contraction!
function contraction!(plan::ContractionPlan, α, a, b, β, c, d, opAB, opABC)
    if (opAB == *) && (opABC == +)
        contraction!(plan, α, a, b, β, c, d)
    elseif (opAB == +) && (opABC == max)
        plan = ContractionPlan(plan, Operator.TropicalFPUOp)
        contraction!(plan, α, a, b, β, c, d)
    end
end

export elementwiseTrinary!
function elementwiseTrinary!(plan::ContractionPlan, α, a, b, β, c, d, opAB, opABC)
    # Future work: reuse GemmKernels building blocks
end

export elementwiseBinary!
function elementwiseBinary!(plan::ContractionPlan, α, a, β, c, d, opAB)
    # Future work: reuse GemmKernels building blocks
end


export reduction!
function reduction!(α, a, β, c, d, opReduce)
    # Future work: reuse GemmKernels building blocks
end

export permutation!
function permutation!(plan::ContractionPlan, α, a, d)
    # Future work: reuse GemmKernels building blocks
end