using GemmKernels.Tensors
using GemmKernels.Operator

export ContractionPlan
mutable struct ContractionPlan
    desc::ContractionDescriptor
    algo::ALGO
    algorithmPlan::AbstractAlgorithmPlan
    operator

    # XXX: this should be done in setUpGETTKernel and stored in the GETT algorithm plan,
    #      but the inputs are currently not known there, so this is a quick fix.
    matmulPlan::GemmKernels.MatmulPlan
    alpha
    beta

    function ContractionPlan(desc::ContractionDescriptor, α, a, b, β, c, d;
                             algo::ALGO=ALGO_GETT, operator=Operator.WMMAOp,
                             blockShape=nothing, warpsPerBlock=nothing, computeWarp=nothing,
                             kernel=Kernel.matmul_singlestage)
        if algo == ALGO_GETT
            algorithmPlan =
                setUpGETTKernel(desc, operator, blockShape, warpsPerBlock, computeWarp)

            unaryOpA = desc.descA.unaryOp
            unaryOpB = desc.descB.unaryOp
            unaryOpC = desc.descC.unaryOp
            unaryOpD = desc.descD.unaryOp

            α = desc.computeType(α)
            β = desc.computeType(β)

            matmulPlan = GemmKernels.plan_matmul(
                algorithmPlan.gemmConf, a, b, c, d;
                transform_shared_to_regs_a = Transform.Elementwise(x -> unaryOpA(α * x)),
                transform_shared_to_regs_b = Transform.Elementwise(x -> unaryOpB(x)),
                transform_global_to_shared_c = Transform.Elementwise(x -> β * unaryOpC(x)),
                transform_shared_to_global_d = Transform.Elementwise(x -> unaryOpD(x)),
                kernel)
            end

        return new(desc, algo, algorithmPlan, operator, matmulPlan, α, β)
    end

    function ContractionPlan(
        α,
        a, descA, modeA::ModeType,
        b, descB, modeB::ModeType,
        β,
        c, descC, modeC::ModeType,
        d, descD, modeD::ModeType;
        algo::ALGO=ALGO_GETT,
        computeType=eltype(a),
        accumulateType=eltype(c),
        operator=Operator.WMMAOp,
        blockShape=nothing,
        warpsPerBlock=nothing,
        computeWarp=nothing,
        kernel=Kernel.matmul_singlestage
    )
        desc = ContractionDescriptor(
            descA, modeA,
            descB, modeB,
            descC, modeC,
            descD, modeD,
            computeType,
            accumulateType
        )
        return ContractionPlan(desc, α, a, b, β, c, d; algo, operator, kernel,
                               blockShape, warpsPerBlock, computeWarp)
    end

    function ContractionPlan(plan::ContractionPlan, operator)
        return ContractionPlan(plan.desc; plan.algo, operator)
    end
end

export contraction!
function contraction!(plan::ContractionPlan, α, a, b, β, c, d;
                      kernel=Kernel.matmul_singlestage)
    unaryOpA = plan.desc.descA.unaryOp
    unaryOpB = plan.desc.descB.unaryOp
    unaryOpC = plan.desc.descC.unaryOp
    unaryOpD = plan.desc.descD.unaryOp

    α = plan.desc.computeType(α)
    @assert α == plan.alpha "α must match the one in the plan"
    β = plan.desc.computeType(β)
    @assert β == plan.beta "β must match the one in the plan"

    if plan.algo == ALGO_GETT
        GemmKernels.matmul(plan.matmulPlan, a, b, c, d)
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
    else
        error("Unsupported operator combination!")
    end
end

export elementwiseTrinary!
function elementwiseTrinary!(plan::ContractionPlan, α, a, b, β, c, d, opAB, opABC)
    # Future work: reuse GemmKernels building blocks
    error("Not implemented yet!")
end

export elementwiseBinary!
function elementwiseBinary!(plan::ContractionPlan, α, a, β, c, d, opAB)
    # Future work: reuse GemmKernels building blocks
    error("Not implemented yet!")
end


export reduction!
function reduction!(α, a, β, c, d, opReduce)
    # Future work: reuse GemmKernels building blocks
    error("Not implemented yet!")
end

export permutation!
function permutation!(plan::ContractionPlan, α, a, d)
    # Future work: reuse GemmKernels building blocks
    error("Not implemented yet!")
end
