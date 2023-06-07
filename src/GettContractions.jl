export GETT
module GETT

# GETT: GEMM-like Tensor Tensor contraction
using CUDA
using GemmKernels
using GemmKernels.Layout
using GemmKernels.Tiling
using GemmKernels.TensorLayout
using GemmKernels.TensorPlan

using KernelAbstractions.Extras: @unroll

# TODO LIST
# - [ ] Test if vectorised store works properly
# - [ ] Also provide a function for the strided store. Improve store.
# - [ ] Experiment with a non-zero C matrix in the run files, and then generalise to LocalLayoutC

export GETTCreateLayoutTypes

function GETTCreateLayoutTypes(plan::PLAN)
    TensorLayout.createLayoutTypes(plan)

    plan.gemm_conf = GemmKernels.get_config(
        gemm_shape = (M = plan.M, N = plan.N, K = plan.K),
        operator = Operator.WMMAOp{16, 16, 16, Float16},

        global_a_layout = plan.TensorLayoutA{Float16},
        global_b_layout = plan.TensorLayoutB{Float16},
        global_c_layout = plan.TensorLayoutC{Float16},
        global_d_layout = plan.TensorLayoutD{Float16},

        shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},
        shared_b_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},
        shared_c_layout = Layout.AlignedColMajor{Float16},
        shared_d_layout = Layout.AlignedColMajor{Float16},

        is_a_col_major = true,
        is_b_col_major = true,
    )
end

export GETTContraction

function GETTContraction(
    plan::PLAN,
    α, A::CuArray, B::CuArray,
    β, C::CuArray,
    D::CuArray,
    )

    GemmKernels.matmul(A, B, C, D, plan.gemm_conf;
        kernel = Kernel.matmul_pipelined)
end

end