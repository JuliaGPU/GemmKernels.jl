# Checks that need no NVIDIA GPU or CUDA driver:
#
#  1. host-side logic: CTA swizzle coverage, async-copy layout support;
#  2. GPU code generation for every kernel (including Kernel.matmul_pipelined_async) on a
#     plain GEMM and on a tensor contraction: type inference + LLVM + PTX generation, which
#     catches most errors in kernel code (InvalidIRError: dynamic dispatch, type
#     instabilities, undefined functions, ...). It needs neither a driver nor the CUDA
#     toolkit, and it does not run anything;
#  3. optionally (PTXAS=/path/to/ptxas, from a CUDA toolkit), assembling that PTX into
#     machine code for the target GPU, which reports per kernel the registers, shared
#     memory and register spills that ptxas needs. Still no GPU or driver needed, so this
#     also checks targets you don't have, e.g. CAP=8.0 (A100) or CAP=8.6 (RTX 30x0, A10).
#
# Usage, from the repository root:
#
#     julia --project test/compile-only.jl
#
# Environment variables: CAP=8.9 (target compute capability), QUICK=1 (fewer cases),
# PTXAS=path to ptxas (assemble the generated PTX for sm_CAP).

using CUDA, GemmKernels
using GemmKernels: Layout, Operator, Kernel, CTASwizzle, Transform, Epilogue, Tensors
const GPUCompiler = CUDA.GPUCompiler

const CAP = VersionNumber(get(ENV, "CAP", "8.9"))
const QUICK = get(ENV, "QUICK", "0") == "1"
const PTXAS = get(ENV, "PTXAS", "")
# opt-in shared memory per block, by compute capability
const MAX_SHMEM = CAP == v"8.0" ? 166912 : CAP == v"8.7" ? 166912 : CAP >= v"9.0" ? 232448 :
                  CAP >= v"8.6" ? 101376 : CAP == v"7.5" ? 65536 : 98304

GemmKernels.OVERRIDE_cap[] = CAP
GemmKernels.OVERRIDE_max_shmem[] = MAX_SHMEM

failures = String[]
warnings = String[]
function check(f, label)
    try
        result = f()
        println("  ok      $label", result === nothing ? "" : "  ($result)")
    catch err
        msg = sprint(showerror, err)
        # Offline, GPUCompiler's exception runtime is not linked, so code that can throw
        # (e.g. a division by a run-time value) shows up as calls to unknown gpu_*
        # functions. That compiles on a real system, but is worth knowing about.
        reasons = [m.captures[1] for m in eachmatch(r"Reason: (.*)", msg)]
        if isa(err, GPUCompiler.InvalidIRError) && !HAVE_CUDA_RUNTIME && !isempty(reasons) &&
           all(r -> occursin("(call to gpu_", r), reasons)
            # the caller of the throwing function (frame [2]), e.g. cta_swizzle rather than div
            where_ = unique([m.captures[1] * " (" * m.captures[2] * ")"
                             for m in eachmatch(r"\n\s*\[2\] ([^\n]*)\n\s*@ ([^\n]+)", msg)])
            println("  ok*     $label  (contains an exception path, at: $(join(where_, "; ")))")
            push!(warnings, label)
            return
        end
        println("  FAILED  $label\n", join(map(l -> "          " * l, first(split(msg, '\n'), 30)), "\n"))
        push!(failures, label)
    end
end

################################################################################
println("== Host-side checks")

check("CTA swizzles: one block per tile") do
    block_shape = (M = 16, N = 32, K = 8)
    for swizzle in [CTASwizzle.Identity, CTASwizzle.HorizontallyTiled{4}, CTASwizzle.HorizontallyTiled{16},
                    CTASwizzle.VerticallyTiled{4}, CTASwizzle.VerticallyTiled{16}],
        rows in 1:23, cols in 1:23
        matmul_shape = (M = rows * block_shape.M, N = cols * block_shape.N, K = 64)
        count = zeros(Int, rows, cols)
        BX, BY = CTASwizzle.number_of_blocks(swizzle, block_shape, matmul_shape)
        for by in 1:BY, bx in 1:BX
            i, j = CTASwizzle.cta_swizzle(swizzle, (x = bx, y = by), block_shape, matmul_shape)
            count[i ÷ block_shape.M + 1, j ÷ block_shape.N + 1] += 1
        end
        all(==(1), count) || error("$swizzle on $rows x $cols tiles is not a bijection")
    end
end

check("async-copy layout support") do
    F16 = Layout.UnsafeAlignedColMajor{Float16}
    P16 = Layout.Padded{Layout.UnsafeAlignedColMajor{Float16}, 8}
    Layout.can_copy_async(F16, P16, (M = 8, K = 1)) || error("16-byte copy rejected")
    Layout.can_copy_async(F16, P16, (M = 1, K = 1)) && error("2-byte copy accepted")
    Layout.can_copy_async(Layout.UnsafeAlignedColMajor{Float32}, P16, (M = 4, K = 1)) && error("type conversion accepted")
    nothing
end

################################################################################
# GPU code generation, mirroring plan_matmul up to (but excluding) ptxas.

devarray(T, dims...) = CuDeviceArray{T, length(dims), CUDA.AS.Global}(
    reinterpret(Core.LLVMPtr{T, CUDA.AS.Global}, C_NULL), Tuple(dims))

# CUDA.compiler_config asks the CUDA runtime library which PTX ISA versions the toolkit
# supports, which fails without a CUDA installation (e.g. on macOS: "libcudart not
# defined"). In that case, build the same configuration ourselves, choosing the PTX ISA
# from what LLVM supports (PTX 7.8 is needed for sm_89; cp.async needs PTX 7.0).
const HAVE_CUDA_RUNTIME = try
    CUDA.runtime_version()
    true
catch
    false
end

function offline_compiler_config(; maxthreads)
    HAVE_CUDA_RUNTIME && return CUDA.compiler_config(nothing; cap = CAP, maxthreads)

    llvm_support = CUDA.llvm_compat()
    ptx = v"7.8" in llvm_support.ptx ? v"7.8" : maximum(llvm_support.ptx)
    llvm_cap = maximum(filter(<=(CAP), llvm_support.cap))
    target = GPUCompiler.PTXCompilerTarget(; cap = llvm_cap, ptx, maxthreads)
    params = CUDA.CUDACompilerParams(; cap = CAP, ptx)
    return GPUCompiler.CompilerConfig(target, params; kernel = true, name = nothing, always_inline = false)
end

function codegen(conf, a, b, c, d; kernel,
                 transforms = ntuple(_ -> Transform.Elementwise(), 8), epilogue = Epilogue.Default())
    # the same host-side checks as plan_matmul
    GemmKernels.check_kernel(conf, kernel, CAP, (; global_to_shared_a = transforms[1],
                                                   global_to_shared_b = transforms[2]))
    shmem = Kernel.shmem_size(conf, kernel)
    shmem ≤ MAX_SHMEM || throw(GemmKernels.ConfigError("needs $shmem bytes of shared memory"))

    kernel_args = map(cudaconvert, (conf, a, b, c, d, transforms..., epilogue))
    source = GPUCompiler.methodinstance(typeof(kernel), Tuple{map(Core.Typeof, kernel_args)...})
    config = offline_compiler_config(maxthreads = conf.warps_per_block * 32)
    job = GPUCompiler.CompilerJob(source, config)
    # Without a CUDA installation, libdevice (NVIDIA's math library, linked when kernels
    # call e.g. __nv_* functions) is unavailable; skip linking it. Type inference and IR
    # validation happen before that, and the calls stay external declarations in the PTX.
    asm = GPUCompiler.JuliaContext() do ctx
        first(GPUCompiler.compile(:asm, job; libraries = HAVE_CUDA_RUNTIME))
    end
    # sanity checks on the generated PTX
    if kernel isa Kernel.MatmulPipelinedAsync
        occursin("cp.async", asm) || error("no cp.async instructions in the generated PTX")
    end
    summary = "$(count(==('\n'), asm)) lines of PTX, $(shmem) bytes of dynamic shared memory"
    isempty(PTXAS) && return summary
    return summary * ", " * assemble(asm)
end

# Assemble PTX into machine code with ptxas, as relocatable code (-c: the exception
# variables are resolved when CUDA.jl links the kernel), and report what ptxas -v prints.
function assemble(asm)
    mktempdir() do dir
        ptx = joinpath(dir, "kernel.ptx")
        write(ptx, asm)
        out = IOBuffer()
        cmd = `$PTXAS -c -v -arch=sm_$(CAP.major)$(CAP.minor) -o $(joinpath(dir, "kernel.o")) $ptx`
        ok = success(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
        log = String(take!(out))
        ok || error("ptxas failed:\n" * log)
        # with -c, ptxas also reports on non-kernel functions; use the kernel's section
        i = findfirst("Compiling entry function", log)
        i === nothing || (log = log[first(i):end])
        regs = match(r"Used (\d+) registers", log)
        spills = match(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
        smem = match(r"(\d+) bytes smem", log)
        "sm_$(CAP.major)$(CAP.minor): $(regs === nothing ? "?" : regs[1]) registers" *
            (spills === nothing ? "" : ", spills $(spills[1])/$(spills[2]) bytes stored/loaded") *
            (smem === nothing ? "" : ", $(smem[1]) bytes static shared memory")
    end
end

const KERNELS = [
    "singlestage" => Kernel.matmul_singlestage,
    "pipelined" => Kernel.matmul_pipelined,
    "pipelined_ng" => Kernel.matmul_pipelined_ng,
    "async_2" => Kernel.matmul_pipelined_async(2),
    "async_3" => Kernel.matmul_pipelined_async(3),
    "async_4" => Kernel.matmul_pipelined_async(4),
]

println("== GPU code generation for sm_$(CAP.major)$(CAP.minor) (no driver needed",
        HAVE_CUDA_RUNTIME ? ")" : "; no CUDA runtime found: building the compiler configuration manually, not linking libdevice)")
isempty(PTXAS) || println("   assembling with $PTXAS (", last(split(readchomp(`$PTXAS --version`), '\n')), ")")

# Plain FP16 GEMM, all transpose combinations.
for transpose_a in [false, true], transpose_b in [false, true],
        swizzle in (QUICK ? [CTASwizzle.Identity] : [CTASwizzle.Identity, CTASwizzle.HorizontallyTiled{4}]),
        (kname, kernel) in KERNELS
    (transpose_a || transpose_b) && swizzle != CTASwizzle.Identity && continue
    M, N, K = 512, 256, 256
    layout_a = transpose_a ? Layout.UnsafeAlignedRowMajor{Float16} : Layout.UnsafeAlignedColMajor{Float16}
    layout_b = transpose_b ? Layout.UnsafeAlignedRowMajor{Float16} : Layout.UnsafeAlignedColMajor{Float16}
    label = "GEMM $(transpose_a ? 'T' : 'N')$(transpose_b ? 'T' : 'N'), $kname, $(nameof(swizzle))"
    check(label) do
        conf = GemmKernels.get_config(
            gemm_shape = (; M, N, K),
            block_shape = (M = 128, N = 64, K = 32),
            warps_per_block = 4,
            compute_warp = (M = 64, N = 32, K = 16),
            global_a_layout = layout_a, global_b_layout = layout_b,
            global_c_layout = Layout.UnsafeAlignedColMajor{Float32},
            global_d_layout = Layout.UnsafeAlignedColMajor{Float32},
            shared_a_layout = Layout.Padded{layout_a, 8}, shared_b_layout = Layout.Padded{layout_b, 8},
            shared_c_layout = Layout.UnsafeAlignedColMajor{Float32},
            shared_d_layout = Layout.UnsafeAlignedColMajor{Float32},
            operator = Operator.WMMAOp{16, 16, 16, Float16, Float32},
            is_a_col_major = !transpose_a, is_b_col_major = !transpose_b,
            cta_swizzle = swizzle)
        a = devarray(Float16, (transpose_a ? (K, M) : (M, K))...)
        b = devarray(Float16, (transpose_b ? (N, K) : (K, N))...)
        c = devarray(Float32, M, N)
        d = devarray(Float32, M, N)
        codegen(conf, a, b, c, d; kernel)
    end
end

# Tensor contraction abc-bda-dc (TC1, with padded extents), via the GETT plan, for a
# non-strided (PERM_M = [2, 1]) and a strided (PERM_M = [1, 2]) layout of A. The async
# kernels must reject the strided one with a ConfigError.
let modes = ([1, 2, 3], [2, 4, 1], [4, 3]), extents = [320, 312, 32, 320]
    packed(ext) = cumprod([1; ext[1:end-1]])
    desc_for(mode, T) = Tensors.TensorDescriptor(length(mode), extents[mode], packed(extents[mode]), T, identity)
    descD = desc_for(modes[1], Float16)
    desc = Tensors.ContractionDescriptor(desc_for(modes[2], Float16), modes[2],
                                         desc_for(modes[3], Float16), modes[3],
                                         descD, modes[1], descD, modes[1], Float16, Float32)
    for perm_m in [[2, 1], [1, 2]], (kname, kernel) in KERNELS
        QUICK && kname in ("pipelined", "async_2", "async_4") && continue
        strided = perm_m == [1, 2]
        label = "TC abc-bda-dc, PERM_M = $perm_m, $kname"
        check(label) do
            Tensors.OVERRIDE_do_override = true
            Tensors.OVERRIDE_is_A_col_major = true
            Tensors.OVERRIDE_is_B_col_major = true
            Tensors.OVERRIDE_is_D_col_major = true
            Tensors.OVERRIDE_perm_M = perm_m
            Tensors.OVERRIDE_perm_N = [3]
            Tensors.OVERRIDE_perm_K = [4]
            Tensors.OVERRIDE_cta_swizzle = CTASwizzle.HorizontallyTiled{4}
            plan = try
                Tensors.setUpGETTKernel(desc, Operator.WMMAOp{32, 8, 16, Float16, Float32},
                                        (M = 128, N = 16, K = 32), 2, (M = 64, N = 16, K = 16), 2, 0)
            finally
                Tensors.OVERRIDE_do_override = false
            end
            a = devarray(Float16, extents[modes[2]]...)
            b = devarray(Float16, extents[modes[3]]...)
            d = devarray(Float16, extents[modes[1]]...)
            if strided && kernel isa Kernel.MatmulPipelinedAsync
                try
                    codegen(plan.gemmConf, a, b, d, d; kernel)
                catch err
                    isa(err, GemmKernels.ConfigError) && return "rejected as expected"
                    rethrow()
                end
                error("strided layout was not rejected")
            end
            codegen(plan.gemmConf, a, b, d, d; kernel)
        end
    end
end

println()
isempty(warnings) || println("$(length(warnings)) kernel(s) contain exception paths (ok*); these compile, but cost a branch.")
if isempty(failures)
    println("All checks passed.")
else
    println("$(length(failures)) check(s) failed:\n  ", join(failures, "\n  "))
    exit(1)
end
