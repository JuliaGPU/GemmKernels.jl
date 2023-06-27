using CUDA

function matmul(a, b, c, d, conf;
                transform_global_to_shared_a = Transform.Elementwise(),
                transform_global_to_shared_b = Transform.Elementwise(),
                transform_global_to_shared_c = Transform.Elementwise(),
                transform_shared_to_global_d = Transform.Elementwise(),
                transform_shared_to_regs_a = Transform.Elementwise(),
                transform_shared_to_regs_b = Transform.Elementwise(),
                transform_shared_to_regs_c = Transform.Elementwise(),
                transform_regs_to_shared_d = Transform.Elementwise(),
                epilogue = Epilogue.Default(),
                kernel = Kernel.matmul_singlestage)

    args = [a, b, c, d,
            transform_global_to_shared_a, transform_global_to_shared_b, transform_global_to_shared_c, transform_shared_to_global_d,
            transform_shared_to_regs_a, transform_shared_to_regs_b, transform_shared_to_regs_c, transform_regs_to_shared_d,
            epilogue,
            conf]

    max_shmem = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    if conf.launch_args.shmem > max_shmem
        error("Requested too much shared memory: The current GPU can use at most $(Base.format_bytes(max_shmem)), while this configuration required $(Base.format_bytes(conf.launch_args.shmem))")
    end

    hostkernel = @cuda launch=false kernel(args...)
    attributes(hostkernel.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = conf.launch_args.shmem
    hostkernel(args...; conf.launch_args...)
end
