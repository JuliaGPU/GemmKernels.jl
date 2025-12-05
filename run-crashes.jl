using Arrow
using CUDA, GemmKernels
using DataFrames
using Serialization
using CUDA_SDK_jll

isinteractive() || include("tuning/wmma-contraction.jl")


function main()
    # Read configurations from disk
    config_path = "/home/thomas/configs.arrow"
    all_configs = copy(DataFrame(Arrow.Table(config_path)))

    # we only care about crashed configurations
    all_configs = all_configs[all_configs[!, "status"] .== "crashed_during_measure", :]
    select!(all_configs, Not(:status))

    problems = generate_problems()

    candidate_configs = similar(all_configs, 0)

    for problem in problems
        configs = select_configs(all_configs, problem)
        configs === nothing && continue
        append!(candidate_configs, configs)
    end

    println("Identified $(size(candidate_configs, 1)) crashed configs.")
    configs_finished = 0

    for problem in problems
        data = allocate_data(problem)

        for config in eachrow(select_configs(candidate_configs, problem))
            idx = configs_finished+1

            println("Trying configuration $(idx)/$(size(candidate_configs, 1)): $(repr_row(config))")

            params = create_params(config)
            try
                args = prepare(problem, data...; params...)
            catch ConfigError
                continue
            end

            execute(problem, data...; args...)

            CUDA.synchronize()

            println("Finished configuration $(idx)/$(size(candidate_configs, 1))")

            configs_finished += 1
        end
    end
end

function run_sanitizer()
    compute_sanitizer = joinpath(CUDA_SDK_jll.artifact_dir, "cuda/compute-sanitizer/compute-sanitizer")
    options = ["--launch-timeout=0", "--target-processes=all", "--report-api-errors=no"]
    julia_options = ["-g2", "--check-bounds=yes", "--project=tuning", "./run-crashes.jl", "RUN"]

    if "NOSAN" in ARGS
        run(`$(Base.julia_cmd()) $julia_options`)
    else
        run(`$compute_sanitizer $options $(Base.julia_cmd()) $julia_options`)
    end
end

if "RUN" in ARGS
    isinteractive() || main()
else
    isinteractive() || run_sanitizer()
end
