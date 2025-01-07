using Arrow
using CUDA, GemmKernels
using DataFrames
using Serialization

isinteractive() || include("tuning/wmma-contraction.jl")

function main()
    # Read configurations from disk
    config_path = joinpath(@__DIR__, "tuning", "configs.arrow")
    all_configs = copy(DataFrame(Arrow.Table(config_path)))

    # we only care about successful configurations
    all_configs = all_configs[all_configs[!, "status"] .== "success", :]
    select!(all_configs, Not(:status))

    problems = generate_problems()

    problem_id = parse(Int, ENV["GK_PROBLEM_ID"])
    problems = problems[problem_id:problem_id]

    # gather the best configurations for each problem
    candidate_configs = similar(all_configs, 0)
    for problem in problems
        configs = select_configs(all_configs, problem)
        configs ===  nothing && continue
        append!(candidate_configs, first(sort(configs, :time), 1))
    end

    for problem in problems
        # Run baseline.
        data = allocate_data(problem)
        args = prepare_baseline(problem, data...)
        CUDA.@profile external=true execute_baseline(problem, data...; args...)

        # Run GemmKernels.
        for config in eachrow(select_configs(candidate_configs, problem))
            params = create_params(config)
            args = prepare(problem, data...; params...)
            CUDA.@profile external=true execute(problem, data...; args...)
        end
    end
end

isinteractive() || main()
