function find_sources(path::String, sources=String[])
    if isdir(path)
        for entry in readdir(path)
            find_sources(joinpath(path, entry), sources)
        end
    elseif endswith(path, ".jl")
        push!(sources, path)
    end
    sources
end

function julia_exec(args::Cmd, env...)
    cmd = Base.julia_cmd()
    cmd = `$cmd --project=$(Base.active_project()) --color=no $args`

    out = Pipe()
    err = Pipe()
    proc = run(pipeline(addenv(cmd, env...), stdout=out, stderr=err), wait=false)
    close(out.in)
    close(err.in)
    wait(proc)
    proc, read(out, String), read(err, String)
end

@testset "Examples" begin
    dir = joinpath(@__DIR__, "..", "examples")
    paths = find_sources(dir)
    examples = relpath.(paths, Ref(dir))

    @testcase "$(splitext(example)[1])" for example in examples
        cd(dir) do
            proc, out, err = julia_exec(`$example`)
            isempty(err) || println(err)
            @test success(proc)
        end
    end
end
