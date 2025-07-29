#!/usr/bin/env julia

function get_processes()
    try
        pids = parse.(Int, split(read(`pgrep -f "julia.*--worker"`, String)))

        union(map(pid -> (chopprefix(first(filter(startswith("GEMMKERNELS_WORKER_TAG="), split(read(open("/proc/$pid/environ"), String), "\0"))), "GEMMKERNELS_WORKER_TAG="), pid), pids), [("main", parse(Int, read(`pgrep -f "julia.*tune.jl"`, String)))])
    catch ex
        @show ex
        []
    end
end

get_memory_usage(pid) = parse(Int, split(first(filter(startswith("Rss:"), readlines("/proc/$pid/smaps_rollup"))))[2]) * 1024

HWMs = Dict()

function main()
    while true
        run(`clear`)

        processes_by_type = Dict()

        for (type, pid) in get_processes()
            if !(type in keys(processes_by_type))
                processes_by_type[type] = []
            end

            push!(processes_by_type[type], pid)
        end

        for (type, processes) in processes_by_type
            println("$(length(processes)) processes of type '$type':")

            hwm_total, res_total = 0, 0

            for pid in sort(processes)
                res = get_memory_usage(pid)
                hwm = HWMs[pid] = max(get(HWMs, pid, 0), res)

                res_total += res
                hwm_total += hwm

                println("\tPID $pid: Rss: $(Base.format_bytes(res)), HWM: $(Base.format_bytes(hwm))")
            end

            println()
            println("Total:      Rss: $(Base.format_bytes(res_total)), HWM: $(Base.format_bytes(hwm_total))")
            println("Total/proc: Rss: $(Base.format_bytes(res_total / length(processes))), HWM: $(Base.format_bytes(hwm_total / length(processes)))")
            println()
        end

        sleep(5)
    end
end

isinteractive() || main()
