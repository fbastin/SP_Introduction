# When does decomposition pay off? The extensive form, solved directly by HiGHS, against the
# multicut L-shaped method of LShaped.jl, at equal numbers of cores, as the number of scenarios grows.
#
#     julia --project=@v1.12 LShaped_scaling.jl                    # all families, 1 and 6 cores
#     julia --project=@v1.12 LShaped_scaling.jl quick              # the smallest sizes only
#     julia --project=@v1.12 LShaped_scaling.jl cores=1,6 limit=1200 memory=12000 csv=scaling.csv
#     julia --project=@v1.12 LShaped_scaling.jl families=storm,capacity methods=extensive,multicut
#     julia --project=@v1.12 LShaped_scaling.jl families=capacity sizes=8000 cores=6
#
# Every measure runs in a Julia process of its own, started with `-t cores`:
#   - HiGHS has one task scheduler per process, whose number of threads is fixed by the first model
#     solved (option `threads`): the extensive form gets `threads = cores`, each recourse problem of
#     the decomposition `threads = 1`, the `cores` of them being solved in parallel on Julia's
#     threads. A process per measure is the only way to fix both;
#   - the memory is the peak resident set of the process during the method, polled from
#     `/proc/<pid>/status` by this script, less the resident set before it: it counts what HiGHS
#     allocates, which Julia's allocation counters do not see (Linux only);
#   - the method is first run on a tiny instance of the same family, so that compiling is not
#     timed; the instance itself is built before the clock starts.
# The methods: the extensive form with HiGHS's default LP method (dual simplex) and with its
# interior-point method (`solver = "ipm"`, the one that uses several threads), and the multicut
# L-shaped method (`tol = 1e-6`). A method that exceeds the time limit is not run on the larger
# sizes of the family, nor is one whose process exceeds `memory` MB, killed beforehand.
#
# The families: the capacity expansion of `LShaped_instances.jl` with 20 plants and 20 products,
# and samples of the SMPS instances storm and ssn of Linderoth, Shapiro and Wright (2006),
# kept in `smps/` (downloaded again if missing).

module Ours
include(joinpath(@__DIR__, "LShaped.jl"))
end

using .Ours.LShaped: TwoStageProblem, lshaped, extensive_form, substream, read_smps,
                     sample_scenarios
using HiGHS
using JuMP
using Distributions
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "LShaped_instances.jl"))

const FAMILIES = Dict(
    "capacity" => (sizes = [500, 1000, 2000, 4000, 8000], quick = [100, 200], tiny = 3,
                   make = S -> capacity_problem(20, 20, S).problem),
    "storm" => (sizes = [250, 500, 1000, 2000, 4000], quick = [50, 100], tiny = 3,
                make = S -> lsw_sample("storm", S)),
    "ssn" => (sizes = [250, 500, 1000, 2000], quick = [50, 100], tiny = 3,
              make = S -> lsw_sample("ssn", S)))
const ORDER = ["capacity", "storm", "ssn"]
const METHODS = ["extensive", "extensive-ipm", "multicut"]

lsw_sample(name, S) = (smps = lsw_instance(name);
                       TwoStageProblem(smps; sample_scenarios(smps.law, S; rng = substream(1))...))

# --------------------------------------------------------------------------------------------
# the worker: one method, one instance, one process
# --------------------------------------------------------------------------------------------

function run_method(method, pb, cores)
    if method == "extensive"
        _, _, value = extensive_form(pb; optimizer = optimizer_with_attributes(
            HiGHS.Optimizer, "threads" => cores))
        return (objective = value, iterations = 0, status = "OPTIMAL")
    elseif method == "extensive-ipm"
        _, _, value = extensive_form(pb; optimizer = optimizer_with_attributes(
            HiGHS.Optimizer, "threads" => cores, "solver" => "ipm"))
        return (objective = value, iterations = 0, status = "OPTIMAL")
    else
        res = lshaped(pb; optimizer = optimizer_with_attributes(HiGHS.Optimizer, "threads" => 1),
                      cuts = :multi, tol = 1e-6, maxiter = 10_000, threads = cores > 1,
                      verbose = false)
        return (objective = res.objective, iterations = res.iterations,
                status = res.converged ? "converged" : "not converged")
    end
end

resident() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096

function worker(family, S, method, cores)
    f = FAMILIES[family]
    run_method(method, f.make(f.tiny), cores)              # compiles everything
    pb = f.make(S)
    GC.gc()
    println("BASE\t", resident()); flush(stdout)
    t = @elapsed out = run_method(method, pb, cores)
    println("RESULT\t", t, "\t", out.objective, "\t", out.iterations, "\t", out.status)
    flush(stdout)
end

# --------------------------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------------------------

"""The resident set of the process `pid`, in bytes, or `nothing` once it is gone."""
function rss(pid)
    text = try
        read("/proc/$pid/status", String)
    catch
        return nothing
    end
    m = match(r"VmRSS:\s+(\d+) kB", text)
    return m === nothing ? nothing : parse(Int, m.captures[1]) * 1024
end

"""Run one measure in a child process; returns its outcome, or the reason it has none: `:time`,
`:memory`, or `:failed`."""
function measure(family, S, method, cores, limit, memory)
    cmd = `$(Base.julia_cmd()) -t $cores --project=$(Base.active_project()) $(@__FILE__) worker $family $S $method $cores`
    # the stack trace of a worker killed at the limit would only clutter the output
    process = open(pipeline(cmd; stderr = devnull), "r")
    pid = getpid(process)
    lines = Channel{String}(Inf)
    reader = @async (for line in eachline(process); put!(lines, line); end; close(lines))
    base, peak, result, started = nothing, 0, nothing, time()
    while process_running(process)
        r = rss(pid)
        base !== nothing && r !== nothing && (peak = max(peak, r))
        while isready(lines)
            line = take!(lines)
            startswith(line, "BASE") && (base = parse(Int, split(line, '\t')[2]); started = time())
            startswith(line, "RESULT") && (result = split(line, '\t'))
        end
        if base !== nothing && (time() - started > limit || peak > memory * 2^20)
            kill(process)
            wait(process)
            return time() - started > limit ? :time : :memory
        end
        sleep(0.02)
    end
    wait(reader)
    for line in lines
        startswith(line, "RESULT") && (result = split(line, '\t'))
    end
    (result === nothing || base === nothing) && return :failed
    return (time = parse(Float64, result[2]), objective = parse(Float64, result[3]),
            iterations = parse(Int, result[4]), status = result[5],
            memory = max(peak - base, 0) / 2^20)
end

function main(args)
    option(name, default) = (i = findfirst(a -> startswith(a, name * "="), args);
                             i === nothing ? default : String(split(args[i], "="; limit = 2)[2]))
    quick = "quick" in args
    cores = parse.(Int, split(option("cores", "1,6"), ","))
    limit = parse(Float64, option("limit", quick ? "300" : "1800"))
    memory = parse(Float64, option("memory", "12000"))
    families = split(option("families", join(ORDER, ",")), ",")
    methods = split(option("methods", join(METHODS, ",")), ",")
    sizes = option("sizes", "")
    csv = option("csv", "")
    println("Extensive form against multicut L-shaped, HiGHS.jl ", pkgversion(HiGHS),
            ", limits per measure: ", limit, " s, ", memory, " MB\n")
    rows = []
    for family in families, ncores in cores
        f = FAMILIES[family]
        @printf("%s, %d core(s)\n  %8s  %-14s %16s %10s %7s %12s %12s\n", family, ncores, "S",
                "method", "objective", "status", "iter.", "time (s)", "memory (MB)")
        given_up = Set{String}()
        chosen = !isempty(sizes) ? parse.(Int, split(sizes, ",")) : quick ? f.quick : f.sizes
        for S in chosen, method in methods
            method in given_up && continue
            m = measure(family, S, method, ncores, limit, memory)
            if m isa Symbol
                @printf("  %8d  %-14s %16s\n", S, method,
                        m == :time ? "> $(round(Int, limit)) s" :
                        m == :memory ? "> $(round(Int, memory)) MB" : "failed")
                push!(given_up, method)
                push!(rows, (family, ncores, S, method, string(m), NaN, 0, NaN, NaN))
            else
                @printf("  %8d  %-14s %16.4f %10s %7s %12.2f %12.0f\n", S, method, m.objective,
                        m.status, m.iterations == 0 ? "-" : string(m.iterations), m.time, m.memory)
                push!(rows, (family, ncores, S, method, m.status, m.objective, m.iterations,
                             m.time, m.memory))
            end
            flush(stdout)
        end
        println()
    end
    if !isempty(csv)
        open(csv, "w") do io
            println(io, "family,cores,scenarios,method,status,objective,iterations,time_s,memory_MB")
            foreach(r -> println(io, join(r, ",")), rows)
        end
        println("written to ", csv)
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    if !isempty(ARGS) && ARGS[1] == "worker"
        worker(ARGS[2], parse(Int, ARGS[3]), ARGS[4], parse(Int, ARGS[5]))
    else
        main(ARGS)
    end
end
