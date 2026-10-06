# Benchmark of LShaped.jl against the L-shaped method of StochasticPrograms.jl.
#
# Both libraries solve the same generated instances, with the same LP solver (HiGHS) for the master
# and the recourse problems, in their single-cut and multicut versions; the extensive form, solved
# directly by HiGHS, gives the reference value every result is checked against.
#
#     julia --project=@v1.12 LShaped_benchmark.jl            # all instances, best of 3 runs
#     julia -t 6 --project=@v1.12 LShaped_benchmark.jl       # adds LShaped.jl with 6 threads
#     julia --project=@v1.12 LShaped_benchmark.jl quick      # small instances, one run
#     julia --project=@v1.12 LShaped_benchmark.jl reps=5 csv=results.csv
#     julia --project=@v1.12 LShaped_benchmark.jl quick rd   # adds regularized decomposition
#
# Needs, besides the packages of LShaped.jl (JuMP, HiGHS, Distributions, RandomDataStreams),
# StochasticPrograms.jl from https://github.com/fbastin/StochasticPrograms.jl, whose fixes the
# comparison relies on: the registered release returns wrong L-shaped solutions.
#
# What is measured, per instance and method:
#   - the objective and its relative error with respect to the extensive form;
#   - whether the method converged (StochasticPrograms: its termination status);
#   - the iterations and the cuts (optimality and feasibility) it added: both multicut versions add
#     only the violated cuts after the first round;
#   - the wall-clock time, the best of `reps` runs after a warm-up that compiles everything, building
#     the models included (LShaped.jl builds them inside `lshaped`, StochasticPrograms in
#     `instantiate`), and the memory allocated by one run.
# The tolerances differ in their definitions: LShaped.jl stops when θ ≥ Q - tol (1 + |Q|), here with
# tol = 1e-6, StochasticPrograms when |θ - Q| / |Q| ≤ τ, its default 1e-6. The relative errors show
# what either achieves.

module Ours
include(joinpath(@__DIR__, "LShaped.jl"))
end

using .Ours.LShaped: TwoStageProblem, lshaped, extensive_form, substream, read_smps
using StochasticPrograms
using HiGHS
using Ipopt
using Distributions
using LinearAlgebra
using Printf
using Statistics

include(joinpath(@__DIR__, "LShaped_instances.jl"))

const SPL = StochasticPrograms.LShaped
const MOI = StochasticPrograms.MOI

# --------------------------------------------------------------------------------------------
# instances: the same data, written for each library
# --------------------------------------------------------------------------------------------

struct Instance
    name::String
    ours::TwoStageProblem
    sp_model::StochasticModel
    sp_scenarios::Vector
end

"""
    farmer(S; seed)

The farmer of `farmer_problem` (in `LShaped_instances.jl`), written for both libraries.
"""
function farmer(S::Integer; seed::Integer = 1)
    ours, yields = farmer_problem(S; seed = seed)
    model = @stochastic_model begin
        @stage 1 begin
            @decision(model, x[1:3] >= 0)
            @constraint(model, sum(x) <= 500)
            @objective(model, Min, 150x[1] + 230x[2] + 260x[3])
        end
        @stage 2 begin
            @known(model, x)
            @uncertain t[1:3]
            @recourse(model, w[1:2] >= 0)
            @recourse(model, y[1:4] >= 0)
            @constraint(model, t[1] * x[1] + w[1] - y[1] >= 200)
            @constraint(model, t[2] * x[2] + w[2] - y[2] >= 240)
            @constraint(model, y[3] + y[4] <= t[3] * x[3])
            @constraint(model, y[3] <= 6000)
            @objective(model, Min, 238w[1] + 210w[2] - 170y[1] - 150y[2] - 36y[3] - 10y[4])
        end
    end
    scenarios = [@scenario(t[1:3] = yields[s], probability = 1 / S) for s in 1:S]
    return Instance("farmer, S = $S", ours, model, scenarios)
end

"""
    capacity(P, F, S; shortage = true, seed)

The capacity expansion of `capacity_problem` (in `LShaped_instances.jl`), written for both
libraries.
"""
function capacity(P::Integer, F::Integer, S::Integer; shortage::Bool = true, seed::Integer = 2)
    ours, c, a, demands, penalty, total = capacity_problem(P, F, S; shortage = shortage, seed = seed)
    model = if shortage
        @stochastic_model begin
            @stage 1 begin
                @parameters begin
                    P = P
                    c = c
                    total = total
                end
                @decision(model, x[i in 1:P] >= 0)
                @constraint(model, sum(x) <= total)
                @objective(model, Min, sum(c[i] * x[i] for i in 1:P))
            end
            @stage 2 begin
                @parameters begin
                    P = P
                    F = F
                    a = a
                    penalty = penalty
                end
                @known(model, x)
                @uncertain d[1:F]
                @recourse(model, y[i in 1:P, j in 1:F] >= 0)
                @recourse(model, u[j in 1:F] >= 0)
                @constraint(model, [i in 1:P], sum(y[i, j] for j in 1:F) <= x[i])
                @constraint(model, [j in 1:F], sum(y[i, j] for i in 1:P) + u[j] >= d[j])
                @objective(model, Min, sum(a[i, j] * y[i, j] for i in 1:P, j in 1:F) +
                                       penalty * sum(u[j] for j in 1:F))
            end
        end
    else
        @stochastic_model begin
            @stage 1 begin
                @parameters begin
                    P = P
                    c = c
                    total = total
                end
                @decision(model, x[i in 1:P] >= 0)
                @constraint(model, sum(x) <= total)
                @objective(model, Min, sum(c[i] * x[i] for i in 1:P))
            end
            @stage 2 begin
                @parameters begin
                    P = P
                    F = F
                    a = a
                end
                @known(model, x)
                @uncertain d[1:F]
                @recourse(model, y[i in 1:P, j in 1:F] >= 0)
                @constraint(model, [i in 1:P], sum(y[i, j] for j in 1:F) <= x[i])
                @constraint(model, [j in 1:F], sum(y[i, j] for i in 1:P) >= d[j])
                @objective(model, Min, sum(a[i, j] * y[i, j] for i in 1:P, j in 1:F))
            end
        end
    end
    scenarios = [@scenario(d[1:F] = demands[s], probability = 1 / S) for s in 1:S]
    kind = shortage ? "capacity" : "capacity, no shortage"
    return Instance("$kind, P = $P, F = $F, S = $S", ours, model, scenarios)
end

# --------------------------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------------------------

"""The outcome of a method on an instance, as reported in the tables."""
struct Outcome
    method::String
    status::String
    objective::Float64
    iterations::Int
    cuts::Int
end

function extensive(inst::Instance)
    _, _, obj = extensive_form(inst.ours; optimizer = HiGHS.Optimizer)
    return Outcome("extensive form (HiGHS)", "OPTIMAL", obj, 0, 0)
end

# HiGHS's QP solver fails on the masters of regularized decomposition: Ipopt solves them, without
# relaxing the constraints
const QP_MASTER = optimizer_with_attributes(Ipopt.Optimizer, "bound_relax_factor" => 0.0, "sb" => "yes")

function ours(inst::Instance, cuts; threads::Bool = false, drop_inactive = nothing,
              regularization::Symbol = :none, x0 = nothing)
    master = regularization == :regularized_decomposition ? QP_MASTER : HiGHS.Optimizer
    res = lshaped(inst.ours; master_optimizer = master, recourse_optimizer = HiGHS.Optimizer,
                  cuts = cuts, tol = 1e-6,
                  maxiter = 10_000, threads = threads, drop_inactive = drop_inactive,
                  regularization = regularization, x0 = x0, verbose = false)
    label = "LShaped.jl, " * (cuts isa Integer ? "$cuts clusters" : "$(cuts)-cut") *
            (threads ? ", $(Threads.nthreads()) threads" : "") *
            (drop_inactive === nothing ? "" : ", drop $drop_inactive") *
            (regularization == :regularized_decomposition ? ", RD (Ipopt)" :
             regularization == :trust_region ? ", trust region" : "") *
            (x0 === :mean_value ? ", from x̄" : "")
    return Outcome(label, res.converged ? "converged" : "not converged",
                   res.objective, res.iterations, res.optimality_cuts + res.feasibility_cuts)
end

function theirs(inst::Instance, cuts::Symbol)
    sp = instantiate(inst.sp_model, inst.sp_scenarios, optimizer = SPL.Optimizer)
    set_silent(sp)
    set_optimizer_attribute(sp, MasterOptimizer(), HiGHS.Optimizer)
    set_optimizer_attribute(sp, SubProblemOptimizer(), HiGHS.Optimizer)
    # feasibility cuts are off by default, and StochasticPrograms first evaluates the subproblems
    # at x = 0, where the instances without complete recourse are infeasible
    set_optimizer_attribute(sp, SPL.FeasibilityStrategy(), SPL.FeasibilityCuts())
    cuts == :single && set_optimizer_attribute(sp, SPL.Aggregator(), SPL.Aggregate())
    optimize!(sp)
    status = termination_status(sp)
    lshaped_algorithm = StochasticPrograms.optimizer(sp).lshaped
    return Outcome("StochasticPrograms, $(cuts)-cut", string(status),
                   status == MOI.OPTIMAL ? objective_value(sp) : NaN,
                   StochasticPrograms.num_iterations(StochasticPrograms.optimizer(sp)),
                   lshaped_algorithm.data.num_cuts)
end

# the parallel variant only when Julia has several threads (`julia -t N`); regularized
# decomposition only on demand (`rd`), being much slower: see `RD_METHODS`
const METHODS = [extensive,
                 inst -> ours(inst, :single), inst -> ours(inst, :single; drop_inactive = 5),
                 inst -> ours(inst, :single; x0 = :mean_value),
                 inst -> ours(inst, :single; regularization = :trust_region),
                 inst -> ours(inst, min(10, Ours.LShaped.n_scenarios(inst.ours))),
                 inst -> ours(inst, :multi),
                 inst -> ours(inst, :multi; regularization = :trust_region),
                 (Threads.nthreads() > 1 ? [inst -> ours(inst, :multi; threads = true)] : [])...,
                 inst -> theirs(inst, :single), inst -> theirs(inst, :multi)]

# Regularized decomposition with the fixed `rho = 1` of deck 04 takes small steps on instances whose
# decisions are in the hundreds, and its quadratic masters need Ipopt: hours on the full benchmark.
const RD_METHODS = [inst -> ours(inst, :single; regularization = :regularized_decomposition),
                    inst -> ours(inst, :multi; regularization = :regularized_decomposition)]

# --------------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------------

struct Measure
    instance::String
    outcome::Outcome
    error::Float64          # relative to the extensive form
    time::Float64           # best of the runs, seconds
    memory::Float64         # allocated by one run, MB
end

function measure(inst::Instance, method, reference, reps::Integer)
    best, outcome, bytes = Inf, nothing, 0
    for _ in 1:reps
        GC.gc()
        stats = @timed method(inst)
        if stats.time < best
            best, outcome, bytes = stats.time, stats.value, stats.bytes
        end
    end
    error = abs(outcome.objective - reference) / max(1.0, abs(reference))
    return Measure(inst.name, outcome, error, best, bytes / 2^20)
end

function report(io::IO, measures::Vector{Measure})
    @printf(io, "  %-40s %-14s %16s %10s %7s %7s %10s %10s\n", "method", "status", "objective",
            "rel. error", "iter.", "cuts", "time (s)", "alloc (MB)")
    for m in measures
        o = m.outcome
        @printf(io, "  %-40s %-14s %16.4f %10.1e %7s %7s %10.3f %10.1f\n", o.method, o.status,
                o.objective, m.error, o.iterations == 0 ? "-" : string(o.iterations),
                o.cuts == 0 ? "-" : string(o.cuts), m.time, m.memory)
    end
end

function write_csv(path::AbstractString, measures::Vector{Measure})
    open(path, "w") do io
        println(io, "instance,method,status,objective,relative_error,iterations,cuts,time_s,alloc_MB")
        for m in measures
            o = m.outcome
            println(io, join(("\"$(m.instance)\"", "\"$(o.method)\"", o.status, o.objective,
                              m.error, o.iterations, o.cuts, m.time, m.memory), ","))
        end
    end
end

# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

function main(args)
    quick = "quick" in args
    methods = "rd" in args ? [METHODS[1:end-2]; RD_METHODS; METHODS[end-1:end]] : METHODS
    function option(name, default)
        for a in args
            startswith(a, name * "=") && return String(split(a, "="; limit = 2)[2])
        end
        return default
    end
    reps = parse(Int, option("reps", quick ? "1" : "3"))
    csv = option("csv", "")
    instances = quick ?
        [() -> farmer(50), () -> capacity(5, 5, 20), () -> capacity(5, 5, 20; shortage = false)] :
        [() -> farmer(100), () -> farmer(1000),
         () -> capacity(10, 10, 100), () -> capacity(20, 20, 500),
         () -> capacity(10, 10, 100; shortage = false), () -> capacity(20, 20, 200; shortage = false)]

    # compile everything on a tiny instance first, so that no timing includes it
    for warmup in (farmer(5), capacity(2, 2, 3), capacity(2, 2, 3; shortage = false))
        foreach(method -> method(warmup), methods)
    end

    println("LShaped.jl vs StochasticPrograms.jl, HiGHS.jl ", pkgversion(HiGHS), ", ",
            Threads.nthreads(), " thread(s), best of ", reps, reps == 1 ? " run\n" : " runs\n")
    results = Measure[]
    for make in instances
        inst = make()
        reference = extensive(inst).objective
        measures = [measure(inst, method, reference, reps) for method in methods]
        println(inst.name)
        report(stdout, measures)
        println()
        flush(stdout)       # show each instance as soon as it is done, even when redirected
        append!(results, measures)
    end
    isempty(csv) || (write_csv(csv, results); println("written to ", csv))
    return results
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
