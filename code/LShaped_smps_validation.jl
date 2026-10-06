# Validation of `read_smps` against published results: the instances of Felt's test set of
# stochastic linear programs, solved on their full support, and those of Linderoth, Shapiro and
# Wright (2006), whose supports are too large and are checked through sampling.
#
#     julia --project=@v1.12 LShaped_smps_validation.jl          # Felt's set, LandS and gbd
#     julia -t 6 --project=@v1.12 LShaped_smps_validation.jl full  # adds 20term, ssn and storm
#
# The instances are kept in `smps/`, and downloaded again if missing. What is checked:
#   - Felt's instances (https://www4.uwsp.edu/math/afelt/slptestset/download.html): the optimal
#     value, by the extensive form and by the multicut L-shaped method, against the value published
#     with the instance — its `solution.txt` or `env.soln.*` file, or its documentation;
#   - LandS and gbd: the cost of the solution Linderoth, Shapiro and Wright give (their `.sol`
#     files), estimated on 100 000 scenarios drawn from the law read, against their estimate of the
#     optimal value;
#   - with `full`, 20term, ssn and storm: the sample average approximation on 500 scenarios, and the
#     cost of its solution on 2000 others, against their estimate of the optimal value.
#
# Known differences, explained or not, are stated in the output rather than hidden:
#   - environ: the solutions shipped were computed with the original core file, `env.cor.diss`,
#     before `env.cor` was revised (see its ChangeLog), which is the one used here. For the
#     instances with a BLOCKS section, imp and lrge, the published values also follow a quirk of
#     the solver that produced them, CPA: an entry of value 0 in a block is lost, and the element
#     keeps the value of the first outcome of the block. The script solves them both ways; read
#     as the SMPS format says, they are 1.1% lower.
#   - assets.small: -723.84 against -720.47 published, with the same optimal decision: unexplained.
#   - cargo: its stoch files were revised (December 2001) after the published solutions (2000),
#     which no longer apply; it is solved without a reference.

module Ours
include(joinpath(@__DIR__, "LShaped.jl"))
end

using .Ours.LShaped
using .Ours.LShaped: SMPSLaw, SMPSProblem, support_size, build_recourse
using HiGHS
using JuMP
using Distributions
using LinearAlgebra
using Printf
using Statistics

include(joinpath(@__DIR__, "LShaped_instances.jl"))

const FELT_URL = "https://www4.uwsp.edu/math/afelt/slptestset/slptestset.tar.gz"

"""The directory of Felt's test set, kept in `dir`, downloaded and unpacked again if missing."""
function felt_testset(dir = joinpath(@__DIR__, "smps"))
    root = joinpath(dir, "slptestset")
    if !isdir(root)
        mkpath(dir)
        archive = joinpath(dir, "slptestset.tar.gz")
        download(FELT_URL, archive)
        run(`tar -xzf $archive -C $dir`)
        rm(archive)
    end
    return root
end

"""
CPA's reading of a law, as reverse-engineered from the published values of environ: an entry of
value 0 in a BLOCKS outcome is lost, and the element keeps its value in the first outcome.
"""
function cpa_reading(law::SMPSLaw)
    outcomes = map(law.outcomes) do block
        first = Dict(block[1])
        [[e => (v == 0 && haskey(first, e) ? first[e] : v) for (e, v) in outcome]
         for outcome in block]
    end
    return SMPSLaw(law.base, law.probabilities, outcomes)
end

with_law(smps::SMPSProblem, law) =
    SMPSProblem((getfield(smps, f) for f in fieldnames(SMPSProblem)[1:end-1])..., law)

"""The optimal value on the full support, by the extensive form and by the multicut method."""
function solve_full(smps)
    pb = TwoStageProblem(smps; enumerate_scenarios(smps.law)...)
    _, _, extensive = extensive_form(pb; optimizer = HiGHS.Optimizer)
    res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = :multi, tol = 1e-8, verbose = false)
    res.converged || error("the L-shaped method did not converge")
    return extensive, res.objective
end

"""
`c'x + E Q(x, ξ)` estimated on `n` draws, with twice its standard error; one recourse model is
re-solved per draw, which assumes that only right-hand sides are random (LandS, gbd).
"""
function estimate(smps, x, n; seed = 7)
    all(e -> e[1] == :h, smps.elements) || error("only right-hand sides may be random here")
    draws = rand(substream(seed), smps.law, n)
    pb = TwoStageProblem(smps; ξ = draws[1:1], p = [1.0])
    r = build_recourse(pb, HiGHS.Optimizer, 1)
    values = map(draws) do ξ
        rhs = pb.h(ξ) - pb.T(ξ) * x
        foreach(((i, con),) -> JuMP.set_normalized_rhs(con, rhs[i]), enumerate(r.con))
        optimize!(r.model)
        return objective_value(r.model)
    end
    return dot(smps.c, x) + mean(values), 2 * std(values) / sqrt(n)
end

function report(label, value, reference, note)
    difference = isnan(reference) ? "" : @sprintf("%+.2e", (value - reference) / abs(reference))
    @printf("  %-22s %18.6f %18s %10s  %s\n", label, value,
            isnan(reference) ? "-" : @sprintf("%.6f", reference), difference, note)
end

function main(args)
    full = "full" in args
    root = felt_testset()
    f(dir, cor, tim, sto) = joinpath.(joinpath(root, dir), (cor, tim, sto))
    println("Felt's test set, on the full support")
    @printf("  %-22s %18s %18s %10s  %s\n", "instance", "optimal value", "published", "rel. diff.", "")
    cases = [
        ("airlift.first", f("airlift", "AIRL.cor", "AIRL.tim", "AIRL.sto.first"), 249101.672072, "airlift/solution.txt"),
        ("airlift.second", f("airlift", "AIRL.cor", "AIRL.tim", "AIRL.sto.second"), 269665.498390, "airlift/solution.txt"),
        ("electric", f("electric", "LandS.cor", "LandS.tim", "LandS.sto"), 381.853333, "doc/electric.tex"),
        ("electric (BLOCKS)", f("electric", "LandS.cor", "LandS.tim", "LandS_blocks.sto"), 381.853333, "doc/electric.tex"),
        ("chem", f("chem", "chem.cor", "chem.tim", "chem.sto"), -13009.166667, "doc/chem.tex"),
        ("env.aggr", f("environ", "env.cor.diss", "env.tim", "env.sto.aggr"), 15963.929095, "environ/env.soln.aggr"),
        ("env.loose", f("environ", "env.cor.diss", "env.tim", "env.sto.loose"), 14794.608219, "environ/env.soln.loose"),
        ("env.imp", f("environ", "env.cor.diss", "env.tim", "env.sto.imp"), 21010.235039, "environ/env.soln.imp"),
        ("env.lrge", f("environ", "env.cor.diss", "env.tim", "env.sto.lrge"), 21034.731951, "environ/env.soln.lrge"),
        ("assets.small", f("assets", "assets.cor", "assets.tim", "assets.sto.small"), -720.472240, "assets/solution.txt"),
        ("cargo.8", f("cargo", "4node.cor", "4node.tim", "4node.sto.8"), NaN, "stoch file revised after the solutions"),
        ("cargo.64", f("cargo", "4node.cor", "4node.tim", "4node.sto.64"), NaN, "stoch file revised after the solutions")]
    for (label, files, reference, source) in cases
        smps = read_smps(files...)
        extensive, ours = solve_full(smps)
        abs(extensive - ours) <= 1e-6 * (1 + abs(extensive)) ||
            @warn "$label: the extensive form gives $extensive, the L-shaped method $ours"
        note = "$(support_size(smps.law)) scenarios, $source"
        label == "assets.small" && (note *= "; unexplained")
        report(label, ours, reference, note)
        if label in ("env.imp", "env.lrge")
            report("  CPA's reading", solve_full(with_law(smps, cpa_reading(smps.law)))[2], reference,
                   "zeros of the BLOCKS lost, as CPA does")
        end
    end

    println("\nLinderoth, Shapiro and Wright: their solutions, on 100 000 draws (± 2 standard errors)")
    for name in ("LandS", "gbd")
        smps = lsw_instance(name)
        sol = parse.(Float64, readlines(joinpath(@__DIR__, "smps", LSW[name].prefix * ".sol")))
        value, error = estimate(smps, sol[2:end], 100_000)
        report(name, value, LSW[name].value, @sprintf("± %.3f", error))
    end

    full || return
    println("\nLinderoth, Shapiro and Wright: approximation on 500 scenarios, its solution on 2000 others")
    for name in ("20term", "ssn", "storm")
        smps = lsw_instance(name)
        saa = TwoStageProblem(smps; sample_scenarios(smps.law, 500; rng = substream(1))...)
        res = lshaped(saa; optimizer = HiGHS.Optimizer, cuts = :multi, tol = 1e-6, maxiter = 2000,
                      threads = Threads.nthreads() > 1, verbose = false)
        test = TwoStageProblem(smps; sample_scenarios(smps.law, 2000; rng = substream(99))...)
        cost = expected_result(test, res.x; optimizer = HiGHS.Optimizer)
        report(name, cost, LSW[name].value, @sprintf("approximation %.4f, %d iterations",
                                                     res.objective, res.iterations))
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
