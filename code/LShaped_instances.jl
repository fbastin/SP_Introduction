# The instances of the benchmarks of LShaped.jl, as `TwoStageProblem`s: generated ones, and the
# SMPS instances of Linderoth, Shapiro and Wright (2006), kept in `smps/` and downloaded again if
# missing. Included by
# `LShaped_benchmark.jl` and `LShaped_scaling.jl`, after `TwoStageProblem`, `substream` and
# `read_smps` have been brought into scope, with Distributions and LinearAlgebra loaded.

"""
    farmer_problem(S; seed = 1)

The farmer of Birge and Louveaux (2011, Section 1.1), with `S` equally likely scenarios whose three
yields are drawn independently, uniformly within ±20% of their mean: the randomness is in the
technology matrix `T`. A small first stage, many scenarios. Returns the problem and the yields.
"""
function farmer_problem(S::Integer; seed::Integer = 1)
    rng = substream(seed)
    mean_yield = [2.5, 3.0, 20.0]
    yields = [mean_yield .* rand(rng, Uniform(0.8, 1.2), 3) for _ in 1:S]
    problem = TwoStageProblem(
        c = [150.0, 230.0, 260.0], A = ones(1, 3), senses1 = ['<'], b = [500.0],
        q = [238.0, 210.0, -170.0, -150.0, -36.0, -10.0],
        W = [1.0 0.0 -1.0 0.0 0.0 0.0; 0.0 1.0 0.0 -1.0 0.0 0.0;
             0.0 0.0 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 0.0 1.0 0.0],
        senses2 = ['>', '>', '<', '<'],
        T = t -> [t[1] 0.0 0.0; 0.0 t[2] 0.0; 0.0 0.0 -t[3]; 0.0 0.0 0.0],
        h = [200.0, 240.0, 0.0, 6000.0], ξ = yields, p = fill(1 / S, S))
    return (problem = problem, yields = yields)
end

"""
    capacity_problem(P, F, S; shortage = true, seed = 2)

Capacity expansion, after the ice-cream example: capacity `xᵢ` bought for each of `P` plants, then
`F` products made, `yᵢⱼ`, to meet random demands `dⱼ`, drawn from a log-normal distribution around
a mean of their own; the randomness is in `h`. With `shortage`, unmet demand `uⱼ` costs a penalty
(complete recourse); without, the demand must be met, the recourse problem is infeasible whenever
the capacity falls short, and the methods need feasibility cuts. Returns the problem and its data.
"""
function capacity_problem(P::Integer, F::Integer, S::Integer; shortage::Bool = true,
                          seed::Integer = 2)
    rng = substream(seed)
    c = rand(rng, Uniform(5.0, 15.0), P)                     # capacity costs
    a = rand(rng, Uniform(1.0, 10.0), P, F)                  # production costs
    mean_demand = rand(rng, Uniform(10.0, 50.0), F)
    demands = [mean_demand .* rand(rng, LogNormal(0.0, 0.3), F) for _ in 1:S]
    penalty = 100.0
    total = 2 * maximum(sum, demands)                         # a loose bound on the capacity
    nu = shortage ? F : 0
    idx(i, j) = F * (i - 1) + j
    W = zeros(P + F, P * F + nu)
    for i in 1:P, j in 1:F
        W[i, idx(i, j)] = 1.0                                 # Σⱼ yᵢⱼ ≤ xᵢ
        W[P + j, idx(i, j)] = 1.0                             # Σᵢ yᵢⱼ (+ uⱼ) ≥ dⱼ
    end
    shortage && (W[P+1:end, P*F+1:end] = Matrix(1.0I, F, F))
    problem = TwoStageProblem(
        c = c, A = ones(1, P), senses1 = ['<'], b = [total],
        q = vcat(vec(permutedims(a)), fill(penalty, nu)),
        W = W, senses2 = vcat(fill('<', P), fill('>', F)),
        T = vcat(-Matrix(1.0I, P, P), zeros(F, P)), h = d -> vcat(zeros(P), d),
        ξ = demands, p = fill(1 / S, S))
    return (problem = problem, c = c, a = a, demands = demands, penalty = penalty, total = total)
end

# Linderoth, Shapiro and Wright (2006), "The empirical behavior of sampling methods for stochastic
# programming", and their estimates of the optimal values (Table 4, Latin hypercube sampling,
# N = 5000: the upper bound estimate).
const LSW_URL = "https://pages.cs.wisc.edu/~swright/stochastic/sampling"
const LSW = Dict("20term" => (archive = "20term", prefix = "data-20/20", value = 254311.55),
                 "gbd" => (archive = "gbd", prefix = "data-gbd/gbd", value = 1655.628),
                 "LandS" => (archive = "LandS", prefix = "data-LandS/LandS", value = 225.624),
                 "ssn" => (archive = "ssn", prefix = "data-ssn/ssn", value = 9.913),
                 "storm" => (archive = "storm", prefix = "data-storm/storm", value = 15498739.41))

"""
    lsw_instance(name; dir = joinpath(@__DIR__, "smps"))

The SMPS instance `name` of Linderoth, Shapiro and Wright — `"20term"`, `"gbd"`, `"LandS"`, `"ssn"`
or `"storm"` — read by `read_smps` from `dir`, where it is kept, after downloading and unpacking it
if it is missing. Their supports are far too large to enumerate: sample them with `sample_scenarios`.
"""
function lsw_instance(name::AbstractString; dir::AbstractString = joinpath(@__DIR__, "smps"))
    entry = LSW[name]
    prefix = joinpath(dir, entry.prefix)
    if !isfile(prefix * ".cor")
        mkpath(dir)
        archive = joinpath(dir, entry.archive * ".tar")
        download("$LSW_URL/$(entry.archive).tar", archive)
        run(`tar -xf $archive -C $dir`)
        rm(archive)
    end
    return read_smps(prefix)
end
