# Checks of LShaped.jl: every instance is solved by the single-cut and the multi-cut method, and
# compared with its extensive form. Needs JuMP, HiGHS, GLPK, Distributions and RandomDataStreams in
# the active environment, e.g.
#     julia --project=@v1.12 LShaped_test.jl
include("LShaped.jl")

using .LShaped
using Test
using JuMP
using HiGHS
using GLPK
using LinearAlgebra
using Statistics
using Distributions

const DEMAND_SCENARIOS = [3.0, 5.0, 7.0]
const DEMAND_PROBABILITIES = [0.3, 0.4, 0.3]

# --------------------------------------------------------------------------------------------
# instances
# --------------------------------------------------------------------------------------------

# Ice-cream capacity investment, written the way it is stated in `two_stages.ipynb`: the capacity
# rows are `≤`, the demand rows are `≥`, the budget is a `≤` row of the first stage.
const NPLANTS, NFLAVORS = 4, 3
const OPENING = [10.0, 7.0, 16.0, 6.0]
const PRODUCTION = [40.0 24.0 4.0; 45.0 27.0 4.5; 32.0 19.2 3.2; 55.0 33.0 5.5]

function icecream(; min_capacity = 12.0, budget = 120.0, lb = 0.0, ub = Inf,
                    intercept = :tight)
    idx(i, j) = NFLAVORS * (i - 1) + j
    W = zeros(NPLANTS + NFLAVORS, NPLANTS * NFLAVORS)
    T = zeros(NPLANTS + NFLAVORS, NPLANTS)
    for i in 1:NPLANTS, j in 1:NFLAVORS
        W[i, idx(i, j)] = 1.0              # Σⱼ yᵢⱼ ≤ xᵢ
        W[NPLANTS + j, idx(i, j)] = 1.0    # Σᵢ yᵢⱼ ≥ dⱼ(ξ)
    end
    for i in 1:NPLANTS
        T[i, i] = -1.0
    end
    return TwoStageProblem(
        c = OPENING,
        A = [ones(1, NPLANTS); OPENING'], senses1 = ['>', '<'], b = [min_capacity, budget],
        q = vec(permutedims(PRODUCTION)),
        W = W, senses2 = vcat(fill('<', NPLANTS), fill('>', NFLAVORS)),
        T = T, h = ξ -> vcat(zeros(NPLANTS), ξ, [3.0, 2.0]),
        ξ = DEMAND_SCENARIOS, p = DEMAND_PROBABILITIES, lb = lb, ub = ub, intercept = intercept)
end

# Newsvendor: two *equality* rows in the second stage.
newsvendor() = TwoStageProblem(
    c = [5.0], A = reshape([1.0], 1, 1), senses1 = ['<'], b = [100.0],
    q = [-12.0, -2.0, 0.0],                          # sold, salvaged, unmet
    W = [1.0 1.0 0.0; 1.0 0.0 1.0], senses2 = ['=', '='],
    T = reshape([-1.0, 0.0], 2, 1), h = ξ -> [0.0, ξ],
    ξ = [10.0, 20.0, 30.0, 40.0], p = [0.15, 0.35, 0.35, 0.15])

# One good, one plant: whatever is installed is produced (an `=` row), the demand must be met
# (`≥`), production is capped (`≤`). The recourse problem is infeasible at x = 0, and proving it
# takes the multiplier of the equality row.
production_balance() = TwoStageProblem(
    c = [10.0], A = reshape([1.0], 1, 1), senses1 = ['<'], b = [50.0],
    q = [1.0],
    W = reshape([1.0, 1.0, 1.0], 3, 1), senses2 = ['=', '>', '<'],
    T = reshape([-1.0, 0.0, 0.0], 3, 1), h = ξ -> [0.0, ξ, 20.0],
    ξ = [2.0, 5.0, 8.0], p = [0.3, 0.4, 0.3])

# A small farm, after the farmer of Birge and Louveaux: land is bought in the first stage for each
# of three crops, at most `LAND` acres in all; once the yields are known, each crop is planted on at
# most the land bought for it, its harvest must cover a contracted demand, and all of it is sold.
# The second-stage cost of an acre planted is its planting cost minus the sale of its harvest.
#
# Both instances solve the recourse problem of x = 0 first, which is infeasible: feasibility cuts
# come before any optimality cut. The demand of corn makes the least land it needs fractional,
# 104 1/6 acres, so that whole acres (`integer = true`) change the solution.
const CROPS = 3
const LAND = 500.0
const ACRE = [100.0, 100.0, 100.0]           # per acre bought
const PLANTING = [150.0, 230.0, 260.0]       # per acre planted
const PRICE = [170.0, 150.0, 36.0]           # per ton sold
const CROP_DEMAND = [200.0, 250.0, 260.0]    # tons
const FARM_PROBABILITIES = [1 / 3, 1 / 3, 1 / 3]

# The matrix modelization asks for a single `W`, which carries the yields: they are fixed to their
# average, and the uncertainty sits in the demand, scaled by ξ.
const AVERAGE_YIELD = [2.5, 3.0, 20.0]       # tons per acre
const DEMAND_FACTOR = [0.8, 1.0, 1.25]

function farm(; integer = false)
    W = vcat(Matrix{Float64}(I, CROPS, CROPS), diagm(AVERAGE_YIELD))
    T = vcat(-Matrix{Float64}(I, CROPS, CROPS), zeros(CROPS, CROPS))
    return TwoStageProblem(
        c = ACRE,
        A = ones(1, CROPS), senses1 = ['<'], b = [LAND],
        q = PLANTING .- PRICE .* AVERAGE_YIELD,
        W = W, senses2 = vcat(fill('<', CROPS), fill('>', CROPS)),   # yⱼ ≤ xⱼ, harvest ≥ demand
        T = T, h = ξ -> vcat(zeros(CROPS), CROP_DEMAND .* ξ),
        ξ = DEMAND_FACTOR, p = FARM_PROBABILITIES, integer1 = integer)
end

# The same farm with the yields — not the demand — varying by scenario: `W` and `q` then depend on
# the scenario, which is out of reach of the matrix modelization and the reason
# `JuMPTwoStageProblem` exists.
const YIELD = [2.0 2.4 16.0; 2.5 3.0 20.0; 3.0 3.6 24.0]   # below average, average, above

"""The recourse rows of the farm in scenario `s`, the `w` absorbing their violation if given."""
function farm_rows!(m, y, s; w = nothing)
    slack(k) = w === nothing ? 0.0 : w[k]
    land = [@constraint(m, y[j] - slack(j) <= 0) for j in 1:CROPS]           # yⱼ ≤ xⱼ
    for j in 1:CROPS
        @constraint(m, YIELD[s, j] * y[j] + slack(CROPS + j) >= CROP_DEMAND[j])
    end
    return land
end

farm_cost(s) = PLANTING .- PRICE .* YIELD[s, :]

function farm_jump(; integer = false)
    return JuMPTwoStageProblem(
        n_scenarios = size(YIELD, 1),
        probabilities = s -> FARM_PROBABILITIES[s],
        master_builder = function (optimizer)
            m = new_model(optimizer)
            x = @variable(m, x[1:CROPS] >= 0, integer = integer)
            @constraint(m, sum(x) <= LAND)
            @objective(m, Min, dot(ACRE, x))
            return m, x
        end,
        recourse_builder = function (optimizer, s)
            m = new_model(optimizer)
            y = @variable(m, y[1:CROPS] >= 0)
            land = farm_rows!(m, y, s)
            @objective(m, Min, dot(farm_cost(s), y))
            return m, y, land
        end,
        rhs = (x, s) -> x,
        cut_coefficients = (s, x, π) -> -π,          # T = -I on the land rows
        elastic_builder = function (optimizer, s)
            m = new_model(optimizer)
            y = @variable(m, y[1:CROPS] >= 0)
            w = @variable(m, w[1:2CROPS] >= 0)      # land overrun, then unmet demand
            land = farm_rows!(m, y, s; w = w)
            @objective(m, Min, sum(w))
            return m, land, 1:CROPS, ones(CROPS)
        end)
end

"""The extensive form of the instance `farm_jump` describes, written out."""
function farm_extensive(; integer = false)
    m = new_model(HiGHS.Optimizer)
    x = @variable(m, x[1:CROPS] >= 0, integer = integer)
    @constraint(m, sum(x) <= LAND)
    S = size(YIELD, 1)
    y = [@variable(m, [1:CROPS], lower_bound = 0) for _ in 1:S]
    for s in 1:S
        land = farm_rows!(m, y[s], s)
        set_normalized_coefficient.(land, x, -1.0)   # yⱼ - xⱼ ≤ 0
    end
    @objective(m, Min, dot(ACRE, x) +
                       sum(FARM_PROBABILITIES[s] * dot(farm_cost(s), y[s]) for s in 1:S))
    optimize!(m)
    return value.(x), objective_value(m)
end

# The ice-cream instance as a hand-made JuMP model: surplus variables, every recourse row `≥`, and
# the scenario kept inside the closures. This is what `JuMPTwoStageProblem` is meant for.
function icecream_jump(; min_capacity = 12.0, budget = 120.0)
    plant(i, j) = NFLAVORS * (i - 1) + j
    demand(ξ) = vcat(zeros(NPLANTS), ξ, [3.0, 2.0])
    cost(y) = sum(PRODUCTION[i, j] * y[plant(i, j)] for i in 1:NPLANTS, j in 1:NFLAVORS)
    return JuMPTwoStageProblem(
        n_scenarios = length(DEMAND_SCENARIOS),
        master_builder = function (optimizer)
            m = new_model(optimizer)
            x = @variable(m, x[1:NPLANTS] >= 0)
            @constraint(m, sum(x) >= min_capacity)
            @constraint(m, dot(OPENING, x) <= budget)
            @objective(m, Min, dot(OPENING, x))
            return m, x
        end,
        recourse_builder = function (optimizer, s)
            m = new_model(optimizer)
            y = @variable(m, y[1:NPLANTS * NFLAVORS] >= 0)
            d = demand(DEMAND_SCENARIOS[s])
            # -Σⱼ yᵢⱼ ≥ -xᵢ: the right-hand side is the only thing that moves with x
            con = [@constraint(m, -sum(y[plant(i, j)] for j in 1:NFLAVORS) >= 0)
                   for i in 1:NPLANTS]
            for j in 1:NFLAVORS
                @constraint(m, sum(y[plant(i, j)] for i in 1:NPLANTS) >= d[NPLANTS + j])
            end
            @objective(m, Min, cost(y))
            return m, y, con
        end,
        rhs = (x, s) -> -x,
        cut_coefficients = (s, x, π) -> π,
        elastic_builder = function (optimizer, s)
            m = new_model(optimizer)
            y = @variable(m, y[1:NPLANTS * NFLAVORS] >= 0)
            w = @variable(m, w[1:NPLANTS] >= 0)
            d = demand(DEMAND_SCENARIOS[s])
            con = [@constraint(m, -sum(y[plant(i, j)] for j in 1:NFLAVORS) + w[i] >= 0)
                   for i in 1:NPLANTS]
            for j in 1:NFLAVORS
                @constraint(m, sum(y[plant(i, j)] for i in 1:NPLANTS) >= d[NPLANTS + j])
            end
            @objective(m, Min, sum(w))
            return m, con, 1:NPLANTS, ones(NPLANTS)
        end,
        probabilities = s -> DEMAND_PROBABILITIES[s])
end

reference(pb; optimizer = HiGHS.Optimizer) = extensive_form(pb; optimizer)[3]

# Two plants whose capacity is bought in the first stage, with a *sampled* demand: the same code
# path, the `(ξ, p)` pair coming from `sample_scenarios` rather than from a table. Plant 1 is free
# to run, plant 2 charges 3 per ton, so the recourse problem trades capacity against unmet demand
# and is genuinely infeasible at `x = 0` — the run needs feasibility cuts as well as optimality
# ones.
function sampled_demand(; n = 200, seed = 11, capacity = 30.0)
    ξ, p = sample_scenarios(Normal(5.0, 1.5), n; rng = substream(seed))
    return TwoStageProblem(
        c = [1.0, 2.0],
        A = reshape([1.0, 1.0], 1, 2), senses1 = ['<'], b = [capacity],
        q = [0.0, 3.0],
        W = [1.0 0.0; 0.0 1.0; 1.0 1.0], senses2 = ['<', '<', '>'],
        T = [-1.0 0.0; 0.0 -1.0; 0.0 0.0], h = ξs -> [0.0, 0.0, ξs],
        ξ = ξ, p = p)
end

# One first-stage and one recourse variable, `Q(x, ξₛ) = 10 + s - 2x` on `0 ≤ x ≤ 1`, to be
# spoiled one keyword at a time by the diagnostics: an objective that makes the recourse unbounded,
# a `Max` sense, an integer recourse variable, a master objective term outside `x`.
function one_variable(; objective = y -> y, sense = MIN_SENSE, integer = false,
                      extra_master_variable = false, constant = 0.0)
    return JuMPTwoStageProblem(
        n_scenarios = 3,
        master_builder = function (optimizer)
            m = new_model(optimizer)
            x = @variable(m, 0 <= x[1:1] <= 1)
            z = @variable(m, z >= 0)
            @objective(m, Min, x[1] + constant + (extra_master_variable ? z : 0.0))
            return m, x
        end,
        recourse_builder = function (optimizer, s)
            m = new_model(optimizer)
            y = @variable(m, y[1:1] >= 0, integer = integer)
            con = [@constraint(m, y[1] >= 0)]
            @objective(m, sense, objective(y[1]))
            return m, y, con
        end,
        rhs = (x, s) -> [10.0 + s - 2x[1]],
        cut_coefficients = (s, x, π) -> 2π,
        elastic_builder = function (optimizer, s)
            m = new_model(optimizer)
            y = @variable(m, y[1:1] >= 0)
            w = @variable(m, w[1:1] >= 0)
            con = [@constraint(m, y[1] + w[1] >= 0)]
            @objective(m, Min, w[1])
            return m, con, [1], [1.0]
        end)
end

# --------------------------------------------------------------------------------------------
# scenario sets
# --------------------------------------------------------------------------------------------

@testset "a sample is reproducible, and independent of the session" begin
    @test rand(substream(7), 5) == rand(substream(7), 5)
    @test rand(substream(7), 5) != rand(substream(8), 5)
    @test length(unique(LShaped.seed_words(1))) == 4      # four distinct, mixed words

    ξ, p = sample_scenarios(Normal(0.0, 1.0), 10_000; rng = substream(3))
    @test length(ξ) == length(p) == 10_000
    @test sum(p) ≈ 1
    @test abs(mean(ξ)) < 0.05 && abs(std(ξ) - 1) < 0.05

    ξ, p = sample_scenarios(Categorical([0.3, 0.4, 0.3]), 500; rng = substream(1))
    @test sort(unique(ξ)) == [1, 2, 3]                    # a discrete law stays discrete
    @test 0.25 < count(==(1), ξ) / 500 < 0.35
    @test_throws ErrorException sample_scenarios(Normal(), 10; p = fill(0.2, 10))
    @test_throws ErrorException sample_scenarios(Normal(), 10; p = fill(0.1, 9))
    @test_throws ErrorException sample_scenarios(Normal(), 0)
end

@testset "a sampled instance: the sample is what the two methods both see" begin
    pb = sampled_demand()
    _, x_optimal, optimal = extensive_form(pb; optimizer = HiGHS.Optimizer)
    for cuts in (:single, :multi)
        res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test res.converged
        @test res.objective ≈ optimal atol = 1e-6
        @test res.x ≈ x_optimal atol = 1e-6
        @test res.feasibility_cuts > 0
    end
    @test sampled_demand(n = 60, seed = 4).ξ != sampled_demand(n = 60, seed = 5).ξ
end

# --------------------------------------------------------------------------------------------
# mixed row senses in both stages
# --------------------------------------------------------------------------------------------

@testset "ice cream: `≤` and `≥` rows in both stages" begin
    pb = icecream()
    optimal = 28639 / 75
    @test reference(pb) ≈ optimal

    for cuts in (:single, :multi)
        res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test res.converged
        @test res.objective ≈ optimal atol = 1e-6
        @test res.x ≈ [8 / 3, 4, 10 / 3, 2] atol = 1e-6
        @test res.lower_bound ≈ optimal atol = 1e-6
        @test res.feasibility_cuts == 0
        @test all(h.upper_bound >= optimal - 1e-6 for h in res.history)
    end
    @test lshaped(pb; optimizer = HiGHS.Optimizer, cuts = :multi, verbose = false).iterations <
          lshaped(pb; optimizer = HiGHS.Optimizer, verbose = false).iterations
end

@testset "ice cream: feasibility cuts once the minimum capacity is dropped" begin
    pb = icecream(min_capacity = 0.0)
    optimal = reference(pb)
    for cuts in (:single, :multi)
        res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test res.converged
        @test res.feasibility_cuts > 0
        @test res.objective ≈ optimal atol = 1e-6
    end
end

# --------------------------------------------------------------------------------------------
# equality rows
# --------------------------------------------------------------------------------------------

@testset "newsvendor: two `=` rows in the recourse problem" begin
    pb = newsvendor()
    @test reference(pb) ≈ -145

    for cuts in (:single, :multi)
        res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test res.converged
        @test res.x[1] ≈ 30 atol = 1e-6
        @test res.objective ≈ -145 atol = 1e-6
    end
end

@testset "a feasibility cut that needs the multiplier of an `=` row" begin
    pb = production_balance()
    @test reference(pb) ≈ 88

    for cuts in (:single, :multi)
        res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test res.converged
        @test res.feasibility_cuts == 3                  # every scenario is infeasible at x = 0
        @test res.x[1] ≈ 8 atol = 1e-6
        @test res.objective ≈ 88 atol = 1e-6
    end
end

# --------------------------------------------------------------------------------------------
# bounds on the recourse variables
# --------------------------------------------------------------------------------------------

@testset "bounded recourse variables: `y ≤ 3`" begin
    pb = icecream(ub = 3.0)
    optimal = 1911 / 5
    @test reference(pb) ≈ optimal

    res = lshaped(pb; optimizer = HiGHS.Optimizer, verbose = false)
    @test res.converged
    @test res.objective ≈ optimal atol = 1e-6
    @test res.feasibility_cuts > 0
    @test all(h.lower_bound <= h.upper_bound + 1e-6 for h in res.history)

    # the closed forms of the slides are no longer valid there: the lower bound crosses the upper
    # bound, and the method stops away from the optimum without saying so
    naive = @test_logs (:warn, r"exceeds the upper bound") match_mode = :any lshaped(
        icecream(ub = 3.0, intercept = :textbook); optimizer = HiGHS.Optimizer, verbose = false)
    @test naive.objective > optimal
    @test any(h.lower_bound > h.upper_bound + 1e-6 for h in naive.history)
end

# --------------------------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------------------------

@testset "a different solver for the master and for the recourse" begin
    pb = icecream()
    optimal = reference(pb)
    tuned = JuMP.MOI.OptimizerWithAttributes(GLPK.Optimizer, "presolve" => true)
    for (master, recourse) in ((GLPK.Optimizer, HiGHS.Optimizer),
                               (HiGHS.Optimizer, GLPK.Optimizer),
                               (tuned, HiGHS.Optimizer),
                               (HiGHS.Optimizer, tuned))
        res = lshaped(pb; master_optimizer = master, recourse_optimizer = recourse, verbose = false)
        @test res.converged
        @test res.objective ≈ optimal atol = 1e-6
    end
    @test_throws ErrorException lshaped(pb; recourse_optimizer = HiGHS.Optimizer, verbose = false)
    @test_throws ErrorException lshaped(pb; master_optimizer = HiGHS.Optimizer, verbose = false)
end

@testset "farm: an LP master, and a mixed-integer one" begin
    optimal = Dict(integer => reference(farm(; integer = integer)) for integer in (false, true))
    @test optimal[true] > optimal[false]               # whole acres cost something
    for integer in (false, true)
        pb = farm(; integer = integer)
        for cuts in (:single, :multi)
            res = lshaped(pb; master_optimizer = HiGHS.Optimizer,
                          recourse_optimizer = GLPK.Optimizer, cuts = cuts, verbose = false)
            @test res.converged
            @test res.feasibility_cuts > 0
            @test res.objective ≈ optimal[integer] rtol = 1e-7
            integer && @test res.x ≈ round.(res.x) atol = 1e-6
        end
    end
end

# --------------------------------------------------------------------------------------------
# any JuMP model
# --------------------------------------------------------------------------------------------

@testset "the ice-cream instance as a hand-made JuMP model" begin
    for cuts in (:single, :multi)
        res = lshaped(icecream_jump(); optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test res.converged
        @test res.objective ≈ 28639 / 75 atol = 1e-6
        @test res.x ≈ [8 / 3, 4, 10 / 3, 2] atol = 1e-6
    end
    @testset "and with feasibility cuts" begin
        pb = icecream_jump(min_capacity = 0.0)
        optimal = reference(icecream(min_capacity = 0.0))
        for cuts in (:single, :multi)
            res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
            @test res.converged
            @test res.feasibility_cuts > 0
            @test res.objective ≈ optimal atol = 1e-6
        end
    end
end

@testset "the farm as a hand-made JuMP model: scenario-dependent yields" begin
    for integer in (false, true)
        x_optimal, optimal = farm_extensive(; integer = integer)
        for cuts in (:single, :multi)
            res = lshaped(farm_jump(; integer = integer); master_optimizer = HiGHS.Optimizer,
                          recourse_optimizer = GLPK.Optimizer, cuts = cuts, verbose = false)
            @test res.converged
            @test res.feasibility_cuts > 0
            @test res.objective ≈ optimal rtol = 1e-7
            @test res.x ≈ x_optimal atol = 1e-5
        end
    end
end

# --------------------------------------------------------------------------------------------
# the decisions
# --------------------------------------------------------------------------------------------

"""`Wy ⋛ h(ξ) - T(ξ)x`, row by row, with the senses of `pb`."""
function satisfies_recourse_rows(pb, x, y, ξ; atol = 1e-7)
    lhs, rhs = pb.W * y, pb.h(ξ) - pb.T(ξ) * x
    return all(zip(pb.senses2, lhs, rhs)) do (sense, l, r)
        sense == LShaped.LEQ ? l <= r + atol : sense == LShaped.GEQ ? l >= r - atol : abs(l - r) <= atol
    end
end

@testset "first- and second-stage decisions" begin
    for (pb, cuts) in ((icecream(), :single), (icecream(), :multi), (icecream(ub = 3.0), :single))
        res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test first_stage_decision(res) == res.x
        x = first_stage_decision(res)
        expected = dot(pb.c, x)
        for s in eachindex(pb.ξ)
            y = second_stage_decision(res, s)
            @test length(y) == length(pb.lb)
            @test all(pb.lb .- 1e-7 .<= y .<= pb.ub .+ 1e-7)
            @test satisfies_recourse_rows(pb, x, y, pb.ξ[s])
            expected += pb.p[s] * dot(pb.q(pb.ξ[s]), y)
        end
        # each yₛ is feasible, and together they reach the optimal value: each is optimal
        @test expected ≈ res.objective atol = 1e-6
    end
    res = lshaped(icecream(); optimizer = HiGHS.Optimizer, verbose = false)
    @test_throws ArgumentError second_stage_decision(res, 0)
    @test_throws ArgumentError second_stage_decision(res, length(DEMAND_SCENARIOS) + 1)
end

@testset "the decisions are displayed" begin
    res = lshaped(icecream(); optimizer = HiGHS.Optimizer, verbose = false)
    first = sprint(print_first_stage, res)
    @test startswith(first, "First-stage decision, objective 381.853333")
    @test count(==('\n'), first) == 1 + NPLANTS
    @test occursin("  x[1] = 2.666667", first) && occursin("  x[4] = 2.0", first)
    second = sprint((io, r) -> print_second_stage(io, r, 2), res)
    @test startswith(second, "Second-stage decision, scenario 2 of 3: ξ = 5.0, probability 0.4, Q(x, ξ) = ")
    @test count(==('\n'), second) == 1 + NPLANTS * NFLAVORS
    @test occursin("  y[12] = ", second)
    @test !occursin("-0.0", second)
    # a JuMP model keeps its own names; without `data`, nothing is known of ξ
    resj = lshaped(icecream_jump(); optimizer = HiGHS.Optimizer, verbose = false)
    secondj = sprint((io, r) -> print_second_stage(io, r, 1; digits = 3), resj)
    @test startswith(secondj, "Second-stage decision, scenario 1 of 3: probability 0.3, Q(x, ξ) = ")
    @test occursin("  y[1]  = ", secondj)
end

# --------------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------------

@testset "what is wrong with the input is reported" begin
    @test_throws ErrorException to_sense('?')
    @test_throws AssertionError TwoStageProblem(      # b has one row too few
        c = [1.0], A = ones(1, 1), senses1 = ['<'], b = [0.0, 0.0],
        q = [1.0], W = ones(1, 1), senses2 = ['='], T = zeros(1, 1), h = [0.0],
        ξ = [0.0], p = [1.0])
    @test_throws AssertionError TwoStageProblem(      # T(ξ) of the wrong size
        c = [1.0], A = ones(1, 1), senses1 = ['<'], b = [0.0],
        q = [1.0], W = ones(1, 1), senses2 = ['='], T = zeros(2, 1), h = [0.0],
        ξ = [0.0], p = [1.0])
    @test_throws AssertionError TwoStageProblem(      # the probabilities do not sum up to one
        c = [1.0], A = ones(1, 1), senses1 = ['<'], b = [0.0],
        q = [1.0], W = ones(1, 1), senses2 = ['='], T = zeros(1, 1), h = [0.0],
        ξ = [0.0, 1.0], p = [0.3, 0.3])
    @test_throws ErrorException lshaped(icecream(); optimizer = HiGHS.Optimizer, cuts = :both,
                                        verbose = false)

    # an unbounded recourse problem: the elastic model says there is no violation to cut on
    @test_throws ErrorException lshaped(one_variable(objective = y -> -y); optimizer = HiGHS.Optimizer,
                                        verbose = false)
    # a recourse problem MOI would return multipliers of the wrong sign for, or none at all
    @test_throws ErrorException lshaped(one_variable(sense = MAX_SENSE); optimizer = HiGHS.Optimizer,
                                        verbose = false)
    @test_throws ErrorException lshaped(one_variable(integer = true); optimizer = HiGHS.Optimizer,
                                        verbose = false)
    # a master objective term outside x would be left out of both bounds
    @test_throws ErrorException lshaped(one_variable(extra_master_variable = true);
                                        optimizer = HiGHS.Optimizer, verbose = false)
end

@testset "the incumbent, and a constant in the master objective" begin
    pb = one_variable(constant = 1000.0)
    res = lshaped(pb; optimizer = HiGHS.Optimizer, verbose = false)
    @test res.converged
    @test res.objective ≈ 1000 + 1 + (mean(10.0 .+ (1:3)) - 2) atol = 1e-6   # at x = 1
    @test res.lower_bound ≈ res.objective atol = 1e-6
    for cuts in (:single, :multi)
        res = lshaped(icecream(); optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test res.objective == minimum(h.value for h in res.history)
        @test issorted([h.upper_bound for h in res.history], rev = true)
    end
end

@testset "a sample splats into the keyword arguments" begin
    pb = TwoStageProblem(; c = [1.0], A = ones(1, 1), senses1 = ['<'], b = [10.0],
                         q = [1.0], W = ones(1, 1), senses2 = ['>'], T = zeros(1, 1),
                         h = ξ -> [ξ], sample_scenarios(Uniform(0, 1), 50; rng = substream(2))...)
    @test pb.ξ == sample_scenarios(Uniform(0, 1), 50; rng = substream(2)).ξ
    @test length(pb.p) == 50
end
