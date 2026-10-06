# Checks of LShaped.jl: every instance is solved by the single-cut and the multi-cut method, and
# compared with its extensive form. Needs JuMP, HiGHS, GLPK, Distributions and RandomDataStreams in
# the active environment, e.g.
#     julia --project=@v1.12 LShaped_test.jl
include("LShaped.jl")

using .LShaped
using Test
using JuMP
using HiGHS
using Ipopt
using GLPK
using LinearAlgebra
using Statistics
using Distributions
using SparseArrays

const DEMAND_SCENARIOS = [3.0, 5.0, 7.0]
const SCENARIOS_ICECREAM = length(DEMAND_SCENARIOS)
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

"""The rows of a master problem that involve a θ: its optimality cuts."""
count_theta_rows(model) =
    count(con -> occursin("θ", string(con)),
          all_constraints(model; include_variable_in_set_constraints = false))

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
    # the counters count cuts, not rounds: one per iteration but the last in the single-cut
    # version, at most one per scenario and per such iteration in the multicut version, which
    # only adds the violated ones; each is a row of the master involving θ
    single = lshaped(pb; optimizer = HiGHS.Optimizer, verbose = false)
    multi = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = :multi, verbose = false)
    @test single.optimality_cuts == single.iterations - 1
    @test multi.optimality_cuts <= SCENARIOS_ICECREAM * (multi.iterations - 1)
    for res in (single, multi)
        @test res.optimality_cuts == count_theta_rows(res.master.model)
    end
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
# performance options: violated cuts only, threads, inactive cuts
# --------------------------------------------------------------------------------------------

@testset "multicut: only the violated cuts are added" begin
    pb = sampled_demand()
    S = length(pb.ξ)
    res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = :multi, verbose = false)
    @test res.converged
    @test res.objective ≈ reference(pb) atol = 1e-6
    # the first round adds the S cuts; after it, only a few scenarios are violated at a time
    @test S <= res.optimality_cuts < S * (res.iterations - 1)
    @test res.optimality_cuts == count_theta_rows(res.master.model)
end

@testset "threads: the same results, in parallel or not" begin
    # GLPK frees its problems in finalizers, which crash Julia when the garbage collector runs them
    # on another thread than the one that created them: collect the GLPK models of the earlier
    # tests here, on the main thread, before any thread may trigger a collection
    GC.gc()
    instances = [("ice cream", icecream(), :single), ("ice cream", icecream(), :multi),
                 ("feasibility cuts", icecream(min_capacity = 0.0), :multi),
                 ("sampled", sampled_demand(), :multi), ("JuMP model", farm_jump(), :single)]
    for (name, pb, cuts) in instances
        serial = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, threads = false, verbose = false)
        parallel = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, threads = true, verbose = false)
        @test parallel.objective == serial.objective
        @test parallel.x == serial.x
        @test parallel.iterations == serial.iterations
        @test (parallel.optimality_cuts, parallel.feasibility_cuts) ==
              (serial.optimality_cuts, serial.feasibility_cuts)
    end
    # an error raised in a thread surfaces as it would serially
    @test_throws ErrorException lshaped(one_variable(objective = y -> -y);
                                        optimizer = HiGHS.Optimizer, threads = true, verbose = false)
    @info "threads tested with $(Threads.nthreads()) thread(s)"
end

@testset "inactive cuts are dropped" begin
    pb = icecream(ub = 3.0)
    kept = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = :multi, verbose = false)
    dropped = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = :multi, drop_inactive = 1,
                      verbose = false)
    @test kept.dropped_cuts == 0
    @test dropped.converged
    @test dropped.objective ≈ reference(pb) atol = 1e-6
    @test dropped.dropped_cuts > 0
    # the master holds what was added minus what was dropped, and fewer rows than without dropping
    @test count_theta_rows(dropped.master.model) == dropped.optimality_cuts - dropped.dropped_cuts
    @test count_theta_rows(dropped.master.model) < count_theta_rows(kept.master.model)
    # the single-cut version too, on an instance that needs feasibility cuts, which are never dropped
    pb = icecream(min_capacity = 0.0)
    res = lshaped(pb; optimizer = HiGHS.Optimizer, drop_inactive = 1, verbose = false)
    @test res.converged
    @test res.objective ≈ reference(pb) atol = 1e-6
    @test_throws ErrorException lshaped(pb; optimizer = HiGHS.Optimizer, drop_inactive = 0)
end

# --------------------------------------------------------------------------------------------
# the decisions
# --------------------------------------------------------------------------------------------

"""`Wy ⋛ h(ξ) - T(ξ)x`, row by row, with the senses of `pb`."""
function satisfies_recourse_rows(pb, x, y, ξ; atol = 1e-7)
    lhs, rhs = pb.W(ξ) * y, pb.h(ξ) - pb.T(ξ) * x
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
# the value of perfect information and of the stochastic solution
# --------------------------------------------------------------------------------------------

# The example of deck 03: Q(x, ξ) = |x - ξ| with ξ uniform over {1, 2, 4} and c = 0, written
# y⁺ - y⁻ = ξ - x. RP = 1, WS = 0, EV = 0 at x̄ = 7/3, EEV = 10/9.
absolute_deviation() = TwoStageProblem(
    c = [0.0], A = ones(1, 1), senses1 = ['<'], b = [10.0],
    q = [1.0, 1.0], W = [1.0 -1.0], senses2 = ['='], T = ones(1, 1), h = ξ -> [ξ],
    ξ = [1.0, 2.0, 4.0], p = fill(1 / 3, 3))

# The farmer of Birge and Louveaux (2011, Section 1.1): 500 acres of wheat, corn and sugar beets,
# random yields. Second stage: wheat and corn bought (w) or sold (y₁, y₂) to cover the cattle feed
# requirements, beets sold at 36 up to the quota of 6000 tons (y₃), at 10 beyond it (y₄).
# WS = -115405.56, RP = -108390, EV = -118600, EEV = -107240: EVPI = 7015.56 and VSS = 1150.
birge_louveaux_farmer() = TwoStageProblem(
    c = [150.0, 230.0, 260.0], A = ones(1, 3), senses1 = ['<'], b = [500.0],
    q = [238.0, 210.0, -170.0, -150.0, -36.0, -10.0],          # w₁, w₂, y₁, y₂, y₃, y₄
    W = [1.0 0.0 -1.0 0.0 0.0 0.0;                              # t₁x₁ + w₁ - y₁ ≥ 200
         0.0 1.0 0.0 -1.0 0.0 0.0;                              # t₂x₂ + w₂ - y₂ ≥ 240
         0.0 0.0 0.0 0.0 1.0 1.0;                               # y₃ + y₄ ≤ t₃x₃
         0.0 0.0 0.0 0.0 1.0 0.0],                              # y₃ ≤ 6000
    senses2 = ['>', '>', '<', '<'],
    T = t -> [t[1] 0.0 0.0; 0.0 t[2] 0.0; 0.0 0.0 -t[3]; 0.0 0.0 0.0],   # rows Wy ⋛ h - Tx
    h = [200.0, 240.0, 0.0, 6000.0],
    ξ = [[3.0, 3.6, 24.0], [2.5, 3.0, 20.0], [2.0, 2.4, 16.0]], p = fill(1 / 3, 3))

@testset "EVPI and VSS: the example of the slides" begin
    pb = absolute_deviation()
    ws = wait_and_see(pb; optimizer = HiGHS.Optimizer)
    @test ws.value ≈ 0 atol = 1e-6
    @test ws.decisions ≈ [[1.0], [2.0], [4.0]] atol = 1e-6
    ev = expected_value_problem(pb; optimizer = HiGHS.Optimizer)
    @test ev.value ≈ 0 atol = 1e-6
    @test ev.x ≈ [7 / 3] atol = 1e-6
    @test expected_result(pb, ev.x; optimizer = HiGHS.Optimizer) ≈ 10 / 9 atol = 1e-6
    @test evpi(pb; optimizer = HiGHS.Optimizer) ≈ 1 atol = 1e-6
    @test vss(pb; optimizer = HiGHS.Optimizer) ≈ 1 / 9 atol = 1e-6
end

@testset "EVPI and VSS: the farmer of Birge and Louveaux" begin
    pb = birge_louveaux_farmer()
    rp = reference(pb)
    @test rp ≈ -108390 atol = 1e-6
    @test wait_and_see(pb; optimizer = HiGHS.Optimizer).value ≈ -115405.5555555 atol = 1e-4
    ev = expected_value_problem(pb; optimizer = HiGHS.Optimizer)
    @test ev.value ≈ -118600 atol = 1e-6
    @test ev.x ≈ [120, 80, 300] atol = 1e-6
    @test expected_result(pb, ev.x; optimizer = HiGHS.Optimizer) ≈ -107240 atol = 1e-6
    for cuts in (:single, :multi)
        @test evpi(pb; optimizer = HiGHS.Optimizer, cuts = cuts) ≈ 7015.5555555 atol = 1e-4
        @test vss(pb; optimizer = HiGHS.Optimizer, cuts = cuts) ≈ 1150 atol = 1e-4
    end
    # given RP is used as is
    @test evpi(pb; rp = rp, optimizer = HiGHS.Optimizer) ≈ 7015.5555555 atol = 1e-4
    @test vss(pb; rp = rp + 1, optimizer = HiGHS.Optimizer) ≈ 1149 atol = 1e-4
end

@testset "EVPI and VSS: a JuMP model, and an infeasible mean-value decision" begin
    # the same instance, as data and as JuMP models
    data, jump = icecream(), icecream_jump()
    @test wait_and_see(jump; optimizer = HiGHS.Optimizer).value ≈
          wait_and_see(data; optimizer = HiGHS.Optimizer).value atol = 1e-6
    x̄ = expected_value_problem(data; optimizer = HiGHS.Optimizer).x
    @test vss(jump, x̄; optimizer = HiGHS.Optimizer) ≈ vss(data; optimizer = HiGHS.Optimizer) atol = 1e-6
    @test evpi(jump; optimizer = HiGHS.Optimizer) ≈ evpi(data; optimizer = HiGHS.Optimizer) atol = 1e-6
    @test evpi(data; optimizer = HiGHS.Optimizer) >= -1e-6
    @test vss(data; optimizer = HiGHS.Optimizer) >= -1e-6
    # without the minimum capacity, the capacity bought for the mean demand is too small for the
    # largest one: EEV, and VSS, are infinite
    pb = icecream(min_capacity = 0.0)
    x̄ = expected_value_problem(pb; optimizer = HiGHS.Optimizer).x
    @test sum(x̄) < 12
    @test expected_result(pb, x̄; optimizer = HiGHS.Optimizer) == Inf
    @test vss(pb; optimizer = HiGHS.Optimizer) == Inf
    # the scenarios of a JuMP model cannot be averaged
    @test_throws MethodError expected_value_problem(jump; optimizer = HiGHS.Optimizer)
end

# --------------------------------------------------------------------------------------------
# partial aggregation, regularization, starting point, callback
# --------------------------------------------------------------------------------------------

# Birge and Louveaux (2011), Section 5.1, Exercise 5: two scenarios, -20 ≤ x ≤ 20, c = 0, and the
# second stage Wy = h - Tx with W, q and h given; the example of regularized decomposition of
# deck 04.
exercise_5() = TwoStageProblem(
    c = [0.0], A = zeros(0, 1), senses1 = Char[], b = Float64[],
    q = ξ -> ξ[1], W = [1.0 -1 -1 -1 0 0; 0 1 0 0 1 0; 0 0 1 0 0 1], senses2 = ['=', '=', '='],
    T = reshape([1.0, 0, 0], 3, 1), h = ξ -> ξ[2],
    ξ = [([1.0, 0, 0, 0, 0, 0], [-1.0, 2, 7]), ([1.5, 0, 2 / 7, 1, 0, 0], [0.0, 2, 7])],
    p = [0.5, 0.5], lb1 = -20.0, ub1 = 20.0)

# Exercise 6: Example 2 of the same section, Q(x, ξ) = |x - ξ| on 0 ≤ x ≤ 10, ξ taking the values
# 0.5, 1, 1.5, 3, 4, 5 with probability 1/9 and 2 with probability 1/3.
exercise_6() = TwoStageProblem(
    c = [0.0], A = ones(1, 1), senses1 = ['<'], b = [10.0],
    q = [1.0, 1.0], W = [1.0 -1.0], senses2 = ['='], T = ones(1, 1), h = ξ -> [ξ],
    ξ = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], p = [1, 1, 1, 3, 1, 1, 1] ./ 9)

@testset "partial aggregation: Exercise 6(c) of Birge and Louveaux" begin
    res = lshaped(exercise_6(); optimizer = HiGHS.Optimizer, cuts = [[1, 2, 3], [4], [5, 6, 7]],
                  x0 = [0.0], verbose = false)
    # the cuts θ₁ ≥ 1 - x, θ₂ ≥ 2 - x, θ₃ ≥ 4 - x at x¹ = 0, then x² = 10 and the cuts θ₁ ≥ x - 1,
    # θ₂ ≥ x - 2, θ₃ ≥ x - 4; "only two major iterations are needed"
    @test [h.x[1] for h in res.history] ≈ [0, 10, 2] atol = 1e-9
    @test res.converged && res.iterations == 3 && res.optimality_cuts == 6
    @test res.objective ≈ 1 atol = 1e-9
    @test count_theta_rows(res.master.model) == 6
    # the partitions `cuts` describes
    clusters = LShaped.scenario_clusters
    @test clusters(:single, 4) == [[1, 2, 3, 4]]
    @test clusters(:multi, 3) == [[1], [2], [3]]
    @test clusters(3, 7) == [[1, 2], [3, 4, 5], [6, 7]]
    @test clusters([[2, 1], [3]], 3) == [[2, 1], [3]]
    @test_throws ErrorException clusters(0, 3)
    @test_throws ErrorException clusters([[1, 2], [2, 3]], 3)
    @test_throws ErrorException clusters([[1], [3]], 3)
    # any partition gives the same optimum, with as many θ as clusters
    pb = sampled_demand()
    for C in (2, 10, 50)
        res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = C, verbose = false)
        @test res.converged
        @test length(res.clusters) == C
        @test res.objective ≈ reference(pb) atol = 1e-6
    end
end

@testset "regularized decomposition: the example of deck 04" begin
    # from a¹ = -0.5, the solution of the expected value problem, and ρ = 1: x² = 0.25, a serious
    # step, then x³ = 0.25, whose model value reaches f(a³) = 0: a³ is optimal
    res = lshaped(exercise_5(); optimizer = HiGHS.Optimizer, cuts = :multi, x0 = [-0.5],
                  regularization = :regularized_decomposition, rho = 1.0, verbose = false)
    @test [h.x[1] for h in res.history] ≈ [-0.5, 0.25] atol = 1e-6
    @test res.converged && res.iterations == 3
    @test res.x ≈ [0.25] atol = 1e-6
    @test res.objective ≈ 0 atol = 1e-9
    @test res.lower_bound ≈ 0 atol = 1e-6            # computed once at the end, without ρ
    # the plain method from x¹ = -2 follows the path of the book: x² = 20, x³ = 12/7
    res = lshaped(exercise_5(); optimizer = HiGHS.Optimizer, x0 = [-2.0], verbose = false)
    @test [h.x[1] for h in res.history][1:3] ≈ [-2, 20, 12 / 7] atol = 1e-9
end

@testset "regularizations reach the optimum" begin
    instances = [("ice cream", icecream()), ("feasibility cuts", icecream(min_capacity = 0.0)),
                 ("bounded recourse", icecream(ub = 3.0)), ("farmer", birge_louveaux_farmer()),
                 ("exercise 5", exercise_5())]
    # HiGHS's QP solver is not reliable on the masters of regularized decomposition: Ipopt solves
    # them, if it does not relax the constraints
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "bound_relax_factor" => 0.0, "sb" => "yes")
    for (name, pb) in instances, regularization in (:regularized_decomposition, :trust_region),
        cuts in (:single, :multi)
        master = regularization == :trust_region ? HiGHS.Optimizer : ipopt
        res = lshaped(pb; master_optimizer = master, recourse_optimizer = HiGHS.Optimizer, cuts = cuts,
                      regularization = regularization, maxiter = 2000, tol = 1e-7, verbose = false)
        @test res.converged
        optimal = reference(pb)
        @test res.objective ≈ optimal rtol = 1e-6 atol = 1e-6
        @test res.lower_bound <= optimal + 1e-6 * (1 + abs(optimal))
    end
    # a small trust region keeps the first step close to the starting point
    res = lshaped(exercise_5(); optimizer = HiGHS.Optimizer, x0 = [-2.0],
                  regularization = :trust_region, radius = 0.5, verbose = false)
    @test abs(res.history[2].x[1] - (-2.0)) <= 0.5 + 1e-9
    @test res.converged
    @test res.objective ≈ 0 atol = 1e-9
    # what is wrong with the parameters is reported
    pb = icecream()
    @test_throws ErrorException lshaped(pb; optimizer = HiGHS.Optimizer, regularization = :proximal)
    @test_throws ErrorException lshaped(pb; optimizer = HiGHS.Optimizer, rho = 0)
    @test_throws ErrorException lshaped(pb; optimizer = HiGHS.Optimizer, eta1 = 0.5, eta2 = 0.1)
    @test_throws ErrorException lshaped(pb; optimizer = HiGHS.Optimizer, gamma = 1)
end

@testset "regularized decomposition: a quadratic master solved, or an error" begin
    # HiGHS's QP solver may declare a wrong solution of the master optimal: the method checks the
    # last master before stopping, and raises an error rather than return a wrong decision
    # the farmer with three scenarios of sampled yields, on which HiGHS 1.15 fails
    rng = substream(2)
    yields = [[2.5, 3.0, 20.0] .* rand(rng, Uniform(0.8, 1.2), 3) for _ in 1:3]
    sampled_farmer = TwoStageProblem(
        c = [150.0, 230.0, 260.0], A = ones(1, 3), senses1 = ['<'], b = [500.0],
        q = [238.0, 210.0, -170.0, -150.0, -36.0, -10.0],
        W = [1.0 0 -1 0 0 0; 0 1 0 -1 0 0; 0 0 0 0 1 1; 0 0 0 0 1 0], senses2 = ['>', '>', '<', '<'],
        T = t -> [t[1] 0 0; 0 t[2] 0; 0 0 -t[3]; 0 0 0], h = [200.0, 240.0, 0.0, 6000.0],
        ξ = yields, p = fill(1 / 3, 3))
    for pb in (sampled_demand(), farm(), birge_louveaux_farmer(), sampled_farmer)
        res = try
            lshaped(pb; optimizer = HiGHS.Optimizer, cuts = :multi,
                    regularization = :regularized_decomposition, verbose = false)
        catch e
            e
        end
        if res isa Exception
            @test occursin("the quadratic master problem was not solved", sprint(showerror, res))
        else
            @test res.converged
            @test res.objective ≈ reference(pb) rtol = 1e-6
        end
    end
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "bound_relax_factor" => 0.0, "sb" => "yes")
    res = lshaped(sampled_farmer; master_optimizer = ipopt, recourse_optimizer = HiGHS.Optimizer,
                  cuts = :multi, regularization = :regularized_decomposition, verbose = false)
    @test res.converged
    @test res.objective ≈ reference(sampled_farmer) rtol = 1e-6
    @test res.lower_bound <= res.objective + 1e-6 * (1 + abs(res.objective))
    @test isfinite(res.lower_bound)
end

@testset "starting point, callback, and bounds on the EVPI and the VSS" begin
    pb = birge_louveaux_farmer()
    # x0 = :mean_value starts from the solution of the expected value problem
    res = lshaped(pb; optimizer = HiGHS.Optimizer, x0 = :mean_value, verbose = false)
    @test res.history[1].x ≈ [120, 80, 300] atol = 1e-6
    @test res.converged
    @test res.objective ≈ -108390 atol = 1e-6
    @test_throws ErrorException lshaped(pb; optimizer = HiGHS.Optimizer, x0 = [1.0, 2.0])
    @test_throws ErrorException lshaped(icecream_jump(); optimizer = HiGHS.Optimizer, x0 = :mean_value)
    # the bounds bracket the EVPI and the VSS at every iteration, and the method can stop as soon
    # as the VSS is known to be positive
    ws = wait_and_see(pb; optimizer = HiGHS.Optimizer).value
    eev = expected_result(pb, expected_value_problem(pb; optimizer = HiGHS.Optimizer).x;
                          optimizer = HiGHS.Optimizer)
    full = lshaped(pb; optimizer = HiGHS.Optimizer, verbose = false)
    for entry in full.history
        lo, hi = evpi_bounds(entry, ws)
        @test lo - 1e-6 <= 7015.5555555 <= hi + 1e-6
        lo, hi = vss_bounds(entry, eev)
        @test lo - 1e-6 <= 1150 <= hi + 1e-6
    end
    @test all(isapprox.(evpi_bounds(full, ws), 7015.5555555; atol = 1e-4))
    early = lshaped(pb; optimizer = HiGHS.Optimizer, verbose = false,
                    callback = entry -> vss_bounds(entry, eev)[1] > 0)
    @test early.stopped && !early.converged
    @test early.iterations < full.iterations
    @test vss_bounds(early, eev)[1] > 0
end

# --------------------------------------------------------------------------------------------
# sparse data, a random recourse matrix, SMPS files
# --------------------------------------------------------------------------------------------

@testset "sparse matrices, and a recourse matrix that depends on the scenario" begin
    pb = sampled_demand(n = 50)
    sp = TwoStageProblem(c = pb.c, A = sparse(pb.A), senses1 = pb.senses1, b = pb.b,
                         q = pb.q, W = sparse(pb.W(nothing)), senses2 = pb.senses2,
                         T = ξ -> sparse(pb.T(ξ)), h = pb.h, ξ = pb.ξ, p = pb.p)
    @test sp.A isa SparseMatrixCSC
    dense, sparse_ = (lshaped(q; optimizer = HiGHS.Optimizer, cuts = :multi, verbose = false)
                      for q in (pb, sp))
    @test sparse_.objective ≈ dense.objective rtol = 1e-9
    @test sparse_.x ≈ dense.x atol = 1e-7
    # W(ξ): the yield of the second-stage technology is random
    random_W = TwoStageProblem(c = [1.0], A = zeros(0, 1), senses1 = Char[], b = Float64[],
        q = [3.0], W = ξ -> fill(ξ, 1, 1), senses2 = ['>'], T = ones(1, 1), h = [10.0],
        ξ = [1.0, 2.0], p = [0.5, 0.5], ub1 = 20.0)
    res = lshaped(random_W; optimizer = HiGHS.Optimizer, cuts = :multi, verbose = false)
    @test res.converged
    @test res.objective ≈ reference(random_W) atol = 1e-8
    @test res.x ≈ [10.0] atol = 1e-8        # x = 10 costs 10, the recourse 0.5(30 + 15) = 22.5
    # without the bound on x, the first cuts leave the master unbounded: it is solved in a box,
    # which yields cuts but no lower bound, until they bound θ
    unbounded = TwoStageProblem(c = [1.0], A = zeros(0, 1), senses1 = Char[], b = Float64[],
        q = [3.0], W = ξ -> fill(ξ, 1, 1), senses2 = ['>'], T = ones(1, 1), h = [10.0],
        ξ = [1.0, 2.0], p = [0.5, 0.5])
    for cuts in (:single, :multi)
        res = lshaped(unbounded; optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test res.converged
        @test res.objective ≈ 10 atol = 1e-8
        @test res.history[1].lower_bound == -Inf
    end
end

# A small instance written in the SMPS format: an integer first-stage variable, a bound on each
# stage, rows of the three senses, and four random elements — a right-hand side (INDEP), and in one
# BLOCKS block a cost, a coefficient of T and one of W.
const TOY_CORE = """
NAME          TOY
ROWS
 N  COST
 L  BUDGET
 G  DEMAND
 L  CAP1
 E  BAL
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X1        COST         2.0   BUDGET       1.0
    X1        CAP1        -1.0
    MARKER                 'MARKER'                 'INTEND'
    X2        COST         3.0   BUDGET       1.0
    X2        BAL         -1.0
    Y1        COST         4.0   DEMAND       1.0
    Y1        CAP1         1.0
    Y2        COST         6.0   DEMAND       1.0
    Y2        BAL          1.0
    Y3        COST        10.0   DEMAND       1.0
RHS
    RHS       BUDGET      10.0   DEMAND       5.0
BOUNDS
 UP BND       X1           6.0
 UP BND       Y3          20.0
ENDATA
"""
const TOY_TIME = """
TIME          TOY
PERIODS       LP
    X1        BUDGET                   STAGE1
    Y1        DEMAND                   STAGE2
ENDATA
"""
const TOY_TIME_EXPLICIT = """
TIME          TOY
PERIODS       EXPLICIT
    STAGE1
    STAGE2
COLUMNS
    X1        STAGE1
    X2        STAGE1
    Y1        STAGE2
    Y2        STAGE2
    Y3        STAGE2
ROWS
    BUDGET    STAGE1
    DEMAND    STAGE2
    CAP1      STAGE2
    BAL       STAGE2
ENDATA
"""
const TOY_STOCH = """
STOCH         TOY
* the demand, and the prices of the second stage, independent of it
INDEP         DISCRETE
    RHS       DEMAND       4.0   STAGE2   0.3
    RHS       DEMAND       7.0   STAGE2   0.7
BLOCKS        DISCRETE
 BL PRICES    STAGE2       0.5
    Y3        COST        10.0
    X1        CAP1        -1.0
 BL PRICES    STAGE2       0.5
    Y3        COST        14.0
    X1        CAP1        -0.5
    Y1        CAP1         2.0
ENDATA
"""
const TOY_SCENARIOS = """
STOCH         TOY
SCENARIOS     DISCRETE
 SC SCEN1     ROOT         0.4       STAGE2
    RHS       DEMAND       4.0
    Y3        COST        12.0
 SC SCEN2     ROOT         0.6       STAGE2
    RHS       DEMAND       8.0
ENDATA
"""

"""The toy instance written by hand; ξ = (demand, cost of y₃, T entry, W entry)."""
toy_by_hand(ξ, p) = TwoStageProblem(
    c = [2.0, 3.0], A = [1.0 1.0], senses1 = ['<'], b = [10.0], ub1 = [6.0, Inf],
    integer1 = [true, false],
    q = ξ -> [4.0, 6.0, ξ[2]], W = ξ -> [1.0 1.0 1.0; ξ[4] 0.0 0.0; 0.0 1.0 0.0],
    senses2 = ['>', '<', '='], T = ξ -> [0.0 0.0; ξ[3] 0.0; 0.0 -1.0], h = ξ -> [ξ[1], 0.0, 0.0],
    ub = [Inf, Inf, 20.0], ξ = ξ, p = p)

function write_toy(dir, stoch; time = TOY_TIME)
    for (ext, text) in (("cor", TOY_CORE), ("tim", time), ("sto", stoch))
        write(joinpath(dir, "toy.$ext"), text)
    end
    return joinpath(dir, "toy")
end

@testset "SMPS files: the core, the time file, INDEP and BLOCKS sections" begin
    mktempdir() do dir
        smps = read_smps(write_toy(dir, TOY_STOCH))
        @test smps.name == "TOY"
        @test smps.columns1 == ["X1", "X2"] && smps.columns2 == ["Y1", "Y2", "Y3"]
        @test smps.rows1 == ["BUDGET"] && smps.rows2 == ["DEMAND", "CAP1", "BAL"]
        @test smps.integer1 == [true, false]
        @test smps.ub1 == [6.0, Inf] && smps.ub == [Inf, Inf, 20.0]
        @test smps.elements == [(:h, 1, 0), (:q, 3, 0), (:T, 2, 1), (:W, 2, 1)]
        @test smps.law.base == [5.0, 10.0, -1.0, 1.0]
        @test support_size(smps.law) == 4
        ξ, p = enumerate_scenarios(smps.law)
        @test sort(collect(zip(ξ, p))) == sort([([4.0, 10.0, -1.0, 1.0], 0.15),
                                                ([4.0, 14.0, -0.5, 2.0], 0.15),
                                                ([7.0, 10.0, -1.0, 1.0], 0.35),
                                                ([7.0, 14.0, -0.5, 2.0], 0.35)])
        # the problem read, and the same written by hand, have the same optimum
        pb = TwoStageProblem(smps; ξ = ξ, p = p)
        @test pb.A isa SparseMatrixCSC
        _, x, value = extensive_form(pb; optimizer = HiGHS.Optimizer)
        _, xh, valueh = extensive_form(toy_by_hand(ξ, p); optimizer = HiGHS.Optimizer)
        @test value ≈ valueh atol = 1e-9
        @test x ≈ xh atol = 1e-9
        res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = :multi, verbose = false)
        @test res.converged
        @test res.objective ≈ value atol = 1e-7
        # the explicit time format reads the same
        explicit = read_smps(joinpath(dir, "toy.cor"),
                             (write(joinpath(dir, "toy2.tim"), TOY_TIME_EXPLICIT); joinpath(dir, "toy2.tim")),
                             joinpath(dir, "toy.sto"))
        @test explicit.rows2 == smps.rows2 && explicit.columns2 == smps.columns2
        @test explicit.T == smps.T && explicit.W == smps.W
        # the law is a sampler: reproducible, with the right frequencies
        sample = rand(substream(5), smps.law, 4000)
        @test sample == rand(substream(5), smps.law, 4000)
        @test 0.27 < count(v -> v[1] == 4.0, sample) / 4000 < 0.33
        @test all(v -> (v[2] == 10.0) == (v[3] == -1.0) == (v[4] == 1.0), sample)   # one block
        ξs, ps = sample_scenarios(smps.law, 100; rng = substream(2))
        @test length(ξs) == 100 && ps == fill(0.01, 100)
        @test TwoStageProblem(smps; ξ = ξs, p = ps) isa TwoStageProblem
        # the mean-value problem averages each element
        ev = expected_value_problem(pb; optimizer = HiGHS.Optimizer)
        evh = expected_value_problem(toy_by_hand(ξ, p); optimizer = HiGHS.Optimizer)
        @test ev.value ≈ evh.value atol = 1e-9
    end
end

@testset "SMPS files: SCENARIOS, and what is not supported" begin
    mktempdir() do dir
        smps = read_smps(write_toy(dir, TOY_SCENARIOS))
        ξ, p = enumerate_scenarios(smps.law)
        @test ξ == [[4.0, 12.0], [8.0, 10.0]] && p ≈ [0.4, 0.6]
        @test smps.elements == [(:h, 1, 0), (:q, 3, 0)]
        @test_throws ErrorException enumerate_scenarios(smps.law; limit = 1)
        # three stages
        three = replace(TOY_TIME, "ENDATA" => "    Y3        BAL                      STAGE3\nENDATA")
        @test_throws ErrorException read_smps(write_toy(dir, TOY_STOCH; time = three))
        # a random first-stage cost, a continuous distribution, probabilities not summing to one
        for stoch in (replace(TOY_SCENARIOS, "Y3        COST        12.0" => "X1        COST        12.0"),
                      replace(TOY_STOCH, "INDEP         DISCRETE" => "INDEP         NORMAL"),
                      replace(TOY_STOCH, "STAGE2   0.7" => "STAGE2   0.6"))
            @test_throws ErrorException read_smps(write_toy(dir, stoch))
        end
    end
end

# --------------------------------------------------------------------------------------------
# integer recourse: the integer L-shaped method
# --------------------------------------------------------------------------------------------

# Birge and Louveaux (2011), Section 7.2, Example 1: binary x, the second stage
# min -2y₁ - 3y₂ s.t. y₁ + 2y₂ ≤ ξ₁ - x₁, y₁ ≤ ξ₂ - x₂, y integer, ξ = (2, 2) or (4, 3); the
# first-stage costs `c` are ours.
integer_example(c) = TwoStageProblem(
    c = c, A = zeros(0, 2), senses1 = Char[], b = Float64[],
    q = [-2.0, -3.0], W = [1.0 2.0; 1.0 0.0], senses2 = ['<', '<'], T = [1.0 0.0; 0.0 1.0],
    h = ξ -> ξ, ξ = [[2.0, 2.0], [4.0, 3.0]], p = [0.5, 0.5], ub1 = 1.0, integer1 = true,
    integer2 = true)

# A server location problem in the spirit of SSLP (Ntaimo and Sen, 2005): open servers j (binary
# x, cost f), clients i present at random; each present client assigned to one server (binary y,
# revenue r), d per client on a server of capacity u xⱼ, an overflow z ≥ 0 at a penalty.
function server_location(J, I, S; seed = 1)
    rng = substream(seed)
    f = rand(rng, 40:80, J) .* 1.0
    r = rand(rng, 5:25, I, J) .* 1.0
    d = rand(rng, 5:25, I) .* 1.0
    u = sum(d) / 2
    yi(i, j) = (i - 1) * J + j
    W, T = zeros(J + I, I * J + J), zeros(J + I, J)
    for j in 1:J
        for i in 1:I
            W[j, yi(i, j)] = d[i]
            W[J + i, yi(i, j)] = 1.0
        end
        W[j, I * J + j] = -1.0
        T[j, j] = -u
    end
    return TwoStageProblem(c = f, A = zeros(0, J), senses1 = Char[], b = Float64[],
        q = vcat([-r[i, j] for i in 1:I for j in 1:J], fill(1000.0, J)), W = W,
        senses2 = vcat(fill('<', J), fill('=', I)), T = T, h = ξ -> vcat(zeros(J), ξ),
        ξ = [Float64.(rand(rng, I) .< 0.5) for _ in 1:S], p = fill(1 / S, S),
        ub1 = 1.0, integer1 = true, ub = vcat(ones(I * J), fill(Inf, J)),
        integer2 = vcat(trues(I * J), falses(J)))
end

@testset "integer recourse: the cut of Birge and Louveaux, Section 7.2" begin
    # at x = (0, 1): L = -5.75, q_S = Q(x) = -5, and the cut θ ≥ 0.75(x₂ - x₁) - 5.75
    res = lshaped(integer_example([-1.0, -1.0]); optimizer = HiGHS.Optimizer, x0 = [0.0, 1.0],
                  maxiter = 1, verbose = false)
    @test res.history[1].Q ≈ -5
    @test res.integer_cuts == 1
    cuts = [constraint_object(con) for con in all_constraints(res.master.model, AffExpr,
                                                               MOI.GreaterThan{Float64})]
    integer_cut = only(filter(o -> length(o.func.terms) == 3 &&
                                   any(v -> name(v) == "x[1]" && o.func.terms[v] > 0, keys(o.func.terms)), cuts))
    coefficient(o, label) = only(a for (v, a) in o.func.terms if name(v) == label)
    @test coefficient(integer_cut, "x[1]") ≈ 0.75
    @test coefficient(integer_cut, "x[2]") ≈ -0.75
    @test coefficient(integer_cut, "θ[1]") ≈ 1
    @test MOI.constant(integer_cut.set) ≈ -5.75
    # and the optimum, for several first-stage costs
    for c in ([-1.0, -1.0], [-0.5, -2.5], [-3.0, -0.2], [0.0, 0.0]), cuts in (:single, :multi)
        pb = integer_example(c)
        res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test res.converged
        @test res.objective ≈ reference(pb) atol = 1e-9
        @test res.lower_bound ≈ res.objective atol = 1e-9
    end
end

@testset "integer recourse: server location, a feasibility cut, given bounds" begin
    for (J, I, S) in ((3, 8, 10), (5, 15, 20)), cuts in (:single, :multi, 4)
        pb = server_location(J, I, S)
        res = lshaped(pb; optimizer = HiGHS.Optimizer, cuts = cuts, verbose = false)
        @test res.converged
        @test res.objective ≈ reference(pb) rtol = 1e-6
        @test all(isinteger, res.x)
        @test res.integer_cuts > 0
        # the second-stage decisions are integer, and their cost is the recourse value
        y = second_stage_decision(res, 1)
        @test all(isinteger, round.(y[1:(I * J)], digits = 6))
    end
    GC.gc()
    pb = server_location(3, 8, 10)
    @test lshaped(pb; optimizer = HiGHS.Optimizer, cuts = :multi, threads = true,
                  verbose = false).objective ≈ reference(pb) rtol = 1e-6
    # 2y = ξ - x with y integer: x = 1 leaves the LP relaxation feasible and the integer program
    # not; the integer feasibility cut x ≤ 0 removes it, although x = 1 is cheaper
    parity = TwoStageProblem(c = [-1.0], A = zeros(0, 1), senses1 = Char[], b = Float64[],
        q = [1.0], W = fill(2.0, 1, 1), senses2 = ['='], T = ones(1, 1), h = ξ -> [ξ],
        ξ = [2.0, 4.0], p = [0.5, 0.5], ub1 = 1.0, integer1 = true, integer2 = true)
    res = lshaped(parity; optimizer = HiGHS.Optimizer, verbose = false)
    @test res.converged && res.x == [0.0] && res.feasibility_cuts == 1
    @test res.objective ≈ reference(parity) atol = 1e-9
    # the bounds L, given rather than computed
    pb = integer_example([-1.0, -1.0])
    for bound in ([-4.0, -7.5], -100.0)
        res = lshaped(pb; optimizer = HiGHS.Optimizer, recourse_bound = bound, verbose = false)
        @test res.converged
        @test res.objective ≈ -6 atol = 1e-9
    end
    # what the method cannot do
    general = integer_example([-1.0, -1.0])
    general.integer1 .= false            # continuous tender variables
    @test_throws ErrorException lshaped(general; optimizer = HiGHS.Optimizer, verbose = false)
    @test_throws ErrorException lshaped(pb; optimizer = HiGHS.Optimizer, verbose = false,
                                        regularization = :trust_region)
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
    # integer recourse needs binary tender variables, here continuous
    @test_throws "must be binary" lshaped(one_variable(integer = true); optimizer = HiGHS.Optimizer,
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
