#=
nested_benders.jl -- a pedagogical implementation of nested Benders decomposition, the nested
L-shaped method of Birge (1985), for multistage stochastic linear programs on a scenario tree.

Slides: "06. Multistage problems" (nested decomposition, cuts, forward and backward passes) and
"07. SDDP" (the same method, then its sampled version under stagewise independence). The notebook
nested_benders.ipynb develops this code step by step, and runs the examples.

The aim is to expose the method, not to be fast: one JuMP model per node of the tree, every node
solved at every iteration, every cut kept. The user writes the problem of a node with JuMP, as an
ordinary JuMP model; the algorithm only adds what decomposition needs (a copy of the parent's
decision, the variable theta and the cuts).

    include("nested_benders.jl")              # the definitions only
    julia --project=. nested_benders.jl       # also runs the example of Birge and Louveaux, Section 1.2
=#

using JuMP
using HiGHS
using Printf

# ── 1. Scenario tree ─────────────────────────────────────────────────────────────────────────────

"""
    Node(stage, parent, probability, data)

A node of the scenario tree: its stage `t`, its parent `a(k)` (`0` at the root), the conditional
probability `p(k, t | a(k), t-1)` of the branch from its parent, and the data `ξ_t` observed when the
node is reached, in whatever form the model builder expects.
"""
struct Node
    stage::Int
    parent::Int
    probability::Float64
    data::Any
end

"""
    ScenarioTree(root_data = nothing)

A tree reduced to its root, node 1. Nodes are added with [`add_node!`](@ref) and numbered in that
order, so that a parent always comes before its children.
"""
struct ScenarioTree
    nodes::Vector{Node}
    children::Vector{Vector{Int}}
end

ScenarioTree(root_data = nothing) = ScenarioTree([Node(1, 0, 1.0, root_data)], [Int[]])

"""
    add_node!(tree, parent, probability, data) -> k

Add a child to node `parent`, reached with the conditional probability `probability`, where `data`
is observed. Return its index `k`.
"""
function add_node!(tree::ScenarioTree, parent::Int, probability::Real, data)
    push!(tree.nodes, Node(tree.nodes[parent].stage + 1, parent, probability, data))
    push!(tree.children, Int[])
    k = length(tree.nodes)
    push!(tree.children[parent], k)
    return k
end

nstages(tree::ScenarioTree) = maximum(node.stage for node in tree.nodes)
stage_nodes(tree::ScenarioTree, t) = [k for (k, node) in enumerate(tree.nodes) if node.stage == t]
isleaf(tree::ScenarioTree, k) = isempty(tree.children[k])

"""Unconditional probability of node `k`: the product of the branch probabilities from the root."""
function node_probability(tree::ScenarioTree, k)
    p = 1.0
    while k != 0
        p *= tree.nodes[k].probability
        k = tree.nodes[k].parent
    end
    return p
end

"""
    stagewise_tree(outcomes, probabilities; root_data = nothing)

The full tree in which every node of stage `t` has one child per element of `outcomes[t]`, reached
with the probabilities `probabilities[t]`: stagewise independent data, `length(outcomes) + 1` stages.
Any other tree is built with [`add_node!`](@ref).
"""
function stagewise_tree(outcomes, probabilities; root_data = nothing)
    tree = ScenarioTree(root_data)
    frontier = [1]
    for (ξs, ps) in zip(outcomes, probabilities)
        frontier = [add_node!(tree, k, p, ξ) for k in frontier for (ξ, p) in zip(ξs, ps)]
    end
    return tree
end

"""Check that the branch probabilities out of every node sum up to one."""
function check_tree(tree::ScenarioTree)
    for (j, K) in enumerate(tree.children)
        isempty(K) && continue
        total = sum(tree.nodes[k].probability for k in K)
        isapprox(total, 1; atol = 1e-9) ||
            error("the branch probabilities out of node $j sum up to $total, not 1")
    end
end

# ── 2. The problem of a node, written with JuMP ──────────────────────────────────────────────────
#
# The user describes the problem of a node by a function
#
#     build!(model, node, xin) -> xout
#
# which adds to the JuMP model `model` the variables and constraints of node `node`, sets its stage
# cost with @objective(model, Min, ...), and returns the vector of the decisions passed on to the
# children (empty at a leaf). `xin` stands for the decision of the parent, x_{t-1}, in the same order
# as the parent returned it (empty at the root); the builder uses it in its constraints, but does not
# bound it. With W_t x_t = h_t - T_{t-1} x_{t-1}, the builder simply writes W_t x_t + T_{t-1} xin == h_t.

"""Turn what a builder returns into a vector of variables."""
collect_variables(x::Nothing) = VariableRef[]
collect_variables(x::VariableRef) = [x]
collect_variables(x) = VariableRef[v for v in x]

"""The stage cost `c_t' x_t`, read off the objective the builder has set."""
function stage_cost_expression(model::Model)
    sense = objective_sense(model)
    sense == FEASIBILITY_SENSE && return AffExpr(0.0)
    sense == MIN_SENSE || error("write the stage cost as a minimization: max f is min -f")
    return convert(AffExpr, objective_function(model))
end

"""
    NodeProblem

The JuMP model of one node, as decomposition sees it:

    min  c' x + θ
    s.t. constraints of the builder, in xin and x,
         xin + s⁺ - s⁻ = x̂        (copy of the parent's trial decision; multipliers λ)
         feasibility and optimality cuts on xout,
         θ ≥ lower bound,  s⁺, s⁻ ∈ [0, 0]

`s⁺` and `s⁻` are only freed in phase 1, when the node is infeasible at `x̂` (see `phase_one!`).
At a leaf there is no `θ`.
"""
mutable struct NodeProblem
    model::Model
    xin::Vector{VariableRef}
    xout::Vector{VariableRef}
    copy::Vector{ConstraintRef}
    s⁺::Vector{VariableRef}
    s⁻::Vector{VariableRef}
    stage_cost::AffExpr
    θ::Union{VariableRef,Nothing}
    optimality_cuts::Vector{Any}   # (value, slope, point) of each cut θ ≥ value + slope'(xout - point)
    feasibility_cuts::Vector{Any}  # (value, slope, point) of each cut value + slope'(xout - point) ≤ 0
    solves::Int                    # number of linear programs solved at this node
end

function NodeProblem(build!::Function, node::Node, nin::Int, leaf::Bool, optimizer, lower_bound)
    model = Model(optimizer)
    set_silent(model)
    xin = @variable(model, [1:nin], base_name = "xin")
    xout = collect_variables(build!(model, node, xin))
    stage_cost = stage_cost_expression(model)
    s⁺ = @variable(model, [1:nin], lower_bound = 0, upper_bound = 0, base_name = "s⁺")
    s⁻ = @variable(model, [1:nin], lower_bound = 0, upper_bound = 0, base_name = "s⁻")
    copy = ConstraintRef[@constraint(model, xin[i] + s⁺[i] - s⁻[i] == 0) for i in 1:nin]
    θ = leaf ? nothing : @variable(model, lower_bound = lower_bound, base_name = "θ")
    np = NodeProblem(model, xin, xout, copy, s⁺, s⁻, stage_cost, θ, [], [], 0)
    set_phase_two_objective!(np)
    return np
end

set_phase_two_objective!(np::NodeProblem) =
    @objective(np.model, Min, np.θ === nothing ? np.stage_cost : np.stage_cost + np.θ)

# ── 3. Solving a node: phase 2, and phase 1 when it is infeasible ────────────────────────────────

"""
    solve_node!(np, x̂) -> (feasible, value, λ)

Solve the node for the trial decision `x̂` of its parent. If the node is feasible, `value` is its
optimal value `v_{t,k}` (stage cost plus θ) and `λ` the Lagrange multipliers of the copy
constraints `xin = x̂` (JuMP's `dual`), a subgradient of `v_{t,k}` with respect to `x̂`: its
derivative wherever `v_{t,k}` is differentiable. Otherwise, see [`phase_one!`](@ref).
"""
function solve_node!(np::NodeProblem, x̂::Vector{Float64})
    set_normalized_rhs.(np.copy, x̂)
    optimize!(np.model)
    np.solves += 1
    status = termination_status(np.model)
    if status == OPTIMAL
        return (feasible = true, value = objective_value(np.model), λ = dual.(np.copy))
    elseif status in (INFEASIBLE, INFEASIBLE_OR_UNBOUNDED)
        return phase_one!(np, x̂)
    else
        error("node problem not solved: $status")
    end
end

"""
    phase_one!(np, x̂) -> (feasible = false, value = w, λ = μ)

The node is infeasible at `x̂`. Free the artificial variables of the copy constraints and minimize
their sum: the optimal value `w(x̂) > 0` is the distance (in norm 1) from `x̂` to the parent decisions
for which the node is feasible. `w` is convex, zero exactly on these decisions, and `μ`, the
multipliers of the copy constraints, is a subgradient of `w` at `x̂`, so that

    w(x̂) + μ'(x - x̂) ≤ w(x) = 0    for every parent decision x at which the node is feasible:

the feasibility cut. Relaxing the copy constraints, rather than every row as in the phase 1 of the
L-shaped slides, leaves the builder's model untouched.
"""
function phase_one!(np::NodeProblem, x̂::Vector{Float64})
    isempty(np.copy) && error("the first-stage problem is infeasible")
    delete_upper_bound.(np.s⁺)
    delete_upper_bound.(np.s⁻)
    @objective(np.model, Min, sum(np.s⁺) + sum(np.s⁻))
    optimize!(np.model)
    np.solves += 1
    status = termination_status(np.model)
    status == INFEASIBLE &&     # its own constraints and feasibility cuts admit no solution at all
        error("a node is infeasible whatever the decision of its parent: the problem is infeasible")
    status == OPTIMAL || error("phase 1 not solved: $status")
    w, μ = objective_value(np.model), dual.(np.copy)
    set_upper_bound.(np.s⁺, 0.0)
    set_upper_bound.(np.s⁻, 0.0)
    set_phase_two_objective!(np)
    w > 1e-7 || error("the node problem is unbounded at x̂ = $x̂ (phase 1 value $w)")
    return (feasible = false, value = w, λ = μ)
end

# ── 4. Cuts ──────────────────────────────────────────────────────────────────────────────────────

"""
    add_optimality_cut!(np, x̂, results, probabilities)

Node `j` (`np`) has solved its children at its trial decision `x̂`, with the values `v_k` and the
multipliers `λ_k` in `results`. The supporting hyperplanes `v_k + λ_k'(x - x̂)`, averaged with the
conditional probabilities, give the cut

    θ_j ≥ Σ_k p(k | j) (v_k + λ_k'(x_j - x̂)).

In the notation of the slides, `λ_k = -T_{t-1,k}' π_{t,k}`.
"""
function add_optimality_cut!(np::NodeProblem, x̂, results, probabilities)
    value = sum(p * r.value for (p, r) in zip(probabilities, results))
    slope = sum(p * r.λ for (p, r) in zip(probabilities, results))
    @constraint(np.model, np.θ >= value + sum(slope[i] * (np.xout[i] - x̂[i]) for i in eachindex(x̂)))
    push!(np.optimality_cuts, (value = value, slope = slope, point = copy(x̂)))
end

"""
    add_feasibility_cut!(np, x̂, r)

A child of node `j` (`np`) is infeasible at the trial decision `x̂`; `r` holds its phase 1 value `w`
and multipliers `μ`. Add `w + μ'(x_j - x̂) ≤ 0`, which `x̂` violates.
"""
function add_feasibility_cut!(np::NodeProblem, x̂, r)
    @constraint(np.model, r.value + sum(r.λ[i] * (np.xout[i] - x̂[i]) for i in eachindex(x̂)) <= 0)
    push!(np.feasibility_cuts, (value = r.value, slope = r.λ, point = copy(x̂)))
end

# ── 5. Nested decomposition: forward and backward passes ─────────────────────────────────────────

"""
    NestedProblem(tree, build!; optimizer, lower_bound)

One [`NodeProblem`](@ref) per node of `tree`, built by `build!`. `lower_bound` must be a valid lower
bound on the expected cost-to-go of every node: θ is bounded below by it until its first cut, as in
the L-shaped method, so that the first-stage value is a lower bound from the first iteration on.
"""
struct NestedProblem
    tree::ScenarioTree
    nodes::Vector{NodeProblem}
end

function NestedProblem(tree::ScenarioTree, build!::Function; optimizer, lower_bound::Real)
    check_tree(tree)
    nodes = NodeProblem[]
    for (k, node) in enumerate(tree.nodes)              # a parent comes before its children
        nin = node.parent == 0 ? 0 : length(nodes[node.parent].xout)
        push!(nodes, NodeProblem(build!, node, nin, isleaf(tree, k), optimizer, lower_bound))
    end
    return NestedProblem(tree, nodes)
end

"""The trial decision a node receives: its parent's, or nothing at the root."""
parent_decision(nb::NestedProblem, x̂, k) =
    nb.tree.nodes[k].parent == 0 ? Float64[] : x̂[nb.tree.nodes[k].parent]

"""
    forward_pass!(nb, x̂, results) -> (feasible, lower, upper)

Solve the nodes stage by stage, each at the trial decision of its parent, and store the trial
decisions in `x̂` and the solutions in `results`. The first-stage value is a lower bound. If every
node is feasible, the trial decisions are a nonanticipative policy, and its expected cost,
`Σ_k P(k) c_k' x̂_k`, an upper bound. If some nodes of a stage are infeasible, their parents receive
feasibility cuts and the pass stops there.
"""
function forward_pass!(nb::NestedProblem, x̂, results)
    tree, lower, upper = nb.tree, NaN, 0.0
    for t in 1:nstages(tree)
        infeasible = false
        for k in stage_nodes(tree, t)
            r = solve_node!(nb.nodes[k], parent_decision(nb, x̂, k))
            results[k] = r
            if r.feasible
                x̂[k] = value.(nb.nodes[k].xout)
                upper += node_probability(tree, k) * value(nb.nodes[k].stage_cost)
                t == 1 && (lower = r.value)
            else
                parent = tree.nodes[k].parent
                add_feasibility_cut!(nb.nodes[parent], x̂[parent], r)
                infeasible = true
            end
        end
        infeasible && return (feasible = false, lower = lower, upper = Inf)
    end
    return (feasible = true, lower = lower, upper = upper)
end

"""
    backward_pass!(nb, x̂, results)

For `t = H, …, 2`, every node of stage `t - 1` receives the optimality cut built from its children,
and is solved again at its parent's trial decision; its new value and multipliers are what its own
parent's cut is built from, so that the cuts carry information up the tree.
"""
function backward_pass!(nb::NestedProblem, x̂, results)
    tree = nb.tree
    for t in nstages(tree):-1:2
        for j in stage_nodes(tree, t - 1)
            isleaf(tree, j) && continue
            K = tree.children[j]
            add_optimality_cut!(nb.nodes[j], x̂[j], results[K], [tree.nodes[k].probability for k in K])
            results[j] = solve_node!(nb.nodes[j], parent_decision(nb, x̂, j))
            results[j].feasible || error("node $j has become infeasible")   # cannot happen
        end
    end
end

"""
    nested_benders!(nb; max_iterations = 100, tolerance = 1e-6, verbose = true)

Alternate forward and backward passes until the bounds meet, up to the relative `tolerance`. When it
stops, every node model holds the solution of the last forward pass: the policy whose expected cost is
the upper bound. Returns the status, the bounds, the trial decisions `x̂` per node, and the history.
"""
function nested_benders!(nb::NestedProblem; max_iterations::Int = 100, tolerance::Real = 1e-6,
                         verbose::Bool = true)
    n = length(nb.tree.nodes)
    x̂ = Vector{Vector{Float64}}(undef, n)
    results = Vector{Any}(undef, n)
    history = NamedTuple[]
    verbose && @printf("%5s %14s %14s %11s %6s %6s %6s\n",
                       "iter", "lower bound", "upper bound", "gap", "opt.", "feas.", "LPs")
    for ν in 1:max_iterations
        fp = forward_pass!(nb, x̂, results)
        gap = fp.upper - fp.lower
        record = (iteration = ν, lower = fp.lower, upper = fp.upper, gap = gap,
                  optimality_cuts = sum(length(np.optimality_cuts) for np in nb.nodes),
                  feasibility_cuts = sum(length(np.feasibility_cuts) for np in nb.nodes),
                  solves = sum(np.solves for np in nb.nodes))
        push!(history, record)
        verbose && @printf("%5d %14.6f %14.6f %11.3e %6d %6d %6d\n", ν, fp.lower, fp.upper, gap,
                           record.optimality_cuts, record.feasibility_cuts, record.solves)
        if fp.feasible && gap <= tolerance * max(1.0, abs(fp.upper))
            return (status = :optimal, lower = fp.lower, upper = fp.upper, x̂ = x̂, history = history)
        end
        fp.feasible && backward_pass!(nb, x̂, results)
    end
    return (status = :iteration_limit, lower = history[end].lower, upper = history[end].upper,
            x̂ = x̂, history = history)
end

# ── 6. Reference: the extensive form, with the same builder ──────────────────────────────────────

"""
    extensive_form(tree, build!; optimizer) -> (model, xout)

The deterministic equivalent, in one JuMP model: `build!` is called once per node, and receives the
parent's own variables as `xin`, so that nonanticipativity is built in, as in the nodal formulation of
the slides. The objective is the sum of the stage costs weighted by the node probabilities. `xout[k]`
are the decisions of node `k`. The names a builder registers are dropped after each call, since every
node would register the same ones.
"""
function extensive_form(tree::ScenarioTree, build!::Function; optimizer)
    check_tree(tree)
    model = Model(optimizer)
    set_silent(model)
    xout = Vector{Vector{VariableRef}}(undef, length(tree.nodes))
    total = AffExpr(0.0)
    for (k, node) in enumerate(tree.nodes)
        @objective(model, Min, 0)                       # so that a node without cost adds nothing
        xin = node.parent == 0 ? VariableRef[] : xout[node.parent]
        xout[k] = collect_variables(build!(model, node, xin))
        add_to_expression!(total, node_probability(tree, k), stage_cost_expression(model))
        for name in collect(keys(object_dictionary(model)))
            unregister(model, name)
        end
    end
    @objective(model, Min, total)
    return model, xout
end

# ── 7. Example: financial planning (Birge and Louveaux, Section 1.2) ─────────────────────────────

"""
    financial_builder(; b = 55, G = 80, q = 1, r = 4, H = 3, borrowing_limit = Inf)

The financial planning problem of the slides, in thousands of dollars: invest `b` in stocks and bonds,
rebalance at the beginning of each of the `H` periods, and reach the goal `G`; a surplus `y` earns `q`,
a shortfall `w` costs `r`, and at most `borrowing_limit` can be borrowed. Stages `1, …, H` decide the
investments `x_t`, stage `H + 1` the surplus and the shortfall; a node of stage `t ≥ 2` observes the
returns of period `t - 1`. We maximize `qy - rw`, that is, minimize `-qy + rw`.
"""
function financial_builder(; b = 55.0, G = 80.0, q = 1.0, r = 4.0, H = 3, borrowing_limit = Inf)
    return function build!(model, node, xin)
        t, ω = node.stage, node.data
        if t == 1
            @variable(model, x[1:2] >= 0)
            @constraint(model, budget, sum(x) == b)
            return x
        elseif t <= H
            @variable(model, x[1:2] >= 0)
            @constraint(model, reinvest, sum(x) == sum(ω[i] * xin[i] for i in 1:2))
            return x
        else
            @variable(model, y >= 0)
            @variable(model, w >= 0)
            isfinite(borrowing_limit) && set_upper_bound(w, borrowing_limit)
            @constraint(model, goal, sum(ω[i] * xin[i] for i in 1:2) - y + w == G)
            @objective(model, Min, -q * y + r * w)
            return VariableRef[]
        end
    end
end

"""The tree of the example: each period, the returns of (stocks, bonds) are (1.25, 1.14) or (1.06, 1.12),
with probability 1/2, independently of the past."""
financial_tree(H = 3) = stagewise_tree(fill([[1.25, 1.14], [1.06, 1.12]], H), fill([0.5, 0.5], H))

if abspath(PROGRAM_FILE) == @__FILE__
    H, b, q = 3, 55.0, 1.0
    tree = financial_tree(H)
    build! = financial_builder(H = H)
    println("Scenario tree: $(length(tree.nodes)) nodes, $(nstages(tree)) stages\n")

    model, xout = extensive_form(tree, build!; optimizer = HiGHS.Optimizer)
    optimize!(model)
    @printf("Extensive form: optimal value %.6f\n\n", objective_value(model))

    # θ ≥ -q y, and y is at most the final wealth, at most b 1.25^H: a valid bound, not a tight one
    nb = NestedProblem(tree, build!; optimizer = HiGHS.Optimizer, lower_bound = -q * b * 1.25^H)
    result = nested_benders!(nb)
    @printf("\nNested Benders: %s, value %.6f\n", result.status, result.upper)
    @printf("First-stage decision: stocks %.3f, bonds %.3f\n", result.x̂[1]...)
end
