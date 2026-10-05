"""
    LShaped

A general L-shaped method for two-stage stochastic linear programs with recourse,

    min  c'x + E_ξ[Q(x, ξ)]   s.t.  Ax ⋛ b,  x ∈ X

    Q(x, ξ) = min  q(ξ)'y   s.t.  Wy ⋛ h(ξ) - T(ξ)x,  y ∈ Y,

where `⋛` is read row by row (`≤`, `=`, `≥` in any mix, in both stages), `X` and `Y` are boxes,
and `ξ` is a finite vector of scenarios with probabilities `p`.

Three things are meant to be independent of each other, and are:

* **the modelization** — either the data of the problem, given to [`TwoStageProblem`](@ref), or
  any JuMP model you like, plugged in through [`JuMPTwoStageProblem`](@ref);
* **the solver** — `master_optimizer` and `recourse_optimizer` are independent, and any JuMP
  optimizer or `MOI.OptimizerWithAttributes` is accepted;
* **the algorithm** — one cut per iteration, [`lshaped`](@ref) with `cuts = :single`, or one cut
  per scenario, with `cuts = :multi`.

The scenarios themselves may be tabulated or drawn: [`sample_scenarios`](@ref) turns a sampler into
the `(ξ, p)` pair both modelizations take, and [`substream`](@ref) is the reproducible stream it
draws from — one per experiment rather than a `Random.seed!` for the session.

The senses of the rows are never normalized away, and no correction is applied to the multipliers
a solver returns: for a minimization problem MOI returns `πᵢ ≥ 0` on a `≥` row, `πᵢ ≤ 0` on a `≤`
row and a free `πᵢ` on an `=` row, which are exactly the signs for which
`Q(x, ξ) = π'(h(ξ) - T(ξ)x)` holds. Both cuts are built as supporting hyperplanes — a
subgradient, plus the value of the subproblem, so that the cut is tight at the current iterate —
which stays valid when the recourse variables have finite bounds, where the closed forms
`e = Σₛ pₛ h(ξₛ)'πₛ` and `e_f = σ'h(ξₛ)` of the slides do not.
"""
module LShaped

using JuMP
using LinearAlgebra
using Printf
using RandomDataStreams

const MOI = JuMP.MOI

export Sense, LEQ, EQ, GEQ, to_sense,
       TwoStageProblem, JuMPTwoStageProblem,
       MasterTemplate, RecourseTemplate,
       lshaped, single_cut_lshaped, multi_cut_lshaped,
       extensive_form, multipliers, feasibility_certificate, new_model,
       first_stage_decision, second_stage_decision, print_first_stage, print_second_stage,
       sample_scenarios, substream

# --------------------------------------------------------------------------------------------
# row senses
# --------------------------------------------------------------------------------------------

"""Sense of a constraint row."""
@enum Sense LEQ EQ GEQ

"""Accept `'<'`, `'='`, `'>'` (or `"<="`, `"=="`, `">="`, `"≤"`, `"≥"`, and the matching symbols)
as well as `Sense` values."""
to_sense(s::Sense) = s
to_sense(s::AbstractChar) =
    s in ('<', '≤') ? LEQ : s in ('>', '≥') ? GEQ : s == '=' ? EQ : error("unknown sense: $s")
to_sense(s::AbstractString) = to_sense(first(s))
to_sense(s::Symbol) = to_sense(String(s)[1])

# --------------------------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------------------------

"""
    new_model(optimizer; silent = true, kwargs...)

A JuMP model attached to `optimizer`, which may be a `MOI.OptimizerFactory` such as
`HiGHS.Optimizer`, an `MOI.OptimizerWithAttributes` — the way to tune a solver — or an
`MOI.AbstractOptimizer` already bound to a model.
"""
function new_model(optimizer; silent::Bool = true, kwargs...)
    m = JuMP.Model(optimizer; kwargs...)
    silent && JuMP.set_silent(m)
    return m
end

"""A short name for a solver, for the header of the log."""
solver_label(optimizer::Type) = _label(string(optimizer))
solver_label(optimizer::MOI.OptimizerWithAttributes) =
    solver_label(getfield(optimizer, :optimizer_constructor))
solver_label(optimizer::MOI.AbstractOptimizer) = string(nameof(typeof(optimizer)))
solver_label(optimizer::Function) = string(nameof(optimizer))
solver_label(optimizer) = string(nameof(typeof(optimizer)))

_label(s) = strip(replace(s, "Optimizer" => ""), ['.', ' '])

# --------------------------------------------------------------------------------------------
# scenario sets
# --------------------------------------------------------------------------------------------

"""
    seed_words(seed)

The four 64-bit words `Xoshiro256ppGen` is seeded with, derived from `seed` by SplitMix64 — the
integer is spread over the four words instead of filling one of them, so that two nearby seeds give
two genuinely different states rather than two neighbouring ones. The expansion is the one
RandomDataStreams applies to the integer seeds of its other generators; a negative seed is read
modulo 2⁶⁴.
"""
function seed_words(seed::Integer)
    golden_ratio = 0x9E3779B97F4A7C15
    state = seed % UInt64
    words = Vector{UInt64}(undef, 4)
    for i in eachindex(words)
        state += golden_ratio
        z = state
        z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
        words[i] = z ⊻ (z >> 31)
    end
    return words
end

"""
    substream(seed = 1)

The first independent stream of a `Xoshiro256ppGen` seeded with `seed` — the reproducible stand-in
for a call to `Random.seed!`.

Pass one `substream(seed)` per experiment, and `next_stream!` on it per worker, rather than seeding
the session: a *global* seed makes every generator in the process advance together, so the
single-cut run, the multi-cut run and the extensive form of one study then quietly see different
samples, and the comparison between them says nothing. Streams obtained this way are reproducible on
their own and leave the rest of the session alone.

The registered version of RandomDataStreams (v0.1.0) takes four seed words and not an integer,
which is why the integer is expanded here by [`seed_words`](@ref); the call reads the same on the
master branch, which takes the integer directly.
"""
substream(seed::Integer = 1) = next_stream!(Xoshiro256ppGen(seed_words(seed)))

"""
    sample_scenarios(sampler, n; rng = substream(1), p = nothing)

`n` samples of `sampler`, as the named tuple `(ξ = ..., p = ...)` [`TwoStageProblem`](@ref) takes:
it can be splatted into the keyword arguments, or destructured,

    TwoStageProblem(; ..., sample_scenarios(Normal(5, 1.5), 500, rng = substream(7))...)
    ξ, p = sample_scenarios(Normal(5, 1.5), 500, rng = substream(7))

**The law of ξ is the modeller's, and is expected to be given as a sampler.** `sampler` is anything
`rand(rng, sampler, n)` accepts — a `Distribution`, a `Categorical`, a `Mixture` — and not a
`Distribution` field of the problem: how ξ is distributed, whether the draws are independent,
stratified or antithetic, is part of the model, and a `TwoStageProblem` that drew its own scenarios
would hand out a different sample every time it was built, so that no figure, no test and no commit
could be reproduced from the code that produced it. What the library owns is the part that is easy
to get wrong, the bookkeeping: a reproducible stream, and a sample that travels with the problem as
plain data, inspectable and replayable.

`p` defaults to the uniform weights; pass your own, summing up to one, when the sample is not
equally weighted — quasi-Monte Carlo points, a scenario-reduction weighting. To stratify, call this
once per stratum `k`, each with its own sampler and stream, and concatenate: `ξ = vcat(ξ₁, ξ₂)`
and `p = vcat(w₁ * p₁, w₂ * p₂)`, `wₖ` being the probability of stratum `k`.

[`JuMPTwoStageProblem`](@ref) needs none of this: its scenario set is opaque to `lshaped`, which only
asks how many there are and how much each weighs, so a sampler closed over by `data` and
`probabilities` works as it stands.
"""
function sample_scenarios(sampler, n::Integer; rng = substream(1), p = nothing)
    n > 0 || error("`n` must be positive, got $n")
    ξ = rand(rng, sampler, Int(n))
    weights = p === nothing ? fill(1 / n, Int(n)) : Float64.(collect(p))
    length(weights) == length(ξ) ||
        error("ξ and p have different lengths, $(length(ξ)) and $(length(weights))")
    isapprox(sum(weights), 1; atol = 1e-9) ||
        error("the probabilities must sum up to one, they sum up to $(sum(weights))")
    return (ξ = ξ, p = weights)
end

# --------------------------------------------------------------------------------------------
# the interface the algorithms are written against
# --------------------------------------------------------------------------------------------

"""
    AbstractTwoStageModel

Supertype of the modelizations [`lshaped`](@ref) accepts. The algorithm only ever calls

| function | role |
|:--|:--|
| `n_scenarios(md)` | number of scenarios `S` |
| `scenario_probability(md, s)` | `pₛ`, needed for the expected value |
| `scenario_data(md, s)` | whatever the callbacks of a `JuMPTwoStageProblem` are keyed on |
| `build_master(md, optimizer)` | the first-stage JuMP model, as a [`MasterTemplate`](@ref) |
| `build_recourse(md, optimizer, s)` | the recourse JuMP model of scenario `s`, as a [`RecourseTemplate`](@ref) |
| `build_elastic(md, optimizer, s)` | the elastic model of scenario `s`, or `nothing` |
| `cut_coefficients(md, r, s, x, π)` | the `n_x` coefficients of a cut built from the multipliers `π` |
| `cut_intercept(md, r, s, x, π, v)` | the right-hand side of that cut, `v` being the subproblem value |

plus [`set_recourse_rhs!`](@ref), [`multipliers`](@ref), [`recourse_value`](@ref) and
[`master_cost`](@ref), which have a generic implementation. A recourse model is built **once** per
scenario and its right-hand side updated in place, so `build_recourse` is not called again inside
the loop: `r.rhs(x, s)` is what does the updating.

`cut_coefficients(md, r, s, x, π)` returns `E` with `π` the multipliers of the rows listed in
`r.con`, in that order: `T(ξₛ)'πₛ` for the recourse problem, the same expression with the
multipliers of the elastic rows for the feasibility cut. `cut_intercept` returns the `e` of the
cut, and defaults to `v + E'x`, which makes the cut tight at the iterate and keeps it valid on
bounded recourse variables. The single-cut average is the probability-weighted sum of the
per-scenario quantities, which reduces to `e = Σₛ pₛ h(ξₛ)'πₛ` only in the textbook case.
"""
abstract type AbstractTwoStageModel end

"""
    MasterTemplate(model, x)

The first-stage JuMP model `model` and its decision variables `x`. The algorithm adds `θ` and the
cuts to `model`, and reads the objective off it, so it must be linear in `x`.
"""
struct MasterTemplate
    model::JuMP.Model
    x::Vector{JuMP.VariableRef}
end

"""
    RecourseTemplate(model, y, con, rhs)
    RecourseTemplate(model, y, con, rhs, row, sign)

A JuMP model of a subproblem — the recourse problem of one scenario, or its elastic version —
built once and re-solved at every iterate by updating right-hand sides only.

* `y` are the recourse variables (empty for an elastic model, whose artificial variables the
  objective keeps at its lower bound);
* `con` are the constraints whose right-hand side depends on the first-stage solution, in a fixed
  order; their multipliers are what a cut is built from;
* `rhs` is `(x, s) -> values of length length(con)`, the right-hand side at the first-stage
  solution `x` in scenario `s`;
* `row` and `sign` tell which row of the recourse problem each constraint of `con` stands for, and
  with which sign, so that [`multipliers`](@ref) assembles one multiplier per recourse row. They
  are the identity on a recourse model, and they are what lets an equality row be relaxed by two
  elastic rows read in the same direction (see [`build_elastic`](@ref)).
"""
struct RecourseTemplate
    model::JuMP.Model
    y::Vector{JuMP.VariableRef}
    con::Vector{Any}
    rhs::Function
    row::Vector{Int}
    sign::Vector{Float64}
end

RecourseTemplate(model, y, con, rhs) =
    RecourseTemplate(model, y, con, rhs, collect(eachindex(con)), ones(length(con)))

"""Update the right-hand side of `r` for the first-stage solution `x` in scenario `s`."""
set_recourse_rhs!(md::AbstractTwoStageModel, r::RecourseTemplate, x, s) =
    JuMP.set_normalized_rhs.(r.con, r.rhs(x, s))

"""The multiplier of each row of the recourse problem, assembled from the multipliers of `r.con`."""
function multipliers(r::RecourseTemplate)
    σ = zeros(isempty(r.row) ? 0 : maximum(r.row))
    for k in eachindex(r.con)
        σ[r.row[k]] += r.sign[k] * JuMP.dual(r.con[k])
    end
    return σ
end

"""The value of the subproblem: `Q(x, ξ)` on a recourse model, the total violation on an elastic one."""
recourse_value(r::RecourseTemplate) = JuMP.objective_value(r.model)

"""
    feasibility_certificate(e)

The value `v` of an elastic model and the multipliers `σ` proving that the recourse problem of the
same scenario is infeasible, as returned by [`multipliers`](@ref).
"""
feasibility_certificate(e::RecourseTemplate) = (value = recourse_value(e), σ = multipliers(e))

"""
    master_cost(master)

The cost `c` of the first stage, and a constant, read off the objective function of the master
model, so that no modelization has to provide them. Every variable of the objective must be one of
`master.x`: a term on any other variable would be dropped from the bounds without notice.
"""
function master_cost(master::MasterTemplate)
    JuMP.objective_sense(master.model) == MOI.MIN_SENSE ||
        error("the master problem must be a minimization")
    f = JuMP.objective_function(master.model)
    f isa JuMP.VariableRef && (f = 1.0 * f)
    f isa JuMP.AffExpr ||
        error("the master objective must be linear in the first-stage variables")
    index = Dict(v => j for (j, v) in enumerate(master.x))
    c = zeros(length(master.x))
    for (v, coef) in f.terms
        j = get(index, v, nothing)
        j === nothing &&
            error("the master objective involves `$v`, which is not a first-stage variable `x`")
        c[j] += coef
    end
    return c, f.constant
end

"""
    check_recourse(r, s)

The cuts rely on `Q(·, ξₛ)` being the value of a linear program, minimized: its multipliers are
then subgradients. A `Max` objective flips their sign under the MOI convention, and an integer
recourse variable leaves no multiplier at all, so both are refused rather than turned into wrong
cuts.
"""
function check_recourse(r::RecourseTemplate, s)
    JuMP.objective_sense(r.model) == MOI.MIN_SENSE ||
        error("scenario $s: the recourse problem must be a minimization; write `Min -f` for `Max f`")
    integers = JuMP.num_constraints(r.model, JuMP.VariableRef, MOI.Integer) +
               JuMP.num_constraints(r.model, JuMP.VariableRef, MOI.ZeroOne)
    integers == 0 ||
        error("scenario $s: the recourse problem has $integers integer variable(s); " *
              "the L-shaped cuts need a linear recourse problem")
    return r
end

cut_intercept(md::AbstractTwoStageModel, r, s, x, π, v) =
    v + dot(cut_coefficients(md, r, s, x, π), x)

# --------------------------------------------------------------------------------------------
# models with mixed row senses, written out by hand
# --------------------------------------------------------------------------------------------

"""`lhs[i] ⋛ rhs[i]`, one sense per row; returns the constraint references, which are of mixed
type by construction and therefore collected in a `Vector{Any}`."""
function add_rows!(m::JuMP.Model, lhs::AbstractVector, senses::Vector{Sense}, rhs::AbstractVector)
    con = Vector{Any}(undef, length(senses))
    for i in eachindex(senses)
        con[i] = senses[i] == LEQ ? @constraint(m, lhs[i] <= rhs[i]) :
                 senses[i] == GEQ ? @constraint(m, lhs[i] >= rhs[i]) :
                                    @constraint(m, lhs[i] == rhs[i])
    end
    return con
end

"""Impose on the variables `v` the box `lb ≤ v ≤ ub`, either bound of which may be infinite, and
integrality wherever `integer` asks for it."""
function declare_box!(m::JuMP.Model, v::Vector{JuMP.VariableRef}, lb::AbstractVector,
                      ub::AbstractVector, integer::AbstractVector = falses(length(v)))
    for j in eachindex(v)
        isfinite(lb[j]) && JuMP.set_lower_bound(v[j], lb[j])
        isfinite(ub[j]) && JuMP.set_upper_bound(v[j], ub[j])
        integer[j] && JuMP.set_integer(v[j])
    end
    return v
end

# --------------------------------------------------------------------------------------------
# modelization 1: the data of the problem
# --------------------------------------------------------------------------------------------

"""
    TwoStageProblem(; c, A, senses1, b, q, W, senses2, T, h, ξ, p,
                      lb1 = 0, ub1 = Inf, lb = 0, ub = Inf,
                      integer1 = false, intercept = :tight)

The two-stage stochastic linear program of the docstring of the module, given as data. Every row
of `A` and of `W` carries its own sense, given by `senses1` and `senses2` as `Sense` values or as
`'<'`, `'='`, `'>'`; `T`, `h` and `q` are functions of the scenario or constants; `lb1`, `ub1` are
the bounds of `x` and `lb`, `ub` those of `y`, scalars or vectors, and `integer1` — a boolean or a
vector of them — asks for some of the `x` to be integer, which makes the master a mixed-integer
program and nothing else change.

`intercept = :tight` (the default) builds both cuts as supporting hyperplanes, tight at the current
iterate, which stays valid with bounded recourse variables. `intercept = :textbook` uses instead
the closed forms of the slides, `e = Σₛ pₛ h(ξₛ)'πₛ` and `e_f = σ'h(ξₛ)`, and is kept for
comparison only: with finite bounds on `y` they can err in either direction, and erring upwards
makes the method stop at a point that is not optimal.
"""
struct TwoStageProblem <: AbstractTwoStageModel
    c::Vector{Float64}
    A::Matrix{Float64}
    senses1::Vector{Sense}
    b::Vector{Float64}
    q::Function
    W::Matrix{Float64}
    senses2::Vector{Sense}
    T::Function
    h::Function
    ξ::Vector
    p::Vector{Float64}
    lb1::Vector{Float64}
    ub1::Vector{Float64}
    integer1::Vector{Bool}
    lb::Vector{Float64}
    ub::Vector{Float64}
    intercept::Symbol
end

function TwoStageProblem(; c, A, senses1, b, q, W, senses2, T, h, ξ, p,
                           lb1 = 0.0, ub1 = Inf, lb = 0.0, ub = Inf,
                           integer1 = false, intercept::Symbol = :tight)
    intercept in (:tight, :textbook) || error("unknown intercept: $intercept")
    nx = length(c)
    ny = q isa AbstractVector ? length(q) : size(W, 2)
    pb = TwoStageProblem(
        Float64.(c), Float64.(A), to_sense.(collect(senses1)), Float64.(b),
        q isa Function ? q : (_ -> Float64.(q)),
        Float64.(W), to_sense.(collect(senses2)),
        T isa AbstractMatrix ? (_ -> Float64.(T)) : T,
        h isa AbstractVector ? (_ -> Float64.(h)) : h,
        collect(ξ), Float64.(p),
        lb1 isa Number ? fill(Float64(lb1), nx) : Float64.(lb1),
        ub1 isa Number ? fill(Float64(ub1), nx) : Float64.(ub1),
        integer1 isa Bool ? fill(integer1, nx) : Bool.(integer1),
        lb isa Number ? fill(Float64(lb), ny) : Float64.(lb),
        ub isa Number ? fill(Float64(ub), ny) : Float64.(ub),
        intercept)
    @assert length(pb.integer1) == nx "integer1 and c disagree on the number of variables"
    @assert size(pb.A, 2) == nx "A and c disagree on the number of columns"
    @assert size(pb.A, 1) == length(pb.b) == length(pb.senses1) "A, b and senses1 disagree"
    @assert size(pb.W, 2) == ny "W and q disagree on the number of columns"
    @assert size(pb.W, 1) == length(pb.senses2) "W and senses2 disagree"
    @assert length(pb.ξ) == length(pb.p) "one probability per scenario is required"
    @assert isapprox(sum(pb.p), 1; atol = 1e-9) "the probabilities must sum up to one"
    for s in eachindex(pb.ξ)
        @assert size(pb.T(pb.ξ[s])) == (size(pb.W, 1), nx) "T(ξ) has a wrong size"
        @assert length(pb.h(pb.ξ[s])) == size(pb.W, 1) "h(ξ) has a wrong length"
        @assert length(pb.q(pb.ξ[s])) == ny "q(ξ) has a wrong length"
    end
    return pb
end

n_x(pb::TwoStageProblem)         = length(pb.c)
n_y(pb::TwoStageProblem)         = length(pb.lb)
n_rows(pb::TwoStageProblem)      = size(pb.W, 1)
n_scenarios(pb::TwoStageProblem) = length(pb.ξ)
scenario_probability(pb::TwoStageProblem, s) = pb.p[s]
scenario_data(pb::TwoStageProblem, s) = pb.ξ[s]

"""Right-hand side of the second stage at `(x, ξ)`."""
recourse_rhs(pb::TwoStageProblem, x, ξ) = pb.h(ξ) - pb.T(ξ) * x

recourse_variables!(m::JuMP.Model, pb::TwoStageProblem; base_name = "y") =
    declare_box!(m, @variable(m, [1:n_y(pb)], base_name = base_name), pb.lb, pb.ub)

function build_master(pb::TwoStageProblem, optimizer)
    m = new_model(optimizer)
    x = declare_box!(m, @variable(m, [1:n_x(pb)], base_name = "x"), pb.lb1, pb.ub1, pb.integer1)
    add_rows!(m, pb.A * x, pb.senses1, pb.b)
    @objective(m, Min, dot(pb.c, x))
    return MasterTemplate(m, x)
end

function build_recourse(pb::TwoStageProblem, optimizer, s)
    ξ = pb.ξ[s]
    m = new_model(optimizer)
    y = recourse_variables!(m, pb)
    con = add_rows!(m, pb.W * y, pb.senses2, zeros(n_rows(pb)))
    @objective(m, Min, dot(pb.q(ξ), y))
    return RecourseTemplate(m, y, con, (x, _) -> recourse_rhs(pb, x, ξ))
end

"""
    build_elastic(pb::TwoStageProblem, optimizer, s)

The elastic version of the recourse problem of scenario `s`: an artificial variable per row
absorbs its violation, and the minimum of their sum is the smallest total violation, which is zero
if and only if the recourse problem is feasible.

* a `≥` row is relaxed by `Wᵢy + wᵢ ≥ rᵢ` and a `≤` row by `Wᵢy - wᵢ ≤ rᵢ`, so that the multiplier
  of the elastic row is the multiplier of the recourse row;
* an `=` row is relaxed by the **pair** `Wᵢy + wᵢ⁺ ≥ rᵢ` and `-Wᵢy + wᵢ⁻ ≥ -rᵢ`, read in the same
  direction, whose free combination `αᵢ - βᵢ` is the multiplier of the equality row. The pair
  `wᵢ⁺, wᵢ⁻` on a single `=` row instead would be useless: the coefficient of `wᵢ⁺` forces the
  multiplier to be `≤ 0` and the one of `wᵢ⁻` forces it to be `≥ 0`, so it would always come back
  zero and the cut would say nothing about that row.
"""
function build_elastic(pb::TwoStageProblem, optimizer, s)
    ξ, rows, senses = pb.ξ[s], n_rows(pb), pb.senses2
    equalities = count(==(EQ), senses)
    m = new_model(optimizer)
    y = recourse_variables!(m, pb)
    @variable(m, w[1:(rows + 2equalities)] >= 0)   # one per row, two for an `=` row
    lhs = pb.W * y
    con, row, sign = Vector{Any}(undef, 0), Int[], Float64[]
    k = 0
    for i in 1:rows
        if senses[i] == LEQ
            push!(con, @constraint(m, lhs[i] - w[i] <= 0.0))
            push!(row, i); push!(sign, 1.0)
        elseif senses[i] == GEQ
            push!(con, @constraint(m, lhs[i] + w[i] >= 0.0))
            push!(row, i); push!(sign, 1.0)
        else
            k += 1
            plus, minus = rows + 2k - 1, rows + 2k
            push!(con, @constraint(m, lhs[i] + w[plus] >= 0.0))
            push!(row, i); push!(sign, 1.0)
            push!(con, @constraint(m, -lhs[i] + w[minus] >= 0.0))
            push!(row, i); push!(sign, -1.0)
        end
    end
    @objective(m, Min, sum(w))
    return RecourseTemplate(m, y, con,
                            (x, _) -> sign .* recourse_rhs(pb, x, ξ)[row], row, sign)
end

cut_coefficients(pb::TwoStageProblem, r, s, x, π) = pb.T(pb.ξ[s])' * π

cut_intercept(pb::TwoStageProblem, r, s, x, π, v) =
    pb.intercept == :textbook ? dot(π, pb.h(pb.ξ[s])) :
                               v + dot(cut_coefficients(pb, r, s, x, π), x)

"""
    extensive_form(pb::TwoStageProblem; optimizer, kwargs...)

The deterministic equivalent, `min c'x + Σₛ pₛ qₛ'yₛ s.t. Ax ⋛ b, T(ξₛ)x + Wyₛ ⋛ h(ξₛ)`, solved
in one go. Returns `(model, x, objective_value)` and serves as the reference the L-shaped
solutions are checked against.
"""
function extensive_form(pb::TwoStageProblem; optimizer, kwargs...)
    m = new_model(optimizer; kwargs...)
    x = declare_box!(m, @variable(m, [1:n_x(pb)], base_name = "x"), pb.lb1, pb.ub1, pb.integer1)
    y = [recourse_variables!(m, pb; base_name = "y_$s") for s in 1:n_scenarios(pb)]
    add_rows!(m, pb.A * x, pb.senses1, pb.b)
    for s in 1:n_scenarios(pb)
        add_rows!(m, pb.T(pb.ξ[s]) * x + pb.W * y[s], pb.senses2, pb.h(pb.ξ[s]))
    end
    @objective(m, Min, dot(pb.c, x) +
                       sum(pb.p[s] * dot(pb.q(pb.ξ[s]), y[s]) for s in 1:n_scenarios(pb)))
    optimize!(m)
    termination_status(m) == MOI.OPTIMAL || error("extensive form: $(termination_status(m))")
    return m, value.(x), objective_value(m)
end

# --------------------------------------------------------------------------------------------
# modelization 2: any JuMP model
# --------------------------------------------------------------------------------------------

"""
    JuMPTwoStageProblem(; n_scenarios, master_builder, recourse_builder, rhs,
                          cut_coefficients, probabilities, data,
                          cut_intercept, elastic_builder)

A two-stage stochastic linear program written as JuMP models of your own — whatever constraints,
reformulations or bounds you prefer, and integer first-stage variables if you need them; the
recourse problems must stay linear programs, minimized, since the cuts are built from their
multipliers. `lshaped` only has to be able to re-solve the second stage when the right-hand side
changes and to read the multipliers back, so four callbacks do the job:

* `master_builder(optimizer)` returns `(model, x)`: the first-stage model, minimized, whose
  objective is read off `model` — it may only involve `x` — and to which `θ` and the cuts are added;
* `recourse_builder(optimizer, s)` returns `(model, y, con)`: the recourse model of scenario `s`,
  `con` being the constraints whose right-hand side depends on `x`, **in a fixed order**;
* `rhs(x, s)` returns their new right-hand side, a vector of `length(con)`;
* `cut_coefficients(s, x, π)` returns the `length(x)` coefficients of the cut built from the
  multipliers `π` — `T(ξₛ)'πₛ` if the rows are written the usual way.

`probabilities` (default: uniform) and `data` (default: `s`, the scenario index) are free for you
to use inside the callbacks; `cut_intercept(s, x, π, v)` defaults to `v + E'x`, the supporting
hyperplane tight at `x`, and `elastic_builder(optimizer, s)`, returning `(model, con, row, sign)`
as in [`RecourseTemplate`](@ref), is what lets the algorithm cut feasibility as well.
"""
struct JuMPTwoStageProblem <: AbstractTwoStageModel
    n_scenarios::Int
    master_builder::Function
    recourse_builder::Function
    rhs::Function
    cut_coefficients::Function
    cut_intercept::Function
    elastic_builder::Union{Nothing,Function}
    probabilities::Function
    data::Function
end

function JuMPTwoStageProblem(; n_scenarios::Integer, master_builder::Function,
                             recourse_builder::Function, rhs::Function,
                             cut_coefficients::Function,
                             cut_intercept::Union{Nothing,Function} = nothing,
                             elastic_builder::Union{Nothing,Function} = nothing,
                             probabilities = s -> 1 / n_scenarios,
                             data = _scenario_index)
    n_scenarios > 0 || error("at least one scenario is required")
    tight = cut_intercept === nothing ?
            (s, x, π, v) -> v + dot(cut_coefficients(s, x, π), x) : cut_intercept
    return JuMPTwoStageProblem(n_scenarios, master_builder, recourse_builder, rhs,
                               cut_coefficients, tight, elastic_builder,
                               probabilities, data)
end

_scenario_index(s) = s   # the default `data`: the scenario index, which says nothing of ξ

n_scenarios(md::JuMPTwoStageProblem) = md.n_scenarios
scenario_probability(md::JuMPTwoStageProblem, s) = md.probabilities(s)
scenario_data(md::JuMPTwoStageProblem, s) = md.data(s)

function build_master(md::JuMPTwoStageProblem, optimizer)
    model, x = md.master_builder(optimizer)
    return MasterTemplate(model, collect(x))
end

function build_recourse(md::JuMPTwoStageProblem, optimizer, s)
    model, y, con = md.recourse_builder(optimizer, s)
    return RecourseTemplate(model, collect(y), collect(Any, con), (x, _) -> md.rhs(x, s))
end

function build_elastic(md::JuMPTwoStageProblem, optimizer, s)
    md.elastic_builder === nothing && return nothing
    model, con, row, sign = md.elastic_builder(optimizer, s)
    return RecourseTemplate(model, JuMP.VariableRef[], collect(Any, con),
                            (x, _) -> md.rhs(x, s), collect(row), Float64.(sign))
end

cut_coefficients(md::JuMPTwoStageProblem, r, s, x, π) = md.cut_coefficients(s, x, π)
cut_intercept(md::JuMPTwoStageProblem, r, s, x, π, v) = md.cut_intercept(s, x, π, v)

# --------------------------------------------------------------------------------------------
# the algorithm
# --------------------------------------------------------------------------------------------

"""
    lshaped(md; optimizer, master_optimizer = optimizer, recourse_optimizer = optimizer,
              cuts = :single, maxiter = 500, tol = 1e-8, feastol = 1e-7,
              verbose = true, log = stdout)

Solve a two-stage stochastic linear program by L-shaped decomposition, on the modelization `md`.

At iteration `k` the master problem is

    min  c'x + θ   s.t.  Ax ⋛ b,  cuts

`θ` joins the objective with the first optimality cut. Solving the second stage at `xᵏ` gives, per
scenario, either the multipliers `πₛ` of an optimal solution or — should the recourse problem be
infeasible — an elastic model proving it, and its multipliers `σₛ`. Both are assembled into cuts
by `cut_coefficients` and `cut_intercept`: `Q(·, ξ)` and the total violation are convex in `x`,
and the cuts are their supporting hyperplanes at `xᵏ`.

`cuts = :single` averages over the scenarios and adds **one** optimality cut per iteration,

    E = Σₛ pₛ T(ξₛ)'πₛ ,  E'x + θ ≥ E'xᵏ + Q(xᵏ),

`cuts = :multi` keeps one `θₛ` per scenario, weighted by `pₛ` in the objective, and adds **one cut
per scenario**,

    Eₛ'x + θₛ ≥ Eₛ'xᵏ + Q(xᵏ, ξₛ),

which is the multicut version of Birge and Louveaux (2011, Section 5.1.d): the same subproblems per
iteration, a larger master, and usually a better approximation of `Q` around `xᵏ`.

`master_optimizer` and `recourse_optimizer` are independent — the master is a sequence of LPs that
grows, the recourse problems are small and re-solved at every iterate, so the two rarely deserve
the same solver; both accept anything [`new_model`](@ref) accepts.

Convergence is declared when no cut separates `θ` from `Q(xᵏ)` any more, up to `tol` relative to
`1 + |Q(xᵏ)|` (`θₛ` and `Q(xᵏ, ξₛ)` scenario by scenario for the multicut version): an absolute
tolerance would ask more of the solvers than they deliver on objectives of large magnitude.
`feastol` is the total violation below which the elastic model counts as proof of feasibility.

The upper bound is that of the **incumbent**, the best first-stage solution met so far: the value
`c'xᵏ + Q(xᵏ)` of the current iterate does not decrease monotonically. Returns a named tuple with
the incumbent `x` and its `objective`, the last `lower_bound`, the `gap` between them,
`converged`, the counters of both kinds of cut, the `problem` `md` and the models, and the
`history`, whose entries record per iteration the bounds and the `value` of the iterate `x`.
[`first_stage_decision`](@ref), [`second_stage_decision`](@ref), [`print_first_stage`](@ref) and
[`print_second_stage`](@ref) read the decisions off this result.
"""
function lshaped(md::AbstractTwoStageModel;
                 optimizer = nothing,
                 master_optimizer = optimizer,
                 recourse_optimizer = optimizer,
                 cuts::Symbol = :single,
                 maxiter::Integer = 500,
                 tol::Real = 1e-8,
                 feastol::Real = 1e-7,
                 verbose::Bool = true,
                 log::IO = stdout)
    cuts in (:single, :multi) || error("`cuts` must be :single or :multi, got :$cuts")
    master_optimizer === nothing &&
        error("pass `optimizer`, or both `master_optimizer` and `recourse_optimizer`")
    recourse_optimizer === nothing &&
        error("pass `optimizer`, or both `master_optimizer` and `recourse_optimizer`")
    scenarios = n_scenarios(md)
    p = [scenario_probability(md, s) for s in 1:scenarios]
    isapprox(sum(p), 1; atol = 1e-9) ||
        error("the probabilities must sum up to one, they sum up to $(sum(p))")

    master = build_master(md, master_optimizer)
    recourse = [check_recourse(build_recourse(md, recourse_optimizer, s), s) for s in 1:scenarios]
    elastic = Any[nothing for _ in 1:scenarios]
    c, constant = master_cost(master)
    x = master.x
    n = length(x)

    # anonymous, so that a master model of the user's may have a variable of its own named θ
    θ = @variable(master.model, [1:(cuts == :single ? 1 : scenarios)], base_name = "θ")
    bounded = falses(length(θ))
    function set_objective!()
        f = dot(c, x) + constant        # the constant too, or the lower bound would miss it
        for t in eachindex(θ)
            bounded[t] && (f += (cuts == :single ? 1.0 : p[t]) * θ[t])
        end
        @objective(master.model, Min, f)
    end

    if verbose
        @printf(log, "%s-cut L-shaped method\n", cuts == :single ? "Single" : "Multi")
        @printf(log, "  master: %s | recourse: %s | scenarios: %d\n\n",
                solver_label(master_optimizer), solver_label(recourse_optimizer), scenarios)
        @printf(log, " iter      lower bound   upper bound          gap %14s %14s\n",
                cuts == :single ? "θ" : "min θₛ", "Q(x)")
    end

    n_optimality = n_feasibility = 0
    history = NamedTuple[]
    lower, upper, xstar, converged = -Inf, Inf, fill(NaN, n), false
    iteration = 0

    for k in 1:maxiter
        iteration = k
        optimize!(master.model)
        status = termination_status(master.model)
        status == MOI.OPTIMAL || error("master problem: $status")
        xk = value.(x)
        lower = any(bounded) ? objective_value(master.model) : -Inf
        θk = [bounded[t] ? value(θ[t]) : -Inf for t in eachindex(θ)]

        Qs, es = zeros(scenarios), zeros(scenarios)
        Es = [zeros(n) for _ in 1:scenarios]
        infeasible = Int[]
        for s in 1:scenarios
            set_recourse_rhs!(md, recourse[s], xk, s)
            optimize!(recourse[s].model)
            if termination_status(recourse[s].model) != MOI.OPTIMAL
                # Either infeasible, or the solver cannot tell: the elastic model decides which.
                if elastic[s] === nothing
                    elastic[s] = build_elastic(md, recourse_optimizer, s)
                end
                elastic[s] === nothing && error("""
                    scenario $s: the recourse problem is $(termination_status(recourse[s].model)) \
                    and no elastic model is available; provide `elastic_builder` to cut feasibility""")
                set_recourse_rhs!(md, elastic[s], xk, s)
                optimize!(elastic[s].model)
                termination_status(elastic[s].model) == MOI.OPTIMAL ||
                    error("elastic problem, scenario $s: $(termination_status(elastic[s].model))")
                certificate = feasibility_certificate(elastic[s])
                if certificate.value <= feastol
                    error("scenario $s: the recourse problem is " *
                          "$(termination_status(recourse[s].model)) yet its elastic relaxation has " *
                          "value $(certificate.value); the second stage is unbounded, or the " *
                          "solver could not solve it")
                end
                σ = certificate.σ
                E_f = cut_coefficients(md, elastic[s], s, xk, σ)
                length(E_f) == n ||
                    error("`cut_coefficients` returned $(length(E_f)) coefficients, expected $n")
                e_f = cut_intercept(md, elastic[s], s, xk, σ, certificate.value)
                @constraint(master.model, dot(E_f, x) >= e_f)
                n_feasibility += 1
                push!(infeasible, s)
                continue
            end
            # π as the solver returns them: πᵀ(h - Tx) = Q(x, ξ) whatever the mix of senses
            π = multipliers(recourse[s])
            Qs[s] = recourse_value(recourse[s])
            Es[s] = cut_coefficients(md, recourse[s], s, xk, π)
            length(Es[s]) == n ||
                error("`cut_coefficients` returned $(length(Es[s])) coefficients, expected $n")
            es[s] = cut_intercept(md, recourse[s], s, xk, π, Qs[s])
        end
        if !isempty(infeasible)
            verbose && @printf(log, "%5d   feasibility cut(s) on scenario(s) %s\n",
                               k, join(infeasible, ", "))
            continue
        end

        Q = dot(p, Qs)
        current = dot(c, xk) + constant + Q
        if current < upper              # a new incumbent
            upper, xstar = current, xk
        end
        push!(history, (iteration = k, lower_bound = lower, upper_bound = upper,
                        gap = upper - lower, value = current, Q = Q, x = xk))
        verbose && @printf(log, "%5d   %12s %12.6f %12.4g %14s %14.6f\n", k, _num(lower),
                           upper, upper - lower, _num(minimum(θk)), Q)

        # checked before convergence: crossing bounds often show up on the very iteration where
        # the invalid cuts make θ look converged
        if lower > upper + tol * (1 + abs(upper))
            @warn """the lower bound exceeds the upper bound ($lower > $upper): the cuts are \
                     invalid, which `intercept = :textbook` causes on bounded recourse variables"""
        end
        done = cuts == :single ? θk[1] >= Q - tol * (1 + abs(Q)) :
                              all(bounded) && all(θk .>= Qs .- tol .* (1 .+ abs.(Qs)))
        if done
            converged = true
            break
        end

        if cuts == :single
            E, e = zeros(n), 0.0
            for s in 1:scenarios
                E .+= p[s] .* Es[s]
                e += p[s] * es[s]
            end
            @constraint(master.model, dot(E, x) + θ[1] >= e)
        else
            for s in 1:scenarios
                @constraint(master.model, dot(Es[s], x) + θ[s] >= es[s])
            end
        end
        bounded .= true
        set_objective!()
        n_optimality += 1
    end

    if verbose
        if converged
            @printf(log, "converged in %d iteration(s): %d optimality cut(s), %d feasibility cut(s)\n",
                    iteration, n_optimality, n_feasibility)
        else
            @printf(log, "no convergence in %d iteration(s), last gap %g\n", maxiter, upper - lower)
        end
    end
    return (x = xstar, objective = upper, lower_bound = lower, gap = upper - lower,
            converged = converged, iterations = iteration, optimality_cuts = n_optimality,
            feasibility_cuts = n_feasibility, problem = md, master = master,
            recourse = recourse, history = history)
end

_num(v) = isfinite(v) ? @sprintf("%.6f", v) : "-Inf"

single_cut_lshaped(md; kwargs...) = lshaped(md; kwargs..., cuts = :single)
multi_cut_lshaped(md; kwargs...) = lshaped(md; kwargs..., cuts = :multi)

# --------------------------------------------------------------------------------------------
# the decisions
# --------------------------------------------------------------------------------------------

"""
    first_stage_decision(res)

The first-stage decision of the result `res` of [`lshaped`](@ref): the incumbent, the best
first-stage solution met.
"""
first_stage_decision(res::NamedTuple) = res.x

"""
    second_stage_decision(res, s)

The second-stage decision of scenario `s` for the first-stage decision of `res`, obtained by solving
the recourse problem of that scenario at `res.x`.

The recourse models `res.recourse` were last solved at the last iterate of the method, which need
not be the incumbent `res.x`: reading their solution directly could give the recourse of another
first-stage decision. When the recourse problem has several optimal solutions, this is one of them.
"""
function second_stage_decision(res::NamedTuple, s::Integer)
    r = _solve_recourse_at_incumbent!(res, s)
    return JuMP.value.(r.y)
end

function _solve_recourse_at_incumbent!(res::NamedTuple, s::Integer)
    md = res.problem
    1 <= s <= n_scenarios(md) ||
        throw(ArgumentError("there is no scenario $s: the scenarios are numbered 1 to $(n_scenarios(md))"))
    all(isfinite, res.x) || error("no feasible first-stage decision was found")
    r = res.recourse[s]
    set_recourse_rhs!(md, r, res.x, s)
    optimize!(r.model)
    status = termination_status(r.model)
    status == MOI.OPTIMAL ||
        error("scenario $s: the recourse problem is $status at the first-stage decision")
    return r
end

"""
    print_first_stage([io,] res; digits = 6)

Display the first-stage decision of `res`, variable by variable, with its objective value.
"""
function print_first_stage(io::IO, res::NamedTuple; digits::Integer = 6)
    @printf(io, "First-stage decision, objective %s\n", _round(res.objective, digits))
    _print_values(io, res.master.x, first_stage_decision(res), "x", digits)
end
print_first_stage(res::NamedTuple; kwargs...) = print_first_stage(stdout, res; kwargs...)

"""
    print_second_stage([io,] res, s; digits = 6)

Display the second-stage decision of scenario `s` for the first-stage decision of `res`, with the
scenario, its probability and the value `Q(x, ξₛ)` of its recourse problem.
"""
function print_second_stage(io::IO, res::NamedTuple, s::Integer; digits::Integer = 6)
    md = res.problem
    r = _solve_recourse_at_incumbent!(res, s)
    @printf(io, "Second-stage decision, scenario %d of %d: %sprobability %s, Q(x, ξ) = %s\n",
            s, n_scenarios(md), _scenario_label(md, s), _round(scenario_probability(md, s), digits),
            _round(recourse_value(r), digits))
    _print_values(io, r.y, JuMP.value.(r.y), "y", digits)
end
print_second_stage(res::NamedTuple, s::Integer; kwargs...) = print_second_stage(stdout, res, s; kwargs...)

_round(v, digits) = string(round(v; digits = digits) + 0.0)   # + 0.0 turns -0.0 into 0.0

_scenario_label(md::TwoStageProblem, s) = "ξ = $(scenario_data(md, s)), "
_scenario_label(md::JuMPTwoStageProblem, s) =
    md.data === _scenario_index ? "" : "ξ = $(scenario_data(md, s)), "

"""One line per variable, named as in its model, or `fallback[j]` if it has no name."""
function _print_values(io::IO, variables, values, fallback, digits)
    labels = [isempty(JuMP.name(v)) ? "$fallback[$j]" : JuMP.name(v) for (j, v) in enumerate(variables)]
    width = maximum(length, labels; init = 0)
    for (label, v) in zip(labels, values)
        println(io, "  ", rpad(label, width), " = ", _round(v, digits))
    end
end

end # module