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
       evpi_bounds, vss_bounds,
       wait_and_see, expected_value_problem, expected_result, evpi, vss,
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
    new_model(optimizer; silent = true, direct = false, kwargs...)

A JuMP model attached to `optimizer`, which may be a `MOI.OptimizerFactory` such as
`HiGHS.Optimizer`, an `MOI.OptimizerWithAttributes` — the way to tune a solver — or an
`MOI.AbstractOptimizer` already bound to a model.

`direct = true` builds the model in direct mode (`JuMP.direct_model`), without the cache and the
bridges between JuMP and the solver: about three times faster to build and solve once, with far
less memory, but the solver must then support natively every constraint the model uses. The
models built for a [`TwoStageProblem`](@ref) are direct; `kwargs` only apply otherwise.
"""
function new_model(optimizer; silent::Bool = true, direct::Bool = false, kwargs...)
    m = if direct
        JuMP.direct_model(optimizer isa MOI.AbstractOptimizer ? optimizer : MOI.instantiate(optimizer))
    else
        JuMP.Model(optimizer; kwargs...)
    end
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
    groups::Vector{Any}      # the constraints of `con` by concrete type, and their positions
end

RecourseTemplate(model, y, con, rhs, row, sign) =
    RecourseTemplate(model, y, con, rhs, row, sign, constraint_groups(con))
RecourseTemplate(model, y, con, rhs) =
    RecourseTemplate(model, y, con, rhs, collect(eachindex(con)), ones(length(con)))

"""
    constraint_groups(con)

The constraints of `con` grouped by concrete type, each group as a typed vector with the positions
of its constraints in `con`. `con` mixes the types of the rows of different senses; reading and
writing them group by group, through typed vectors, avoids resolving every call at run time, and
lets JuMP set the right-hand sides of a group in one call.
"""
function constraint_groups(con::AbstractVector)
    positions = Dict{DataType,Vector{Int}}()
    for (k, c) in enumerate(con)
        push!(get!(positions, typeof(c), Int[]), k)
    end
    return Any[(T[con[k] for k in ks], ks) for (T, ks) in positions]
end

"""Update the right-hand side of `r` for the first-stage solution `x` in scenario `s`."""
function set_recourse_rhs!(md::AbstractTwoStageModel, r::RecourseTemplate, x, s)
    values = r.rhs(x, s)
    for (refs, ks) in r.groups
        _set_rhs!(refs, values, ks)
    end
end
_set_rhs!(refs, values, ks) = JuMP.set_normalized_rhs(refs, values[ks])

"""The multiplier of each row of the recourse problem, assembled from the multipliers of `r.con`."""
function multipliers(r::RecourseTemplate)
    σ = zeros(isempty(r.row) ? 0 : maximum(r.row))
    for (refs, ks) in r.groups
        _add_multipliers!(σ, r, refs, ks)
    end
    return σ
end
function _add_multipliers!(σ, r::RecourseTemplate, refs, ks)
    for (c, k) in zip(refs, ks)
        σ[r.row[k]] += r.sign[k] * JuMP.dual(c)
    end
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
        q isa Function ? q : (qc = Float64.(q); _ -> qc),     # converted once, not at every call
        Float64.(W), to_sense.(collect(senses2)),
        T isa AbstractMatrix ? (Tc = Float64.(T); _ -> Tc) : T,
        h isa AbstractVector ? (hc = Float64.(h); _ -> hc) : h,
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
    m = new_model(optimizer; direct = true)
    x = declare_box!(m, @variable(m, [1:n_x(pb)], base_name = "x"), pb.lb1, pb.ub1, pb.integer1)
    add_rows!(m, pb.A * x, pb.senses1, pb.b)
    @objective(m, Min, dot(pb.c, x))
    return MasterTemplate(m, x)
end

function build_recourse(pb::TwoStageProblem, optimizer, s)
    ξ = pb.ξ[s]
    m = new_model(optimizer; direct = true)
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
    m = new_model(optimizer; direct = true)
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
              threads = false, drop_inactive = nothing,
              regularization = :none, rho = 1.0, radius = nothing, max_radius = Inf,
              eta1 = 1e-4, eta2 = 1e-4, gamma = 2.0, patience = 3,
              x0 = nothing, callback = nothing, verbose = true, log = stdout)

Solve a two-stage stochastic linear program by L-shaped decomposition, on the modelization `md`.

At iteration `k` the master problem is

    min  c'x + Σₖ Pₖ θₖ   s.t.  Ax ⋛ b,  cuts

where the scenarios are partitioned into clusters `𝒮ₖ` of probability `Pₖ = Σ_{s ∈ 𝒮ₖ} pₛ`, and
`θₖ` approximates `Σ_{s ∈ 𝒮ₖ} (pₛ / Pₖ) Q(x, ξₛ)`, the expected recourse within the cluster;
`θₖ` joins the objective with its first cut. Solving the second stage at `xᵏ` gives, per scenario,
either the multipliers `πₛ` of an optimal solution or — should the recourse problem be infeasible —
an elastic model proving it, and its multipliers `σₛ`. Both are assembled into cuts by
`cut_coefficients` and `cut_intercept`: `Q(·, ξ)` and the total violation are convex in `x`, and
the cuts are their supporting hyperplanes at `xᵏ`. The cut of cluster `k` averages those of its
scenarios,

    Eₖ = Σ_{s ∈ 𝒮ₖ} (pₛ / Pₖ) T(ξₛ)'πₛ ,  Eₖ'x + θₖ ≥ Eₖ'xᵏ + Σ_{s ∈ 𝒮ₖ} (pₛ / Pₖ) Q(xᵏ, ξₛ),

and is only added if it is violated at `xᵏ`, that is if `θₖ` falls short of the expected recourse
within the cluster beyond the tolerance (always, at the first round, which bounds every `θₖ`).

`cuts` sets the partition (Birge and Louveaux, 2011, Section 5.1.d, and the "hybrid approaches" of
deck 04):
* `:single`, one cluster: **one** optimality cut per iteration, `E = Σₛ pₛ T(ξₛ)'πₛ`;
* `:multi`, one cluster per scenario: up to **one cut per scenario**, a larger master and usually
  fewer iterations;
* an integer `C`: `C` clusters of consecutive scenarios, of sizes as equal as possible;
* a vector of vectors: the clusters themselves, a partition of `1:S`.

`regularization` stabilizes the iterates around a centre `a`, the first second-stage-feasible
iterate to begin with (`x0`, if given and feasible), as presented in deck 04:
* `:regularized_decomposition` (Ruszczyński, 1986; Birge and Louveaux, Section 5.2) adds
  `‖x - a‖² / (2 rho)` to the objective of the master, which then is a quadratic program. GLPK
  cannot solve it, and HiGHS 1.15's QP solver often returns wrong solutions, declared optimal, even
  on small masters: before stopping, `lshaped` checks that the last master was solved, and
  raises an error if not. Ipopt solves them, provided it is not allowed to relax the constraints, or the
  iterates violate the feasibility cuts slightly and the same cut comes back forever:
  `master_optimizer = optimizer_with_attributes(Ipopt.Optimizer, "bound_relax_factor" => 0.0)`.
  A feasibility cut is a null step; when no cut is violated at `xᵏ`, or when `xᵏ` is not worse than
  `a`, `xᵏ` becomes the centre.
* `:trust_region` (Linderoth and Wright, 2003) restricts the master to `‖x - a‖∞ ≤ Δ`, starting
  from `radius`, by default `0.1 max(1, ‖a‖∞)`. The step is judged by the ratio `τ` of the actual
  reduction `f(a) - f(xᵏ)` to the reduction `m(a) - m(xᵏ)` the model predicted, `f` being the
  objective and `m` the master's model: if `τ ≥ eta1`, `xᵏ` becomes the centre, and if moreover
  `τ ≥ eta2`, the radius grows to `min(max_radius, gamma Δ)`; after `patience` unsuccessful
  iterations in a row, the radius is halved (deck 04 leaves this rule open).
Both stop when the model value of the master solution reaches `f(a)`, up to `tol` relative to
`1 + |f(a)| + Σₖ Pₖ |Qₖ(a)|`, the scale of the cut tests: then `a` is optimal. Up to a tolerance,
this test is weaker for regularized decomposition, whose proximal term flattens the model: there,
the master is then also solved without it, and the method only stops if this lower bound meets the
incumbent; otherwise the solution of this LP is the next iterate. The master of a regularized iteration gives no lower bound; a valid one is computed once at
the end, by solving the master without the regularization.

`x0` is a first-stage decision at which the subproblems are evaluated before the first master
problem, instead of the solution of a master without any cut. `x0 = :mean_value` takes the solution
of the expected value problem (a [`TwoStageProblem`](@ref) only), as in the example of regularized
decomposition of deck 04. It must satisfy the first-stage constraints.

`callback(entry)` is called with each new entry of the `history` (below); if it returns `true`, the
method stops there, with `stopped = true` in the result. [`evpi_bounds`](@ref) and
[`vss_bounds`](@ref) turn an entry into intervals containing the EVPI and the VSS, so that the method
can stop as soon as these intervals answer the question asked.

`threads = true` solves the recourse problems of an iteration in parallel, on the threads Julia was
started with (`julia -t N`). The master is only modified once all of them are solved, in the order
of the scenarios, so the results do not depend on it. The recourse solver must be thread-safe when
its instances are solved concurrently: HiGHS is, GLPK is not and crashes. GLPK is unsafe even
unused here: its models are freed by finalizers, which crash Julia when the garbage collector runs
them on a worker thread; call `GC.gc()` before `lshaped` once GLPK models have been discarded. The
callbacks of a [`JuMPTwoStageProblem`](@ref) must also be safe to call from several threads at
once, which they are if they only build and read their own models.

`drop_inactive = k` deletes an optimality cut from the master once it has been slack at `k`
consecutive iterates, which keeps the master small on long runs (Linderoth and Wright, 2003).
Feasibility cuts are kept. A master with fewer cuts is still a relaxation, so the lower bound and
the stopping test remain valid, but the finite convergence of the method is no longer guaranteed:
`maxiter` then matters.

`master_optimizer` and `recourse_optimizer` are independent — the master is a sequence of LPs that
grows, the recourse problems are small and re-solved at every iterate, so the two rarely deserve
the same solver; both accept anything [`new_model`](@ref) accepts.

Without regularization, convergence is declared when no cut is violated any more, up to `tol`
relative to `1 + |·|`: an absolute tolerance would ask more of the solvers than they deliver on
objectives of large magnitude. `feastol` is the total violation below which the elastic model counts
as proof of feasibility.

The upper bound is that of the **incumbent**, the best first-stage solution met so far: the value
`c'xᵏ + Q(xᵏ)` of the current iterate does not decrease monotonically. Returns a named tuple with
the incumbent `x` and its `objective`, the last `lower_bound`, the `gap` between them,
`converged`, `stopped`, the counters of both kinds of cut and of the `dropped_cuts`, the
`clusters`, the `regularization`, the `problem` `md` and the models, and the `history`, whose
entries record per iteration the bounds, the `value` of the iterate `x`, and the `model` value of
the master solution. [`first_stage_decision`](@ref), [`second_stage_decision`](@ref),
[`print_first_stage`](@ref) and [`print_second_stage`](@ref) read the decisions off this result.
"""
function lshaped(md::AbstractTwoStageModel;
                 optimizer = nothing,
                 master_optimizer = optimizer,
                 recourse_optimizer = optimizer,
                 cuts = :single,
                 maxiter::Integer = 500,
                 tol::Real = 1e-8,
                 feastol::Real = 1e-7,
                 threads::Bool = false,
                 drop_inactive::Union{Nothing,Integer} = nothing,
                 regularization::Symbol = :none,
                 rho::Real = 1.0,
                 radius::Union{Nothing,Real} = nothing,
                 max_radius::Real = Inf,
                 eta1::Real = 1e-4,
                 eta2::Real = 1e-4,
                 gamma::Real = 2.0,
                 patience::Integer = 3,
                 x0 = nothing,
                 callback = nothing,
                 verbose::Bool = true,
                 log::IO = stdout)
    drop_inactive === nothing || drop_inactive >= 1 ||
        error("`drop_inactive` must be a positive number of iterations, got $drop_inactive")
    regularization in (:none, :regularized_decomposition, :trust_region) ||
        error("`regularization` must be :none, :regularized_decomposition or :trust_region, " *
              "got :$regularization")
    rho > 0 || error("`rho` must be positive, got $rho")
    radius === nothing || radius > 0 || error("`radius` must be positive, got $radius")
    0 < eta1 <= eta2 < 1 || error("0 < eta1 ≤ eta2 < 1 is required, got $eta1 and $eta2")
    gamma > 1 || error("`gamma` must exceed 1, got $gamma")
    patience >= 1 || error("`patience` must be positive, got $patience")
    master_optimizer === nothing &&
        error("pass `optimizer`, or both `master_optimizer` and `recourse_optimizer`")
    recourse_optimizer === nothing &&
        error("pass `optimizer`, or both `master_optimizer` and `recourse_optimizer`")
    scenarios = n_scenarios(md)
    p = [scenario_probability(md, s) for s in 1:scenarios]
    isapprox(sum(p), 1; atol = 1e-9) ||
        error("the probabilities must sum up to one, they sum up to $(sum(p))")
    clusters = scenario_clusters(cuts, scenarios)
    C = length(clusters)
    P = [sum(p[s] for s in cluster) for cluster in clusters]

    master = build_master(md, master_optimizer)
    recourse = [check_recourse(build_recourse(md, recourse_optimizer, s), s) for s in 1:scenarios]
    elastic = Any[nothing for _ in 1:scenarios]
    c, constant = master_cost(master)
    x = master.x
    n = length(x)
    if x0 === :mean_value
        md isa TwoStageProblem ||
            error("`x0 = :mean_value` needs a TwoStageProblem, whose scenarios can be averaged")
        x0 = expected_value_problem(md; optimizer = master_optimizer).x
    elseif x0 !== nothing
        length(x0) == n || error("x0 has $(length(x0)) components, the first stage $n variables")
        x0 = Float64.(collect(x0))
    end
    regularized = regularization != :none

    # anonymous, so that a master model of the user's may have a variable of its own named θ
    θ = @variable(master.model, [1:C], base_name = "θ")
    bounded = falses(C)
    center, f_center = Float64[], Inf          # the centre `a` of a regularization, and f(a)
    scale_center = Inf                         # the scale of the stopping test at `a`
    Δ = radius === nothing ? NaN : Float64(radius)
    box = Any[]                                # the trust region, once there is a centre
    unsuccessful = 0
    stale = true                               # whether the master objective must be rebuilt
    function set_objective!(; proximal::Bool = true)
        f = dot(c, x) + constant               # the constant too, or the lower bound would miss it
        for k in 1:C
            bounded[k] && (f += P[k] * θ[k])
        end
        if proximal && regularization == :regularized_decomposition && !isempty(center)
            f += sum((x[j] - center[j])^2 for j in 1:n) / (2rho)
        end
        @objective(master.model, Min, f)
    end
    function set_center!(a, fa, scale)
        center, f_center, scale_center = copy(a), fa, scale
        if regularization == :trust_region
            isnan(Δ) && (Δ = 0.1 * max(1.0, norm(a, Inf)))
            set_box!()
        end
        stale = true
    end
    function set_box!()
        if isempty(box)
            for j in 1:n
                push!(box, @constraint(master.model, x[j] >= center[j] - Δ))
                push!(box, @constraint(master.model, x[j] <= center[j] + Δ))
            end
        else
            for j in 1:n
                JuMP.set_normalized_rhs(box[2j-1], center[j] - Δ)
                JuMP.set_normalized_rhs(box[2j], center[j] + Δ)
            end
        end
    end
    function unsuccessful!()                   # the trust region shrinks after `patience` of them
        unsuccessful += 1
        if unsuccessful >= patience
            Δ /= 2
            unsuccessful = 0
            set_box!()
        end
    end
    # Certify that xᵏ solves the proximal master, before stopping on it: the master is convex, so
    # xᵏ minimizes m(x) + ‖x - a‖²/(2ρ) if and only if it minimizes the LP m(x) + g'x, where
    # g = (xᵏ - a)/ρ is the gradient of the quadratic term at xᵏ. A wrong solution of the QP,
    # declared optimal, would otherwise stop the method at a wrong point.
    function check_proximal_step(xk, model)
        g = (xk .- center) ./ rho
        f = dot(c, x) + constant + dot(g, x)
        for k in 1:C
            f += P[k] * θ[k]
        end
        @objective(master.model, Min, f)
        stale = true
        optimize!(master.model)
        termination_status(master.model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) || return nothing
        expected, best = model + dot(g, xk), objective_value(master.model)
        best >= expected - 1e-6 * (1 + abs(expected)) && return nothing
        error("""the quadratic master problem was not solved: $(solver_label(master_optimizer)) \
                 reports it optimal, but its solution does not minimize it (the linearized master \
                 reaches $best < $expected). Pass another quadratic programming solver as \
                 `master_optimizer`, such as \
                 `optimizer_with_attributes(Ipopt.Optimizer, "bound_relax_factor" => 0.0)`, or use \
                 `regularization = :trust_region`, whose master is linear.""")
    end
    """The value of the model of the master at `a`: the cuts, not yet those of this iteration."""
    function model_value(a)
        θa = fill(-Inf, C)
        for cut in stored
            θa[cut.cluster] = max(θa[cut.cluster], cut.e - dot(cut.E, a))
        end
        return dot(c, a) + constant + dot(P, θa)
    end

    if verbose
        label = C == 1 ? "Single-cut" : C == scenarios ? "Multi-cut" : "Hybrid ($C clusters)"
        extra = regularization == :regularized_decomposition ? ", regularized decomposition (rho = $rho)" :
                regularization == :trust_region ? ", trust region" : ""
        @printf(log, "%s L-shaped method%s\n", label, extra)
        @printf(log, "  master: %s | recourse: %s | scenarios: %d\n\n",
                solver_label(master_optimizer), solver_label(recourse_optimizer), scenarios)
        @printf(log, " iter %16s   upper bound          gap %14s %14s\n",
                regularized ? "model value" : "lower bound", C == 1 ? "θ" : "min θₖ", "Q(x)")
    end

    n_optimality = n_feasibility = n_dropped = 0
    stored = StoredCut[]                       # the optimality cuts in the master
    history = NamedTuple[]
    lower, upper, xstar, converged, stopped = -Inf, Inf, fill(NaN, n), false, false
    iteration = 0

    for it in 1:maxiter
        iteration = it
        if it == 1 && x0 !== nothing            # start from x0 rather than from a master
            xk, θk, model = x0, fill(-Inf, C), -Inf
        else
            stale && (set_objective!(); stale = false)
            optimize!(master.model)
            status = termination_status(master.model)
            # a local optimum of the convex master, as Ipopt reports it, is a global one
            status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) ||
                error("master problem: $status" *
                      (regularization == :regularized_decomposition && !isempty(center) ?
                       "; this quadratic master may need another solver, see `regularization`" : ""))
            xk = value.(x)
            θk = [bounded[k] ? value(θ[k]) : -Inf for k in 1:C]
            model = all(bounded) ? dot(c, xk) + constant + dot(P, θk) : -Inf
            lower = !regularized && any(bounded) ? objective_value(master.model) : -Inf
            if drop_inactive !== nothing
                n_dropped += drop_inactive_cuts!(master.model, stored, it, drop_inactive)
            end
            # the stopping test of the regularized methods: the model reaches f(a) (deck 04)
            if regularized && !isempty(center) && model >= f_center - tol * scale_center
                regularization == :trust_region && (converged = true; break)
                check_proximal_step(xk, model)
                # Up to a tolerance ε, the test only bounds the error of `a` by about
                # ‖a - x*‖ √(2ε/ρ): the master without the proximal term, an LP, says whether the
                # incumbent is optimal. If not, its solution is the next iterate, an L-shaped step
                # where the model is too loose.
                set_objective!(proximal = false)
                stale = true
                optimize!(master.model)
                status = termination_status(master.model)
                status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) || error("master problem: $status")
                lower = objective_value(master.model)
                upper - lower <= tol * scale_center && (converged = true; break)
                xk = value.(x)
                θk = [value(θ[k]) for k in 1:C]
                model = lower
            end
        end

        # the subproblems, in parallel or not; the master is only modified afterwards, in the
        # order of the scenarios, so that the results do not depend on the threads
        pieces = Vector{Any}(undef, scenarios)
        foreach_scenario(threads, scenarios) do s
            pieces[s] = solve_scenario!(md, recourse, elastic, recourse_optimizer, s, xk, n, feastol)
        end
        infeasible = [s for s in 1:scenarios if !pieces[s].feasible]
        for s in infeasible
            @constraint(master.model, dot(pieces[s].E, x) >= pieces[s].e)
            n_feasibility += 1
        end
        if !isempty(infeasible)                # a null step for the regularizations
            regularization == :trust_region && !isempty(center) && unsuccessful!()
            verbose && @printf(log, "%5d   feasibility cut(s) on scenario(s) %s\n",
                               it, join(infeasible, ", "))
            continue
        end
        Qs = [pieces[s].Q for s in 1:scenarios]
        Q = dot(p, Qs)
        current = dot(c, xk) + constant + Q
        if current < upper                     # a new incumbent
            upper, xstar = current, xk
        end
        push!(history, (iteration = it, lower_bound = lower, upper_bound = upper,
                        gap = upper - lower, value = current, Q = Q, x = xk, model = model))
        verbose && @printf(log, "%5d   %14s %12.6f %12.4g %14s %14.6f\n", it,
                           _num(regularized ? model : lower), upper,
                           (regularized ? f_center : upper) - (regularized ? model : lower),
                           _num(minimum(θk)), Q)

        # checked before convergence: crossing bounds often show up on the very iteration where
        # the invalid cuts make θ look converged
        if !regularized && lower > upper + tol * (1 + abs(upper))
            @warn """the lower bound exceeds the upper bound ($lower > $upper): the cuts are \
                     invalid, which `intercept = :textbook` causes on bounded recourse variables"""
        end
        Qk = [sum(p[s] * Qs[s] for s in cluster) / P[k] for (k, cluster) in enumerate(clusters)]
        violated = [!bounded[k] || θk[k] < Qk[k] - tol * (1 + abs(Qk[k])) for k in 1:C]
        # the stopping test of the regularizations, f(a) - m(x) ≤ tol × scale, must not ask more
        # than the cuts: without a violated cut, m(xᵏ) may still fall short of f(xᵏ) by up to
        # Σₖ Pₖ tol (1 + |Qₖ|), and the method would neither cut nor stop
        scale = 1 + abs(current) + dot(P, abs.(Qk))
        if !regularized && !any(violated)
            converged = true
            break
        end
        if callback !== nothing && callback(history[end]) === true
            stopped = true
            break
        end

        # the centre of a regularization, before the cuts of this iteration change the model
        if regularized && isempty(center)
            set_center!(xk, current, scale)    # the first feasible iterate
        elseif regularization == :regularized_decomposition
            # exact serious step if no cut is violated, approximate if xᵏ is not worse than a
            (!any(violated) || current <= f_center) && set_center!(xk, current, scale)
        elseif regularization == :trust_region
            predicted = model_value(center) - model
            τ = predicted > 0 ? (f_center - current) / predicted : -Inf
            if τ >= eta1                       # successful: move the centre
                unsuccessful = 0
                τ >= eta2 && (Δ = min(max_radius, gamma * Δ))
                set_center!(xk, current, scale)
            else
                unsuccessful!()
            end
        end

        for (k, cluster) in enumerate(clusters)
            violated[k] || continue
            E = sum(p[s] / P[k] * pieces[s].E for s in cluster)
            e = sum(p[s] / P[k] * pieces[s].e for s in cluster)
            con = @constraint(master.model, dot(E, x) + θ[k] >= e)
            push!(stored, StoredCut(con, it, k, E, e))
            n_optimality += 1
        end
        # the objective only changes when some θ is bounded for the first time
        if !all(bounded)
            bounded .= true
            stale = true
        end
    end

    # a regularized master gives no lower bound: solve it once without the regularization
    if regularized && any(bounded)
        foreach(con -> JuMP.delete(master.model, con), box)
        empty!(box)
        set_objective!(proximal = false)
        optimize!(master.model)
        termination_status(master.model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) &&
            (lower = objective_value(master.model))
    end
    if verbose
        if converged
            @printf(log, "converged in %d iteration(s): %d optimality cut(s), %d feasibility cut(s)%s\n",
                    iteration, n_optimality, n_feasibility,
                    drop_inactive === nothing ? "" : ", $n_dropped dropped")
        elseif stopped
            @printf(log, "stopped by the callback at iteration %d\n", iteration)
        else
            @printf(log, "no convergence in %d iteration(s), last gap %g\n", maxiter, upper - lower)
        end
    end
    return (x = xstar, objective = upper, lower_bound = lower, gap = upper - lower,
            converged = converged, stopped = stopped, iterations = iteration,
            optimality_cuts = n_optimality, feasibility_cuts = n_feasibility,
            dropped_cuts = n_dropped, clusters = clusters, regularization = regularization,
            problem = md, master = master, recourse = recourse, history = history)
end

"""An optimality cut `E'x + θₖ ≥ e` of the master, and the last iteration at which it was tight."""
mutable struct StoredCut
    con::Any
    last::Int
    cluster::Int
    E::Vector{Float64}
    e::Float64
end

"""
    scenario_clusters(cuts, S)

The partition of the scenarios `1:S` that `cuts` describes: one cluster for `:single`, one per
scenario for `:multi`, `C` clusters of consecutive scenarios for an integer `C`, or the given
vector of clusters, checked to be a partition.
"""
function scenario_clusters(cuts, S::Integer)
    cuts === :single && return [collect(1:S)]
    cuts === :multi && return [[s] for s in 1:S]
    if cuts isa Integer
        1 <= cuts <= S || error("the number of clusters must be between 1 and $S, got $cuts")
        bounds = [round(Int, k * S / cuts) for k in 0:cuts]
        return [collect(bounds[k]+1:bounds[k+1]) for k in 1:cuts]
    end
    if cuts isa AbstractVector && all(cluster -> cluster isa AbstractVector{<:Integer}, cuts)
        clusters = [collect(Int, cluster) for cluster in cuts]
        all(!isempty, clusters) && sort(reduce(vcat, clusters)) == 1:S ||
            error("the clusters must be a partition of the scenarios 1:$S")
        return clusters
    end
    error("`cuts` must be :single, :multi, a number of clusters or a vector of clusters, got $cuts")
end

"""
    evpi_bounds(entry, ws)
    vss_bounds(entry, eev)

Intervals containing the EVPI and the VSS, from the bounds `L ≤ RP ≤ U` of an entry of the
`history` of [`lshaped`](@ref), or of its result, and the wait-and-see value `ws` or the expected
result `eev` of the mean-value decision (deck 04, "Using the L-shaped iterations"):

    L - WS ≤ EVPI ≤ U - WS,      EEV - U ≤ VSS ≤ EEV - L.

Before the first optimality cut, and along a regularized run, `L` is `-Inf`.
"""
evpi_bounds(entry, ws::Real) = (entry.lower_bound - ws, _upper(entry) - ws)
vss_bounds(entry, eev::Real) = (eev - _upper(entry), eev - entry.lower_bound)
_upper(entry) = hasproperty(entry, :upper_bound) ? entry.upper_bound : entry.objective

"""
    solve_scenario!(md, recourse, elastic, optimizer, s, xk, n, feastol)

Solve the recourse problem of scenario `s` at `xk`. Returns `(feasible = true, Q, E, e)`, its value
and the coefficients of its optimality cut, or, if it is infeasible, `(feasible = false, E, e)`,
those of a feasibility cut obtained from the elastic model, built on first need. Only touches the
models of scenario `s`, so that the scenarios can be solved in parallel.
"""
function solve_scenario!(md, recourse, elastic, optimizer, s, xk, n, feastol)
    r = recourse[s]
    set_recourse_rhs!(md, r, xk, s)
    optimize!(r.model)
    status = termination_status(r.model)
    if status != MOI.OPTIMAL
        # Either infeasible, or the solver cannot tell: the elastic model decides which.
        if elastic[s] === nothing
            elastic[s] = build_elastic(md, optimizer, s)
        end
        elastic[s] === nothing && error("""
            scenario $s: the recourse problem is $status and no elastic model is available; \
            provide `elastic_builder` to cut feasibility""")
        set_recourse_rhs!(md, elastic[s], xk, s)
        optimize!(elastic[s].model)
        termination_status(elastic[s].model) == MOI.OPTIMAL ||
            error("elastic problem, scenario $s: $(termination_status(elastic[s].model))")
        certificate = feasibility_certificate(elastic[s])
        certificate.value <= feastol &&
            error("scenario $s: the recourse problem is $status yet its elastic relaxation has " *
                  "value $(certificate.value); the second stage is unbounded, or the solver " *
                  "could not solve it")
        E = cut_coefficients(md, elastic[s], s, xk, certificate.σ)
        length(E) == n || error("`cut_coefficients` returned $(length(E)) coefficients, expected $n")
        return (feasible = false, Q = NaN, E = E,
                e = cut_intercept(md, elastic[s], s, xk, certificate.σ, certificate.value))
    end
    # π as the solver returns them: πᵀ(h - Tx) = Q(x, ξ) whatever the mix of senses
    π = multipliers(r)
    Q = recourse_value(r)
    E = cut_coefficients(md, r, s, xk, π)
    length(E) == n || error("`cut_coefficients` returned $(length(E)) coefficients, expected $n")
    return (feasible = true, Q = Q, E = E, e = cut_intercept(md, r, s, xk, π, Q))
end

"""
    foreach_scenario(f, threads, scenarios)

`f(s)` for every scenario, on Julia's threads if `threads` and there are several. An error is
raised once all the scenarios are done, that of the first scenario in which one occurred.
"""
function foreach_scenario(f, threads::Bool, scenarios::Integer)
    if !threads || Threads.nthreads() == 1
        foreach(f, 1:scenarios)
        return nothing
    end
    errors = Vector{Any}(nothing, scenarios)
    Threads.@threads for s in 1:scenarios
        try
            f(s)
        catch err
            errors[s] = err
        end
    end
    i = findfirst(!isnothing, errors)
    i === nothing || throw(errors[i])
    return nothing
end

"""
    drop_inactive_cuts!(model, cuts, k, window)

Record which of the optimality `cuts` are tight at the current solution of the master `model`, and
delete those slack at the last `window` iterations; returns how many were deleted.
"""
function drop_inactive_cuts!(model::JuMP.Model, cuts::Vector{StoredCut}, k::Integer, window::Integer)
    # all the slacks are read first: deleting a row invalidates the solution of the model
    tight = map(cuts) do cut
        rhs = JuMP.normalized_rhs(cut.con)
        return JuMP.value(cut.con) - rhs <= 1e-7 * (1 + abs(rhs))
    end
    keep = trues(length(cuts))
    for (i, cut) in enumerate(cuts)
        if tight[i]
            cut.last = k
        elseif k - cut.last >= window
            JuMP.delete(model, cut.con)
            keep[i] = false
        end
    end
    keepat!(cuts, keep)
    return count(!, keep)
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

# --------------------------------------------------------------------------------------------
# the value of perfect information and of the stochastic solution
# --------------------------------------------------------------------------------------------

"""
    ScenarioRestriction(md, s)

The two-stage problem `md` restricted to its scenario `s`, taken with probability one: what the
decision maker would solve knowing that `ξ = ξₛ`. It is solved by [`lshaped`](@ref) like any other
modelization, which makes [`wait_and_see`](@ref) work on both.
"""
struct ScenarioRestriction{M <: AbstractTwoStageModel} <: AbstractTwoStageModel
    md::M
    s::Int
end

n_scenarios(::ScenarioRestriction) = 1
scenario_probability(::ScenarioRestriction, _) = 1.0
scenario_data(r::ScenarioRestriction, _) = scenario_data(r.md, r.s)
build_master(r::ScenarioRestriction, optimizer) = build_master(r.md, optimizer)
build_recourse(r::ScenarioRestriction, optimizer, _) = build_recourse(r.md, optimizer, r.s)
build_elastic(r::ScenarioRestriction, optimizer, _) = build_elastic(r.md, optimizer, r.s)
cut_coefficients(r::ScenarioRestriction, rec, _, x, π) = cut_coefficients(r.md, rec, r.s, x, π)
cut_intercept(r::ScenarioRestriction, rec, _, x, π, v) = cut_intercept(r.md, rec, r.s, x, π, v)

function _solve_converged(md; kwargs...)
    res = lshaped(md; kwargs..., verbose = false)
    res.converged || error("the L-shaped method did not converge, gap $(res.gap)")
    return res
end

"""
    wait_and_see(md; optimizer, kwargs...)

The **wait-and-see** value `WS = E[min_x z(x, ξ)]` of the two-stage problem `md`: the expected
optimal value if `ξ` were known before deciding `x`, each scenario being solved on its own by
[`lshaped`](@ref), to which `kwargs` are passed. Returns a named tuple with the `value` `WS`, and
per scenario the optimal `values` and first-stage `decisions`.
"""
function wait_and_see(md::AbstractTwoStageModel; kwargs...)
    results = [_solve_converged(ScenarioRestriction(md, s); kwargs...) for s in 1:n_scenarios(md)]
    values = [res.objective for res in results]
    p = [scenario_probability(md, s) for s in 1:n_scenarios(md)]
    return (value = dot(p, values), values = values, decisions = [res.x for res in results])
end

"""
    expected_value_problem(pb::TwoStageProblem; optimizer, kwargs...)

The **expected value problem**: `pb` with its scenarios replaced by their mean `ξ̄ = Σₛ pₛ ξₛ`, solved
by [`extensive_form`](@ref). Returns a named tuple with its optimal `value` `EV` and its optimal
first-stage decision `x`, the mean-value decision `x̄(ξ̄)`.

The scenarios must support averaging (numbers or vectors); for a [`JuMPTwoStageProblem`](@ref),
whose scenarios are opaque, compute `x̄(ξ̄)` yourself and pass it to [`expected_result`](@ref) or
[`vss`](@ref).
"""
function expected_value_problem(pb::TwoStageProblem; optimizer, kwargs...)
    ξ̄ = try
        sum(pb.p[s] * pb.ξ[s] for s in eachindex(pb.ξ))
    catch err
        err isa MethodError || rethrow()
        error("the scenarios of this problem cannot be averaged; compute the mean-value " *
              "decision yourself and pass it to `expected_result` or `vss`")
    end
    ev = TwoStageProblem(pb.c, pb.A, pb.senses1, pb.b, pb.q, pb.W, pb.senses2, pb.T, pb.h,
                         [ξ̄], [1.0], pb.lb1, pb.ub1, pb.integer1, pb.lb, pb.ub, pb.intercept)
    _, x̄, value = extensive_form(ev; optimizer, kwargs...)
    return (value = value, x = x̄)
end

"""
    expected_result(md, x; optimizer, master_optimizer = optimizer, recourse_optimizer = optimizer)

The expected cost `c'x + Σₛ pₛ Q(x, ξₛ)` of the first-stage decision `x`: with the mean-value
decision `x̄(ξ̄)` of [`expected_value_problem`](@ref), the **expected result of the EV solution**
`EEV`. It is `+Inf` if `x` leaves the recourse problem of some scenario infeasible. `x` is assumed
to satisfy the first-stage constraints.
"""
function expected_result(md::AbstractTwoStageModel, x::AbstractVector;
                         optimizer = nothing, master_optimizer = optimizer,
                         recourse_optimizer = optimizer)
    (master_optimizer === nothing || recourse_optimizer === nothing) &&
        error("pass `optimizer`, or both `master_optimizer` and `recourse_optimizer`")
    c, constant = master_cost(build_master(md, master_optimizer))
    length(x) == length(c) ||
        error("x has $(length(x)) components, the first stage $(length(c)) variables")
    total = dot(c, x) + constant
    for s in 1:n_scenarios(md)
        total += scenario_probability(md, s) * _recourse_value(md, recourse_optimizer, s, x)
        isinf(total) && return total
    end
    return total
end

"""`Q(x, ξₛ)`, `+Inf` if the recourse problem is infeasible, `-Inf` if it is unbounded."""
function _recourse_value(md, optimizer, s, x)
    r = check_recourse(build_recourse(md, optimizer, s), s)
    set_recourse_rhs!(md, r, x, s)
    optimize!(r.model)
    status = termination_status(r.model)
    status == MOI.OPTIMAL && return recourse_value(r)
    status == MOI.INFEASIBLE && return Inf
    status == MOI.DUAL_INFEASIBLE && return -Inf
    # the solver could not tell infeasible from unbounded: the elastic model decides
    e = build_elastic(md, optimizer, s)
    e === nothing && error("scenario $s: the recourse problem is $status, and no elastic model " *
                           "is available to tell whether it is infeasible")
    set_recourse_rhs!(md, e, x, s)
    optimize!(e.model)
    return recourse_value(e) > 1e-7 ? Inf : -Inf
end

"""
    evpi(md; rp = nothing, optimizer, kwargs...)

The **expected value of perfect information** `EVPI = RP - WS`: the most the decision maker should
pay for a perfect forecast of `ξ`. `RP`, the optimal value of `md`, is computed by
[`lshaped`](@ref) unless given as `rp`; `kwargs` are passed to `lshaped`.
"""
function evpi(md::AbstractTwoStageModel; rp = nothing, kwargs...)
    rp = rp === nothing ? _solve_converged(md; kwargs...).objective : rp
    return rp - wait_and_see(md; kwargs...).value
end

"""
    vss(pb::TwoStageProblem; rp = nothing, optimizer, kwargs...)
    vss(md, x̄; rp = nothing, optimizer, kwargs...)

The **value of the stochastic solution** `VSS = EEV - RP`: what is gained by modelling the
uncertainty rather than replacing `ξ` by its mean. The mean-value decision `x̄` is computed by
[`expected_value_problem`](@ref), or given, which is the only way for a
[`JuMPTwoStageProblem`](@ref). `VSS` is `+Inf` when `x̄` is infeasible for some scenario. `RP` is
computed by [`lshaped`](@ref) unless given as `rp`.
"""
function vss(md::AbstractTwoStageModel, x̄::AbstractVector; rp = nothing, kwargs...)
    rp = rp === nothing ? _solve_converged(md; kwargs...).objective : rp
    return expected_result(md, x̄; _optimizers(; kwargs...)...) - rp
end
vss(pb::TwoStageProblem; rp = nothing, kwargs...) =
    vss(pb, expected_value_problem(pb; optimizer = _optimizers(; kwargs...).master_optimizer).x;
        rp = rp, kwargs...)

"""The optimizer keywords among `kwargs`, as `expected_result` takes them."""
function _optimizers(; optimizer = nothing, master_optimizer = optimizer,
                     recourse_optimizer = optimizer, kwargs...)
    return (master_optimizer = master_optimizer, recourse_optimizer = recourse_optimizer)
end

"""One line per variable, named as in its model, or `fallback[j]` if it has no name."""
function _print_values(io::IO, variables, values, fallback, digits)
    labels = [isempty(JuMP.name(v)) ? "$fallback[$j]" : JuMP.name(v) for (j, v) in enumerate(variables)]
    width = maximum(length, labels; init = 0)
    for (label, v) in zip(labels, values)
        println(io, "  ", rpad(label, width), " = ", _round(v, digits))
    end
end

end # module