# TraceProblem constructors and SciML interface integration

@inline ODE_DEFAULT_ISOUTOFDOMAIN(u, p, t) = false

"""
    _named_params(p)

Name the slots of positional full-orbit parameters `(q2m, m, E, B, F)`. Other containers
pass through.
"""
_named_params(p::NTuple{5, Any}) = NamedTuple{(:q2m, :m, :E, :B, :F)}(p)
_named_params(p) = p

# Only the particle constants follow the state's float type; other entries of `p`, such as
# user data, keep theirs.
const PARTICLE_CONSTANTS = (:q, :q2m, :m, :c, :μ)

@inline _promote_p(::Type{T}, p::NamedTuple{K}) where {T <: AbstractFloat, K} = NamedTuple{K}(
    map((k, x) -> k in PARTICLE_CONSTANTS && x isa Real ? T(x) : x, K, values(p))
)
@inline _promote_p(::Type{T}, p) where {T} = p

@inline function _promote_trace_args(u0, tspan, p)
    p = _named_params(p)
    T = eltype(u0)
    if T <: AbstractFloat
        tspan_T = (T(tspan[1]), T(tspan[2]))
        p_T = _promote_p(T, p)
        return tspan_T, p_T
    else
        return tspan, p
    end
end

# Default `(in-place, out-of-place)` equations of each model.
_equations(::Type{FullOrbit}) = (trace!, trace)
_equations(::Type{RelativisticOrbit}) = (trace_relativistic!, trace_relativistic)
_equations(::Type{Canonical}) = (trace_canonical!, trace_canonical)
_equations(::Type{RelativisticCanonical}) =
    (trace_canonical_relativistic!, trace_canonical_relativistic)
_equations(::Type{GuidingCenter}) = (trace_gc!, trace_gc)

_relativistic(::Type{FullOrbit}) = RelativisticOrbit
_relativistic(::Type{Canonical}) = RelativisticCanonical
_relativistic(::Type{M}) where {M} =
    throw(ArgumentError("$M has no relativistic equations."))

_default_f(::Type{M}, isinplace) where {M} = ODEFunction{isinplace, DEFAULT_SPECIALIZATION}(
    _equations(M)[isinplace ? 1 : 2]
)

_is_default_f(::Type{M}, f) where {M} = SciMLBase.unwrapped_f(f.f) in _equations(M)

# Passing relativistic equations as `f` selects their model.
_resolve_model(::Type{M}, f) where {M} = M
_resolve_model(::Type{FullOrbit}, f::Union{typeof(trace_relativistic!), typeof(trace_relativistic)}) =
    RelativisticOrbit
_resolve_model(
    ::Type{Canonical},
    f::Union{typeof(trace_canonical_relativistic!), typeof(trace_canonical_relativistic)}
) = RelativisticCanonical

"""
    _param_model(p)

Model that the parameters `p` of a bare state are written for: the state-level diagnostics'
default where no problem names the model.
"""
@inline _param_model(p) = hasproperty(p, :μ) ? GuidingCenter() :
    hasproperty(p, :pf) ? Canonical() : FullOrbit()

_rhs_model(f, p) = _param_model(p)
_rhs_model(::Union{typeof(trace_relativistic!), typeof(trace_relativistic)}, p) =
    RelativisticOrbit()
_rhs_model(
    ::Union{typeof(trace_canonical_relativistic!), typeof(trace_canonical_relativistic)}, p
) = RelativisticCanonical()

"""
    _model(prob)

Model of `prob`. A plain `ODEProblem` has none of its own, so it follows from the equations
and the parameters.
"""
_model(::TraceProblem{M}) where {M} = M()
_model(prob) = hasproperty(prob.f, :f) ?
    _rhs_model(SciMLBase.unwrapped_f(prob.f.f), prob.p) : _param_model(prob.p)

"""
    TraceProblem(u0, tspan, p; relativistic = false, prob_func = DEFAULT_PROB_FUNC)
    TraceProblem(f, u0, tspan, p; prob_func = DEFAULT_PROB_FUNC)
    TraceProblem{M}(...)

Problem specification for particle tracing in the model `M`: `FullOrbit` (the default),
`RelativisticOrbit`, `Canonical`, `RelativisticCanonical` or `GuidingCenter`, also written
`TraceCanonicalProblem` and `TraceGCProblem`. Supports the native solvers of each model and
standard ODE solvers (e.g. Tsit5, Vern7).

When `f` is not provided, the model's equations are used: in-place for mutable arrays,
out-of-place for `StaticArray`s. `relativistic = true` selects the relativistic variant of
the model, in which case a full-orbit `u0` holds `γv` in place of `v`. Boris solvers accept
only the default `FullOrbit` equations without an external force.

Normalized units need no separate equations: prepare `p` with `q = 1, m = 1`, and
`c = 1` for the relativistic case.
"""
TraceProblem(args...; kw...) = TraceProblem{FullOrbit}(args...; kw...)

function TraceProblem{M}(
        u0, tspan, p; relativistic::Bool = false, prob_func = DEFAULT_PROB_FUNC
    ) where {M}
    N = relativistic ? _relativistic(M) : M
    return TraceProblem{N}(_default_f(N, !(u0 isa StaticArray)), u0, tspan, p; prob_func)
end

function TraceProblem{M}(f, u0, tspan, p; prob_func = DEFAULT_PROB_FUNC) where {M}
    _f = ODEFunction{!(u0 isa StaticArray), DEFAULT_SPECIALIZATION}(f)
    return TraceProblem{M}(_f, u0, tspan, p; prob_func)
end

function TraceProblem{M}(
        f::AbstractODEFunction{iip}, u0, tspan, p; prob_func = DEFAULT_PROB_FUNC
    ) where {M, iip}
    tspan, p = _promote_trace_args(u0, tspan, p)
    N = _resolve_model(M, SciMLBase.unwrapped_f(f.f))
    return TraceProblem{
        N, typeof(u0), typeof(tspan), iip, typeof(p), typeof(f), typeof(prob_func),
    }(f, u0, tspan, p, prob_func)
end

function SciMLBase.remake(
        prob::TraceProblem{M}; f = prob.f, u0 = prob.u0, tspan = prob.tspan, p = prob.p,
        prob_func = prob.prob_func, kwargs...
    ) where {M}
    # Default equations follow `u0` between mutable and static arrays.
    if f === prob.f && _is_default_f(M, prob.f)
        f = _default_f(M, !(u0 isa StaticArray))
    end
    return TraceProblem{M}(f, u0, tspan, p; prob_func)
end

function solve(
        prob::TraceProblem, alg::SciMLBase.AbstractODEAlgorithm,
        ensemblealg::BasicEnsembleAlgorithm;
        trajectories::Int = 1,
        seed::Union{Nothing, Integer} = nothing,
        safetycopy::Bool = false,
        kwargs...
    )
    ensemble_prob = EnsembleProblem(
        prob; prob_func = prob.prob_func, safetycopy
    )

    return solve(ensemble_prob, alg, ensemblealg; trajectories, seed, kwargs...)
end
