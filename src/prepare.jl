# Construction of tracing parameters.

"""
    is_time_dependent(f::Function)

Judge whether the field function is time dependent.
"""
is_time_dependent(f) = applicable(f, zeros(3), 0.0) || applicable(f, zeros(6), 0.0)

is_time_dependent(::AbstractField{itd}) where {itd} = itd
is_time_dependent(::AbstractFieldInterpolator) = false

"""
    Field(f; time_dependent=is_time_dependent(f))

Wrap `f(x)` or `f(x, t)` as a field. Automatic detection prefers the two-argument
signature; set `time_dependent` explicitly for ambiguous or input-type-specific callables.
"""
struct Field{itd, F} <: AbstractField{itd}
    field_function::F
    function Field{itd, F}(field_function::F) where {itd, F}
        return isa(itd, Bool) ? new(field_function) : throw(ArgumentError("itd must be a boolean."))
    end
end

Field(f; time_dependent::Bool = is_time_dependent(f)) =
    Field{time_dependent, typeof(f)}(f)

@inline (f::Field{true})(xu, t) = getfield(f, :field_function)(xu, t)
@inline function (f::Field{true})(xu)
    throw(ArgumentError("Time-dependent field function must have a time argument."))
end
@inline (f::Field{false})(xu, t) = getfield(f, :field_function)(xu)
@inline (f::Field{false})(xu) = getfield(f, :field_function)(xu)

function jacobian(f::Field{true}, xu, t)
    return jacobian(f.field_function, xu, t)
end

function jacobian(f::Field{false}, xu, t)
    return jacobian(f.field_function, xu)
end

function jacobian(f, xu, t)
    return ForwardDiff.jacobian(r -> f(r, t), xu)
end

function derivative_t(f::Field{true}, xu, t)
    return derivative_t(f.field_function, xu, t)
end

function derivative_t(f::Field{false}, xu, t)
    return zero(xu)
end

function derivative_t(f, xu, t)
    return ForwardDiff.derivative(τ -> f(xu, τ), t)
end

function Base.show(io::IO, f::Field)
    println(io, "Field with interpolation support")
    return println(io, "Time-dependent: ", is_time_dependent(f))
end

function Base.getproperty(f::Field, sym::Symbol)
    if sym === :field_function
        fn = getfield(f, :field_function)
        return fn isa ErasedFunction ? fn.raw : fn
    end
    return getfield(f, sym)
end

include("erased.jl")

prepare_field(f; erased = false) = erased ? ErasedField(f) : Field(f)
prepare_field(f::Field{itd, ErasedFunction}; erased = false) where {itd} = f
prepare_field(f::Field; erased = false) = erased ? ErasedField(f) : f
prepare_field(f::ZeroField; erased = false) = f
prepare_field(::Nothing; erased = false) = ZeroField()


"""
    prepare(; E=nothing, B=nothing, F=nothing, species=Proton, q, m, c, erased=false)

Build particle parameters `(; q, q2m, m, c, E, B, F)` from callable electric, magnetic, and
external force fields. Omitted fields are zero. Each callable accepts `x` or `(x, t)` and
returns three components. Construct numerical fields with [`build_interpolator`](@ref)
before calling `prepare`. See [`Field`](@ref) to select the calling convention.

`q` and `m` override `species`. `c` defaults to the SI speed of light, except with `q` or
`m` given, whose units are not known: relativistic equations then require an explicit `c`.
Particle constants take the float type of the state when traced.
"""
function prepare(;
        E = nothing, B = nothing, F = nothing, species = Proton,
        q = nothing, m = nothing, c = nothing, erased::Bool = false
    )
    any(f -> f isa AbstractArray, (E, B, F)) && throw(
        ArgumentError(
            "Array fields require build_interpolator(A, grids; bc)."
        )
    )
    q, m, c = _particle_constants(species, q, m, c)
    E, B, F = map(f -> prepare_field(f; erased), (E, B, F))
    return (; q, q2m = q / m, m, c, E, B, F)
end

function _particle_constants(species, q, m, c_)
    light = isnothing(c_) && isnothing(q) && isnothing(m) ? c : c_
    return (@something q species.q), (@something m species.m), light
end

"""
    prepare_erased(; E = nothing, B = nothing, F = nothing, kwargs...)

Tracing parameters with type-erased fields for uniform dispatch and static compilation.
Equivalent to `prepare(; erased = true, kwargs...)`.
"""
prepare_erased(; kw...) = prepare(; erased = true, kw...)

"""
    PotentialField(;
        phi = nothing, A = nothing, grad_phi = nothing, grad_A = nothing, erased = false
    )

Scalar potential `phi(x[, t])` and vector potential `A(x[, t])` for canonical tracing;
`nothing` is an absent potential. Gradients that are not given are computed with
`ForwardDiff`. `grad_A(x[, t])[i, j]` is ``∂A_j/∂x_i``, so that `grad_A * v` is
``∇(\\mathbf{A} ⋅ \\mathbf{v})`` at fixed ``\\mathbf{v}``.
"""
_prepare_potential_scalar(f, erased) = erased ? ErasedField(f) : Field(f)
_prepare_potential_scalar(f::ZeroField, erased) = f
_prepare_potential_scalar(::Nothing, erased) = ZeroField()

function PotentialField(;
        phi = nothing, A = nothing, grad_phi = nothing, grad_A = nothing,
        erased::Bool = false
    )
    f_phi, f_A = _prepare_potential_scalar(phi, erased), prepare_field(A; erased)
    return PotentialField(
        f_phi, f_A, _grad_phi(grad_phi, f_phi; erased), _grad_A(grad_A, f_A; erased)
    )
end

_grad_phi(g, phi; erased = false) = prepare_field(g; erased)
_grad_phi(::Nothing, ::ZeroField; erased = false) = ZeroField()
_grad_phi(::Nothing, phi; erased = false) =
    (x, t) -> SVector{3}(ForwardDiff.gradient(r -> phi(r, t), x))

_grad_A(g::ZeroField, A; erased = false) = g
_grad_A(g, A; erased = false) = erased ? ErasedField(g) : Field(g)
_grad_A(::Nothing, ::ZeroField; erased = false) = ZeroField()
_grad_A(::Nothing, A; erased = false) =
    (x, t) -> SMatrix{3, 3}(ForwardDiff.jacobian(r -> A(r, t), x)')

"""
    prepare(pf::PotentialField; species = Proton, q = nothing, m = nothing, c = nothing)

Return parameters `(; q, m, c, pf)` for `TraceCanonicalProblem`. `c` defaults as in
[`prepare`](@ref).
"""
function prepare(
        pf::PotentialField; species = Proton, q = nothing, m = nothing, c = nothing
    )
    q, m, c = _particle_constants(species, q, m, c)
    return (; q, m, c, pf)
end

"""
    prepare_canonical(; phi = nothing, A = nothing, grad_phi = nothing, grad_A = nothing, kwargs...)

[`PotentialField`](@ref) and [`prepare`](@ref) in one call, for `TraceCanonicalProblem`.
"""
prepare_canonical(;
    phi = nothing, A = nothing, grad_phi = nothing, grad_A = nothing,
    erased::Bool = false, kw...
) =
    prepare(PotentialField(; phi, A, grad_phi, grad_A, erased); kw...)
