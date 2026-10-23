# Tracing equations.

get_x(u) = @inbounds SA[u[1], u[2], u[3]]
get_v(u) = @inbounds SA[u[4], u[5], u[6]]

function get_dv(v::AbstractVector{T}, x, p, t) where {T <: AbstractFloat}
    E = SVector{3, T}(p.E(x, t))
    B = SVector{3, T}(p.B(x, t))
    F = SVector{3, T}(p.F(x, t))

    return T(p.q2m) * (v × B + E) + F / T(p.m)
end

get_dv(v, x, p, t) = p.q2m * (v × p.B(x, t) + p.E(x, t)) + p.F(x, t) / p.m

"""
    get_dx!(dx, v, x, p, t)

In-place solver components for `DynamicalODEProblem` (location).
"""
function get_dx!(dx, v, x, p, t)
    @inbounds for i in eachindex(dx, v)
        dx[i] = v[i]
    end
    return
end

"""
    get_dv!(dv, v, x, p, t)

In-place solver components for `DynamicalODEProblem` (velocity).
"""
function get_dv!(dv, v, x, p, t)
    val = get_dv(v, x, p, t)
    @inbounds for i in eachindex(dv, val)
        dv[i] = val[i]
    end

    return
end

"""
    get_relativistic_v(γv; c = c)

Velocity of a particle with the relativistic velocity `γv`.
"""
function get_relativistic_v(γv; c = c)
    γ²v² = γv[1]^2 + γv[2]^2 + γv[3]^2
    return γv / √(1 + γ²v² / oftype(float(γ²v²), c)^2)
end

@inline function _store!(dy, dx, dv)
    @inbounds for i in 1:3
        dy[i] = dx[i]
        dy[i + 3] = dv[i]
    end
    return
end

@inline function _trace_rhs(y, p, t)
    v = get_v(y)
    return v, get_dv(v, y, p, t)
end

@inline function _trace_relativistic_rhs(y, p, t)
    v = get_relativistic_v(get_v(y))
    return v, get_dv(v, y, p, t)
end

@inline function _trace_normalized_rhs(y, p, t)
    v = get_v(y)
    return v, v × p.B(y, t) + p.E(y, t)
end

@inline function _trace_relativistic_normalized_rhs(y, p, t)
    v = get_relativistic_v(get_v(y); c = 1)
    return v, v × p.B(y, t) + p.E(y, t)
end

"""
    trace!(dy, y, p, t)

ODE equations for charged particle moving in EM field and external force field with in-place form.
"""
trace!(dy, y, p, t) = _store!(dy, _trace_rhs(y, p, t)...)

"""
    trace(y, p, t)::SVector{6}

ODE equations for charged particle moving in EM field and external force field with out-of-place form.
"""
trace(y, p, t) = vcat(_trace_rhs(y, p, t)...)

"""
    trace_relativistic!(dy, y, p, t)

ODE equations for relativistic charged particle (x, γv) moving in EM field with in-place form.
"""
trace_relativistic!(dy, y, p, t) = _store!(dy, _trace_relativistic_rhs(y, p, t)...)

"""
    trace_relativistic(y, p, t) -> SVector{6}

ODE equations for relativistic charged particle (x, γv) moving in static EM field with out-of-place form.
"""
trace_relativistic(y, p, t) = vcat(_trace_relativistic_rhs(y, p, t)...)

"""
    trace_normalized!(dy, y, p, t)

Normalized ODE equations for charged particle moving in EM field with in-place form.
If the field is in 2D X-Y plane, periodic boundary should be applied for the field in z via
the extrapolation function.
"""
trace_normalized!(dy, y, p, t) = _store!(dy, _trace_normalized_rhs(y, p, t)...)

"""
    trace_normalized(y, p, t)

Normalized ODE equations for charged particle moving in EM field with out-of-place form.
"""
trace_normalized(y, p, t) = vcat(_trace_normalized_rhs(y, p, t)...)

"""
    trace_relativistic_normalized!(dy, y, p, t)

Normalized ODE equations for relativistic charged particle (x, γv) moving in EM field with in-place form.
"""
trace_relativistic_normalized!(dy, y, p, t) =
    _store!(dy, _trace_relativistic_normalized_rhs(y, p, t)...)

"""
    trace_relativistic_normalized(y, p, t)

Normalized ODE equations for relativistic charged particle (x, γv) moving in EM field with out-of-place form.
"""
trace_relativistic_normalized(y, p, t) = vcat(_trace_relativistic_normalized_rhs(y, p, t)...)

"""
    _lorentz_factor(u, f)

Lorentz factor of the velocity part `u` of a state traced with the equations `f`,
which store `γv` in place of `v` for the relativistic equations.
"""
_lorentz_factor(u, f) = one(eltype(u))
_lorentz_factor(γv, ::Union{typeof(trace_relativistic!), typeof(trace_relativistic)}) =
    sqrt(1 + (γv ⋅ γv) / c^2)
_lorentz_factor(
    γv,
    ::Union{typeof(trace_relativistic_normalized!), typeof(trace_relativistic_normalized)}
) = sqrt(1 + γv ⋅ γv)

"""
    trace_gc_exb!(dx, x, p, t)

Equations for tracing the guiding center using the ExB drift and parallel velocity from a reference trajectory.
"""
function trace_gc_exb!(dx, x, p, t)
    # TODO: support external forces
    (; q2m, sol) = p
    Efunc, Bfunc = p.E, p.B
    xu = sol(t)
    v = get_v(xu)
    E = Efunc(x, t)
    B = Bfunc(x, t)

    Bmag = norm(B)
    b = B / Bmag
    v_par = (v ⋅ b) .* b

    @inbounds dx[1:3] = (E × b) / Bmag + v_par

    return
end

"""
    trace_gc_flr!(dx, x, p, t)

Equations for tracing the guiding center using the ExB drift with FLR corrections and parallel velocity.
"""
function trace_gc_flr!(dx, x, p, t)
    # TODO: support external forces
    (; q2m, sol) = p
    Efunc, Bfunc = p.E, p.B
    xu = sol(t)
    xp = get_x(xu)
    v = get_v(xu)
    E = Efunc(x, t)
    B = Bfunc(x, t)

    # B at particle position
    Bx = Bfunc(xp, t)
    Bmag_particle = norm(Bx)
    b_particle = Bx / Bmag_particle
    v_par = (v ⋅ b_particle) .* b_particle
    v_perp = v - v_par

    r4 = (v_perp ⋅ v_perp) / (4 * (q2m * Bmag_particle)^2)

    # Helper for FLR term: (E × B) / B²
    EB(x_in) = begin
        E_in = Efunc(x_in, t)
        B_in = Bfunc(x_in, t)
        (E_in × B_in) / (B_in ⋅ B_in)
    end

    # dx = EB(x) + r^2/4 * ∇²(EB) + v_par
    # EB(x) is redundant, use E and B directly
    @inbounds dx[1:3] =
        (E × B) / (B ⋅ B) + r4 * vector_laplacian(EB, SVector{3}(x)) + v_par

    return
end

# Row 3(j - 1) + i, column k of the nested Jacobian is ∂²fᵢ/∂xⱼ∂xₖ.
function vector_laplacian(f, x::SVector{3})
    H = ForwardDiff.jacobian(y -> vec(ForwardDiff.jacobian(f, y)), x)
    return SVector(ntuple(i -> H[i, 1] + H[3 + i, 2] + H[6 + i, 3], 3))
end

"""
    get_B_parameters(x, t, Bfunc)

Evaluate the magnetic field `Bfunc` at `(x, t)` and return the field vector `B`,
its magnitude `Bmag`, the unit vector `b̂ = B / Bmag`, the gradient `∇B`, and the
Jacobian `JB` of the field.
"""
@inline function get_B_parameters(x, t, Bfunc)
    B, JB = _get_B_jacobian(x, t, Bfunc)
    Bmag = norm(B)
    b̂ = B / Bmag
    # Grad-B from Jacobian
    ∇B = JB' * b̂

    return B, Bmag, b̂, ∇B, JB
end

@inline function get_E_parameters(x, t, Efunc)
    E = Efunc(x, t)
    JE = jacobian(Efunc, x, t)

    return E, JE
end

"""
    trace_gc_drifts!(dx, x, p, t)

Equations for tracing the guiding center using analytical drifts, including the grad-B drift, curvature drift, and ExB drift.
Parallel velocity is also added. This expression requires the full particle trajectory `p.sol`.
"""
function trace_gc_drifts!(dx, x, p, t)
    # TODO: support external forces
    (; q2m, sol) = p
    Efunc, Bfunc = p.E, p.B
    xu = sol(t)
    v = get_v(xu)
    E = Efunc(x, t)

    B, Bmag, b, ∇B, JB = get_B_parameters(x, t, Bfunc)

    v_par_val = v ⋅ b
    v_par = v_par_val .* b
    v_perp = v - v_par
    Ω = q2m * Bmag

    # Curvature vector κ = (b̂ ⋅ ∇) b̂
    # κ = (JB * b̂ - b̂ * (∇B ⋅ b̂)) / Bmag
    κ = (JB * b + b * (-∇B ⋅ b)) / Bmag

    v_E = (E × b) / Bmag
    w = v_perp - v_E

    # w^2*(b×∇|B|)/(2*Ω*B) + v∥^2*(b×κ)/Ω + v_E + v∥
    @inbounds dx[1:3] = (w[1]^2 + w[2]^2 + w[3]^2) * (b × ∇B) / (2 * Ω * Bmag) +
        v_par_val^2 * (b × κ) / Ω +
        v_E + v_par

    return
end


"""
    trace_gc!(dy, y, p, t)

Guiding center equations for nonrelativistic charged particle moving in EM field with in-place form.
Variable `y = (x, y, z, u)`, where `u` is the velocity along the magnetic field at (x,y,z).
"""
function trace_gc!(dy, y, p, t)
    v1, v2, v3, du = get_gc_derivatives(y, p, t)

    @inbounds dy[1] = v1
    @inbounds dy[2] = v2
    @inbounds dy[3] = v3
    @inbounds dy[4] = du

    return
end

"""
    trace_gc(y, p, t)

Guiding center equations for nonrelativistic charged particle moving in EM field with out-of-place form.
"""
function trace_gc(y, p, t)
    v1, v2, v3, du = get_gc_derivatives(y, p, t)
    return SVector{4}(v1, v2, v3, du)
end

"""
    get_gc_velocity(y, p, t)

Get the guiding center velocity.
"""
function get_gc_velocity(y, p, t)
    v1, v2, v3, _ = get_gc_derivatives(y, p, t)
    return SVector{3}(v1, v2, v3)
end

function get_gc_derivatives(y, p, t)
    # TODO: support external forces
    (; q, q2m, μ) = p
    X = get_x(y)

    E = p.E(X, t)
    B, Bmag, b̂, ∇B, JB = get_B_parameters(X, t, p.B)

    # ∇ × b̂ = (∇ × B + b̂ × ∇B) / B
    # ∇ × B from JB (Jacobian of B)
    curlB = SVector{3}(JB[3, 2] - JB[2, 3], JB[1, 3] - JB[3, 1], JB[2, 1] - JB[1, 2])
    curlb = (curlB + b̂ × ∇B) / Bmag

    # effective EM fields
    # B* = B + (m/q) u (∇ × b)
    # E* = E - (μ/q) ∇B
    # In CGS: B* = B + (c p_par / q) (∇ × b). In SI, c -> 1.
    Bᵉ = B + (y[4] / q2m) * curlb
    Eᵉ = E - (μ / q) * ∇B

    inv_Bparᵉ = inv(b̂ ⋅ Bᵉ)

    # dx/dt = (p_par/m * B* +  E* × b ) / B*_par
    #       = (u * B* + E* × b) / B*_par
    # In CGS: c/q * q E* x b. In SI, c=1.
    v = (y[4] * Bᵉ + Eᵉ × b̂) * inv_Bparᵉ

    # dp_par/dt = q/B*_par * B* ⋅ E* => du/dt = (q/m)/B*_par * B* ⋅ E*
    du = q2m * inv_Bparᵉ * Bᵉ ⋅ Eᵉ

    return v[1], v[2], v[3], du
end

"""
    _work_rate_terms(q, m, v_par, μ, E, ∇B, κ, b̂, Bmag, dBdt) -> SVector{4}

Leading-order guiding-center work rates [W] shared by the full-orbit and the
guiding-center flavors. Returns `(P_par, P_fermi, P_grad, P_betatron)`:

| Term | Formula | Physics |
|:---|:---|:---|
| `P_par` | `q v∥ (E⋅b̂)` | direct acceleration by `E∥` |
| `P_fermi` | `(m v∥²/B) (b̂×κ)⋅E` | work by `E` on the curvature drift |
| `P_grad` | `(μ/B) (b̂×∇B)⋅E` | convective part of the betatron channel, `μ v_E⋅∇B` |
| `P_betatron` | `μ ∂B/∂t` | local part of the betatron channel |

Here `v_E = (E×b̂)/B` is the ExB drift, so `P_grad` is the perpendicular heating a
particle picks up by being convected across `∇B` at fixed `μ`, while `P_betatron` is
the heating from the local change of `|B|`. Together they are `μ (∂_t + v_E⋅∇) B`.

Reference: Northrop, T. G. (1963), *The Adiabatic Motion of Charged Particles*.
"""
@inline function _work_rate_terms(q, m, v_par::Number, μ, E, ∇B, κ, b̂, Bmag, dBdt)
    # 1. Parallel work: q v∥ (E ⋅ b̂)
    P_par = q * v_par * (E ⋅ b̂)
    # 2. Fermi work: m v∥² / B (b̂ × κ) ⋅ E
    P_fermi = (m * v_par^2 / Bmag) * ((b̂ × κ) ⋅ E)
    # 3. Gradient drift work: μ / B (b̂ × ∇B) ⋅ E = μ v_E ⋅ ∇B
    P_grad = (μ / Bmag) * ((b̂ × ∇B) ⋅ E)
    # 4. Betatron work: μ ∂B/∂t
    P_betatron = μ * dBdt

    return SVector{4}(P_par, P_fermi, P_grad, P_betatron)
end

"""
    get_work_rates(xv, p, t[, magnetic_properties, E_field]) -> SVector{4}

Calculate the work rates [W] acting on a full-orbit particle or a guiding center.
Returns the four leading-order adiabatic terms `(P_par, P_fermi, P_grad, P_betatron)`
documented in [`_work_rate_terms`](@ref); see [`get_work_channels`](@ref) for the
grouping into physical acceleration channels.

The state is taken as a guiding center when `p` carries a magnetic moment `μ`. Saved
guiding-center states can carry diagnostics appended by `save_fields`/`save_work`, so
their length is not a reliable discriminator.

`magnetic_properties` and `E_field` are optional precomputed values of
[`get_magnetic_properties`](@ref) and of the electric field at `(get_x(xv), t)`, used
to avoid recomputing the field Jacobian when the caller already has them.

For a full-orbit state these are gyro-averaged quantities: they do not reproduce the
instantaneous `q v⋅E`, which oscillates at the gyrofrequency. They are therefore best
compared against the energy change accumulated over many gyroperiods.

For the guiding-center equations solved by [`trace_gc!`](@ref), the four terms sum to
`dK/dt` with `K = ½ m v∥² + μB` up to `O(ε)` corrections. The sum is *exact* when
`b̂⋅(∇×b̂) = 0`, as in axisymmetric fields without a `B_φ` component. In this case
`B*_∥ = |B|` and the residual twist term vanishes. Planar field lines or translation
invariance alone do not imply this condition. When it holds, any remaining mismatch
is numerical (e.g. time-integration or quadrature error).

All four terms are zero in a magnetic null.
"""
@inline function get_work_rates(xv, p, t, magnetic_properties = nothing, E_field = nothing)
    return hasproperty(p, :μ) ?
        _get_work_rates_gc(xv, p, t, magnetic_properties, E_field) :
        _get_work_rates_fo(xv, p, t, magnetic_properties, E_field)
end

@inline function _get_work_rates_fo(xu, p, t, magnetic_properties = nothing, E_field = nothing)
    (; q2m, m) = p
    Efunc, Bfunc = p.E, p.B
    r = get_x(xu)
    q = q2m * m

    E = E_field === nothing ? Efunc(r, t) : E_field

    if magnetic_properties === nothing
        _, ∇B, κ, b̂, Bmag = get_magnetic_properties(r, t, Bfunc)
    else
        _, ∇B, κ, b̂, Bmag = magnetic_properties
    end

    if Bmag == 0
        return SVector{4, eltype(xu)}(0, 0, 0, 0)
    end

    v = get_v(xu)
    v_par = v ⋅ b̂
    v_perp = _get_v_perp(v, b̂)
    μ = m * (v_perp ⋅ v_perp) / (2 * Bmag)
    dBdt = derivative_t(Bfunc, r, t) ⋅ b̂

    return _work_rate_terms(q, m, v_par, μ, E, ∇B, κ, b̂, Bmag, dBdt)
end

@inline function _get_work_rates_gc(xv, p, t, magnetic_properties = nothing, E_field = nothing)
    (; q, q2m, μ) = p
    Efunc, Bfunc = p.E, p.B
    r = get_x(xv)
    v_par = xv[4]

    E = E_field === nothing ? Efunc(r, t) : E_field

    if magnetic_properties === nothing
        _, ∇B, κ, b̂, Bmag = get_magnetic_properties(r, t, Bfunc)
    else
        _, ∇B, κ, b̂, Bmag = magnetic_properties
    end

    if Bmag == 0
        return SVector{4, eltype(xv)}(0, 0, 0, 0)
    end

    m = q / q2m
    dBdt = derivative_t(Bfunc, r, t) ⋅ b̂

    return _work_rate_terms(q, m, v_par, μ, E, ∇B, κ, b̂, Bmag, dBdt)
end

"""
    get_work_channels(xv, p, t) -> SVector{3}

Group the work rates into the three physical acceleration channels
`(P_parallel, P_fermi, P_betatron)` [W].

The mapping between the four leading-order terms of [`get_work_rates`](@ref) and the
channels is:

- `P_parallel = P_par`: direct acceleration by `E∥`;
- `P_fermi = P_fermi`: work by `E` on the curvature drift;
- `P_betatron = P_betatron + P_grad`: perpendicular heating under conservation of `μ`.
  Grad-B drift work is *not* a separate mechanism; since
  `(μ/B)(b̂×∇B)⋅E = μ v_E⋅∇B`, it is the convective (spatial) part of the betatron
  channel, while `μ ∂B/∂t` is its local (temporal) part.

Both a full-orbit and a guiding-center state are accepted, as in `get_work_rates`.
"""
function get_work_channels(xv, p, t)
    P_par, P_fermi, P_grad, P_betatron = get_work_rates(xv, p, t)

    return SVector{3}(P_par, P_fermi, P_grad + P_betatron)
end

@inline _vector_potential(pf, x, t) = pf.A isa ZeroField ? zero(x) : SVector{3}(pf.A(x, t))

@inline _kinetic_to_velocity(p_kin, m, c, relativistic::Bool) =
    relativistic ? p_kin / (m * √(1 + (p_kin ⋅ p_kin) / (m * c)^2)) : p_kin / m

@inline function _canonical_rhs(y, p, t, relativistic::Bool)
    (; q, m, pf) = p
    x = get_x(y)
    c = relativistic ? p.c : nothing
    v = _kinetic_to_velocity(get_v(y) - q * _vector_potential(pf, x, t), m, c, relativistic)
    grad_phi = pf.grad_phi isa ZeroField ? zero(x) : SVector{3}(pf.grad_phi(x, t))
    return v, q * (pf.grad_A(x, t) * v) - q * grad_phi
end

trace_canonical!(dy, y, p, t) = _store!(dy, _canonical_rhs(y, p, t, false)...)
trace_canonical(y, p, t) = vcat(_canonical_rhs(y, p, t, false)...)
trace_canonical_relativistic!(dy, y, p, t) = _store!(dy, _canonical_rhs(y, p, t, true)...)
trace_canonical_relativistic(y, p, t) = vcat(_canonical_rhs(y, p, t, true)...)

function velocity_to_canonical(
        x0::AbstractVector, v0::AbstractVector, p;
        relativistic::Bool = false, t = 0.0
    )
    (; q, m, c, pf) = p
    x_vec = SA[x0[1], x0[2], x0[3]]
    v_vec = SA[v0[1], v0[2], v0[3]]
    γ = relativistic ? 1 / √(1 - (v_vec ⋅ v_vec) / c^2) : 1
    return vcat(x_vec, γ * m * v_vec + q * _vector_potential(pf, x_vec, t))
end

function canonical_to_velocity(
        x::AbstractVector, p_can::AbstractVector, p;
        relativistic::Bool = false, t = 0.0
    )
    (; q, m, c, pf) = p
    x_vec = SA[x[1], x[2], x[3]]
    p_kin = SA[p_can[1], p_can[2], p_can[3]] - q * _vector_potential(pf, x_vec, t)
    return _kinetic_to_velocity(p_kin, m, c, relativistic)
end

function canonical_to_velocity(u::AbstractVector, p; relativistic::Bool = false, t = 0.0)
    return canonical_to_velocity(u[1:3], u[4:6], p; relativistic, t)
end

function canonical_hamiltonian(u::AbstractVector, p, t = 0.0; relativistic::Bool = false)
    (; q, m, c, pf) = p
    x = get_x(u)
    phi = pf.phi isa ZeroField ? zero(eltype(x)) : pf.phi(x, t)
    p_kin = get_v(u) - q * _vector_potential(pf, x, t)
    p² = p_kin ⋅ p_kin

    if relativistic
        return √(m^2 * c^4 + c^2 * p²) + q * phi
    else
        return p² / (2 * m) + q * phi
    end
end
