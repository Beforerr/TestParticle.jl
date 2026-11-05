# Guiding center.

"Split a full-orbit state into position and velocity."
@inline _split_xv(xv) = xv[SA[1:3...]], xv[SA[4:6...]]

"""
    _gc_position(x, v, B, q, m, t)

Guiding-center position `X = x - ρ(x)`, with the Larmor radius `ρ` evaluated at
the particle: the zeroth-order map.
"""
@inline function _gc_position(x, v, B, q, m, t)
    b = B(x, t)
    Bmag = norm(b)
    return x - ((b / Bmag) × v) / ((q / m) * Bmag)
end

"""
    _gc_frame(x, v, E, B, m, t) -> (; b̂, w, vpar, μ)

Field-aligned frame at position `x`: unit field `b̂`, gyration velocity
`w = v - vpar b̂ - v_E`, parallel velocity `vpar = b̂ ⋅ v` and magnetic moment
`μ = m|w|² / (2|B|)`.
"""
@inline function _gc_frame(x, v, E, B, m, t)
    b = B(x, t)
    Bmag = norm(b)
    b̂ = b / Bmag
    vpar = b̂ ⋅ v
    w = v - vpar * b̂ - (E(x, t) × b̂) / Bmag
    return (; b̂, w, vpar, μ = m * (w ⋅ w) / (2 * Bmag))
end

"Gyrophase of the gyration velocity `w` in the frame built from `b̂`."
@inline function _gc_phase(b̂, w)
    e1, e2 = get_perp_vector(b̂)
    return atan(w ⋅ e2, w ⋅ e1)
end

"""
    _get_gc_parameters(xv, E, B, q, m, t) -> (X, vpar, q, q / m, μ, phase)

Extract the guiding-center position `X`, parallel velocity `vpar`, magnetic
moment `μ`, and gyrophase `phase` from a full-particle state `xv` at time `t`.
`E` and `B` must be callable as `E(x, t)` / `B(x, t)`; static fields simply
ignore `t`.

#TODO : support external forces
"""
@inline function _get_gc_parameters(xv, E, B, q, m, t)
    x, v = _split_xv(xv)
    X = _gc_position(x, v, B, q, m, t)
    f = _gc_frame(X, v, E, B, m, t)
    return X, f.vpar, q, q / m, f.μ, _gc_phase(f.b̂, f.w)
end

"""
    _gc_adia_parameters(xv, E, B, q, m, t) -> (X, μ)

Guiding-center position and magnetic moment of the full-orbit state `xv` — the
two quantities an adiabaticity estimate needs. The gyrophase (`get_perp_vector`
and an `atan2`) is skipped because the estimate never uses it; the values are
identical to those of [`_get_gc_parameters`](@ref).
"""
@inline function _gc_adia_parameters(xv, E, B, q, m, t)
    x, v = _split_xv(xv)
    X = _gc_position(x, v, B, q, m, t)
    return X, _gc_frame(X, v, E, B, m, t).μ
end

"""
    _gc_state_at_switch(xv, E, B, q, m, t, niter) -> (X, vpar, μ, phase)

Guiding-center state of the full-orbit state `xv`, refined by up to `niter`
fixed-point iterations of `X = x - ρ(X)`: the Larmor radius is evaluated at the
guiding center instead of at the particle, which is the first-order correction
that removes the `O(ε)` error of the zeroth-order map. `niter = 0` recovers the
zeroth-order map.

This map is the exact inverse of [`_gc_to_full`](@ref), so a switch to the
guiding center and back returns the particle where it was; the reverse
conversion needs no correction of its own.

An iteration is kept only if it reduces the residual `‖x - (X + ρ(X))‖`, so a
strongly non-adiabatic crossing, where the fixed-point map stops contracting,
cannot make the conversion worse than the zeroth-order one.
"""
@inline function _gc_state_at_switch(xv, E, B, q, m, t, niter)
    x, v = _split_xv(xv)
    X, vpar, _, _, μ, phase = _get_gc_parameters(xv, E, B, q, m, t)
    niter > 0 || return X, vpar, μ, phase

    x_rec = _gc_to_full(SVector(X[1], X[2], X[3], vpar), E, B, q, m, μ, t, phase)
    residual = norm(x - get_x(x_rec))

    for _ in 1:niter
        X_next = X + (x - get_x(x_rec))
        f = _gc_frame(X_next, v, E, B, m, t)
        phase_next = _gc_phase(f.b̂, f.w)
        x_rec_next = _gc_to_full(
            SVector(X_next[1], X_next[2], X_next[3], f.vpar), E, B, q, m,
            f.μ, t, phase_next
        )
        residual_next = norm(x - get_x(x_rec_next))
        residual_next < residual || break

        X, vpar, μ, phase = X_next, f.vpar, f.μ, phase_next
        x_rec, residual = x_rec_next, residual_next
    end

    return X, vpar, μ, phase
end

"""
    prepare_gc(xv, p; t = 0) -> (state_gc, param_gc)

Guiding-center state `[X..., vpar]` of the full-orbit state `xv = [x, y, z, vx, vy, vz]` at
time `t` and the parameters from [`prepare`](@ref) with its magnetic moment `μ` added, in
the float type of `xv`. The state is an `SVector` for a static `xv`.

The guiding-center equations have no external force, so `p.F` must be absent.
"""
function prepare_gc(xv, p; t = 0)
    hasproperty(p, :F) && !(p.F isa ZeroField) && throw(
        ArgumentError("Guiding-center equations do not support the external force `F`.")
    )
    T = float(eltype(xv))
    X, vpar, _, _, μ, _ = _get_gc_parameters(xv, p.E, p.B, p.q, p.m, t)
    state = SVector{4, T}(X[1], X[2], X[3], vpar)

    return (xv isa StaticArray ? state : Vector(state)), (; p..., μ = T(μ))
end

"""
    gc_to_full(state_gc, p; phase = 0, t = 0)

Full particle state `[x, y, z, vx, vy, vz]` of the guiding-center state `state_gc` at
gyrophase `phase` and time `t`, with `p` from [`prepare_gc`](@ref).
"""
function gc_to_full(state_gc, p; phase = 0, t = 0)
    return Vector(_gc_to_full(state_gc, p.E, p.B, p.q, p.m, p.μ, t, phase))
end

"""
    _gc_to_full(state_gc, E_field, B_field, q, m, μ, t, phase = 0)

Internal GC → full-orbit conversion taking the field functions and charges
directly, so it can be shared by the GC and hybrid solvers. `E_field` and
`B_field` must be callable as `E_field(x, t)` / `B_field(x, t)`; static fields
simply ignore `t`.

Returns an `SVector{6}` so that it can be used in a hot loop and stored in a
`Vector{SVector{6}}` without allocating; [`gc_to_full`](@ref) converts it to a
`Vector` for the public API.
"""
function _gc_to_full(state_gc, E_field, B_field, q, m, μ, t, phase = 0)
    R = get_x(state_gc)
    vpar = state_gc[4]

    E = E_field(R, t)
    B = B_field(R, t)
    Bmag = norm(B)
    b̂ = B / Bmag

    v_E = (E × b̂) / Bmag

    # perp speed, μ = m * w^2 / (2B)
    w = sqrt(2 * μ * Bmag / m)

    # perp vector
    e1, e2 = get_perp_vector(b̂)
    v_gyr = w * (cos(phase) * e1 + sin(phase) * e2)
    v_perp = v_gyr + v_E

    v = vpar * b̂ + v_perp

    # gyroradius
    Ω = q * Bmag / m
    ρ_vec = (b̂ × v_perp) / Ω

    x = R + ρ_vec

    return SVector{6}(x[1], x[2], x[3], v[1], v[2], v[3])
end

"""
    get_gc(xu, param, t; use_vE = false)
    get_gc(xu, param; use_vE = false)
    get_gc(x, y, z, vx, vy, vz, bx, by, bz, q2m)

Calculate the coordinates of the guiding center according to the phase space coordinates of a particle.
Reference: [wiki](https://en.wikipedia.org/wiki/Guiding_center)

Without `t`, the time is the last entry of a 7-element `xu = [x..., v..., t]`; a 6-element
`xu` is accepted only for static fields.

Nonrelativistic definition:

```math
\\mathbf{X}=\\mathbf{x}-m\\frac{\\mathbf{b}\\times\\mathbf{v}}{qB}
```

If `use_vE=true`, it calculates the guiding center relative to the E × B drift frame:

```math
\\mathbf{X}=\\mathbf{x}-m\\frac{\\mathbf{b}\\times(\\mathbf{v}-\\mathbf{v}_E)}{qB}
```

which is essential for correctly identifying the polarization drift in time-varying fields.
"""
function get_gc(xu, param; use_vE::Bool = false)
    length(xu) == 7 && return get_gc(xu, param, xu[end]; use_vE)
    (is_time_dependent(param.E) || is_time_dependent(param.B)) && throw(
        ArgumentError("Time-dependent fields require get_gc(xu, param, t).")
    )
    return get_gc(xu, param, zero(eltype(xu)); use_vE)
end

function get_gc(xu, param, t; use_vE::Bool = false)
    # TODO: support external forces
    q2m = param.q2m
    B_field = param.B
    v = xu[SA[4:6...]]
    B = B_field(xu, t)
    B² = B[1]^2 + B[2]^2 + B[3]^2

    if use_vE
        E_field = param.E
        E = E_field(xu, t)
        v = v - (E × B) / B²
    end

    # vector of Larmor radius
    ρ = B × v ./ (q2m * B²)

    return X = @views xu[1:3] - ρ
end

function get_gc(x, y, z, vx, vy, vz, bx, by, bz, q2m)
    l = SVector{3}(x, y, z)
    B = SVector{3}(bx, by, bz)
    v = SVector{3}(vx, vy, vz)

    B² = bx^2 + by^2 + bz^2
    # vector of Larmor radius
    ρ = B × v ./ (q2m * B²)

    return X = l - ρ
end
