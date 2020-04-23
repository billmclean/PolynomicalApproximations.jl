module Remez

using OffsetArrays
using Optim: optimize
using ..Interpolation: divided_difference!, Newton_poly

export minimax

"""
    ddp, pt, zmax, zmin = minimax(T, f, n, max_iterations)

Computes the minimax polynomial of degree `n` to `f(x)` over `[-1,1]` using
at most `max_iterations` iterations of the Remez algorithm.

The minimax polynomial is `p=NewtonPoly(ddp, pt)`.  The difference 
`zmax[k]-zmin[k]` should converge quadratically to zero, and the maximum
error `||f-p||_∞` should be `zmax[end]`.
"""
function minimax(::Type{T}, f::Function, n::Integer,
                max_iterations::Integer,
                clamp=:no) where T <: AbstractFloat
    pt = OffsetArray{T}(undef, 0:n+1)
    pt[0] = -one(T)
    for i = 1:n
        pt[i] = cospi((n+1-i)/(n+1))
    end
    pt[n+1] = one(T)
    ddf = OffsetArray{T}(undef, 0:n)
    ddp = OffsetArray{T}(undef, 0:n)
    E = equi_oscillation!(ddp, pt, ddf, f, clamp)
    new_pt = OffsetArray{T}(undef, 0:n+1)
    zmax = Vector{T}(undef, max_iterations)
    zmin = Vector{T}(undef, max_iterations)
    for k = 1:max_iterations
        pow = ( E > 0 ) ? -one(T) : one(T)
        if ( clamp == :no ) | ( clamp == :right )
            res = optimize(-one(T), pt[1]) do x
                z = f(x) - Newton_poly(ddp[0:n], pt[0:n], x)
                return pow * z
            end
            new_pt[0] = res.minimizer
            nextz = abs(f(new_pt[0])-Newton_poly(ddp[0:n], pt[0:n], new_pt[0]))
        else
            new_pt[0] = -one(T)
            nextz = zero(T)
        end
        zmax[k] = zmin[k] = nextz
        pow = -pow
        for i = 1:n
            res = optimize(pt[i-1], pt[i+1]) do x
                z = f(x) - Newton_poly(ddp[0:n], pt[0:n], x)
                return pow * z
            end
            new_pt[i] = res.minimizer
            nextz = abs(f(new_pt[i])-Newton_poly(ddp[0:n], pt[0:n], new_pt[i]))
            zmax[k] = max(zmax[k], nextz)
            zmin[k] = min(zmin[k], nextz)
            pow = -pow
        end
        if ( clamp == :no ) | ( clamp == :left )
            res = optimize(pt[n], one(T)) do x
                z = f(x) - Newton_poly(ddp[0:n], pt[0:n], x)
                return pow * z
            end
            new_pt[n+1] = res.minimizer
            nextz = abs(f(new_pt[n+1])
                        -Newton_poly(ddp[0:n], pt[0:n], new_pt[n+1]))
        else
            new_pt[n+1] = one(T)
            nextz = zero(T)
        end
        zmax[k] = max(zmax[k], nextz)
        zmin[k] = min(zmin[k], nextz)
        pt .= new_pt
        E = equi_oscillation!(ddp, pt, ddf, f, clamp)
        if zmax[k] - zmin[k] < 10*eps(T)
            zmax = zmax[1:k]
            zmin = zmin[1:k]
            break
        end
    end
    return ddp[0:n], pt[0:n], zmax, zmin
end

"""
    E = equi_oscillation!(ddp, pt, ddf, f, clamp)

Computes a polynomial `p` (in Newton form) and a number `E` such that 

    f(pt[i]) - p(pt[i]) = (-1)^i E    for 1 ≤ i ≤ n.

If `clamp=:no` then this condition holds also for `i=0` and `i=n+1`.
If `clamp=:both` then `f(pt[i]) - p(pt[i]) = 0` for `i=0` and `i=n+1`.
If `clamp=:left` then `f(pt[0]) - p(pt[0]) = 0` and the usual condition holds 
for `i=n+1`.
If `clamp=:right` then `f(pt[n+1]) - p(pt[n+1]) = 0` and the usual condition
holds for `i=0`.
""" 
function equi_oscillation!(ddp::OffsetVector{T}, pt::OffsetVector{T},
                           ddf::OffsetVector{T}, 
                           f::Function, clamp=:no) where T <: AbstractFloat
    n = length(pt) - 2
    for i = 0:n
        ddf[i] = f(pt[i])
    end
    divided_difference!(view(ddf, 0:n), pt[0:n])
    pow = one(T)
    for i = 0:n
        ddp[i] = pow
        pow = -pow
    end
    if ( clamp == :left ) | ( clamp == :both )
        ddp[0] = zero(T)
    end

    divided_difference!(view(ddp, 0:n), pt[0:n])
    p1 = Newton_poly(ddf[0:n], pt[0:n], pt[n+1])
    p2 = Newton_poly(ddp[0:n], pt[0:n], pt[n+1])
    if ( clamp == :no ) | ( clamp == :left )
        E = ( p1 - f(pt[n+1]) ) / ( p2 - pow )
    else
        E = ( p1 - f(pt[n+1]) ) / p2
    end
    for i = 0:n
        ddp[i] = ddf[i] - E * ddp[i]
    end
    return E
end

end # module
