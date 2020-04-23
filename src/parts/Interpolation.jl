module Interpolation

export NewtonPoly

struct NewtonPoly{T<:AbstractFloat}
    ddp::Vector{T}
    pt::Vector{T}
end

function NewtonPoly(f::Function, pt::AbstractVector{T}) where T <: AbstractFloat
    ddp = divided_difference(f, pt)
    return NewtonPoly(ddp, pt)
end

function (p::NewtonPoly{T})(x::T) where T <: AbstractFloat
    return Newton_poly(p.ddp, p.pt, x)
end

"""
    divided_difference(f, pt)

Return a vector `ddf` of principal divided differences.

The entry `ddf[k]` is the divided difference `f(x)` with respect to the
sequence of points `pt[1]`, `pt[2]`, ..., `pt[k]`.
"""
function divided_difference(f::Function,
                            pt::AbstractVector{T}) where T <: AbstractFloat
    ddf = f.(pt)
    divided_difference!(ddf, pt)
    return ddf
end

"""
    divided_difference!(ddf, pt)

Overwrite `ddf` with the principal divided differences.

On entry, `ddf[k]` holds `f(pt[k])`.  On exit, `ddf[k]` holds the divided
difference with respect to `pt[1]`, `pt[2]`, ..., `pt[k]`.
"""
function divided_difference!(ddf::AbstractVector{T},
                             pt::AbstractVector{T}) where T <: AbstractFloat
    n = length(pt)
    if length(ddf) != n
        raise(ArgumentError("Lengths of the arrays ddf and pt do not match"))
    end
    for k = 2:n
        for j = n:-1:k
            ddf[j] = ( ddf[j] - ddf[j-1] ) / ( pt[j] - pt[j-k+1] )
        end
    end
end

"""
    Newton_poly(ddp, pt, x)

Evaluate the Newton interpolating polynomial at `x`.

The array `ddp` holds the principal divided differences with respect to the
points stored in the array `pt`.
"""
function Newton_poly(ddf::AbstractVector{T},
                     pt::AbstractVector{T}, x::T) where T <: AbstractFloat
    n = length(ddf)
    s = ddf[n]
    for k = n-1:-1:1
        s = ddf[k] + ( x - pt[k] ) * s
    end
    return s
end

end # module
