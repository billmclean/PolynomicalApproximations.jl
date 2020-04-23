module Chebyshev

using FFTW: dct!
using OffsetArrays
using ArgCheck
export ChebyshevPoly

struct ChebyshevPoly{T<:AbstractFloat}
    a::OffsetVector{T}
end

function ChebyshevPoly(f::Function, n::Integer)
    a = OffsetArray{Float64}(undef, 0:n-1)
    chebyshev_coefs_dct!(a, f)
    display(a)
    return ChebyshevPoly(a)
end

function (p::ChebyshevPoly{T})(x::T) where T <: AbstractFloat
    return chebyshev_sum(p.a, x)
end

function chebyshev_polys!(Cheb::OffsetArray{T}, x::T
                         ) where T <: AbstractFloat
    nmax = length(Cheb) - 1
    Cheb[0] = one(T)
    if nmax ≥ 1
        Cheb[1] = x
    end
    if nmax ≥ 2
        for n = 1:nmax-1
            Cheb[n+1] = 2x * Cheb[n] - Cheb[n-1]
        end
    end
end

function chebyshev_polys!(Cheb::OffsetArray{T}, x::AbstractVector{T}
                         ) where T <: AbstractFloat
    nmax, M = size(Cheb)
    nmax -= 1
    @argcheck Cheb.offsets == (-1, 0)
    @argcheck length(x) == M
    for m = 1:M
        Cheb[0,m] = one(T)
    end
    if nmax ≥ 1
        for m = 1:M
            Cheb[1,m] = x[m]
        end
    end
    if nmax ≥ 2
        for m = 1:M, n = 1:nmax-1
            Cheb[n+1,m] = 2x[m] * Cheb[n,m] - Cheb[n-1,m]
        end
    end
end

function chebyshev_coefs(::Type{T}, f::Function, nmax::Integer,
                         M::Integer) where T <: AbstractFloat
    a = OffsetArray{T}(undef, 0:nmax)
    chebyshev_coefs!(a, f, M)
    return a
end

function chebyshev_coefs!(a::OffsetArray{T}, f::Function, M::Integer
                         ) where T <: AbstractFloat
    nmax = length(a) - 1
    @argcheck a.offsets == (-1,)
    x = Array{T}(undef, M)
    for i = 1:M
        tim1 = convert(T, 2i-1)
        x[i] = cospi(tim1/(2M))
    end
    Cheb = OffsetArray{T}(undef, 0:nmax, 1:M)
    chebyshev_polys!(Cheb, x)
    f_at_x = x
    for i = 1:M
        xi = f_at_x[i]
        f_at_x[i] = f(xi)
    end
    for n = 0:nmax
        s = zero(T)
        for m = 1:M
            s += f_at_x[m] * Cheb[n,m]
        end
        a[n] = (2s) / M
    end
end

"""
    chebyshev_coefs_dct!(a, f)

Overwrite `a[0]`, `a[1]`, ..., `a[n-1]` with the Fourier-Chebyshev
coefficients of `f` using the `n`-point Gauss-Chebyshev quadrature rule.
"""
function chebyshev_coefs_dct!(a::OffsetArray{Float64}, f::Function) 
    n = length(a)
    @argcheck a.offsets == (-1,)
    for j = 0:n-1
        xj = cospi((2j+1)/(2n))
        a[j] = f(xj) / n
        println("x$j = ", xj, ", a$j = ", a[j])
    end
    dct!(a.parent)
    a[0] *= 2 * sqrt(n)
    for j = 1:n-1
        a[j] *= sqrt(2n)
    end
end

"""
    chebyshev_sum(a, x)

Evaluate `a[0]/2 + a[1]T_1(x) + ⋯ + a[n-1]T_(n-1)(x)` using Clenshaw's 
algorithm.
"""
function chebyshev_sum(a::OffsetArray{T}, x::T) where T <: AbstractFloat
    n = length(a)
    bkp1 = bkp2 = zero(T)
    for k = n-1:-1:1
        bk = a[k] + 2x*bkp1 - bkp2
        bkp2 = bkp1
        bkp1 = bk
    end
    bk = a[0] + 2x*bkp1 - bkp2
    return ( bk - bkp2 ) / 2
end

end # module
