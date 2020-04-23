import Remez

pt = range(-1, 1, length=5)
f(x) = 3 - 2x^2 + x^3  - x^4
ddf = Remez.divided_difference(f, pt)

function interpolation_error(f::Function, ddf::AbstractVector{T},
                             pt::AbstractVector{T}, 
                             M::Integer) where T <: AbstractFloat
    n = length(pt)
    max_err = 0.0
    for x in range(pt[1], pt[n], length=M)
        err_at_x = abs(Remez.Newton_poly(ddf, pt, x) - f(x))
        max_err = max(max_err, err_at_x)
    end
    return max_err
end

max_err = interpolation_error(f, ddf, pt, 100) 

@test max_err < 10.0 * eps(1.0)
