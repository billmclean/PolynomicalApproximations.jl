module PolynomialApproximations

include("parts/Interpolation.jl")
include("parts/Remez.jl")
include("parts/Chebyshev.jl")

using .Interpolation
using .Remez
using .Chebyshev
export NewtonPoly, minimax, ChebyshevPoly


end # module
