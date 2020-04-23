using PolynomialApproximations: minimax, NewtonPoly, ChebyshevPoly
using PyPlot

T = Float64
n = 6
iterations = 8
#clamp = :left
#clamp = :right
clamp = :both
ddp, pt, zmax, zmin = minimax(T, exp, n, iterations, clamp)

display([zmin  zmax  (zmax-zmin)])

x = range(-one(T), one(T), length=401)
p = NewtonPoly(ddp, pt)

figure(1)
plot(x, p.(x) - exp.(x))
grid(true)
title("Error in the minimax polynomial of degree $n")

figure(2)
p2 = ChebyshevPoly(exp, n)
plot(x, p2.(x) - exp.(x))
grid(true)
title("Error in the Chebyshev polynomial of degree $n")
