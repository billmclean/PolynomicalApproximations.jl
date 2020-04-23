PolynomialApproximations.jl
===========================

Julia package to compute simple polynomial approximations to univariate 
functions.

Interpolating Polynomial
------------------------

Given a vector `pt` of distinct points on the real line, and a function `f`,

    p = NewtonPoly(f, pt)

generates a `NewtonPoly` object that can be evaluated using function call
syntax.  For example

    f(x) = exp(x)
    pt = range(-1, 1, length=5)
    p = NewtonPoly(f, pt)
    p(0.3)
    x = range(-1, 1, length=201)
    y = p.(x)

Minimax Polynomial
------------------

