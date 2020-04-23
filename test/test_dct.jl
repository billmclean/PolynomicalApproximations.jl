using FFTW
using OffsetArrays
using PyPlot

function diy_dct(X::OffsetVector{Float64})
    Y = similar(X)
    n = length(X)
    Y[0] = sqrt(1/n) * sum(X)
    for k = 1:n-1
        s = 0.0
        for j = 0:n-1
            s += X[j] * cospi(k*(j+0.5)/n)
        end
        Y[k] = sqrt(2/n)*s
    end
    return Y
end

n = 50
X = OffsetVector(zeros(n), 0:n-1)
X[2] = 1.0
Y1 = diy_dct(X)
Y2 = OffsetVector(dct(X.parent), 0:n-1)

figure(1)
θ = Float64[ (j+1/2)/n for j = 0:n-1 ]
plot(θ, Y1[0:n-1], θ, Y2[0:n-1])

