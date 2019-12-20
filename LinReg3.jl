import Random
Random.seed!(0)

using Plots
using LinearAlgebra

# clear all files
open("norm_gradient_w.txt", "w") do io
end

open("error.txt", "w") do io
end

open("learnrate.txt", "w") do io
end

"Sum from k=`from` to `to` of `a(k)`"
Σ(from::Integer, to::Integer, a::Function, zero = 0) = mapreduce(a, (+), from:to; init = zero)

"""Linear Regression
# Args:
    𝐰: Parameters
    Φ(j, 𝐱): Basis function of type (Int, Vector{T}) -> T
    𝐱: Input vector
"""
function y(
  𝐰::Vector{<:Number},
  Φ::(T where T <: Function),
  𝐱::Vector{<:Number})::Number
    Σ(1, size(𝐰)[1], j->𝐰[j] * Φ(j, 𝐱))
end

"""Derivative of E_D with respect to 𝐰ₖ
# Args:
    Φ(k, 𝐱ₙ): Basis function
    𝐗: Set of inputs 𝐱ₙ where 𝐱ₙ is an input vector to Φ
    t: corresponding target values for each 𝐱ₙ
    k: Index for 𝐰ₖ in respect to which the derivative is taken
    𝐰: Parameters
"""
function ∂E_D∂w_k(
  Φ::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  𝐰::Vector{<:Number},
  k::Integer)::Number
    N = size(t)[1]
    - Σ(1, N, n->Φ(k, 𝐗[n]) * (t[n] - y(𝐰, Φ, 𝐗[n,:])))
end

"""Error function
# Args:
    Φ(k, 𝐱ₙ): Basis function
    𝐗: Set of inputs 𝐱ₙ where 𝐱ₙ is an input vector to Φ
    t: corresponding target values for each 𝐱ₙ
    𝐰: Parameters
"""
function E_D(
  Φ::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  𝐰::Vector{<:Number})::Number
    N = size(t)[1]
    1 // 2 * Σ(1, N, n-> (t[n] - y(𝐰, Φ, 𝐗[n,:]))^2)
end

"""Gradient descent iteration
# Args:
    Φ: Basis Function
    𝐗: Set of inputs 𝐱ₙ where 𝐱ₙ is an input vector to Φ
    t: corresponding target values for each 𝐱ₙ
    𝐰: Parameters
    η: Learning rate
    ∇𝐰_prior: Gradient of parameters from prior iteration
    γ: Momentum factor
"""
function gd_iteration(
  Φ::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  𝐰::Vector{<:Number},
  η::Number,
  ∇𝐰_prior::Vector{<:Number},
  γ::Number)::Tuple{Vector{<:Number}, Vector{<:Number}}
    M = size(𝐰)[1]
    ∇𝐰 = γ * ∇𝐰_prior
    for j = 1:M
        ∂E_D∂w_jk(k) = ∂E_D∂w_k(Φ, 𝐗, t, 𝐰, k)
        ∇𝐰 += collect(map(∂E_D∂w_jk, 1:M))
    end
    # println("||∇𝐰|| = ", norm(∇𝐰))
    open("norm_gradient_w.txt", "a") do io
        write(io, string(norm(∇𝐰)), "\n")
    end
    (𝐰 - η * ∇𝐰, ∇𝐰)
end

"""Find regression model using gradient descent
# Args:
    Φ: Basis Function
    𝐗: Set of inputs 𝐱ₙ where 𝐱ₙ is an input vector to Φ
    t: corresponding target values for each 𝐱ₙ
    η: learning rate with which to train
    M: Number of model parameters
    iters: Number of iterations
    ε: Gradient descent stops once the difference between two iterations (𝐰 and 𝐰') is less than ε
    γ: Momentum Parameter
"""
function gd(
  Φ::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  η::Number,
  M::Integer,
  iters::Integer,
  ε = 10e-12::Number,
  γ = 0.9::Real)::Function
    𝐰 = randn(M)
    ∇𝐰 = zero(𝐰)
    did_iters = 0
    for i = 1:iters
        did_iters += 1
        𝐰_old = 𝐰
        (𝐰, ∇𝐰) = gd_iteration(Φ, 𝐗, t, 𝐰, η, ∇𝐰, γ)
        model(𝐱ₙ) = y(𝐰, Φ, 𝐱ₙ)

        open("error.txt", "a") do ioE
            write(ioE, string(E_D(Φ, 𝐗, t, 𝐰)), "\n")
        end
        open("learnrate.txt", "a") do ioη
            write(ioη, string(η), "\n")
        end
        
        if norm(𝐰_old - 𝐰) < ε || any(isnan.(𝐰)) || any(isinf.(𝐰))
            break
        end
    end
    println(𝐰, " after ", did_iters, " iterations. Residual error: ", E_D(Φ, 𝐗, t, 𝐰))
    𝐱->y(𝐰, Φ, 𝐱)
end

# optimal solution using a linear model for comparison

x_b = [1, 2, 3]
y_b = [1, 1, 2]

function fitline(x::Vector{<:Number}, y::Vector{<:Number})::Function
    n = size(x)[1]
    a = (sum(y) * sum(x.^2) - sum(x) * sum(x .* y)) / (n * sum(x.^2) - sum(x)^2) # t
    b = (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x.^2) - sum(x)^2)# m
    x->b * x + a
end

optimal_linear_model = fitline(x_b, y_b)

# end of optimal solution

# a few different basis functions for testing
Φ1(j, 𝐱) =
    if j == 0
        1
    else
        𝐱[1]
end

Φ2(j, 𝐱) = 𝐱[1]^j
Φ3(j, 𝐱) = sin(1/j*𝐱[1])
σ(a) = 1 / (1 + exp(-a))
function Φ4(j, 𝐱)
    μ = 0.2
    s = 0.2
    σ((𝐱[1] - μ) / s)
end

"""
# test1

𝐗 = [[1], [2], [3]]
t = [1, 1, 2]

model1 = gd(Φ1, 𝐗, t, 0.001, 2, 20000, 10e-12)

x = 1:0.1:5

p = scatter(map(x->x[1], 𝐗), t, label = "training");
plot!(x, model1.(map(x->[x], x)), label = "prediction")
plot!(x, optimal_linear_model.(x), label = "optimal", line = :dot)
display(p)
readline()
"""

"""
# test2

𝐗 = [[0], [1], [2], [3], [4], [5]]
t = [0, 1, 4, 9, 16, 25]
t += randn(size(t)[1]) * 3

model1 = gd(Φ3, 𝐗, t, 0.00001, 5, 200000, 10e-12)

x = 0:0.1:5

p = scatter(map(x->x[1], 𝐗), t, label = "training");
plot!(x, model1.(map(x->[x], x)), label = "prediction")
display(p)
readline()
"""

# test3
𝐗 = hcat([0; 1; 2; 3; 4; 5]) # hcat to convert to matrix because julia is weird like that
t = [0, 1, 4, 9, 16, 25]
t += randn(size(t)[1]) * 3

model1 = gd(Φ2, 𝐗, t, 0.00005, 2, 300000, 10e-12, 0.9)

x = 0:0.1:5

p = scatter(map(x->x[1], 𝐗), t, label = "training");
plot!(x, model1.(map(x->[x], x)), label = "prediction")
display(p)
readline()
