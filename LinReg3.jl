import Random
Random.seed!(0)

using Plots
using LinearAlgebra

# clear all files
open("norm_gradient_w.txt", "w") do io
end

open("error.txt", "w") do io
end

open("learning_rate.txt", "w") do io
end

"Sum from k=`from` to `to` of `a(k)`"
Σ(from::Integer, to::Integer, a::Function, zero = 0) = mapreduce(a, (+), from:to; init = zero)

"""Linear Regression
# Args:
    𝐰: Parameters
    Φ(j, 𝐱): Basis function of type (Int, Vector{T}) -> T
    𝐱: Input vector
"""
function y(𝐰::Vector{<:Number},
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
function ∂E_D∂w_k(Φ::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  𝐰::Vector{<:Number},
  k::Integer)::Number
    N = size(t)[1]
    - Σ(1, N, n->Φ(k, 𝐗[n, :]) * (t[n] - y(𝐰, Φ, 𝐗[n,:])))
end

"""Error function
# Args:
    Φ(k, 𝐱ₙ): Basis function
    𝐗: Set of inputs 𝐱ₙ where 𝐱ₙ is an input vector to Φ
    t: corresponding target values for each 𝐱ₙ
    𝐰: Parameters
"""
function E_D(Φ::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  𝐰::Vector{<:Number})::Number
    N = size(t)[1]
    1 // 2 * Σ(1, N, n->(t[n] - y(𝐰, Φ, 𝐗[n,:]))^2)
end

"""One iteration of the gradient descent algorithm
# Args:
    ∂E_D∂w_k: Partial derivative of error function with respect to
        the k-th parameter
    𝐗: Column vector of inputs 𝐱ₙ where 𝐱ₙ is an input vector to the
        error function
    t: corresponding target values for each 𝐱ₙ
    𝐰: Parameters
    η: Learning rate
    ∇𝐰_prior: Gradient of parameters from prior iteration
    γ: Momentum factor
"""
function gradient_descent_iteration(∂E_D∂w_k::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  𝐰::Vector{<:Number},
  η::Number,
  ∇𝐰_prior::Vector{<:Number},
  γ::Number)::Tuple{Vector{<:Number},Vector{<:Number}}
    M = size(𝐰)[1]
    ∇𝐰 = γ * ∇𝐰_prior
    for j = 1:M
        ∂E_D∂w_jk(k) = ∂E_D∂w_k(𝐗, t, 𝐰, k)
        ∇𝐰 += collect(map(∂E_D∂w_jk, 1:M))

        open("norm_gradient_w.txt", "a") do io
            write(io, string(norm(∇𝐰)), "\n")
        end
    end
    (𝐰 - η * ∇𝐰, ∇𝐰)
end

"""Minimize function E_D(𝐗, t, 𝐰)
# Args:
    ∂E_D∂w_k: Partial derivative of error function with respect to
        the k-th parameter
    𝐗: Column vector of inputs 𝐱ₙ where 𝐱ₙ is an input vector to the
        error function
    t: corresponding target values for each 𝐱ₙ
    𝐰: Initial parameters - usually `randn(M)`
    η: Learning rate
    M: Number of model parameters
    iters: Number of iterations
    ε: Gradient descent stops once the difference between two iterations
        (𝐰 and 𝐰') is less than ε
    γ: Momentum factor

# Fails:
    Fails on encountering NaN in computation or on Divergence to Inf
"""
function gradient_descent(∂E_D∂w_k::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  η::Number,
  M::Integer,
  iters::Integer,
  𝐰::Vector{<:Number},
  ε = 10e-12::Number,
  γ = 0.9::Real)::Tuple{Vector{<:Number},Integer}
    ∇𝐰 = zero(𝐰)
    did_iters = 0
    for i = 1:iters
        did_iters += 1
        𝐰_old = 𝐰
        open("error.txt", "a") do io
            write(io, string(E_D(Φ, 𝐗, t, 𝐰)), "\n")
        end
        open("learning_rate.txt", "a") do io
            write(io, string(η), "\n")
        end
        (𝐰, ∇𝐰) = gradient_descent_iteration(∂E_D∂w_k, 𝐗, t, 𝐰, η, ∇𝐰, γ)
         if any(isnan.(𝐰))
           error("Encountered NaN") 
        end
        if any(isinf.(𝐰))
            error("Divergence in calculation")
        end
        if norm(𝐰_old - 𝐰) < ε
            break
        end
    end
    (𝐰, did_iters)
end

"""Find regression model 
# Args:
    Φ: Basis Function
    𝐗: Set of inputs 𝐱ₙ where 𝐱ₙ is an input vector to Φ
    t: corresponding target values for each 𝐱ₙ
    η: learning rate with which to train
    M: Number of model parameters
    iters: Number of iterations
    ε: Gradient descent stops once the difference between
        two iterations (𝐰 and 𝐰') is less than ε
    γ: Momentum Parameter
    optimizer: Parameter to select optimizer that's used

# Fails:
    On unknown optimizers or error inside the optimizer
"""
function fit_linear_model(Φ::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  η::Number,
  M::Integer,
  iters::Integer,
  ε = 10e-12::Number,
  γ = 0.9::Real,
  optimizer = :gradient_descent)::Tuple{Function,Number}
    if optimizer == :gradient_descent
        (𝐰, did_iters) = gradient_descent((𝐗, t, 𝐰, k)->∂E_D∂w_k(Φ, 𝐗, t, 𝐰, k),
            𝐗, t, η, M, iters, randn(M), ε, γ)
        residual_error = E_D(Φ, 𝐗, t, 𝐰)
        println(𝐰, " after ", did_iters, " iterations. Residual error: ", residual_error)
        (𝐱->y(𝐰, Φ, 𝐱), residual_error)
    else
        error("Invalid optimizer")
    end
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
Φ3(j, 𝐱) = sin(1 / j * 𝐱[1])
σ(a) = 1 / (1 + exp(-a))
function Φ4(j, 𝐱)
    μ = 0.2
    s = 0.2
    σ((𝐱[1] - μ) / s)
end

# test3
𝐗 = hcat([0; 1; 2; 3; 4; 5]) # hcat to convert to matrix because julia is weird like that
t = [0, 1, 4, 9, 16, 25]
t += randn(size(t)[1]) * 3


Φ = Φ2

(model1, residual_error) = fit_linear_model(Φ2, 𝐗, t, 0.00005, 2, 300000, 10e-3, 0.5) # Φ2, 𝐗, t, 0.00005, 2, 300000, 10e-5, 0.2)


x = 0:0.1:5

p = scatter(map(x->x[1], 𝐗), t, label = "training");
plot!(x, model1.([[x] for x in x]), label = "prediction")
display(p)
readline()
