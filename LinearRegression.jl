module LinearRegression

import Random

using LinearAlgebra

export fit_linear_model

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
    𝐰: Parameters
    k: Index for 𝐰ₖ in respect to which the derivative is taken
    ω: Weight decay factor
"""
function ∂E_D∂w_k(Φ::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  𝐰::Vector{<:Number},
  k::Integer,
  ω::Real)::Number
    N = size(t)[1]
    - Σ(1, N, n->Φ(k, 𝐗[n, :]) * (t[n] - y(𝐰, Φ, 𝐗[n,:])) - ω*𝐰[k])
end

"""Error function
# Args:
    Φ(k, 𝐱ₙ): Basis function
    𝐗: Set of inputs 𝐱ₙ where 𝐱ₙ is an input vector to Φ
    t: corresponding target values for each 𝐱ₙ
    𝐰: Parameters
    ω: Weight decay factor
"""
function E_D(Φ::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  𝐰::Vector{<:Number},
  ω::Real)::Number
    N = size(t)[1]
    1 // 2 * Σ(1, N, n->(t[n] - y(𝐰, Φ, 𝐗[n,:]))^2 + ω*norm(𝐰)^2)
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
    ω: Weight decay factor
"""
function gradient_descent_iteration(∂E_D∂w_k::Function,
  𝐗::Matrix{<:Number},
  t::Vector{<:Number},
  𝐰::Vector{<:Number},
  η::Number,
  ∇𝐰_prior::Vector{<:Number},
  γ::Number,
  ω::Real)::Tuple{Vector{<:Number},Vector{<:Number}}
    M = size(𝐰)[1]
    ∇𝐰 = γ * ∇𝐰_prior
    for j = 1:M
        ∂E_D∂w_jk(k) = ∂E_D∂w_k(𝐗, t, 𝐰, k, ω)
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
    ω: Weight decay factor

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
  γ = 0.9::Real,
  ω = 0.01::Real)::Tuple{Vector{<:Number},Integer}
    ∇𝐰 = zero(𝐰)
    did_iters = 0
    for i = 1:iters
        did_iters += 1
        𝐰_old = 𝐰
        open("error.txt", "a") do io
            write(io, string(E_D(Φ, 𝐗, t, 𝐰, ω)), "\n")
        end
        open("learning_rate.txt", "a") do io
            write(io, string(η), "\n")
        end
        (𝐰, ∇𝐰) = gradient_descent_iteration(∂E_D∂w_k, 𝐗, t, 𝐰, η, ∇𝐰, γ, ω)
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
    ω: Weight decay factor
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
  ω = 0.01::Real,
  optimizer = :gradient_descent)::Tuple{Function,Number}
    if optimizer == :gradient_descent
        (𝐰, did_iters) = gradient_descent((𝐗, t, 𝐰, k, ω)->∂E_D∂w_k(Φ, 𝐗, t, 𝐰, k, ω),
            𝐗, t, η, M, iters, randn(M), ε, γ, ω)
        residual_error = E_D(Φ, 𝐗, t, 𝐰, ω)
        println(𝐰, " after ", did_iters, " iterations. Residual error: ", residual_error)
        (𝐱->y(𝐰, Φ, 𝐱), residual_error)
    else
        error("Invalid optimizer")
    end
end

Φ = (j, 𝐱) -> 𝐱[1]^j

end