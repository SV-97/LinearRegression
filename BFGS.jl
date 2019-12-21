using LinearAlgebra

"""Perform a line search on some function f
# Args:
    ∇f: Gradient-function of f
    x_0: Initial guess
    η: learning rate
    ε: cancellation tolerance
"""
function line_search_gradient_descent(∇f::Function,
  x_0,
  η::Number,
  ε::Number)
    # println("Bims im line search")
    x_k = x_0
    a = ∇f(x_k)
    # println(a)
    while norm(∇f(x_k)) > ε
        # println("Calling with x_k = ", x_k)
        p_k = -∇f(x_k)
        # println("jup")
        x_k += η * p_k
    end
    x_k
end

"""BFGS Optimization Algorithm
Broyden–Fletcher–Goldfarb–Shanno algorithm
for details see:
    https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm

# Args:
    ∇f: Gradient of function to optimize
    x_0: Initial guess for optimal value
    iters: Maximum number of iterations before cancellation
    line_search: Line search function that's used
    ε: Optimization stops once the norm of the gradient is below this value
    ls_x_0: Initial value for the line search
    ls_η: Learning rate for the line search
    ε_ls: same as ε but for the line search
"""
function BFGS(∇f::Function, 
    x_0::Vector{<:Number}, 
    iters::Integer,
    line_search::Function,
    ε = 10e-12::Real, 
    ls_x_0 = 0::Number, 
    ls_η = 0.000001::Number,
    ε_ls = 10e-12::Real)
    n = size(x_0)[1]
    x_k = x_0
    B_k = Matrix{typeof(x_0[1])}(I, n, n)

    for i = 1:iters
        if norm(∇f(x_k)) < ε
        break
        end
        # Step 1: obtain direction p_k by solving B_k ∙ p_k = - (gradient of f at x_k)
        p_k = B_k \ -∇f(x_k)
        # println("p_k = ", p_k)
        # println("x_k = ", x_k)
        # Step 2.: Find stepsize α_k such that α_k = arg min f(x_k + α_k * p_k)
        g = α->∇f(x_k + α * p_k)
        α_k = line_search(g, ls_x_0, ls_η, ε_ls)
    
        # Step 3.
        s_k = α_k * p_k
        x_k_prime = x_k + s_k
    
        # Step 4.
        y_k = ∇f(x_k_prime) - ∇f(x_k)
    
        # Step 5.
        B_k += (y_k * y_k') / (y_k' * s_k) - (B_k * s_k * s_k' * B_k) / (s_k' * B_k * s_k)
        x_k = x_k_prime
    end
    x_k
end

∇f(x) = [-5 + 20 * x[1]]
println(BFGS(∇f,
    [-5],
    20000,
    line_search_gradient_descent,
    1e-12,
    0, ))


# f(w, x) = w[1] * x[1] + w[2] * x[1]^2

# ∇f(w, x) = [w[1] + 2 * x[1] * w[2]] # [x; x^2]

# # initialize parameters
# w_0 = [-5, 10]
# x_0 = [-5]
# B_0 = Matrix{Float64}(I, 1, 1) # initialize approximate hessian with identity matrix
# ε = 1e-10 # cancellation parameter

# w_k = w_0
# x_k = x_0
# B_k = B_0

# while begin
#     x = norm(∇f(w_k, x_k))
#     println("∇f = ", x)
#     x > ε
# end
#     global B_k
#     global x_k
#     # Step 1.: obtain direction p_k by solving B_k*p_k = - (gradient of f at x_k)

#     p_k = B_k \ -∇f(w_k, x_k)

#     # Step 2.: Find stepsize α_k such that α_k minimizes f(x_k + α_k * p_k)

#     g_k(α) = w_k[1] + 2 * (x_k[1] + α * p_k[1]) * w_k[2]
#     α_k = line_search_gradient_descent(g_k, 0, 0.000001, 1e-10)
#     println("α_k = ", α_k)

#     # Step 3.
#     s_k = α_k * p_k
#     x_k_prime = x_k + s_k

#     # Step 4.
#     y_k = ∇f(w_k, x_k_prime) - ∇f(w_k, x_k)

#     # Step 5.
#     B_k += (y_k * y_k') / (y_k' * s_k) - (B_k * s_k * s_k' * B_k) / (s_k' * B_k * s_k)
#     x_k = x_k_prime
# end

# println(x_k)

"""
begin
    inv_B = inv(B_k)
    (s_k' * y_k + y_k' * inv_B * y_k) * (s_k * s_k') / (s_k' * y_k)^2 - (inv_B * y_k * s_k' + s_k * y_k' * inv_B) / (s_k' * y_k)
end
"""
