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
    println("Entered line search")
    x_k = x_0
    a = ∇f(x_k)
    println("a   = ", a)
    count = 0
    while norm(∇f(x_k)) > ε
        count += 1
        # if count > 100000
        #     break
        # end
        p_k = -∇f(x_k)
        println("x_k = ", x_k)
        println("p_k = ", p_k)
        x_k += η * p_k
        println("p_k* = ", -∇f(x_k))
    end
    println("End line search")
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
function BFGS(f::Function,
    ∇f::Function, 
    x_0::Vector{<:Number}, 
    iters::Integer,
    line_search::Function,
    ε = 10e-12::Real, 
    ls_x_0 = 0::Number, 
    ls_η = 0.00000001::Number,
    ε_ls = 10e-12::Real)
    n = size(x_0)[1]
    x_k = x_0
    B_k = Matrix{typeof(x_0[1])}(I, n, n)

    println("initial gradient = ", ∇f(x_k))

    for i = 1:iters
        if norm(∇f(x_k)) < ε
            break

        end
        # Step 1: obtain direction p_k by solving B_k ∙ p_k = - (gradient of f at x_k)
        p_k = B_k \ -∇f(x_k)
        println("p_k = ", p_k)
        println("x_k = ", x_k)
        println("B_k = ", B_k)
        # Step 2.: Find stepsize α_k such that α_k = arg min f(x_k + α_k * p_k)
        # α->(f(x_k + α * p_k) - f(x_k + (α + 1e-01) * p_k)) / 1e-10 # numeric derivative
        # α->-5 * p_k[1] + 20 * x_k[1] * p_k[1] + 20 * α * x_k[1]  # exact derivative
        α_k = line_search(α->(f(x_k + (α + 1e-20) * p_k) - f(x_k + α * p_k)) / 1e-20,
            ls_x_0, ls_η, ε_ls)
        α_k = abs(α_k) # just trying around - this doesn't really belong here
        # Step 3.
        s_k = α_k * p_k
        x_k_prime = x_k + s_k
    
        # Step 4.
        y_k = ∇f(x_k_prime) - ∇f(x_k)
        println("y_k = ", y_k)
        println("s_k = ", s_k)
        # Step 5.
        B_k += (y_k * y_k') / (y_k' * s_k) - (B_k * s_k * s_k' * B_k) / (s_k' * B_k * s_k)
        x_k = x_k_prime
        println()

    end
    x_k
end

# f(x) = -5 * x[1] + 10 * x[1]^2
# ∇f(x) = [-5 + 20 * x[1]]
f(x) = x[1]^5 / 5000 + 21 * x[1]^4 / 4000 + 17 * x[1]^3 / 375 + 293 * x[1]^2 / 1000 + 521 * x[1] / 1000
∇f(x) = [x[1]^4 / 1000 + 21 * x[1]^3 / 1000 + 17 * x[1]^2 / 125 + 293 * x[1] / 500 + 521 / 1000]

println("optimal x = ", BFGS(f,
    ∇f,
    [-2],
    50000,
    line_search_gradient_descent,
    10e-12,
    1e-10,
    0.00005,
    1e-10))


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
