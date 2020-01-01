# run `export JULIA_LOAD_PATH=.:` to enable loading of LinearRegression module

# using Plots
using LinearRegression

import Random

Random.seed!(0)

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

for i = 0:100
    (model1, residual_error) = @time fit_linear_model(Φ2, 𝐗, t, 0.00005, 2, 300000, 10e-3, 0.9, 0) # Φ2, 𝐗, t, 0.00005, 2, 300000, 10e-5, 0.2)
end
x = 0:0.1:5

# p = scatter(map(x->x[1], 𝐗), t, label = "training");
# plot!(x, model1.([[x] for x in x]), label = "prediction")
# display(p)
# readline()


"""
# test4
using StatsBase

f(x) = sin(3*x + 10)
x = -1:0.1:2
y = f.(x)
𝐗 = hcat([x for x in sample(x, 20)]) # hcat to convert to matrix because julia is weird like that
t = f.(𝐗)[:]
t += randn(size(t)[1]) * 0.3

(model1, residual_error) = fit_linear_model(Φ2, 𝐗, t, 0.00008, 5, 200000, 10e-8, 0.95, 0.008)

x = -1:0.1:2

p = scatter(map(x->x[1], 𝐗), t, label = "training");
plot!(x, model1.([[x] for x in x]), label = "prediction")
plot!(x, y, label = "real")
display(p)
readline()
"""
