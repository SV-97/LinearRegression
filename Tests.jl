
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
