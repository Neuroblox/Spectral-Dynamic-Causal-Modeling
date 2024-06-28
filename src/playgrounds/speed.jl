"""
Playground to perform general speed comparisons.
"""

N = 100

function test1()
    for i = 1:100
        for j = 1:10
            A = zeros(Float64, N, N, 10)
            A[:,:,j] = rand(N, N)
        end
    end
end

function test2()
    A = zeros(Int, N, N, 10)
    for i = 1:100
        A = Float64.(A)
        for j = 1:10
            A[:,:,j] = rand(N, N)
        end
    end
end

function test3()
    for i = 1:100
        A = zeros(Float64, N, N, 10)
        for j = 1:10
            A[:,:,j] = rand(N, N)
        end
    end
end

@benchmark test1()
@benchmark test2()
@benchmark test3()



#=

    Compare matrix free vs matrix based solutions for computing the Jacobian

=#

function f(y,x) # in-place
  global fcalls += 1
  for i in 2:length(x)-1
    y[i] = x[i-1] - 2x[i] + x[i+1]
  end
  y[1] = -2x[1] + x[2]
  y[end] = x[end-1] - 2x[end]
  nothing
end

using SparseDiffTools
using ForwardDiff

fcalls = 0
function g(x) # out-of-place
    global fcalls += 1
    y = zero(x)
    for i in 2:length(x)-1
      y[i] = x[i-1] - 2x[i] + x[i+1]
    end
    y[1] = -2x[1] + x[2]
    y[end] = x[end-1] - 2x[end]
    y
end


using Plots

"""
Even though, theoretically, a VJP (Vector-Jacobian product - reverse autodiff) and a JVP (Jacobian-Vector product - forward-mode autodiff) 
are similar—they compute a product of a Jacobian and a vector—they differ by the computational complexity of the operation. In short, 
when you have a large number of parameters (hence a wide matrix), a JVP is less efficient computationally than a VJP, 
and, conversely, a JVP is more efficient when the Jacobian matrix is a tall matrix.

from: https://lux.csail.mit.edu/dev/tutorials/beginner/1_Basics
"""
nvarset = [10, 50, 100, 250, 500]
ncolset = [10, 50, 100, 250, 500]
nvarset = vcat(2,10:10:50)
ncolset = vcat(2, 10:10:50)

speed_Jfree = zeros(length(nvarset)*length(ncolset))
speed_Jbased = zeros(length(nvarset)*length(ncolset))
for (i, (nvars, ncols)) in enumerate(Iterators.product(nvarset, ncolset))
    x = rand(nvars)
    J = JacVec(g, x)
    V = rand(nvars, ncols)
    tmp = @benchmark stack(J*c for c in eachcol(V))
    speed_Jfree[i] = mean(tmp.times)*1e-6
    tmp = @benchmark ForwardDiff.jacobian(g, x)*V
    speed_Jbased[i] = mean(tmp.times)*1e-6
end
p1 = contourf(nvarset, ncolset, speed_Jbased, title="Matrix-based", color=:turbo, clim=(0.0,0.04));
p2 = contourf(nvarset, ncolset, speed_Jfree, xlabel="number of variables", title="Matrix-free", color=:turbo, clim=(0.0,0.04));
plot(p1, p2, layout=(2, 1), ylabel="number of columns")


res = similar(v)
v = foo[][3][:, 1]
J = JacVec(foo[][1], foo[][2])
mul!(res, J, v)
J* foo[][3]
dfdp = zeros(Complex, 288, 21)
for i = 1:size(foo[][3], 2)
    dfdp[:, i] = J*foo[][3][:, i]
end
