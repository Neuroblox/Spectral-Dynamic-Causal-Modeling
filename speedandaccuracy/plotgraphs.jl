using Plots
using MAT

t_mat = Vector{Float64}()
t_julad = Vector{Float64}()
t_julspm = Vector{Float64}()
list = collect(2:8)
# append!(list,10)
for i = list
    tmp = matread("speedandaccuracy/n" * string(i) * ".mat");
    print(tmp)
    append!(t_mat, tmp["t_mat"])
    append!(t_julad, tmp["t_jad"])
    append!(t_julspm, tmp["t_jspm"])
end

plot(list, (t_mat./t_jul.-1).*100, legend=false,xlabel="number of regions",ylabel="percentage speed increase", linewidth=3)
scatter!(list, (t_mat./t_jul.-1).*100, markersize=6)
title!("speed comparison between Julia and Matlab")
