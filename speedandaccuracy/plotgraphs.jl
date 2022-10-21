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

plot(list, (t_mat./t_julad.-1).*100, legend=:topleft, xlabel="number of regions",ylabel="percentage speed increase", linewidth=3, label="AD")
scatter!(list, (t_mat./t_julad.-1).*100, markersize=6, label=nothing)
plot!(list, (t_mat./t_julspm.-1).*100, color="red", linewidth=3, label="SPM12")
scatter!(list, (t_mat./t_julspm.-1).*100, markersize=6, label=nothing)
title!("speed comparison between Julia and Matlab")
