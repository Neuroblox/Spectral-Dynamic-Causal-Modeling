using Plots
using MAT

t_mat = Vector{Float64}()
t_jul = Vector{Float64}()
list = collect(2:8)
# append!(list,10)
for i = list
    tmp = matread("data_speedtest/n" * string(i) * ".mat");
    print(tmp)
    append!(t_mat, tmp["t_matlab"])
    append!(t_jul, tmp["t_julia"])
end

plot(list, (t_mat./t_jul.-1).*100, legend=false,xlabel="number of regions",ylabel="percentage speed increase", linewidth=3)
scatter!(list, (t_mat./t_jul.-1).*100, markersize=6)
title!("speed comparison between Julia and Matlab")
