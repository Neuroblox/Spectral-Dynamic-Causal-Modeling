using Plots
using MAT
using Serialization
using LaTeXStrings

t_mat = Vector{Float64}()
t_julad = Vector{Float64}()
t_julspm = Vector{Float64}()
iter_spm = Vector{Float64}()
iter_ad = Vector{Float64}()
F_spm = Vector{Float64}()
F_mat = Vector{Float64}()
F_ad =  Vector{Float64}()
list = collect(2:10)
# append!(list,10)
for i = list
    tmp = matread("speedandaccuracy/PMn" * string(i) * ".mat");
    # append!(t_mat, tmp["t_mat"]/tmp["iter_spm"])
    # append!(t_julad, tmp["t_jad"]/tmp["iter_ad"])
    # append!(t_julspm, tmp["t_jspm"]/tmp["iter_spm"])
    append!(t_mat, tmp["t_mat"])
    append!(t_julad, tmp["t_jad"])
    append!(t_julspm, tmp["t_jspm"])
    append!(iter_ad, tmp["iter_ad"])
    append!(iter_spm, tmp["iter_spm"])
    append!(F_mat, tmp["F_mat"])
    append!(F_spm, tmp["F_jspm"])
    append!(F_ad, tmp["F_jad"])
end
# plot(list, (t_mat./t_julad.-1), legendfontsize=12, xtickfontsize=14, xguidefontsize=14, ytickfontsize=14, yguidefontsize=14, legend=:topleft, xlabel="number of regions",ylabel="speed increase", linewidth=3, label="Julia AD")
# plot!(list, (t_mat./t_julspm.-1), linewidth=3, label="Julia SPM12")
# scatter!(list, (t_mat./t_julad.-1), markersize=6, color="blue", label=nothing)
# scatter!(list, (t_mat./t_julspm.-1), markersize=6, color="red", label=nothing)
plot(list, t_julad./60, legendfontsize=12, xtickfontsize=14, xguidefontsize=14, ytickfontsize=14, yguidefontsize=14, legend=:topleft, xlabel="number of regions",ylabel="computation time [min]", linewidth=3, label="Julia AD")
plot!(list, t_julspm./60, linewidth=3, label="Julia SPM12")
plot!(list, t_mat./60, linewidth=3, color="black", label="Matlab SPM12")
scatter!(list, t_julad, markersize=6, color="blue", label=nothing)
scatter!(list, t_julspm, markersize=6, color="red", label=nothing)
title!("speed comparison between Julia and Matlab")
savefig("speedandaccuracy/plots/speedcomp10regions.png")


### plot ADVI results ###
using JLD2
using StatsPlots
using LinearAlgebra

r = 3
vals = matread("speedandaccuracy/matlab0.01_" * string(r) * "regions.mat");
A_true = vals["true_params"]["A"]
A_std = reshape(sqrt.(diag(vals["Cp"][1:9,1:9])),3,3)
d = size(A_true, 1)
ADVIsteps = 1000
ADVIsamples = 10
iter = 1
(q, advi, cond_model) = load_object("speedandaccuracy/ADVIADA" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(r) * ".jld2");
# (q, advi, cond_model) = deserialize("speedandaccuracy/ADVIADA" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(r) * ".dat");
Turing.elbo(advi, q, cond_model, 10)
Fs = zeros(100)
for iter in 1:15
    (q, advi, cond_model) = deserialize("speedandaccuracy/ADVIADA" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(r) * ".dat");
    for i = 1:100
        Fs[i] = Turing.elbo(advi, q, cond_model, 10)
    end
    histogram(Fs, xlims=(-5000, 0))
    savefig("speedandaccuracy/plots/F_ADVI_hist" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(r) * ".png")
end

X = []
Yj = []
Yj_total = []
Ym = []
Ym_err = []
Yt = []
for i = 1:d
    for j = 1:d
        if i == j
            continue
        end
        push!(X, latexstring("a_{", string(i), string(j), "}"))
        push!(Yj, reshape(q.dist.m[1:d^2], d, d)[i,j])
        push!(Ym, vals["Ep"]["A"][i, j])
        push!(Ym_err, A_std[i, j])
        push!(Yt, A_true[i, j])
    end
end
push!(Yj_total, (Yj .- Yt)./abs.(Yt))
# scatter(X, Yj, label="ADVI",color=:orange, legendfontsize=10, xtickfontsize=12, xguidefontsize=12, ytickfontsize=12, yguidefontsize=12, markeralpha=0.3)
for iter = 2:15
    q = load_object("speedandaccuracy/ADVIADA" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(r) * ".jld2")[1];
    Yj = []
    for i = 1:d
        for j = 1:d
            if i == j
                continue
            end
            push!(Yj, reshape(q.dist.m[1:d^2], d, d)[i,j])
        end
    end
    # scatter!(X, Yj, label=false, color=:orange, markeralpha=0.3)
    push!(Yj_total, (Yj .- Yt)./abs.(Yt))
end
default(size=(800,600),
        legendfontsize=14,
        xtickfontsize=22,
        xguidefontsize=16,
        ytickfontsize=16,
        yguidefontsize=16,
        markersize=8,
        ylabel="relative distance from ground truth")

boxplot(repeat(X,outer=5), vcat(Yj_total...), label="ADVI")
scatter!(X, (Ym .- Yt)./abs.(Yt), yerr=Ym_err./abs.(Yt), label="Laplace", legend=:best)
hline!([0], linestyle=:dash, color=:black, label=false)
savefig("speedandaccuracy/plots/ADVIboxplot_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(r) * ".png")

scatter!(X, Ym  .- Yt, label="Laplace", color=:blue)
scatter!(repeat(X,outer=5),Yj_total, label=false, markeralpha=0.3, color=:orange)
scatter!(X, mean(Yj_total), yerr=std(Yj_total), label=L"$\mu$(ADVI)", color=:green, markerhsape=:diamond)
title!("standard deviation of interaction = 0.01\n ADVI samples = " * string(ADVIsamples) * ", steps =" * string(ADVIsteps) * ", regions = " * string(r))
violin(repeat(X,outer=5), vcat(Yj_total...), linewidth=0)
scatter!(X, Ym  .- Yt, label="Laplace", color=:blue)
boxplot!(repeat(X,outer=5), vcat(Yj_total...), fillalpha=0.75, linewidth=2)
