
using Pkg; Pkg.activate("QuadraticKalman\\")
using Revise, QuadraticKalman
using LinearAlgebra, Parameters, Random, StochasticDiffEq, 
    Plots, Statistics, ForwardDiff, BenchmarkTools
Random.seed!(123124)

#Define Parameters
θ_z_init = 0.5
σ_z_init = 1.0
θ_y_init = 2.0
σ_y_init = 0.05 * σ_z_init
ξ₀_init = 1.0
ξ₁_init = 0.5
α_init = 0.2
θ0 = [α_init, log(θ_z_init), log(θ_y_init), log(σ_z_init), log(σ_y_init),
     ξ₀_init, ξ₁_init]
t0 = 0.0
T = 0.25
const Δt = T - t0

#Simulate data
function drift!(du,u,p,t)
    du[1] = θ_y_init * (α_init - u[1]) + ξ₀_init * u[2] + ξ₁_init * u[2]^2
    du[2] = -θ_z_init * u[2]
end

function diffusion!(du,u,p,t)
    du[1] = σ_y_init
    du[2] = σ_z_init
end

u0 = [0.0, 0.0]
tspan = (0.0, 150)
prob = SDEProblem(drift!, diffusion!, u0, tspan)
sol = solve(prob, SOSRA(), saveat = Δt)
plot(sol)
Y = [sol.u[i][1] for i in 1:length(sol.u)]
X = [sol.u[i][2] for i in 1:length(sol.u)]
function θ_to_qkparams(θ::Vector{T}) where T <: Real
    αy = θ[1]
    θz = exp(θ[2])
    θy = exp(θ[3])
    σz = exp(θ[4])
    σy = exp(θ[5])
    ξ₀ = θ[6]
    ξ₁ = θ[7]

    ulm = UnivariateLatentModel(θz, σz, αy, θy, σy, ξ₀, ξ₁, Δt)

    return convert_to_qkparams(ulm)

end

function compute_nll(θ::Vector{T1}, Y::Vector{T2}) where {T1 <: Real, T2 <: Real}
    qkparams = θ_to_qkparams(θ)
    qkdata = QKData(Y)
    llₜ, ~, ~, ~, ~, ~, ~, ~ = qkf_filter(qkdata, qkparams)
    return -sum(llₜ)
end

compute_nll(θ0, Y)
#Compute gradient
nll_grad(x) = ForwardDiff.gradient(θ -> compute_nll(θ,Y), x)
nll_grad(θ0)
#@btime nll_grad(θ0)
#
#nll_hess(x) = ForwardDiff.hessian(θ -> compute_nll(θ,Y,T̄), θ0)
#nll_hess(θ0)
#@btime nll_hess(θ0)

#OptimizationProblem
using Optim
lcons = [-100., -5., -5., -5., -5., -100., -100.]
ucons = [100., 100., 100., 100., 100., 100., 100.]
x0 = θ0
df = TwiceDifferentiable(x -> compute_nll(x,Y,T̄), x0, autodiff = :forward)
dfc = TwiceDifferentiableConstraints(lcons, ucons)
res = optimize(df, dfc, x0, IPNewton(), Optim.Options(show_trace = true))

@code_warntype compute_nll(θ0,Y,T̄)
fparams = θ_to_fparams(res.minimizer)

llₜOpt, ZₜₜOpt, PₜₜOpt, Yₜₜ₋₁Opt, Mₜₜ₋₁Opt, KₜOpt, Zₜₜ₋₁Opt, Pₜₜ₋₁Opt = filter(T̄, Y, M,fparams)
-sum(llₜOpt)

plot(1:T̄, Y[2:end], label="Y")
plot!(1:T̄, Yₜₜ₋₁[1:end], label="Ŷ", linestyle = :dash)
plot!(1:T̄, Yₜₜ₋₁Opt[1:end], label="Ŷ", linestyle = :dash)

mean((Y[2:end] - Yₜₜ₋₁[1:end]).^2)
mean((Y[2:end] - Yₜₜ₋₁Opt[1:end]).^2)

using DifferentiableEigen
A = rand(5,5)
vals, vecs = LinearAlgebra.eigen(A)
vals
vecs

vals2, vecs2 = d_eigen(A)



vals2 == vals
vecs2 == vecs