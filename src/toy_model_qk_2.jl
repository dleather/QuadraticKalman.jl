# This code simulates and estimates the Latent-OU State, Linear-Quadratic Measurement model
# found in the PDF "Thoughts on NL-TSM" in Section 2 - Section 3
#using Pkg; Pkg.activate("QuadraticKalman\\")
using Revise, QuadraticKalman, ModelingToolkit, Plots, Statistics, StochasticDiffEq,
    LinearAlgebra, SparseArrays, Random, FLoops, Enzyme, DifferentiationInterface,
    CSV, DataFrames, BenchmarkTools, Distributions
Random.seed!(123124)

θ_z_init = 0.1
σ_z_init = 1.0
θ_y_init = 2.0
σ_y_init = 0.05 * σ_z_init
ξ₀_init = 1.0 
ξ₁_init = 0.5
α_init = 0.2
t0 = 0.0
T = 0.25
Δt = T - t0

u0 = [0.0, 0.0]
#params = [θ_z => 1.0, σ_z => 1.0, θ_y => 1.0, σ_y => 1.0, ξ₀ => 1.0, ξ₁ => 1.0, α => 1.0]

#Define the model
function drift!(du,u,p,t)
    du[1] = θ_y_init * (α_init - u[1]) + ξ₀_init * u[2] + ξ₁_init * u[2]^2
    du[2] = -θ_z_init * u[2]
end
   
function diffusion!(du,u,p,t)
    du[1] = σ_y_init
    du[2] = σ_z_init
end

dt = 0.25
tspan = (0.0, 150)
prob = SDEProblem(drift!, diffusion!, u0, tspan)

sol = solve(prob, SOSRA(), saveat = dt)

#plot(sol, idxs=(0,1), linewidth=2, label="Y")
#plot!(sol, idxs=(0,2), linewidth=2, label="Z")

#Compute correlation
cor([sol.u[i][1] for i in 1:length(sol.u)], [sol.u[i][2] for i in 1:length(sol.u)])

#Let's prepare the data for saving
data = hcat(sol.u...)
df = DataFrame(Y = data[1,:], X = data[2,:])
CSV.write("C:/Users/davle/Dropbox (Personal)/CapRates/NonlinearTSM/QKF_R_code/QKF code/data.csv", df)


#Try to filter out z using QuadraticKalman.jl\
#Compute mean of var_y
mean_z =  [0.0
0.0
0.22119921692859523
0.1952076237936233
0.1952076237936233
0.22119921692859523];

qkdata = QKData(data[1,:])


function nll(θ::Vector{Float64}, qkdata::QKData{Float64, N}; Δt = 0.25) where N
    #θ = [α, log(θz), log(θy), log(σz), log(σy), ξ0, ξ1]
    α = θ[1]
    θz = exp(θ[2])
    θy = exp(θ[3])
    σz = exp(θ[4])
    σy = exp(θ[5])
    ξ0 = θ[6]
    ξ1 = θ[7]

    ulm = UnivariateLatentModel(θz, σz, α, θy, σy, ξ0, ξ1, Δt)
    qkparams = convert_to_qkparams(ulm)
    llₜ, ~, ~, ~, ~, ~, ~, ~ = qkf_filter(qkdata, qkparams)

    out = -sum(llₜ)
    return out
end

function nll!(θ::Vector{Float64}, qkdata::QKData{Float64, N}; Δt = 0.25) where N
    #θ = [α, log(θz), log(θy), log(σz), log(σy), ξ0, ξ1]
    α = θ[1]
    θz = exp(θ[2])
    θy = exp(θ[3])
    σz = exp(θ[4])
    σy = exp(θ[5])
    ξ0 = θ[6]
    ξ1 = θ[7]

    ulm = UnivariateLatentModel(θz, σz, α, θy, σy, ξ0, ξ1, Δt)
    qkparams = convert_to_qkparams(ulm)
    llₜ, ~, ~, ~, ~, ~, ~, ~ , ~ = qkf_filter!(qkdata, qkparams)

    out = -sum(llₜ)
    return out
end


function nll(θ1::Float64, θ2::Float64, θ3::Float64, θ4::Float64, θ5::Float64, θ6::Float64,
    θ7::Float64,qkdata::QKData{Float64, N}; Δt = 0.25) where N
    #θ = [α, log(θz), log(θy), log(σz), log(σy), ξ0, ξ1]
    α = θ1
    θz = exp(θ2)
    θy = exp(θ3)
    σz = exp(θ4)
    σy = exp(θ5)
    ξ0 = θ6
    ξ1 = θ7

    ulm = UnivariateLatentModel(θz, σz, α, θy, σy, ξ0, ξ1, Δt)
    qkparams = convert_to_qkparams(ulm)
    llₜ, ~, ~, ~, ~, ~, ~, ~, ~ = qkf_filter(qkdata, qkparams)

    out = -sum(llₜ)
    return out
end



ulm = UnivariateLatentModel(θ_z_init, σ_z_init, α_init, θ_y_init, σ_y_init, ξ₀_init, ξ₀_init, Δt)
qkparams = convert_to_qkparams(ulm)


θ0 = [α_init, log(θ_z_init), log(θ_y_init), log(σ_z_init), log(σ_y_init), ξ₀_init, ξ₁_init]

nll!(θ0, qkdata)

#=
compute_loglik!(llₜ::AbstractVector{T}, Yₜ::AbstractVector{T},
        Yₜₜ₋₁::AbstractVector{T}, Mₜₜ₋₁::AbstractArray{T}, t::Int)

μ = zeros(2)
Φ = [exp(-θ_z_init * Δt) 0.0; 1.0 0.0]
Ω = [sqrt(((σ_z_init^2)/(2.0 * θ_z_init))*(1-exp(-2*θ_z_init*Δt))) 0.0; 0.0 0.0]
Σ = Ω * Ω'
e = compute_e(1)
Λ =  compute_Λ(2)
compute_μ̃_old(μ, Σ) == compute_μ̃(μ, Σ)

tmp_fn = (x) -> compute_μ̃(x, Σ)[1]
tmp_fn(μ)

DifferentiationInterface.gradient(tmp_fn, AutoEnzyme(), μ)

compute_Φ̃(μ, Φ) == compute_Φ̃_old(μ, Φ)
@btime compute_Φ̃(μ, Φ) 
@btime compute_Φ̃_old(μ, Φ)
M = 1
N = 2
e = compute_e(M)
Λ =  compute_Λ(N)

compute_L1(Σ, Λ)
function f1(θ)
    σ_z_init = θ[1]
    θ_z_init = θ[2]
    Ω = [sqrt(((σ_z_init^2)/(2.0 * θ_z_init))*(1-exp(-2*θ_z_init*Δt))) 0.0; 0.0 0.0]
    Σ = Ω * Ω'

    L1 = compute_L1(Σ, Λ)
    return L1[1]
end

θ = [σ_z_init, θ_z_init]
f1(θ)
DifferentiationInterface.gradient(f1, AutoEnzyme(), θ)

Λ = 

=#