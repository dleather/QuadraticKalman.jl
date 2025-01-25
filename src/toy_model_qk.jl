# This code simulates and estimates the Latent-OU State, Linear-Quadratic Measurement model
# found in the PDF "Thoughts on NL-TSM" in Section 2 - Section 3
using Pkg; Pkg.activate("QuadraticKalman\\")
using Revise, QuadraticKalman, Statistics, StochasticDiffEq, LinearAlgebra, 
    Random, DifferentiationInterface, CSV, DataFrames, BenchmarkTools, Distributions,
    Zygote, Optimization, OptimizationOptimJL
Random.seed!(123124)

println("Checkpoint 1")

θ_z_init = 0.5
σ_z_init = 1.0
θ_y_init = 2.0
σ_y_init = 0.05 * σ_z_init
ξ₀_init = 1.0 
ξ₁_init = 0.5
α_init = 0.2
t0 = 0.0
T = 0.25
const Δt = T - t0

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

const qkdata = QKData(data[1,:])


θ0 = [α_init, log(θ_z_init), log(θ_y_init), log(σ_z_init), log(σ_y_init), ξ₀_init, ξ₁_init]
α = θ0[1]
θz = exp(θ0[2])
θy = exp(θ0[3])
σz = exp(θ0[4])
σy = exp(θ0[5])
ξ0 = θ0[6]
ξ1 = θ0[7]


ulm = UnivariateLatentModel(3000.0, σz, α, θy, σy, ξ0, ξ1, Δt)
qkparams = convert_to_qkparams(ulm)
llₜ, ~, ~, ~, ~, ~, ~, ~ = qkf_filter(qkdata, qkparams)
sum(llₜ)

α = θ[1]
θz = exp(θ[2])
θy = exp(θ[3])
σz = exp(θ[4])
σy = exp(θ[5])
ξ0 = θ[6]
ξ1 = θ[7]

    ulm_in = UnivariateLatentModel(θz, σz, α, θy, σy, ξ0, ξ1, Δt)
    qkparams = convert_to_qkparams(ulm_in)


    has_inf_or_nan(x::Real) = isinf(x) || isnan(x)
    function has_inf_or_nan(s::T) where T
        return any(f -> has_inf_or_nan(getfield(s, f)), fieldnames(T))
    end
function nll(θ::Vector{Float64}, qkdata::QKData{Float64, N}, Δt) where N
    #θ = [α, log(θz), log(θy), log(σz), log(σy), ξ0, ξ1]
    α = θ[1]
    θz = exp(θ[2])
    θy = exp(θ[3])
    σz = exp(θ[4])
    σy = exp(θ[5])
    ξ0 = θ[6]
    ξ1 = θ[7]
    if σz <= 0.0 
        println("hi")
        println(σz)
        println(θ[4])
    end
    ulm_in = UnivariateLatentModel(θz, σz, α, θy, σy, ξ0, ξ1, Δt)
    
    if has_inf_or_nan(ulm_in)
        return Inf
    else
        qkparams = convert_to_qkparams(ulm_in)
        llₜ, ~, ~, ~, ~, ~, ~, ~ = qkf_filter(qkdata, qkparams)

        out = -sum(llₜ)
        return out
    end
end

p = 0.0
f = (x) -> nll(x, qkdata, Δt)
f(θ0)

ulm_in = UnivariateLatentModel(θz, σz, α, θy, σy, ξ0, ξ1, Δt)
qkparams = convert_to_qkparams(ulm_in)
f = (x) -> nll(x, qkdata, Δt)
f(θ0)
θ1 = θ0 + [0.0;0.001*randn(1); zeros(5)]
f(θ1)
grad = Zygote.gradient(f, θ0)

using FiniteDiff
h = (x,p) -> f(x)
of = OptimizationFunction(h, AutoFiniteDiff())
prob = OptimizationProblem(of, θ0, lcons = [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0], 
    ucons = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
callback4 = function (state, obj_value)
    println("Iteration: ", state.iter)
    println("Current parameters: ", state.u)
    println("Current objective value: ", state.objective)
    println("---")
    return false
end

sol = solve(prob, BFGS(), callback = callback4)

θ2 = [0.24499242934099064, -0.6910029279553279, 0.6719177826601989, -0.050208834982104086, -2.9840374237978917, 0.9579444146809127, 0.5067772858992915]

f(θ2)

Σ = qkparams.Σ

eigvals1, eigvecs1 = eigen(A)
eigvals2, eigvecs2 = DifferentiableEigen.eigen(A)
hcat([eigvecs2[1+(i-1)*2*size(A,1):2:i*2*size(A,1)] for i in 1:size(A,1)]...)

A = rand(3,3)


out = Zygote.jacobian((x) -> eigvals(Symmetric(x)), A)[1]

function g(A)
    B = make_positive_definite(A)
    C = sum(B)
    return C
end

#Differentiate with zygote
grad = Zygote.gradient(g, A)

