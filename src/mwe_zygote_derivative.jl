
using Pkg; Pkg.activate("QuadraticKalman\\")
using Revise, QuadraticKalman
using Zygote, LinearAlgebra, Parameters, Random, StochasticDiffEq, 
    Plots, Statistics, ForwardDiff, BenchmarkTools, ReverseDiff
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

#Simple linear-Quadratic form
# Yₜ = A + BXₜ + αYₜ₋₁ + ∑ₖ₌₁ᵐ eₖXₜ'C⁽ᵏ⁾Xₜ + D_t ηₜ
# Xₜ = μ + ΦXₜ₋₁ + Ωεₜ 

#Set the model parameters

N = 1
M = 1
A = α_init
B = ξ₀_init
C = ξ₁_init
D = σ_y_init
α = exp(-θ_y_init * Δt)
μ = 0.0
Φ = exp(-θ_z_init * Δt)
Ω = σ_z_init

Σ = Ω * Ω'
V = D^2
T = Float64

e = compute_e(M, T) #DONE
Λ = compute_Λ(N) #DONE
μ̃ = compute_μ̃(μ, Σ) #DONE
Φ̃ = compute_Φ̃(μ, Φ) #DONE
L1 = compute_L1(Σ, Λ) #DONE
L2 = compute_L2(Σ, Λ) #DONE
L3 = compute_L3(Σ, Λ) #DONE
#ν = compute_ν(L1, L2, L3, Λ, Σ, μ) #DONE
#Ψ = compute_Ψ(L1, L2, L3, Φ̃, N) #DONE
μᵘ = compute_μᵘ(μ ,Φ) #DONE
Σᵘ = compute_Σᵘ(Φ, Σ) #DONE
μ̃ᵘ = compute_μᵘ(μ̃, Φ̃) #DONE
Σ̃ᵘ = compute_Σ̃ᵘ(μ̃ᵘ, Σ, μ, Φ, Φ̃)
B̃ = compute_B̃(B, C) #DONE
H = selection_matrix(N, T) #DONE
G = duplication_matrix(N, T) #DONE
H̃ = compute_H̃(N, H) #DONE
G̃ = compute_G̃(N, G) #DONE
P = N * (1 + N)

#SImulate model
T̄ = 150
Y = zeros(T̄+1)
X = zeros(T̄+1)
Y_pred = zeros(T̄)
X_pred = zeros(T̄)
for t = 1:T̄
    X_pred[t] = μ + Φ * X[t]
    X[t+1] = X_pred[t]  + Ω * randn()
    Y_pred[t] = A + α * Y[t] + B * X[t] + C * X[t]^2
    Y[t+1] = Y_pred[t] + D * randn()
end

plot(Y[2:end], label="Y")
plot!(Y_pred, label="Y_pred")
plot(Y[2:end] - Y_pred)

plot(X[2:end], label="X")
plot!(X_pred, label="X_pred", linestyle = :dash)
plot(X[2:end] - X_pred)
@with_kw struct FilterParams{T<:Real}
    
    P::Int
    μ::Vector{T}
    Φ::Matrix{T}
    A::T
    B::Matrix{T}
    α::T
    Σ::T
    V::T
    Z_init::Vector{T}
    P_init::Matrix{T}

end


fp = FilterParams(P, μ̃, Φ̃, A, B̃, α, Σ, V, μ̃ᵘ, Σ̃ᵘ)

function filter(T̄, Y, M, params::FilterParams{T}) where T<:Real

    @unpack P, μ, Φ, A, B, α, Σ, V, Z_init, P_init = params

    Y_concrete = vec(Y)  # Convert to vector if it's not already
    Yₜ = @view Y_concrete[2:end]

    Zₜₜ = zeros(T, P, T̄ + 1)
    Pₜₜ = zeros(T, P, P, T̄ + 1)
    Zₜₜ₋₁ = zeros(T, P, T̄)
    Σₜₜ₋₁ = zeros(T, P, P, T̄)
    Pₜₜ₋₁ = zeros(T, P, P, T̄)
    Kₜ = zeros(T, P, M, T̄)
    Yₜₜ₋₁ = zeros(T, T̄)
    Mₜₜ₋₁ = zeros(T, M, M, T̄)
    llₜ = zeros(T, T̄)
    pred = zero(T)
    pmP = zeros(T, P, P)

    # Initialize
    Zₜₜ[:, 1] = Z_init
    Pₜₜ[:, :, 1] = P_init

    for t in 1:T̄

        #Predict mean and variance of Z_t
        Zₜₜ₋₁[:, t] = μ + Φ * Zₜₜ[:, t]
        pred = copy(Zₜₜ₋₁[1, t])
        #Σₜₜ₋₁[:,:,t] = compute_Σₜₜ₋₁(Zₜₜ[:, t], params, t)
        Σₜₜ₋₁[1, 1, t] = Σ
        Σₜₜ₋₁[1, 2, t] = Σ * pred
        Σₜₜ₋₁[2, 1, t] = Σ * pred
        Σₜₜ₋₁[2, 2, t] = 2. * Σ^2 + 4. * Σ * pred^2
        Σₜₜ₋₁[:, :, t] = make_positive_definite(Σₜₜ₋₁[:, :, t])
        Pₜₜ₋₁[:, :, t] = make_positive_definite(Φ * Pₜₜ[:, :, t] * Φ' + Σₜₜ₋₁[:, :, t])

        #Predict mean and variance of Y_t
        Yₜₜ₋₁[t] = A + (B * Zₜₜ₋₁[:,t])[1] + α * Y[t] 
        Mₜₜ₋₁[:, :, t] = [(B * Pₜₜ₋₁[:, :, t] * B')[1] + V]

        #Compute Kalman gain
        Kₜ[:, :, t] = Pₜₜ₋₁[:, :, t] * B' / (B * Pₜₜ₋₁[:, :, t] * B' + Mₜₜ₋₁[:, :, t])
        
        #Update mean and variance of Z_t
        Zₜₜ[:, t + 1] = Zₜₜ₋₁[:, t] + Kₜ[:, :, t] * (Yₜ[t] - Yₜₜ₋₁[t])
        pmP = (I - Kₜ[:, :, t] * B) 
        Pₜₜ[:, :, t + 1] = pmP * Pₜₜ₋₁[:, :, t] * pmP' +
            Kₜ[:, :, t] * V * Kₜ[:, :, t]'

        Zₜₜ[:, t + 1] = correct_Zₜₜ(Zₜₜ[:, t + 1], params, t)

        llₜ[t] = log_pdf_normal(Yₜₜ₋₁[t], Mₜₜ₋₁[1, 1, t], Yₜ[t])
  
    end

    return (llₜ = copy(llₜ), Zₜₜ = copy(Zₜₜ), Pₜₜ = copy(Pₜₜ), Yₜₜ₋₁ = copy(Yₜₜ₋₁),
        Mₜₜ₋₁ = copy(Mₜₜ₋₁), Kₜ = copy(Kₜ), Zₜₜ₋₁ = copy(Zₜₜ₋₁), Pₜₜ₋₁ = copy(Pₜₜ₋₁))
end

function smooth_max(x, threshold=1e-8)
    return ((x + threshold) + sqrt((x - threshold)^2 + sqrt(eps()))) / 2.0
end

function correct_Zₜₜ(Zₜₜ::AbstractVecOrMat{T}, params::FilterParams, t::Int) where T <: Real
    N = 1
    Ztt = Zₜₜ
    Zttcp = Ztt[1:N] * Ztt[1:N]'
    implied_ZZp = reshape(Ztt[N+1:end], (N, N))
    implied_cov = implied_ZZp - Zttcp
    
    #Smooth approximation of max(0, x)
    
    eig_vals, eig_vecs = eigen(Symmetric(implied_cov))
    eig_vals_corrected = smooth_max.(eig_vals)
    corrected_ZZp = vec(eig_vecs * Diagonal(eig_vals_corrected) * eig_vecs' + Zttcp)
    
    return vcat(Ztt[1:N], corrected_ZZp)
end

function log_pdf_normal(μ::Real, σ2::Real, x::Real) 
    return -0.5 * (log(2.0 * π) + log(σ2) + (x - μ)^2 / σ2)
end
M=1

function θ_to_fparams(θ::AbstractVector{T}) where T <: Real

    αy = θ[1]
    θz = exp(θ[2])
    θy = exp(θ[3])
    σz = exp(θ[4])
    σy = exp(θ[5])
    ξ0 = θ[6]
    ξ1 = θ[7]

    A = αy
    B = ξ0
    C = ξ1
    D = σy
    α = exp(-θy * Δt)
    μ = 0.0
    Φ = exp(-θz * Δt)
    Ω = σz
    
    Σ = Ω^2
    V = D^2
    
    μ̃ = compute_μ̃(μ, Σ) #DONE
    Φ̃ = compute_Φ̃(μ, Φ) #DONE
    μ̃ᵘ = compute_μᵘ(μ̃, Φ̃) #DONE
    Σ̃ᵘ = compute_Σ̃ᵘ(μ̃ᵘ, Σ, μ, Φ, Φ̃)
    B̃ = compute_B̃(B, C) #DONE
    P = Int64(2)

    return FilterParams(P, μ̃, Φ̃, A, B̃, α, Σ, V, μ̃ᵘ, Σ̃ᵘ)

end


function compute_nll(θ::AbstractVector{T},Y::AbstractVector{S},T̄::Int) where {T <: Real, S <: Real}
    fp = θ_to_fparams(θ)

    llₜ, ~, ~, ~, ~, ~, ~, ~ = filter(T̄, Y, 1, fp)
    
    negsum_ll = -sum(llₜ)
    return negsum_ll
end

compute_nll(θ0,Y,T̄)


#Compute gradient
nll_grad(x) = ForwardDiff.gradient(θ -> compute_nll(θ,Y,T̄), x)
nll_grad(θ0)
@btime nll_grad(θ0)

nll_hess(x) = ForwardDiff.hessian(θ -> compute_nll(θ,Y,T̄), θ0)
nll_hess(θ0)
@btime nll_hess(θ0)

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

