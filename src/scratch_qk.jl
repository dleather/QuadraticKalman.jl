using Pkg; Pkg.activate("QuadraticKalman")
using Revise, QuadraticKalman, LinearAlgebra, Random, BenchmarkTools, KalmanFilters, Turing,
      Parameters, Plots

N = 1
M = 1
μ = [-0.1]
Φ = reshape([0.9], (1,1))
Ω = reshape([0.2], (1,1))
A = [0.1]
B = reshape([0.6], (1,1))
C = [reshape([0.2],(1,1))]
D = reshape([0.05], (1,1))
α = reshape([0.9], (1,1))

params =  QKParams(N, M, μ, Φ, Ω, A, B, C, D, α)

#Set seed
Random.seed!(1234)
T = 1000
X_sim = zeros(N, T)
Y_sim = zeros(M, T)
for t = 2:T
    X_sim[:,t] = params.μ + params.Φ * X_sim[:,t-1] + params.Ω * randn(N)
    Y_sim[:,t] = params.A + params.α * Y_sim[:, t-1] + params.B * X_sim[:,t] +
                 X_sim[:,t] * params.C[1] * X_sim[:,t]' + params.D * randn(M)
end

data = QKData(Y_sim)

out =  qkf_filter(data, params)

μ = params.μ[1,1]
Φ = params.Φ[1,1]
Ω = params.Ω[1,1]
A = params.A[1,1]
B = params.B[1,1]
C = params.C[1][1,1]
D = params.D[1,1]
α = params.α[1,1]



f11 = plot(X_sim[1,1:T], label="True Xₜ", color="black")
plot!(f11, out.Zₜₜ[1,1:T], ribbon=2*sqrt.(out.Pₜₜ[1,1,1:T]), 
    label="Filtered Xₜ ± 2σ", color="red")
xlabel!(f11, "t")
ylabel!(f11, "Xₜ")
title!(f11, "Xₜ = $μ + $Φ Xₜ₋₁ + $Ω * ϵₜ")
plot!(f11,legend_background_color=:transparent)
f12 = plot(Y_sim[1,1:T], label="True Yₜ", color="black")
plot!(f12, out.Yₜₜ₋₁[1,1:T-1], ribbon=2*sqrt.(out.Mₜₜ₋₁[1,1,1:T-1]), 
        label="Filtered Yₜ ± 2σ", color="red")
xlabel!(f12, "t")
ylabel!(f12, "Yₜ")
title!(f12, "Yₜ = $A + $α * Yₜ₋₁ + $B Xₜ + $C Xₜ² + $D * ηₜ")
plot!(f12,legend_background_color=:transparent)
plot(f11, f12, layout=(2,1), size=(800, 400))

# Define the Turing model
function turing_mdl(data::QKData, params::QKParams)
    @unpack Y, T̄ = data
    @unpack μ, Φ, Ω, A, α, B, e, C, D, μᵘ, Σᵘ, Σ, V = params
    Y = [Y[:, t] for t in 1:(T̄+1)]

    @model function state_space_model(Y, T̄)
        # Prior for the initial state X₀
        X₀ ~ MvNormal(μᵘ, Σᵘ)
        
        # Initialize the state vector
        X = Vector{Vector{Float64}}(undef, T̄)
        X[1] = X₀
        
        for t in 2:T̄
            # State equation: Xₜ = μ + ΦXₜ₋₁ + Ωϵₜ
            X[t] ~ Normal(μ .+ Φ * X[t-1], Σ)
        end
        
        for t in 2:T̄
            # Observation equation: Yₜ = A(Y_{t-1}) + BXₜ + ∑ₖ₌₁ᵐ eₖXₜ'C⁽ᵏ⁾Xₜ + Dηₜ
            quadratic_term = [0.0]
            for k in eachindex(C)
                quadratic_term += e[k] * X[t]' * C[k] * X[t]
            end
            Y[t] ~ Normal(A .+ α * Y[t-1] .+ B * X[t] .+ quadratic_term, V)
        end
    end

    return state_space_model(Y, T)
end

model = turing_mdl(data, params)
chain = sample(model, NUTS(), 1000)