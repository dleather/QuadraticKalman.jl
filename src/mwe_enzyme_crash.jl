using Enzyme, SparseArrays, LinearAlgebra

function compute_L1(Σ::Matrix{T}, Λ::AbstractSparseMatrix{T}) where T <: Real
    # LHS coefficient from Proposition 3.2
    #   L₁ = [Σ ⊗ (Iₙ² + Λₙ)] ⊗ [vec(Iₙ) ⊗ Iₙ]
    N = size(Σ, 1)
    return kron(Σ, I + Λ) * kron(vec(I(N)), I(N))
end

const Λ =  spzeros(4, 4)
Λ[1, 1] = 1.0
Λ[3, 2] = 1.0
Λ[2, 3] = 1.0
Λ[4, 4] = 1.0

const Δt = 0.25

function f(θ, Δt , Λ)
    σ_z = θ[1]
    θ_z = θ[2]
    Ω = [sqrt(((σ_z^2)/(2.0 * θ_z))*(1-exp(-2*θ_z*Δt))) 0.0; 0.0 0.0]
    Σ = Ω * Ω'
    L1 = compute_L1(Σ, Λ)
    return L1[1]
end

θ = [1.0, 0.5]
dθ = similar(θ)
g = (x) -> f(x, Δt, Λ)
@code_warntype g(θ)
Enzyme.autodiff(Reverse, g , Active, Duplicated(θ, dθ))
