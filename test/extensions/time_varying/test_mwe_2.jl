using Enzyme, SparseArrays, LinearAlgebra

function compute_L1(Σ::Matrix{T}, Λ::AbstractSparseMatrix{T}) where T <: Real
    # LHS coefficient from Proposition 3.2
    #   L₁ = [Σ ⊗ (Iₙ² + Λₙ)] ⊗ [vec(Iₙ) ⊗ Iₙ]
    N = size(Σ, 1)
    return kron(Σ, I + Λ) * kron(vec(I(N)), I(N))
end

Λ =  spzeros(4, 4)
Λ[1, 1] = 1.0
Λ[3, 2] = 1.0
Λ[2, 3] = 1.0
Λ[4, 4] = 1.0

Δt = 0.25

function f(θ, Λ, Δt)
    σ_z = θ[1]
    θ_z = θ[2]
    Ω = zeros(2,2)
    Ω[1,1] = sqrt(((σ_z^2)/(2.0 * θ_z))*(1-exp(-2*θ_z*Δt)))
    Σ = Ω * Ω'
    L1 = compute_L1(Σ, Λ)
    return L1[1]
end

θ = [1.0, 0.5]
dθ = similar(θ)
f(θ,  Λ, Δt)
Enzyme.autodiff(Reverse, f, Active, Duplicated(θ, dθ), Const(Λ), Const(Δt))
