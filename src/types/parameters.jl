
struct StateParams{T<:Real}
    N::Int              # State dimension
    mu::Vector{T}       # Drift vector
    Phi::Matrix{T}      # Autoregressive matrix
    Omega::Matrix{T}    # Noise scaling matrix
    Sigma::Matrix{T}    # Covariance of state. Precomputed.
end

function StateParams(N::Int, mu::AbstractVector{T}, Phi::AbstractMatrix{T}, 
    Omega::AbstractMatrix{T}; check_stability::Bool=false) where {T<:Real}
    @assert length(mu) == N "mu must have length N"
    @assert size(Phi,1) == N && size(Phi,2) == N "Phi must be N×N"
    @assert size(Omega,1) == N && size(Omega,2) == N "Omega must be N×N"

    if check_stability
        @assert spectral_radius(Phi) < 1 "Phi must be stable (spectral radius < 1)"
    end

    # Handle UniformScaling without breaking AD
    Phi_final = Phi isa UniformScaling ? Phi * one(T) : Phi
    Omega_final = Omega isa UniformScaling ? Omega * one(T) : Omega
    Sigma = Omega_final * Omega_final'

    return StateParams{T}(N, mu, Phi_final, Omega_final, Sigma)
end

# Measurement equation parameters
struct MeasParams{T<:Real}
    M::Int                  # Observation dimension
    A::Vector{T}            # Constant term in observation equation
    B::Matrix{T}            # Linear observation matrix
    C::Vector{Matrix{T}}    # Quadratic observation matrices
    D::Matrix{T}            # Time-invariant observation noise scaling matrix
    alpha::Matrix{T}        # Autoregressive coefficients of measurement equation
    V::Matrix{T}            # Precomputed DD'
end

function MeasParams(M::Int, N::Int, A::AbstractVector{T}, B::AbstractMatrix{T},
                   C::Vector{<:AbstractMatrix{T}}, D::AbstractMatrix{T}, 
                   alpha::AbstractMatrix{T}) where {T<:Real}
    @assert length(A) == M "A must have length M"
    @assert size(B,1) == M && size(B,2) == N "B must be M×N"
    @assert length(C) == M "C must have length M"
    @assert all(size(Ci) == (N,N) for Ci in C) "Each Ci must be N×N"
    @assert size(D,1) == M && size(D,2) == M "D must be M×M"
    
    D_final = D isa UniformScaling ? D * one(T) : D
    V = D_final * D_final'
    
    return MeasParams{T}(M, A, B, C, D_final, alpha, V)
end

# Augmented state parameters
struct AugStateParams{T<:Real, T2<:Real}
    mu_aug::Vector{T}
    Phi_aug::Matrix{T}
    B_aug::Matrix{T}
    H_aug::Matrix{T}
    G_aug::Matrix{T}
    Lambda::Matrix{T2}
    L1::Matrix{T}
    L2::Matrix{T}
    L3::Matrix{T}
    P::Int
end

function AugStateParams(N::Int, mu::AbstractVector{T}, Phi::AbstractMatrix{T}, 
    Sigma::AbstractMatrix{T}, B::AbstractMatrix{T}, 
    C::Vector{<:AbstractMatrix{T}}) where {T<:Real}

    Lambda = compute_Λ(N)
    mu_aug = compute_μ̃(mu, Sigma)
    Phi_aug = compute_Φ̃(mu, Phi)
    B_aug = compute_B̃(B, C)

    L1 = compute_L1(Sigma, Lambda)
    L2 = compute_L2(Sigma, Lambda)
    L3 = compute_L3(Sigma, Lambda)

    H = selection_matrix(N, T)
    G = duplication_matrix(N, T)
    H_aug = compute_H̃(N, H)
    G_aug = compute_G̃(N, G)
    P = N + N^2

    return AugStateParams{T,eltype(Lambda)}(mu_aug, Phi_aug, B_aug, H_aug, G_aug, Lambda, L1, L2, L3, P)
end

# Unconditional moments
struct Moments{T<:Real}
    state_mean::Vector{T}
    state_cov::Matrix{T}
    aug_mean::Vector{T}
    aug_cov::Matrix{T}
end

function Moments(mu::AbstractVector{T}, Phi::AbstractMatrix{T}, Sigma::AbstractMatrix{T},
                mu_aug::AbstractVector{T}, Phi_aug::AbstractMatrix{T}, 
                L1::AbstractMatrix{T}, L2::AbstractMatrix{T}, L3::AbstractMatrix{T},
                Lambda::AbstractMatrix) where {T<:Real}
    state_mean = compute_state_mean(mu, Phi)
    state_cov = compute_state_cov(Phi, Sigma)
    aug_mean = compute_state_mean(mu_aug, Phi_aug)
    aug_cov = compute_aug_state_uncond_cov(aug_mean, L1, L2, L3, Lambda, Sigma, mu, Phi_aug)
    
    return Moments{T}(state_mean, state_cov, aug_mean, aug_cov)
end

# Main structure
@with_kw struct QKParams{T<:Real, T2<:Real}
    state::StateParams{T}
    meas::MeasParams{T}
    aug::AugStateParams{T,T2}
    moments::Moments{T}
end

# Main constructor
function QKModel(N::Int, M::Int, mu::AbstractVector{T}, Phi::AbstractMatrix{T}, 
                 Omega::AbstractMatrix{T}, A::AbstractVector{T}, B::AbstractMatrix{T}, 
                 C::Vector{<:AbstractMatrix{T}}, D::AbstractMatrix{T},
                 alpha::AbstractMatrix{T}, dt::T2;
                 check_stability::Bool=false) where {T<:Real, T2<:Real}
    
    @assert dt > 0 "dt must be positive"
    
    # Construct sub-components
    state = StateParams(N, mu, Phi, Omega; check_stability=check_stability)
    meas = MeasParams(M, N, A, B, C, D, alpha)
    aug = AugStateParams(N, state.mu, state.Phi, state.Sigma, meas.B, meas.C)
    moments = Moments(state.mu, state.Phi, state.Sigma,
                     aug.mu_aug, aug.Phi_aug, aug.L1, aug.L2, aug.L3, aug.Lambda)
    
    return QKModel(state=state, meas=meas, aug=aug, moments=moments, dt=dt)
end