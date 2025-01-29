@with_kw struct StateParams{T<:Real}
    N::Int              # State dimension
    mu::Vector{T}       # Drift vector
    Phi::Matrix{T}      # Autoregressive matrix
    Omega::Matrix{T}    # Noise scaling matrix
    Sigma::Matrix{T}    # Covariance of state. Precomputed.
end

function StateParams(N::Int, mu::AbstractVector{T}, Phi::Union{AbstractMatrix{T}, UniformScaling}, 
    Omega::Union{AbstractMatrix{T}, UniformScaling}; check_stability::Bool=false) where {T<:Real}
    @assert length(mu) == N "mu must have length N"
    
    # Modified size assertions to handle UniformScaling
    if !(Phi isa UniformScaling)
        @assert size(Phi,1) == N && size(Phi,2) == N "Phi must be N×N"
    end
    if !(Omega isa UniformScaling)
        @assert size(Omega,1) == N && size(Omega,2) == N "Omega must be N×N"
    end

    if check_stability
        @assert spectral_radius(Phi) < 1 "Phi must be stable (spectral radius < 1)"
    end

    # Handle UniformScaling without breaking AD
    Phi_final = Phi isa UniformScaling ? Matrix(Phi, N, N) : Phi
    Omega_final = Omega isa UniformScaling ? Matrix(Omega, N, N) : Omega
    Sigma = Omega_final * Omega_final'

    return StateParams{T}(N, mu, Phi_final, Omega_final, Sigma)
end

# Measurement equation parameters
@with_kw struct MeasParams{T<:Real}
    M::Int                                  # Observation dimension
    A::Vector{T}                            # Constant term in observation equation
    B::Matrix{T}                            # Linear observation matrix
    C::Vector{Matrix{T}}                    # Quadratic observation matrices
    D::Matrix{T}                            # Time-invariant observation noise scaling matrix
    alpha::Matrix{T}                        # Autoregressive coefficients of measurement equation
    V::Matrix{T}                            # Precomputed DD'
end

function MeasParams(M::Int, N::Int, A::AbstractVector{T}, 
                   B::Union{AbstractMatrix{T}, UniformScaling},
                   C::Vector{<:Union{AbstractMatrix{T}, UniformScaling}}, 
                   D::Union{AbstractMatrix{T}, UniformScaling},
                   alpha::Union{AbstractMatrix{T}, UniformScaling}) where {T<:Real}
    @assert length(A) == M "A must have length M"
    
    # Handle UniformScaling for B
    B_final = B isa UniformScaling ? Matrix(B, M, N) : B
    @assert size(B_final,1) == M && size(B_final,2) == N "B must be M×N"
    
    @assert length(C) == M "C must have length M"
    C_final = [Ci isa UniformScaling ? Matrix(Ci, N, N) : Ci for Ci in C]
    @assert all(size(Ci) == (N,N) for Ci in C_final) "Each Ci must be N×N"
    
    # Handle UniformScaling for D and alpha
    D_final = D isa UniformScaling ? Matrix(D, M, M) : D
    @assert size(D_final,1) == M && size(D_final,2) == M "D must be M×M"
    
    alpha_final = alpha isa UniformScaling ? Matrix(alpha, M, M) : alpha
    
    V = D_final * D_final'
    
    return MeasParams{T}(M, A, B_final, C_final, D_final, alpha_final, V)
end

# Augmented state parameters
@with_kw struct AugStateParams{T<:Real, T2<:Real}
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

    Lambda = compute_Lambda(N)
    mu_aug = compute_mu_aug(mu, Sigma)
    Phi_aug = compute_Phi_aug(mu, Phi)
    B_aug = compute_B_aug(B, C)

    L1 = compute_L1(Sigma, Lambda)
    L2 = compute_L2(Sigma, Lambda)
    L3 = compute_L3(Sigma, Lambda)

    H = selection_matrix(N, T)
    G = duplication_matrix(N, T)
    H_aug = compute_H_aug(N, H)
    G_aug = compute_G_aug(N, G)
    P = N + N^2

    return AugStateParams{T,eltype(Lambda)}(mu_aug, Phi_aug, B_aug, H_aug, G_aug, Lambda, L1, L2, L3, P)
end

# Unconditional moments
@with_kw struct Moments{T<:Real}
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
@with_kw struct QKModel{T<:Real, T2<:Real}
    state::StateParams{T}
    meas::MeasParams{T}
    aug_state::AugStateParams{T,T2}
    moments::Moments{T}
end

# Main constructor
function QKModel(N::Int, M::Int, mu::AbstractVector{T}, Phi::AbstractMatrix{T}, 
                 Omega::AbstractMatrix{T}, A::AbstractVector{T}, B::AbstractMatrix{T}, 
                 C::Vector{<:AbstractMatrix{T}}, D::AbstractMatrix{T},
                 alpha::AbstractMatrix{T};
                 check_stability::Bool=false) where {T<:Real}
        
    # Construct sub-components
    state = StateParams(N, mu, Phi, Omega; check_stability=check_stability)
    meas = MeasParams(M, N, A, B, C, D, alpha)
    aug_state = AugStateParams(N, state.mu, state.Phi, state.Sigma, meas.B, meas.C)
    moments = Moments(state.mu, state.Phi, state.Sigma, aug_state.mu_aug, aug_state.Phi_aug,
                     aug_state.L1, aug_state.L2, aug_state.L3, aug_state.Lambda)
    
    return QKModel(state=state, meas=meas, aug_state=aug_state, moments=moments)
end

# Keyword argument constructor
function QKModel(state::StateParams{T}, meas::MeasParams{T}; check_stability::Bool=false) where {T<:Real}
    @unpack N, mu, Phi, Omega = state
    @unpack M, A, B, C, D, alpha = meas
    return QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha; check_stability=check_stability)
end