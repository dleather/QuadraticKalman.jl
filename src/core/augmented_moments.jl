"""
    File: augmented_moments.jl

Utilities for computing augmented means, augmented transition matrices, and auxiliary 
matrices in a Quadratic Kalman Filter (QKF) or similar state-space model.

# Overview

This file implements several scalar and multivariate routines inspired by **Proposition 3.1** 
and **Proposition 3.2** in the QKF context:

1. **Augmented Mean and Transition**  
   - `compute_mu_aug(mu, Sigma)`: Constructs the augmented drift vector, concatenating `[mu; vec(mu*mu' + Sigma)]`.  
   - `compute_Phi_aug(mu, Phi)`: Builds the augmented state transition matrix for both scalar and 
     multivariate cases. 

2. **Unconditional Moments**  
   - `compute_state_mean(mu, Phi)`: Unconditional mean of a stationary (V)AR(1) process.  
   - `compute_state_cov(Phi, Sigma)`: Unconditional covariance for a stable (V)AR(1) process.

3. **Auxiliary Matrices**  
   - `compute_L1`, `compute_L2`, `compute_L3`: Matrix constructions used when updating 
     conditional covariance in the QKF. Each has both a **scalar** and a **multivariate** 
     version. In the scalar case, the matrix operations reduce to simple products; in 
     the multivariate case, Kronecker products and commutation matrices are used.

4. **Conditional Covariances**
   - `compute_cond_cov_state_aug`: Computes conditional covariance of augmented state vector for both scalar and multivariate cases
   - `compute_cond_cov_state`: Computes conditional covariance of state vector
   - `compute_cond_cov_state_ttm1`: Computes conditional covariance at time t given t-1 information
   - `compute_cond_cov_state_ttm1!`: In-place version that updates a pre-allocated array

5. **Observation Matrix**
   - `compute_B_aug`: Constructs augmented observation matrix combining linear and quadratic terms

# Usage

Typically, these functions are called within higher-level filtering routines.
"""

"""
    compute_mu_aug(mu::Real, Sigma::Real)

Compute the augmented drift vector μ̃ for scalar case.

# Arguments
- `mu::Real`: Mean parameter
- `Sigma::Real`: Variance parameter

# Returns
- `Vector`: Augmented drift vector [mu, mu² + Sigma]

# Description
Implements Proposition 3.1 to compute the augmented mean vector μ̃ = [mu, vec(mu*mu' + Sigma)]
for the scalar case where mu and Sigma are real numbers.
"""
function compute_mu_aug(mu::Real, Sigma::Real) 
    return [mu; mu^2 + Sigma]
end

"""
    compute_mu_aug(mu::AbstractVector{T}, Sigma::AbstractMatrix{T}) where T <: Real

Compute the augmented drift vector μ̃ for multivariate case.

# Arguments
- `mu::AbstractVector{T}`: Mean vector of dimension N
- `Sigma::AbstractMatrix{T}`: Covariance matrix of dimension N×N

# Returns
- `Vector{T}`: Augmented drift vector [mu; vec(mu*mu' + Sigma)] of dimension N + N²

# Description
Implements Proposition 3.1 to compute the augmented drift vector μ̃ = [mu; vec(mu*mu' + Sigma)]
for the multivariate case. The result combines:
1. The original mean vector mu
2. The vectorized sum of the outer product mu*mu' and covariance matrix Sigma
"""
function compute_mu_aug(mu::AbstractVector{T}, Sigma::AbstractMatrix{T}) where T <: Real
    # First part: original mean vector mu
    first_part = mu
    
    # Second part: vectorize the sum of outer product mu*mu' and covariance Sigma
    # This implements vec(mu*mu' + Sigma) from Proposition 3.1
    second_part = vec(mu * mu' + Sigma)
    
    # Combine both parts into the augmented mean vector
    mu_aug = vcat(first_part, second_part)
    
    return mu_aug
end

"""
    compute_Phi_aug(mu::Real, Phi::Real)

Compute the augmented state transition matrix Φ̃ for scalar case.

# Arguments
- `mu::Real`: Mean parameter
- `Phi::Real`: State transition parameter

# Returns
- `Matrix`: Augmented state transition matrix [Phi 0.0; 2mu*Phi Phi²]

# Description
Implements Proposition 3.1 to compute the augmented state transition matrix 
Φ̃ = [Phi 0.0; (mu⊗Phi + Phi⊗mu), Phi⊗Phi] for the scalar case where mu and Phi are real numbers.
"""
function compute_Phi_aug(mu::Real, Phi::Real)
    # Prop. 3.1: Φ̃ = [Phi 0.0; (mu⊗Phi + Phi⊗mu), Phi⊗Phi]
    return [Phi 0.0; 2.0*mu*Phi Phi^2]
end

"""
    compute_Phi_aug(mu::AbstractVector{T}, Phi::AbstractMatrix{T}) where T <: Real

Compute the augmented state transition matrix Φ̃ for multivariate case.

# Arguments
- `mu::AbstractVector{T}`: Mean vector of dimension N
- `Phi::AbstractMatrix{T}`: State transition matrix of dimension N×N

# Returns
- `Matrix{T}`: Augmented state transition matrix of dimension (N+N²)×(N+N²)

# Description
Implements Proposition 3.1 to compute the augmented state transition matrix 
Phi_aug = [Phi 0; (mu⊗Phi + Phi⊗mu), Phi⊗Phi] for the multivariate case where:
- Top left block is Phi
- Top right block is zero matrix
- Bottom left block implements mu⊗Phi + Phi⊗mu 
- Bottom right block implements Phi⊗Phi
"""
function compute_Phi_aug(mu::AbstractVector{T}, Phi::AbstractMatrix{T}) where T <: Real
    N = length(mu)
    P = N + N^2

    # Construct the top-left part (N x N)
    top_left = copy(Phi)

    # Construct the top-right part (N x N^2)
    top_right = zeros(eltype(Phi), N, N^2)

    # Construct the bottom part (N^2 x P)
    bottom = [
        let 
            (i, j) = divrem(row - 1, N) .+ 1
            if col <= N
                mu[j] * Phi[i,col] + mu[i] * Phi[j,col]
            else
                k, l = divrem(col - N - 1, N) .+ 1
                Phi[i,k] * Phi[j,l]
            end
        end
        for row in 1:N^2, col in 1:P
    ]

    # Combine all parts
    Phi_aug = [
        [top_left top_right]
        reshape(bottom, N^2, P)
    ]

    return Phi_aug
end
"""
    compute_L1(Sigma::AbstractMatrix{T1}, Lambda::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}

Compute the first auxiliary matrix L₁ for the multivariate case.

# Arguments
- `Sigma::AbstractMatrix{T1}`: State covariance matrix of dimension N×N
- `Lambda::AbstractMatrix{T2}`: Commutation matrix of dimension N²×N²

# Returns
- `Matrix`: The N³ × N  auxiliary matrix: L₁ = [Σ ⊗ (Iₙ² + Λₙ)] × [vec(Iₙ) ⊗ Iₙ]

# Description
Implements the first auxiliary matrix L₁ from Proposition 3.2 for computing the 
conditional covariance matrix in the multivariate case. This matrix helps capture
the interaction between the state covariance and the commutation matrix.
"""
function compute_L1(Sigma::AbstractMatrix{T1}, Lambda::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}
    N = size(Sigma, 1)
    return kron(Sigma, I + Lambda) * kron(vec(I(N)), I(N))
end

"""
    compute_L1(Sigma::Real, Lambda::AbstractMatrix{T}) where T <: Real

Compute the first auxiliary matrix L₁ for the scalar case.

# Arguments
- `Sigma::Real`: State variance parameter
- `Lambda::AbstractMatrix{T}`: Commutation matrix (reduces to scalar in this case)

# Returns
- `Real`: The auxiliary scalar L₁ = Σ(1 + Λ₁₁)

# Description
Implements the first auxiliary matrix L₁ from Proposition 3.2 for the scalar case.
This is a simplified version where the matrix operations reduce to scalar multiplication.
"""
function compute_L1(Sigma::Real, Lambda::AbstractMatrix{T}) where T <: Real
    return Sigma * (1.0 + Lambda[1])
end

"""
    compute_L2(Sigma::Real, Lambda::AbstractMatrix{T}) where T <: Real

Compute the second auxiliary matrix L₂ for the scalar case.

# Arguments
- `Sigma::Real`: State variance parameter
- `Lambda::AbstractMatrix{T}`: Commutation matrix (reduces to scalar in this case)

# Returns
- `Real`: The auxiliary scalar L₂ = (1 + Λ₁) Σ Λ₁

# Description
Implements the second auxiliary matrix L₂ from Proposition 3.2 for the scalar case.
This is a simplified version where the matrix operations reduce to scalar multiplication.
"""
function compute_L2(Sigma::Real, Lambda::AbstractMatrix{T}) where T <: Real
    return (1.0 + Lambda[1]) * Sigma * Lambda[1] 
end

"""
    compute_L2(Sigma::AbstractMatrix{T1}, Lambda::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}

Compute the second auxiliary matrix L₂ for the multivariate case.

# Arguments
- `Sigma::AbstractMatrix{T1}`: State covariance matrix of dimension N×N
- `Lambda::AbstractMatrix{T2}`: Commutation matrix of dimension N²×N²

# Returns
- `Matrix`: The N³ × N auxiliary matrix: L₂ = [(Iₙ² + Λₙ) ⊗ Σ][Iₙ ⊗ Λₙ][vec(Iₙ) ⊗ Iₙ]

# Description
Implements the second auxiliary matrix L₂ from Proposition 3.2 for computing the
conditional covariance matrix in the multivariate case. The distinct types T1 and T2
allow for automatic differentiation frameworks that may pass traced types.
"""
function compute_L2(Sigma::AbstractMatrix{T1}, Lambda::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}
    N = size(Sigma, 1)
    return kron(I + Lambda, Sigma) * kron(I(N), Lambda) * kron(vec(I(N)), I(N))
end

"""
    compute_L3(Sigma::Real, Lambda::AbstractMatrix{T}) where T <: Real

Compute the third auxiliary matrix L₃ for the scalar case.

# Arguments
- `Sigma::Real`: State variance parameter
- `Lambda::AbstractMatrix{T}`: Commutation matrix (reduces to scalar in this case)

# Returns
- `Real`: The auxiliary scalar L₃ = (1 + Λ₁)² Λ₁ Σ

# Description
Implements the third auxiliary matrix L₃ from Proposition 3.2 for the scalar case.
This is a simplified version where the matrix operations reduce to scalar multiplication.

"""
function compute_L3(Sigma::Real, Lambda::AbstractMatrix{T}) where T <: Real
    L3 = (1.0 + Lambda[1]) * (1.0 + Lambda[1]) * Lambda[1] * Sigma
    return L3
end

"""
    compute_L3(Sigma::AbstractMatrix{T1}, Lambda::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}

Compute the third auxiliary matrix L₃ for the multivariate case.

# Arguments
- `Sigma::AbstractMatrix{T1}`: State covariance matrix of dimension N×N
- `Lambda::AbstractMatrix{T2}`: Commutation matrix of dimension N²×N²

# Returns
- `Matrix`: The auxiliary matrix L₃ = [(Iₙ² + Λₙ) ⊗ (Iₙ² + Λₙ)][Iₙ ⊗ Λₙ ⊗ Iₙ][vec(Σ) ⊗ Iₙ²]

# Description
Implements the third auxiliary matrix L₃ from Proposition 3.2 for computing the
conditional covariance matrix in the multivariate case. The distinct types T1 and T2
allow for automatic differentiation frameworks that may pass traced types.
"""
function compute_L3(Sigma::AbstractMatrix{T1}, Lambda::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}
    N = size(Sigma, 1)
    L3 = kron(I + Lambda, I + Lambda) * kron(I(N), kron(Lambda, I(N))) * kron(vec(Sigma), I(N^2))
    return L3
end

"""
    compute_state_mean(mu::Real, Phi::Real)

Compute the unconditional mean for a univariate AR(1) process:
Xₜ = mu + Phi*Xₜ₋₁

For a stationary process (|Phi| < 1), the unconditional mean is mu/(1-Phi).
"""
function compute_state_mean(mu::Real, Phi::Real) 


    return mu / (1.0 - Phi) 

end

"""
    compute_state_mean(mu::AbstractVector{T}, Phi::AbstractMatrix{T}) where T <: Real

Compute the unconditional mean for a multivariate AR(1) process:
Xₜ = mu + Phi*Xₜ₋₁

For a stationary process (eigenvalues of Phi inside unit circle), 
the unconditional mean is (I - Phi)⁻¹mu.
"""
function compute_state_mean(mu::AbstractVector{T}, Phi::AbstractMatrix{T}) where T <: Real

    return (I - Phi)\mu
end

"""
    compute_state_cov(Phi::Real, Sigma::Real)

Compute the unconditional covariance for an AR(1) process:
Xₜ = mu + Phi*Xₜ₋₁

For a stationary process (|Phi| < 1), the unconditional covariance is Sigma/(1-Phi²).
"""     
function compute_state_cov(Phi::Real, Sigma::Real)
    return Sigma / (1.0 - Phi^2) 
end

"""
    compute_state_cov(Phi::AbstractMatrix{T}, Sigma::AbstractMatrix{T}) where T <: Real

Compute the unconditional covariance for a VAR(1) process:
Xₜ = mu + Phi*Xₜ₋₁

For a stationary process (eigenvalues of Phi inside unit circle), 
the unconditional covariance is (I - kron(Phi, Phi))⁻¹Sigma.
"""
function compute_state_cov(Phi::AbstractMatrix{T}, Sigma::AbstractMatrix{T}) where T <: Real
    N = size(Sigma, 1)
    return reshape((I - kron(Phi, Phi))\vec(Sigma), (N, N))
end

"""
    compute_cond_cov_state_aug(Z::AbstractVector{T}, Sigma::Real, mu::Real, Phi::Real) where T <: Real

Compute the conditional covariance matrix of the augmented state vector Zₜ = [Xₜ, XₜXₜ'] for a univariate AR(1) process.

# Arguments
- `Z::AbstractVector{T}`: Current augmented state vector [X, XX']
- `Sigma::Real`: State noise variance
- `mu::Real`: Constant term in state equation
- `Phi::Real`: AR(1) coefficient

# Returns
- `Matrix{T}`: 2×2 conditional covariance matrix of the augmented state
"""
function compute_cond_cov_state_aug(Z::AbstractVector{T}, Sigma::Real, mu::Real, Phi::Real) where T <: Real
    
    u = mu + Phi * Z[1]
    Sigma_aug_11 = Sigma
    Sigma_aug_12 = 2.0 * Sigma * u
    Sigma_aug_21 = 2.0 * Sigma * u
    Sigma_aug_22 = 4.0 * u^2 * Sigma + 2.0 * Sigma^2

    return [Sigma_aug_11 Sigma_aug_12; Sigma_aug_21 Sigma_aug_22]

end


"""
    compute_cond_cov_state_aug(Z::AbstractVector{T1}, L1::AbstractMatrix{T2}, L2::AbstractMatrix{T3}, L3::AbstractMatrix{T4}, 
                          Lambda::AbstractMatrix{T5}, Sigma::AbstractMatrix{T6}, mu::AbstractVector{T7}, 
                          Phi_aug::AbstractMatrix{T8}) where {T1,T2,T3,T4,T5,T6,T7,T8 <: Real}

Compute the conditional covariance matrix of the augmented state vector Zₜ = [Xₜ, vec(XₜXₜ')] for a multivariate VAR(1) process.

# Arguments
- `Z::AbstractVector{T1}`: Current augmented state vector [X, vec(XX')]
- `L1,L2,L3::AbstractMatrix`: Auxiliary matrices for covariance computation
- `Lambda::AbstractMatrix`: Commutation matrix
- `Sigma::AbstractMatrix`: State noise covariance matrix
- `mu::AbstractVector`: Constant term in state equation  
- `Phi_aug::AbstractMatrix`: Augmented transition matrix

# Returns
- `Matrix`: (N+N²)×(N+N²) conditional covariance matrix of the augmented state
"""
function compute_cond_cov_state_aug(Z::AbstractVector{T1}, L1::AbstractMatrix{T2},
    L2::AbstractMatrix{T3}, L3::AbstractMatrix{T4}, Lambda::AbstractMatrix{T5},
    Sigma::AbstractMatrix{T6}, mu::AbstractVector{T7},
    Phi_aug::AbstractMatrix{T8}) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real,
        T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real}

    N = length(mu)
    mu_Phi_z = (mu + Phi_aug[1:N,:] * Z)
    Sigma_11 = Sigma
    Sigma_12 = reshape(L1 * mu_Phi_z, (N^2, N))'
    Sigma_21 = Sigma_12'
    Sigma_22 = reshape(L3 * (kron(mu, mu) + Phi_aug[N+1:end,:] * Z) +
                    kron(I(N^2), I + Lambda) * vec(kron(Sigma, Sigma)), (N^2, N^2))

    return [Sigma_11 Sigma_12; Sigma_21 Sigma_22]
end

"""
    compute_cond_cov_state(X::AbstractVector{T}, model::QKModel{T,T2}) where {T <: Real, T2 <: Real}

Compute the conditional covariance matrix of the state vector Xₜ for a VAR(1) process.

# Arguments
- `X::AbstractVector{T}`: Current state vector
- `model::QKModel{T,T2}`: Model parameters

# Returns
- `Matrix`: N×N conditional covariance matrix of the state
"""
function compute_cond_cov_state(X::AbstractVector{T}, model::QKModel{T,T2}) where {T <: Real, T2 <: Real}
    @unpack mu, Phi, Sigma, N = model.state
    @unpack Lambda = model.aug_state
    N = length(mu)
    Gamma = compute_Gamma_tm1(X, mu, Phi)
    Sigma_aug_11 = Sigma
    Sigma_aug_12 = Sigma * Gamma' 
    Sigma_aug_21 = Gamma * Sigma
    Sigma_aug_22 = Gamma * Sigma * Gamma' + (I + Lambda) * kron(Sigma, Sigma)
    return [Sigma_aug_11 Sigma_aug_12; Sigma_aug_21 Sigma_aug_22]

end

"""
    compute_aug_state_uncond_cov(Z::AbstractVector{T}, Sigma::Real, mu::Real, Phi::Real, Phi_aug::AbstractMatrix{T}) where T <: Real

Compute the unconditional covariance matrix of the augmented state vector Zₜ = [Xₜ, vec(XₜXₜ')] for a VAR(1) process.

# Arguments
- `Z::AbstractVector{T}`: Current augmented state vector
- `Sigma::Real`: State covariance matrix
- `mu::Real`: Constant term in state equation
- `Phi::Real`: State transition matrix
- `Phi_aug::AbstractMatrix{T}`: Augmented state transition matrix

# Returns
- `Matrix{T}`: P×P unconditional covariance matrix of the augmented state vector
"""
function compute_aug_state_uncond_cov(Z::AbstractVector{T}, Sigma::Real, mu::Real, Phi::Real, Phi_aug::AbstractMatrix{T}) where T <: Real

    P = size(Phi_aug, 1)
    return reshape((I - kron(Phi_aug, Phi_aug)) \ 
        vec(compute_cond_cov_state_aug(Z, Sigma, mu, Phi)), (P, P))
end

"""
    compute_aug_state_uncond_cov(aug_mean::AbstractVector{T1}, L1::AbstractMatrix{T2}, L2::AbstractMatrix{T3},
        L3::AbstractMatrix{T4}, Lambda::AbstractMatrix{T5}, Sigma::AbstractMatrix{T6},
        mu::AbstractVector{T7}, Phi_aug::AbstractMatrix{T8}) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real}

Compute the unconditional covariance matrix of the augmented state vector Zₜ = [Xₜ, vec(XₜXₜ')] for a VAR(1) process.

# Arguments
- `aug_mean::AbstractVector{T1}`: Unconditional mean of augmented state vector
- `L1::AbstractMatrix{T2}`: First auxiliary matrix for covariance computation
- `L2::AbstractMatrix{T3}`: Second auxiliary matrix for covariance computation
- `L3::AbstractMatrix{T4}`: Third auxiliary matrix for covariance computation
- `Lambda::AbstractMatrix{T5}`: Commutation matrix
- `Sigma::AbstractMatrix{T6}`: State covariance matrix
- `mu::AbstractVector{T7}`: Constant term in state equation
- `Phi_aug::AbstractMatrix{T8}`: Augmented state transition matrix

# Returns
- `Matrix`: P×P unconditional covariance matrix of the augmented state vector
"""
function compute_aug_state_uncond_cov(aug_mean::AbstractVector{T1}, L1::AbstractMatrix{T2},
    L2::AbstractMatrix{T3}, L3::AbstractMatrix{T4}, Lambda::AbstractMatrix{T5},
    Sigma::AbstractMatrix{T6}, mu::AbstractVector{T7},
    Phi_aug::AbstractMatrix{T8}) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real,
    T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real}

    P = size(Phi_aug, 1)
    return reshape((I - kron(Phi_aug, Phi_aug)) \ 
        vec(compute_cond_cov_state_aug(aug_mean, L1, L2, L3, Lambda, Sigma, mu, Phi_aug)), (P, P))
end

"""
    compute_B_aug(B::Real, C::Real)

Compute the augmented observation matrix B̃ for scalar case.

# Arguments
- `B::Real`: Linear observation coefficient
- `C::Real`: Quadratic observation coefficient

# Returns
- `Vector{Real}`: Augmented observation matrix B̃ = [B C] that maps the augmented state Zₜ = [Xₜ, XₜXₜ'] 
  into the observation space
"""
function compute_B_aug(B::Real, C::Real)
    return [B C]
end

"""
    compute_B_aug(B::AbstractMatrix{T}, C::Vector{<:AbstractMatrix{T}}) where T <: Real

Compute the augmented observation matrix B̃ that maps the augmented state-space Zₜ = [Xₜ, vec(XₜXₜ')] 
into the observation space.

# Arguments
- `B::AbstractMatrix{T}`: M×N linear observation matrix
- `C::Vector{<:AbstractMatrix{T}}`: Vector of M N×N quadratic observation matrices

# Returns
- `Matrix{T}`: M×(N+N²) augmented observation matrix B̃ = [B C̃] where C̃ contains the vectorized 
  quadratic observation matrices

# Throws
- `DimensionMismatch`: If any matrix in C is not N×N
"""
function compute_B_aug(B::AbstractMatrix{T}, C::Vector{<:AbstractMatrix{T}}) where T <: Real
    M, N = size(B)
    
    # Ensure all matrices in C have the correct size
    if any(size.(C) .!= Ref((N, N)))
        throw(DimensionMismatch("All matrices in C must be N x N"))
    end
    
    # Create the C part of B̃ using array comprehension
    #C_part = [C[i][j, k] for i in 1:M, j in 1:N, k in 1:N]
    C_part = reduce(vcat, [vec(C[i])' for i in 1:M])
    
    # Combine B and C_part
    B_aug = hcat(B, C_part)
    
    return B_aug
end

"""
    compute_Sigma_ttm1!(Sigma_ttm1::AbstractArray{Real, 3}, Z_tt::AbstractMatrix{T}, 
                    model::QKModel{T,T2}, t::Int) where {T <: Real, T2 <: Real}

Compute and store the conditional covariance matrix of the augmented state vector at time t 
given information up to time t-1.

# Arguments
- `Sigma_ttm1::AbstractArray{Real, 3}`: Array to store the computed covariance matrix
- `Z_tt::AbstractMatrix{T}`: Matrix of filtered state estimates
- `model::QKModel{T,T2}`: Model parameters
- `t::Int`: Time index

# Description
Updates the slice Sigma_ttm1[:,:,t] with the conditional covariance computed using the 
augmented state vector at time t and model parameters.
"""
function compute_Sigma_ttm1!(Sigma_ttm1::AbstractArray{T, 3}, Z_tt::AbstractMatrix{T},
    model::QKModel{T,T2}, t::Int) where {T <: Real, T2 <: Real}
    @unpack mu, Sigma, Phi, N = model.state
    @unpack Lambda, L1, L2, L3, P, Phi_aug = model.aug_state
    Sigma_ttm1[:, :, t] .= compute_cond_cov_state_aug(Z_tt[:,t], L1, L2, L3, Lambda, Sigma, mu, Phi_aug)
end

"""
    compute_Sigma_ttm1(Z_tt::AbstractVector{T}, model::QKModel{T,T2}, t::Int) where {T <: Real, T2 <: Real}

Compute the conditional covariance matrix of the augmented state vector at time t given 
information up to time t-1.

# Arguments
- `Z_tt::AbstractVector{T}`: Vector of filtered state estimates
- `model::QKModel{T,T2}`: Model parameters  
- `t::Int`: Time index

# Returns
- `Matrix{T}`: The conditional covariance matrix

# Description
Returns the conditional covariance computed using the augmented state vector at time t
and model parameters.
"""
function compute_Sigma_ttm1(Z_tt::AbstractVector{T}, model::QKModel{T,T2}) where {T <: Real, T2 <: Real}

    @unpack mu, Sigma, Phi, N = model.state
    @unpack Lambda, L1, L2, L3, P, Phi_aug = model.aug_state

    return compute_cond_cov_state_aug(Z_tt, L1, L2, L3, Lambda, Sigma, mu, Phi_aug)
end