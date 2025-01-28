"""
    File: augmented_moments.jl

Utilities for computing augmented means, augmented transition matrices, and auxiliary 
matrices in a Quadratic Kalman Filter (QKF) or similar state-space model.

# Overview

This file implements several scalar and multivariate routines inspired by **Proposition 3.1** 
and **Proposition 3.2** in the QKF context:

1. **Augmented Mean and Transition**  
   - `compute_μ̃(μ, Σ)`: Constructs the augmented mean vector, concatenating `[μ; vec(μμ' + Σ)]`.  
   - `compute_Φ̃(μ, Φ)`: Builds the augmented state transition matrix for both scalar and 
     multivariate cases. 

2. **Unconditional Moments**  
   - `compute_μᵘ(μ, Φ)`: Unconditional mean of a stationary (V)AR(1) process.  
   - `compute_Σᵘ(Φ, Σ)`: Unconditional covariance for a stable (V)AR(1) process.

3. **Auxiliary Matrices**  
   - `compute_L1`, `compute_L2`, `compute_L3`: Matrix constructions used when updating 
     conditional covariance in the QKF. Each has both a **scalar** and a **multivariate** 
     version. In the scalar case, the matrix operations reduce to simple products; in 
     the multivariate case, Kronecker products and commutation matrices are used.

4. **Conditional Covariances**
   - `compute_Σ̃condZ`: Computes conditional covariance of augmented state vector for both scalar and multivariate cases
   - `compute_Σ̃condX`: Computes conditional covariance of state vector
   - `compute_Σₜₜ₋₁`: Computes conditional covariance at time t given t-1 information
   - `compute_Σₜₜ₋₁!`: In-place version that updates a pre-allocated array

5. **Observation Matrix**
   - `compute_B̃`: Constructs augmented observation matrix combining linear and quadratic terms

# Usage

Typically, these functions are called within higher-level filtering routines.
"""

"""
    compute_μ̃(μ::Real, Σ::Real)

Compute the augmented mean vector μ̃ for scalar case.

# Arguments
- `μ::Real`: Mean parameter
- `Σ::Real`: Variance parameter

# Returns
- `Vector`: Augmented mean vector [μ, μ² + Σ]

# Description
Implements Proposition 3.1 to compute the augmented mean vector μ̃ = [μ, vec(μμ' + Σ)]
for the scalar case where μ and Σ are real numbers.
"""
function compute_μ̃(μ::Real, Σ::Real) 
    return [μ; μ^2 + Σ]
end

"""
    compute_μ̃(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where T <: Real

Compute the augmented mean vector μ̃ for multivariate case.

# Arguments
- `μ::AbstractVector{T}`: Mean vector of dimension N
- `Σ::AbstractMatrix{T}`: Covariance matrix of dimension N×N

# Returns
- `Vector{T}`: Augmented mean vector [μ; vec(μμ' + Σ)] of dimension N + N²

# Description
Implements Proposition 3.1 to compute the augmented mean vector μ̃ = [μ; vec(μμ' + Σ)]
for the multivariate case. The result combines:
1. The original mean vector μ
2. The vectorized sum of the outer product μμ' and covariance matrix Σ
"""
function compute_μ̃(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where T <: Real
    # First part: original mean vector μ
    first_part = μ
    
    # Second part: vectorize the sum of outer product μμ' and covariance Σ
    # This implements vec(μμ' + Σ) from Proposition 3.1
    second_part = vec(μ * μ' + Σ)
    
    # Combine both parts into the augmented mean vector
    μ̃ = vcat(first_part, second_part)
    
    return μ̃
end

"""
    compute_Φ̃(μ::Real, Φ::Real)

Compute the augmented state transition matrix Φ̃ for scalar case.

# Arguments
- `μ::Real`: Mean parameter
- `Φ::Real`: State transition parameter

# Returns
- `Matrix`: Augmented state transition matrix [Φ 0.0; 2μΦ Φ²]

# Description
Implements Proposition 3.1 to compute the augmented state transition matrix 
Φ̃ = [Φ 0.0; (μ⊗Φ + Φ⊗μ), Φ⊗Φ] for the scalar case where μ and Φ are real numbers.
"""
function compute_Φ̃(μ::Real, Φ::Real)
    # Prop. 3.1: Φ̃ = [Φ 0.0; (μ⊗Φ + Φ⊗μ), Φ⊗Φ]
    return [Φ 0.0; 2.0*μ*Φ Φ^2]
end

"""
    compute_Φ̃(μ::AbstractVector{T}, Φ::AbstractMatrix{T}) where T <: Real

Compute the augmented state transition matrix Φ̃ for multivariate case.

# Arguments
- `μ::AbstractVector{T}`: Mean vector of dimension N
- `Φ::AbstractMatrix{T}`: State transition matrix of dimension N×N

# Returns
- `Matrix{T}`: Augmented state transition matrix of dimension (N+N²)×(N+N²)

# Description
Implements Proposition 3.1 to compute the augmented state transition matrix 
Φ̃ = [Φ 0; (μ⊗Φ + Φ⊗μ), Φ⊗Φ] for the multivariate case where:
- Top left block is Φ
- Top right block is zero matrix
- Bottom left block implements μ⊗Φ + Φ⊗μ 
- Bottom right block implements Φ⊗Φ
"""
function compute_Φ̃(μ::AbstractVector{T}, Φ::AbstractMatrix{T}) where T <: Real
    N = length(μ)
    P = N + N^2

    # Construct the top-left part (N x N)
    top_left = copy(Φ)

    # Construct the top-right part (N x N^2)
    top_right = zeros(eltype(Φ), N, N^2)

    # Construct the bottom part (N^2 x P)
    bottom = [
        let 
            (i, j) = divrem(row - 1, N) .+ 1
            if col <= N
                μ[j] * Φ[i,col] + μ[i] * Φ[j,col]
            else
                k, l = divrem(col - N - 1, N) .+ 1
                Φ[i,k] * Φ[j,l]
            end
        end
        for row in 1:N^2, col in 1:P
    ]

    # Combine all parts
    Φ̃ = [
        [top_left top_right]
        reshape(bottom, N^2, P)
    ]

    return Φ̃
end

"""
    compute_L1(Σ::AbstractMatrix{T1}, Λ::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}

Compute the first auxiliary matrix L₁ for the multivariate case.

# Arguments
- `Σ::AbstractMatrix{T1}`: State covariance matrix of dimension N×N
- `Λ::AbstractMatrix{T2}`: Commutation matrix of dimension N²×N²

# Returns
- `Matrix`: The auxiliary matrix L₁ = [Σ ⊗ (Iₙ² + Λₙ)] ⊗ [vec(Iₙ) ⊗ Iₙ]

# Description
Implements the first auxiliary matrix L₁ from Proposition 3.2 for computing the 
conditional covariance matrix in the multivariate case. This matrix helps capture
the interaction between the state covariance and the commutation matrix.
"""
function compute_L1(Σ::AbstractMatrix{T1}, Λ::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}
    N = size(Σ, 1)
    return kron(Σ, I + Λ) * kron(vec(I(N)), I(N))
end

"""
    compute_L1(Σ::Real, Λ::AbstractMatrix{T}) where T <: Real

Compute the first auxiliary matrix L₁ for the scalar case.

# Arguments
- `Σ::Real`: State variance parameter
- `Λ::AbstractMatrix{T}`: Commutation matrix (reduces to scalar in this case)

# Returns
- `Real`: The auxiliary scalar L₁ = Σ(1 + Λ₁₁)

# Description
Implements the first auxiliary matrix L₁ from Proposition 3.2 for the scalar case.
This is a simplified version where the matrix operations reduce to scalar multiplication.
"""
function compute_L1(Σ::Real, Λ::AbstractMatrix{T}) where T <: Real
    return Σ * (1.0 + Λ[1])
end

"""
    compute_L2(Σ::Real, Λ::AbstractMatrix{T}) where T <: Real

Compute the second auxiliary matrix L₂ for the scalar case.

# Arguments
- `Σ::Real`: State variance parameter
- `Λ::AbstractMatrix{T}`: Commutation matrix (reduces to scalar in this case)

# Returns
- `Real`: The auxiliary scalar L₂ = (1 + Λ₁) Σ Λ₁

# Description
Implements the second auxiliary matrix L₂ from Proposition 3.2 for the scalar case.
This is a simplified version where the matrix operations reduce to scalar multiplication.
"""
function compute_L2(Σ::Real, Λ::AbstractMatrix{T}) where T <: Real
    return (1.0 + Λ[1]) * Σ * Λ[1] 
end

"""
    compute_L2(Σ::AbstractMatrix{T1}, Λ::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}

Compute the second auxiliary matrix L₂ for the multivariate case.

# Arguments
- `Σ::AbstractMatrix{T1}`: State covariance matrix of dimension N×N
- `Λ::AbstractMatrix{T2}`: Commutation matrix of dimension N²×N²

# Returns
- `Matrix`: The auxiliary matrix L₂ = [(Iₙ² + Λₙ) ⊗ Σ][Iₙ ⊗ Λₙ][vec(Iₙ) ⊗ Iₙ]

# Description
Implements the second auxiliary matrix L₂ from Proposition 3.2 for computing the
conditional covariance matrix in the multivariate case. The distinct types T1 and T2
allow for automatic differentiation frameworks that may pass traced types.
"""
function compute_L2(Σ::AbstractMatrix{T1}, Λ::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}
    N = size(Σ, 1)
    return kron(I + Λ, Σ) * kron(I(N), Λ) * kron(vec(I(N)), I(N))
end

"""
    compute_L3(Σ::Real, Λ::AbstractMatrix{T}) where T <: Real

Compute the third auxiliary matrix L₃ for the scalar case.

# Arguments
- `Σ::Real`: State variance parameter
- `Λ::AbstractMatrix{T}`: Commutation matrix (reduces to scalar in this case)

# Returns
- `Real`: The auxiliary scalar L₃ = (1 + Λ₁)² Λ₁ Σ

# Description
Implements the third auxiliary matrix L₃ from Proposition 3.2 for the scalar case.
This is a simplified version where the matrix operations reduce to scalar multiplication.
"""
function compute_L3(Σ::Real, Λ::AbstractMatrix{T}) where T <: Real
    L3 = (1.0 + Λ[1]) * (1.0 + Λ[1]) * Λ[1] * Σ
    return L3
end

"""
    compute_L3(Σ::AbstractMatrix{T1}, Λ::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}

Compute the third auxiliary matrix L₃ for the multivariate case.

# Arguments
- `Σ::AbstractMatrix{T1}`: State covariance matrix of dimension N×N
- `Λ::AbstractMatrix{T2}`: Commutation matrix of dimension N²×N²

# Returns
- `Matrix`: The auxiliary matrix L₃ = [(Iₙ² + Λₙ) ⊗ (Iₙ² + Λₙ)][Iₙ ⊗ Λₙ ⊗ Iₙ][vec(Σ) ⊗ Iₙ²]

# Description
Implements the third auxiliary matrix L₃ from Proposition 3.2 for computing the
conditional covariance matrix in the multivariate case. The distinct types T1 and T2
allow for automatic differentiation frameworks that may pass traced types.
"""
function compute_L3(Σ::AbstractMatrix{T1}, Λ::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}
    N = size(Σ, 1)
    L3 = kron(I + Λ, I + Λ) * kron(I(N), kron(Λ, I(N))) * kron(vec(Σ), I(N^2))
    return L3
end

"""
    compute_μᵘ(μ::Real, Φ::Real)

Compute the unconditional mean for a univariate AR(1) process:
Xₜ = μ + ΦXₜ₋₁

For a stationary process (|Φ| < 1), the unconditional mean is μ/(1-Φ).
"""
function compute_μᵘ(μ::Real, Φ::Real) 

    return μ / (1.0 - Φ) 

end

"""
    compute_μᵘ(μ::AbstractVector{T}, Φ::AbstractMatrix{T}) where T <: Real

Compute the unconditional mean for a multivariate AR(1) process:
Xₜ = μ + ΦXₜ₋₁

For a stationary process (eigenvalues of Φ inside unit circle), 
the unconditional mean is (I - Φ)⁻¹μ.
"""
function compute_μᵘ(μ::AbstractVector{T}, Φ::AbstractMatrix{T}) where T <: Real

    return (I - Φ)\μ
end

"""
    compute_Σᵘ(Φ::Real, Σ::Real)

Compute the unconditional covariance for an AR(1) process:
Xₜ = μ + ΦXₜ₋₁

For a stationary process (|Φ| < 1), the unconditional covariance is Σ/(1-Φ²).
"""     
function compute_Σᵘ(Φ::Real, Σ::Real)
    return Σ / (1.0 - Φ^2) 
end

"""
    compute_Σᵘ(Φ::AbstractMatrix{T}, Σ::AbstractMatrix{T}) where T <: Real

Compute the unconditional covariance for a VAR(1) process:
Xₜ = μ + ΦXₜ₋₁

For a stationary process (eigenvalues of Φ inside unit circle), 
the unconditional covariance is (I - kron(Φ, Φ))⁻¹Σ.
"""
function compute_Σᵘ(Φ::AbstractMatrix{T}, Σ::AbstractMatrix{T}) where T <: Real
    N = size(Σ, 1)
    return reshape((I - kron(Φ, Φ))\vec(Σ), (N, N))
end

"""
    compute_Σ̃condZ(Z::AbstractVector{T}, Σ::Real, μ::Real, Φ::Real) where T <: Real

Compute the conditional covariance matrix of the augmented state vector Zₜ = [Xₜ, XₜXₜ'] for a univariate AR(1) process.

# Arguments
- `Z::AbstractVector{T}`: Current augmented state vector [X, XX']
- `Σ::Real`: State noise variance
- `μ::Real`: Constant term in state equation
- `Φ::Real`: AR(1) coefficient

# Returns
- `Matrix{T}`: 2×2 conditional covariance matrix of the augmented state
"""
function compute_Σ̃condZ(Z::AbstractVector{T}, Σ::Real, μ::Real, Φ::Real) where T <: Real
    
    u = μ + Φ * Z[1]
    Σ̃_11 = Σ
    Σ̃_12 = 2.0 * Σ * u
    Σ̃_21 = 2.0 * Σ * u
    Σ̃_22 = 4.0 * u^2 * Σ + 2.0 * Σ^2

    return [Σ̃_11 Σ̃_12; Σ̃_21 Σ̃_22]

end


"""
    compute_Σ̃condZ(Z::AbstractVector{T1}, L1::AbstractMatrix{T2}, L2::AbstractMatrix{T3}, L3::AbstractMatrix{T4}, 
                   Λ::AbstractMatrix{T5}, Σ::AbstractMatrix{T6}, μ::AbstractVector{T7}, Φ̃::AbstractMatrix{T8}) where {T1,T2,T3,T4,T5,T6,T7,T8 <: Real}

Compute the conditional covariance matrix of the augmented state vector Zₜ = [Xₜ, vec(XₜXₜ')] for a multivariate VAR(1) process.

# Arguments
- `Z::AbstractVector{T1}`: Current augmented state vector [X, vec(XX')]
- `L1,L2,L3::AbstractMatrix`: Auxiliary matrices for covariance computation
- `Λ::AbstractMatrix`: Commutation matrix
- `Σ::AbstractMatrix`: State noise covariance matrix
- `μ::AbstractVector`: Constant term in state equation  
- `Φ̃::AbstractMatrix`: Augmented transition matrix

# Returns
- `Matrix`: (N+N²)×(N+N²) conditional covariance matrix of the augmented state
"""
function compute_Σ̃condZ(Z::AbstractVector{T1} ,L1::AbstractMatrix{T2},
    L2::AbstractMatrix{T3}, L3::AbstractMatrix{T4}, Λ::AbstractMatrix{T5},
    Σ::AbstractMatrix{T6}, μ::AbstractVector{T7}, Φ̃::AbstractMatrix{T8}) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real}
    N = length(μ)
    μΦz = (μ + Φ̃[1:N,:] * Z)
    Σ̃_11 = Σ
    Σ̃_12 = reshape(L1 * μΦz, (N, N^2))
    Σ̃_21 = reshape(L2 * μΦz, (N^2, N))
    Σ̃_22 = reshape(L3 * (kron(μ, μ) + Φ̃[N+1:end,:] * Z) +
                    kron(I(N^2), I + Λ) * vec(kron(Σ, Σ)), (N^2, N^2))

    return [Σ̃_11 Σ̃_12; Σ̃_21 Σ̃_22]

end

"""
    compute_Σ̃condX(X::AbstractVector{T}, params::QKParams{T,T2}) where {T <: Real, T2 <: Real}

Compute the conditional covariance matrix of the state vector Xₜ for a VAR(1) process.

# Arguments
- `X::AbstractVector{T}`: Current state vector
- `params::QKParams{T,T2}`: Parameters of the VAR(1) process

# Returns
- `Matrix`: N×N conditional covariance matrix of the state
"""
function compute_Σ̃condX(X::AbstractVector{T}, params::QKParams{T,T2}) where {T <: Real, T2 <: Real}
    @unpack μ, Φ, Σ, Λ, N = params
    N = length(μ)
    Γ = compute_Γₜ₋₁_old(X, μ, Φ)
    Σ̃_11 = Σ
    Σ̃_12 = Σ * Γ' 
    Σ̃_21 = Γ * Σ
    Σ̃_22 = Γ * Σ * Γ' + (I + Λ) * kron(Σ, Σ)
    return [Σ̃_11 Σ̃_12; Σ̃_21 Σ̃_22]

end

"""
    compute_Γₜ₋₁(Z::AbstractVector{T}, μ::AbstractVector{T}, Φ::AbstractMatrix{T}) where T <: Real

Compute the matrix Γₜ₋₁ for a VAR(1) process.

# Arguments
- `Z::AbstractVector{T}`: Current state vector
- `μ::AbstractVector{T}`: Constant term in state equation
- `Φ::AbstractMatrix{T}`: Transition matrix

# Returns
- `Matrix`: N×N matrix Γₜ₋₁
"""
function compute_Γₜ₋₁(Z::AbstractVector{T}, μ::AbstractVector{T},
    Φ::AbstractMatrix{T}) where T <: Real
    N = length(μ)
    tmp = μ + Φ * Z
    return kron(I(N), tmp) + kron(tmp, I(N)) 
end

"""
    compute_Σ̃ᵘ(Z::AbstractVector{T}, Σ::Real, μ::Real, Φ::Real, Φ̃::AbstractMatrix{T}) where T <: Real

Compute the unconditional covariance matrix of the augmented state vector Zₜ = [Xₜ, vec(XₜXₜ')] for a VAR(1) process.

# Arguments
- `Z::AbstractVector{T}`: Current augmented state vector
- `Σ::Real`: State covariance matrix
- `μ::Real`: Constant term in state equation
- `Φ::Real`: State transition matrix
- `Φ̃::AbstractMatrix{T}`: Augmented state transition matrix

# Returns
- `Matrix{T}`: P×P unconditional covariance matrix of the augmented state vector
"""
function compute_Σ̃ᵘ(Z::AbstractVector{T}, Σ::Real, μ::Real, Φ::Real, Φ̃::AbstractMatrix{T}) where T <: Real

    P = size(Φ̃, 1)
    return reshape((I - kron(Φ̃, Φ̃)) \ 
        vec(compute_Σ̃condZ(Z, Σ, μ, Φ)), (P, P))
end

"""
    compute_Σ̃ᵘ(μ̃ᵘ::AbstractVector{T1}, L1::AbstractMatrix{T2}, L2::AbstractMatrix{T3},
        L3::AbstractMatrix{T4}, Λ::AbstractMatrix{T5}, Σ::AbstractMatrix{T6},
        μ::AbstractVector{T7}, Φ̃::AbstractMatrix{T8}) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real}

Compute the unconditional covariance matrix of the augmented state vector Zₜ = [Xₜ, vec(XₜXₜ')] for a VAR(1) process.

# Arguments
- `μ̃ᵘ::AbstractVector{T1}`: Unconditional mean of augmented state vector
- `L1::AbstractMatrix{T2}`: First auxiliary matrix for covariance computation
- `L2::AbstractMatrix{T3}`: Second auxiliary matrix for covariance computation
- `L3::AbstractMatrix{T4}`: Third auxiliary matrix for covariance computation
- `Λ::AbstractMatrix{T5}`: Commutation matrix
- `Σ::AbstractMatrix{T6}`: State covariance matrix
- `μ::AbstractVector{T7}`: Constant term in state equation
- `Φ̃::AbstractMatrix{T8}`: Augmented state transition matrix

# Returns
- `Matrix`: P×P unconditional covariance matrix of the augmented state vector
"""
function compute_Σ̃ᵘ(μ̃ᵘ::AbstractVector{T1}, L1::AbstractMatrix{T2}, L2::AbstractMatrix{T3},
    L3::AbstractMatrix{T4}, Λ::AbstractMatrix{T5}, Σ::AbstractMatrix{T6},
    μ::AbstractVector{T7}, Φ̃::AbstractMatrix{T8}) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real}

    P = size(Φ̃, 1)
    return reshape((I - kron(Φ̃, Φ̃)) \ vec(compute_Σ̃condZ(μ̃ᵘ, L1, L2, L3, Λ, Σ, μ, Φ̃)),
                    (P, P))
end

"""
    compute_B̃(B::Real, C::Real)

Compute the augmented observation matrix B̃ for scalar case.

# Arguments
- `B::Real`: Linear observation coefficient
- `C::Real`: Quadratic observation coefficient

# Returns
- `Vector{Real}`: Augmented observation matrix B̃ = [B C] that maps the augmented state Zₜ = [Xₜ, XₜXₜ'] 
  into the observation space
"""
function compute_B̃(B::Real, C::Real)
    return [B C]
end

"""
    compute_B̃(B::AbstractMatrix{T}, C::Vector{<:AbstractMatrix{T}}) where T <: Real

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
function compute_B̃(B::AbstractMatrix{T}, C::Vector{<:AbstractMatrix{T}}) where T <: Real
    M, N = size(B)
    
    # Ensure all matrices in C have the correct size
    if any(size.(C) .!= Ref((N, N)))
        throw(DimensionMismatch("All matrices in C must be N x N"))
    end
    
    # Create the C part of B̃ using array comprehension
    C_part = [C[i][j, k] for i in 1:M, j in 1:N, k in 1:N]
    
    # Reshape C_part to be M x N^2
    C_part = reshape(C_part, M, N^2)
    
    # Combine B and C_part
    B̃ = hcat(B, C_part)
    
    return B̃
end

"""
    compute_Σₜₜ₋₁!(Σₜₜ₋₁::AbstractArray{Real, 3}, Zₜₜ::AbstractMatrix{T}, 
                    params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

Compute and store the conditional covariance matrix of the augmented state vector at time t 
given information up to time t-1.

# Arguments
- `Σₜₜ₋₁::AbstractArray{Real, 3}`: Array to store the computed covariance matrix
- `Zₜₜ::AbstractMatrix{T}`: Matrix of filtered state estimates
- `params::QKParams{T,T2}`: Model parameters
- `t::Int`: Time index

# Description
Updates the slice Σₜₜ₋₁[:,:,t] with the conditional covariance computed using the 
augmented state vector at time t and model parameters.
"""
function compute_Σₜₜ₋₁!(Σₜₜ₋₁::AbstractArray{Real, 3}, Zₜₜ::AbstractMatrix{T},
    params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    @unpack Λ, Σ, μ , Φ, N,L1, L2, L3, Λ, Σ, μ, Φ̃ = params
    Σₜₜ₋₁[:, :, t] .= compute_Σ̃condZ(Zₜₜ[:,t], L1, L2, L3, Λ, Σ, μ, Φ̃)
end

"""
    compute_Σₜₜ₋₁(Zₜₜ::AbstractVector{T}, params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

Compute the conditional covariance matrix of the augmented state vector at time t given 
information up to time t-1.

# Arguments
- `Zₜₜ::AbstractVector{T}`: Vector of filtered state estimates
- `params::QKParams{T,T2}`: Model parameters  
- `t::Int`: Time index

# Returns
- `Matrix{T}`: The conditional covariance matrix

# Description
Returns the conditional covariance computed using the augmented state vector at time t
and model parameters.
"""
function compute_Σₜₜ₋₁(Zₜₜ::AbstractVector{T}, params::QKParams{T,T2},t::Int) where {T <: Real, T2 <: Real}

    @unpack Λ, Σ, μ, Φ, N, L1, L2, L3, Λ, Σ, μ, Φ̃ = params

    return compute_Σ̃condZ(Zₜₜ, L1, L2, L3, Λ, Σ, μ, Φ̃)
end

export compute_μ̃, compute_Φ̃, compute_B̃, compute_μᵘ, compute_Σᵘ,
       compute_L1, compute_L2, compute_L3, compute_Σ̃condZ, compute_Σ̃condX, compute_Σ̃ᵘ,
       compute_Σₜₜ₋₁, compute_Σₜₜ₋₁!, compute_B̃