"""
    File: parameters.jl

This file defines the `QKParams` type and its constructor for the **Quadratic Kalman Filter** (QKF). 

# Overview

`QKParams{T<:Real, T2<:Real}` aggregates all model parameters and precomputed matrices 
needed by the QKF, including:

1. **Core Dimensions**:  
   - `N::Int`, `M::Int` for state and observation dimensions

2. **State Equation Parameters** (`μ`, `Φ`, `Ω`) and **Observation Equation Parameters** (`A`, `B`, `C`, `α`, `Δt`)

3. **Derived Quantities**:  
   - `Σ::Matrix{T}` = `Ω*Ω'`  
   - Augmented vectors/matrices (`μ̃`, `Φ̃`, `B̃`, etc.) for the QKF approach  
   - Additional precomputed items like commutation/duplication/selection matrices 
     (`Λ`, `G̃`, `H̃`) and unconditional moments (`μᵘ`, `Σᵘ`, `μ̃ᵘ`, `Σ̃ᵘ`)

4. **Dimension `P::Int`** for the augmented state (often `N + N²`)

# Usage

Typically, you create a `QKParams` instance by providing the minimal "raw" parameters 
(e.g. `N`, `M`, `μ`, `Φ`, `Ω`, ...) to the constructor `QKParams(...)`. The constructor 
checks dimension consistency (like `size(Φ) == (N,N)`) and stability 
(`spectral_radius(Φ) < 1`) and then builds all precomputed fields.  

Example:
```julia
params = QKParams(N=2, M=1, μ, Φ, Ω, A, B, C, D, α, Δt)
```
"""

"""
    QKParams{T<:Real, T2<:Real}

Parameters for the Quadratic Kalman Filter model.

# Fields
## Model Dimensions
- `N::Int`: Dimension of the state vector
- `M::Int`: Dimension of the observation vector

## State Equation Parameters 
- `μ::Vector{T}`: Constant term in state equation
- `Φ::Matrix{T}`: State transition matrix 
- `Ω::Matrix{T}`: State noise scaling matrix

## Observation Equation Parameters
- `A::Vector{T}`: Constant term in observation equation
- `B::Matrix{T}`: Linear observation matrix
- `C::Vector{Matrix{T}}`: Quadratic observation matrices
- `D::Matrix{T}`: Time-invariant observation noise scaling matrix
- `α::Matrix{T}`: Autoregressive coefficient matrix
- `Δt::T2`: Time step

## Precomputed Parameters
- `Σ::Matrix{T}`: State covariance matrix (ΩΩ')
- `V::Matrix{T}`: Observation covariance matrix (DD')
- `e::Vector{Vector{T}}`: Unit vectors
- `μ̃::Vector{T}`: Augmented constant vector
- `Φ̃::Matrix{T}`: Augmented transition matrix
- `Λ::Matrix{T2}`: Commutation matrix
- `L1,L2,L3::Matrix{T}`: Auxiliary matrices for covariance
- `μᵘ,μ̃ᵘ::Vector{T}`: Unconditional means
- `Σᵘ,Σ̃ᵘ::Matrix{T}`: Unconditional covariances
- `B̃::Matrix{T}`: Augmented observation matrix
- `H̃::Matrix{T}`: Augmented selection matrix
- `G̃::Matrix{T}`: Augmented duplication matrix
- `P::Int`: Dimension of augmented state (N + N²)
"""
@with_kw struct QKParams{T <: Real, T2 <: Real} 
    # State space dimensions
    N::Int  # State dimension
    M::Int  # Observation dimension
    
    # State equation parameters (Eq 1a)
    μ::Vector{T}    # Constant term
    Φ::Matrix{T}    # State transition matrix
    Ω::Matrix{T}    # Noise scaling matrix
    
    # Observation equation parameters (Eq 1b)
    A::Vector{T}            # Constant term
    B::Matrix{T}            # Linear observation matrix
    C::Vector{Matrix{T}}    # Quadratic observation matrices
    D::Matrix{T}            # Time-invariant observation noise scaling matrix
    α::Matrix{T}            # Autoregressive coefficients
    Δt::T2                  # Time step
    
    # Precomputed matrices for efficiency
    Σ::Matrix{T}    # State covariance (ΩΩ')
    V::Matrix{T}    # Observation covariance (DD')
    e::Vector{Vector{T}}  # Unit vectors
    μ̃::Vector{T}    # Augmented constant
    Φ̃::Matrix{T}    # Augmented transition
    Λ::Matrix{T2}   # Commutation matrix
    L1::Matrix{T}   # Auxiliary covariance matrix
    L2::Matrix{T}   # Auxiliary covariance matrix
    L3::Matrix{T}   # Auxiliary covariance matrix
    μᵘ::Vector{T}   # Unconditional mean
    Σᵘ::Matrix{T}   # Unconditional covariance
    μ̃ᵘ::Vector{T}   # Augmented unconditional mean
    Σ̃ᵘ::Matrix{T}   # Augmented unconditional covariance
    B̃::Matrix{T}    # Augmented observation matrix
    H̃::Matrix{T}    # Augmented selection matrix
    G̃::Matrix{T}    # Augmented duplication matrix
    P::Int          # Augmented state dimension (N + N²)
end

"""
    QKParams(N::Int, M::Int, μ::Vector{T}, Φ::Matrix{T}, Ω::Matrix{T}, 
            A::Vector{T}, B::Matrix{T}, C::Vector{<:AbstractMatrix{T}},
            wc::T, wu::T, wv::T, wuu::T, wuv::T, wvv::T,
            α::Matrix{T}, Δt::T2) where {T <: Real, T2 <: Real}

Constructor for QKParams that computes all derived matrices from the core parameters.

# Arguments
- `N::Int`: State dimension
- `M::Int`: Observation dimension  
- `μ::Vector{T}`: Constant term in state equation
- `Φ::Matrix{T}`: State transition matrix
- `Ω::Matrix{T}`: Noise scaling matrix
- `A::Vector{T}`: Constant term in observation equation
- `B::Matrix{T}`: Linear observation matrix
- `C::Vector{<:AbstractMatrix{T}}`: Quadratic observation matrices
- `D::Matrix{T}`: Time-invariant observation noise scaling matrix
- `α::Matrix{T}`: Autoregressive coefficients of measurement equation
- `Δt::T2`: Time step
"""
function QKParams(N::Int, M::Int, 
                 μ::Vector{T}, 
                 Φ::Matrix{T},
                 Ω::Matrix{T}, 
                 A::Vector{T}, 
                 B::Matrix{T}, 
                 C::Vector{<:AbstractMatrix{T}},
                 D::Matrix{T},
                 α::Matrix{T}, 
                 Δt::T2) where {T <: Real, T2 <: Real}

    # Validate input dimensions
    @assert size(Φ) == (N, N) "Φ must be N×N"
    @assert spectral_radius(Φ) < 1 "Φ must be stable (spectral radius < 1)"
    @assert size(Ω) == (N, N) "Ω must be N×N"
    @assert length(A) == M "A must have length M"
    @assert size(B) == (M, N) "B must be M×N"
    @assert length(μ) == N "μ must have length N"
    @assert length(C) == M "C must have length M"
    @assert all(size(Ci) == (N, N) for Ci in C) "Each Ci must be N×N"
    @assert size(D) == (M, M) "D must be M×M"
    @assert Δt > 0 "Δt must be positive"

    # Precompute derived matrices
    Σ = Ω * Ω'
    V = D * D'
    e = compute_e(M, T)
    Λ = compute_Λ(N)
    μ̃ = compute_μ̃(μ, Σ)
    Φ̃ = compute_Φ̃(μ, Φ)
    
    # Auxiliary covariance matrices
    L1 = compute_L1(Σ, Λ)
    L2 = compute_L2(Σ, Λ)
    L3 = compute_L3(Σ, Λ)
    
    # Unconditional moments
    μᵘ = compute_μᵘ(μ, Φ)
    Σᵘ = compute_Σᵘ(Φ, Σ)
    μ̃ᵘ = compute_μᵘ(μ̃, Φ̃)
    Σ̃ᵘ = compute_Σ̃ᵘ(μ̃ᵘ, L1, L2, L3, Λ, Σ, μ, Φ̃)
    
    # Augmented matrices
    B̃ = compute_B̃(B, C)
    H = selection_matrix(N, T)
    G = duplication_matrix(N, T)
    H̃ = compute_H̃(N, H)
    G̃ = compute_G̃(N, G)
    P = N + N^2

    return QKParams(N, M, μ, Φ, Ω, A, B, C, D, α, Δt, Σ, V, e, μ̃, Φ̃, Λ, L1, L2, L3, μᵘ, Σᵘ, μ̃ᵘ,
                    Σ̃ᵘ, B̃, H̃, G̃, P)
end

# Export the type and constructor
export QKParams