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
- `α::Matrix{T}`: Autoregressive coefficient matrix
- `Δt::T2`: Time step

## Time-Varying Volatility Parameters
Coefficients for V(z) = wc + wu*u + wv*v + wuu*u² + wuv*u*v + wvv*v²
- `wc::T`: Constant term
- `wu::T`: Linear coefficient for u
- `wv::T`: Linear coefficient for v  
- `wuu::T`: Quadratic coefficient for u²
- `wuv::T`: Cross term coefficient for u*v
- `wvv::T`: Quadratic coefficient for v²

## Precomputed Parameters
- `Σ::Matrix{T}`: State covariance matrix (ΩΩ')
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
    A::Vector{T}    # Constant term
    B::Matrix{T}    # Linear observation matrix
    C::Vector{Matrix{T}}  # Quadratic observation matrices
    α::Matrix{T}    # Autoregressive coefficients
    Δt::T2         # Time step
    
    # Time-varying volatility parameters
    # V(z) = wc + wu*u + wv*v + wuu*u² + wuv*u*v + wvv*v²
    wc::T          # Constant term
    wu::T          # Linear u term
    wv::T          # Linear v term
    wuu::T         # Quadratic u term
    wuv::T         # Cross term
    wvv::T         # Quadratic v term
    
    # Precomputed matrices for efficiency
    Σ::Matrix{T}    # State covariance (ΩΩ')
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
    compute_Gamma_tm1(Z::AbstractVector{T}, mu::AbstractVector{T}, Phi::AbstractMatrix{T}) where T <: Real

Compute the matrix Γₜ₋₁ for a VAR(1) process.

# Arguments
- `Z::AbstractVector{T}`: Current state vector
- `mu::AbstractVector{T}`: Constant term in state equation
- `Phi::AbstractMatrix{T}`: Transition matrix

# Returns
- `Matrix`: N×N matrix Γₜ₋₁
"""
function compute_Gamma_tm1(Z::AbstractVector{T}, mu::AbstractVector{T},
    Phi::AbstractMatrix{T}) where T <: Real
    N = length(mu)
    tmp = mu + Phi * Z
    return kron(I(N), tmp) + kron(tmp, I(N)) 
end