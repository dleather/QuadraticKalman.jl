@with_kw struct StateParams{T<:Real}
    N::Int              
    mu::AbstractVector{T}       
    Phi::AbstractMatrix{T}      
    Omega::AbstractMatrix{T}    
    Sigma::AbstractMatrix{T}    
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
        @assert spectral_radius(Phi) < 1 - 1e-6 "Phi must be stable (spectral radius < 1)"
    end

    # Handle UniformScaling without breaking AD
    Phi_final = Phi isa UniformScaling ? Matrix(Phi, N, N) : Phi
    Omega_final = Omega isa UniformScaling ? Matrix(Omega, N, N) : Omega
    Sigma = Omega_final * Omega_final'

    return StateParams{T}(N, mu, Phi_final, Omega_final, Sigma)
end

# Measurement equation parameters
@with_kw struct MeasParams{T<:Real}
    M::Int                                  
    A::AbstractVector{T}                            
    B::AbstractMatrix{T}                            
    C::Vector{<:AbstractMatrix{T}}                    
    D::AbstractMatrix{T}                            
    alpha::AbstractMatrix{T}                        
    V::AbstractMatrix{T}                            
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
    mu_aug::AbstractVector{T}
    Phi_aug::AbstractMatrix{T}
    B_aug::AbstractMatrix{T}
    H_aug::AbstractMatrix{T}
    G_aug::AbstractMatrix{T}
    Lambda::AbstractMatrix{T2}
    L1::AbstractMatrix{T}
    L2::AbstractMatrix{T}
    L3::AbstractMatrix{T}
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
    state_mean::AbstractVector{T}
    state_cov::AbstractMatrix{T}
    aug_mean::AbstractVector{T}
    aug_cov::AbstractMatrix{T}
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

"""
    QKModel{T<:Real, T2<:Real}

Main structure containing all parameters and moments needed for the quadratic Kalman filter.

# Fields
- `state::StateParams{T}`: Parameters for the state equation
- `meas::MeasParams{T}`: Parameters for the measurement equation  
- `aug_state::AugStateParams{T,T2}`: Parameters for the augmented state space
- `moments::Moments{T}`: Unconditional moments of the state and augmented state

# Type Parameters
- `T`: The main numeric type used for most parameters (e.g. Float64)
- `T2`: A possibly different numeric type used for Lambda in AugStateParams
"""
@with_kw struct QKModel{T<:Real, T2<:Real}
    state::StateParams{T}
    meas::MeasParams{T}
    aug_state::AugStateParams{T,T2}
    moments::Moments{T}
end

"""
    QKModel(N::Int, M::Int, mu::AbstractVector, Phi::AbstractMatrix,
            Omega::AbstractMatrix, A::AbstractVector, B::AbstractMatrix,
            C::Vector{<:AbstractMatrix}, D::AbstractMatrix,
            alpha::AbstractMatrix; check_stability::Bool=false) where {T<:Real}
    
    QKModel(state::StateParams, meas::MeasParams; check_stability::Bool=false)

Creates a Quadratic Kalman Filter model specification.

# Arguments
- `N::Int`: Dimension of the state vector
- `M::Int`: Dimension of the measurement vector
- `mu::AbstractVector`: Initial state mean vector (N×1)
- `Phi::AbstractMatrix`: State transition matrix (N×N)
- `Omega::AbstractMatrix`: State noise covariance matrix (N×N)
- `A::AbstractVector`: Constant term in measurement equation (M×1)
- `B::AbstractMatrix`: Linear measurement matrix (M×N)
- `C::Vector{<:AbstractMatrix}`: Quadratic measurement matrices (Vector of M N×N matrices)
- `D::AbstractMatrix`: Measurement noise covariance matrix (M×M)
- `alpha::AbstractMatrix`: Measurement error higher moment parameter (M×M)

# Keyword Arguments
- `check_stability::Bool=false`: If true, checks if the state transition dynamics are stable

# Alternative Constructor
The second method allows construction from pre-built StateParams and MeasParams objects.

# Returns
Returns a QKModel object containing all parameters needed for the quadratic Kalman filter.

# Examples
```julia
# Direct construction
N, M = 2, 1
model = QKModel(
    N, M,
    [0.0, 0.0],        # mu
    [0.9 0.0; 0.0 0.9], # Phi
    [0.1 0.0; 0.0 0.1], # Omega
    [0.0],              # A
    [1.0 1.0],          # B
    [reshape([1.0 0.0; 0.0 1.0], 2, 2)], # C
    [0.1],              # D
    [0.0]               # alpha
)

# Construction from components
state = StateParams(...)
meas = MeasParams(...)
model = QKModel(state, meas)
```

# Notes
- All matrices must be properly sized according to N and M
- The model components are used to construct augmented state representations
- The stability check verifies that eigenvalues of Phi are within the unit circle
"""
function QKModel(N::Int, M::Int, mu::AbstractVector{T}, Phi::AbstractMatrix{T}, 
                 Omega::AbstractMatrix{T}, A::AbstractVector{T}, B::AbstractMatrix{T}, 
                 C::Vector{<:AbstractMatrix{T}}, D::AbstractMatrix{T},
                 alpha::AbstractMatrix{T};
                 check_stability::Bool=false) where {T<:Real}
    
    # Convert tracked arrays to regular arrays for internal storage
    state = StateParams(N, collect(mu), collect(Phi), collect(Omega); check_stability=check_stability)
    meas = MeasParams(M, N, collect(A), collect(B), [collect(Ci) for Ci in C], collect(D), collect(alpha))
    aug_state = AugStateParams(N, state.mu, state.Phi, state.Sigma, meas.B, meas.C)
    moments = Moments(state.mu, state.Phi, state.Sigma, aug_state.mu_aug, aug_state.Phi_aug,
                     aug_state.L1, aug_state.L2, aug_state.L3, aug_state.Lambda)
    
    return QKModel(state=state, meas=meas, aug_state=aug_state, moments=moments)
end

"""
    QKModel(state::StateParams, meas::MeasParams; check_stability::Bool=false)

Alternative constructor for QKModel that takes pre-built state and measurement parameter objects.

# Arguments
- `state::StateParams`: State parameters object containing N, mu, Phi, and Omega
- `meas::MeasParams`: Measurement parameters object containing M, A, B, C, D, and alpha

# Keyword Arguments
- `check_stability::Bool=false`: If true, checks if the state transition dynamics are stable

# Returns
Returns a QKModel object by unpacking the parameters and calling the primary constructor.

# Example
```julia
# Create parameter objects
state = StateParams(2, [0.0, 0.0], [0.9 0.0; 0.0 0.9], [0.1 0.0; 0.0 0.1])
meas = MeasParams(1, 2, [0.0], [1.0 1.0], [reshape([1.0 0.0; 0.0 1.0], 2, 2)], [0.1], [0.0])

# Create model
model = QKModel(state, meas)
```

See also: `QKModel`, `StateParams`, `MeasParams`
"""
function QKModel(state::StateParams{T}, meas::MeasParams{T}; check_stability::Bool=false) where {T<:Real}
    @unpack N, mu, Phi, Omega = state
    @unpack M, A, B, C, D, alpha = meas
    return QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha; check_stability=check_stability)
end

"""
    params_to_model(params::Vector{T}, N::Int, M::Int) where T<:Real -> QKModel

Convert a parameter vector into a QKModel object with state and measurement parameters.

# Arguments
- `params::Vector{T}`: A vector of unconstrained parameters.
- `N::Int`: Dimension of the state vector.
- `M::Int`: Dimension of the measurement vector.

# Returns
A `QKModel` object containing:
- **State parameters:** (mu, Phi, Omega)  
  where the state equation is
      Xₜ = μ + Φ Xₜ₋₁ + Omega εₜ.
  Here, Omega is constructed as Omega = D_state * D_state′,
  with D_state a lower–triangular matrix (of size N×N) whose diagonal entries are positive.
- **Measurement parameters:** (A, B, C, D, α)  
  where the measurement equation is
      Yₜ = A + B Xₜ + α Yₜ₋₁ + ∑₍ᵢ₌₁₎ᴹ Xₜ′ Cᵢ Xₜ + D εₜ.
  Here, D is constructed as D = D_meas * D_meas′,
  with D_meas a lower–triangular matrix (of size M×M) whose diagonal entries are positive.
- Augmented state parameters and model moments (computed via helper functions).

# Parameter vector layout

The parameter vector is assumed to contain:

1. **State parameters:**
   - First `N` entries: state mean `mu`.
   - Next `N^2` entries: entries of `Phi` (stored columnwise).
   - Next `N(N+1)/2` entries: unconstrained parameters for D_state (used to form Omega).

2. **Measurement parameters:**
   - Next `M` entries: `A`.
   - Next `M×N` entries: entries of `B` (reshaped as an M×N matrix).
   - Next `M×N^2` entries: entries for `C`. (Interpreted as M matrices of size N×N.)
   - Next `M(M+1)/2` entries: unconstrained parameters for D_meas (used to form D).
   - Final `M×M` entries: entries of `α` (reshaped as an M×M matrix).

# Total expected length:

    N + N^2 + N(N+1)/2  +  M + M×N + M×N^2 + M(M+1)/2 + M^2

"""
function params_to_model(params::AbstractVector{T}, N::Int, M::Int) where T<:Real
    # Convert tracked arrays to regular arrays for internal computations
    params_array = collect(params)
    
    # Compute the number of parameters in each block.
    n_mu    = N
    n_Phi   = N^2
    n_Dstate = div(N*(N+1), 2)      # for lower-triangular D_state

    n_A     = M
    n_B     = M * N
    n_C     = M * N^2           # interpreted as M matrices (each N×N)
    n_Dmeas = div(M*(M+1), 2)       # for lower-triangular D_meas
    n_alpha = M^2

    n_state = n_mu + n_Phi + n_Dstate
    n_meas  = n_A + n_B + n_C + n_Dmeas + n_alpha
    expected_length = n_state + n_meas

    @assert length(params_array) == expected_length "Parameter vector has unexpected length: got $(length(params_array)), expected $expected_length"

    # Extract state parameters
    ndx = 1

    # 1. State mean: mu (length N)
    mu = params_array[ndx : ndx + n_mu - 1]
    ndx += n_mu

    # 2. State transition matrix: Phi (N×N, stored columnwise)
    Phi = reshape(params_array[ndx : ndx + n_Phi - 1], (N, N))
    ndx += n_Phi

    # 3. State noise parameters: Build D_state (lower-triangular) from n_Dstate unconstrained parameters.
    Omega_params = params_array[ndx : ndx + n_Dstate - 1]
    ndx += n_Dstate

    Omega = zeros(T, N, N)
    k = 1
    for i in 1:N
        for j in 1:i
            if i == j
                Omega[i, j] = exp(Omega_params[k])  # exponentiate diagonal entries to ensure positivity.
            else
                Omega[i, j] = Omega_params[k]
            end
            k += 1
        end
    end

    # =====================
    # Extract measurement parameters
    # =====================

    # 4. Measurement intercept: A (length M)
    A = params_array[ndx : ndx + n_A - 1]
    ndx += n_A

    # 5. Measurement loading matrix: B (reshape as M×N)
    B = reshape(params_array[ndx : ndx + n_B - 1], (M, N))
    ndx += n_B

    # 6. Quadratic term parameters: C.
    C_entries = params_array[ndx : ndx + n_C - 1]
    ndx += n_C
    C = Vector{Matrix{T}}(undef, M)
    for i in 1:M
        C[i] = reshape(C_entries[(i-1)*N^2 + 1 : i*N^2], (N, N))
    end

    # 7. Measurement noise parameters: Build D_meas (lower-triangular) from n_Dmeas unconstrained parameters.
    D_params = params_array[ndx : ndx + n_Dmeas - 1]
    ndx += n_Dmeas

    D = zeros(T, M, M)
    k = 1
    for i in 1:M
        for j in 1:i
            if i == j
                D[i, j] = exp(D_params[k])
            else
                D[i, j] = D_params[k]
            end
            k += 1
        end
    end

    # 8. Measurement noise auto-regressive parameter: α (reshape as M×M)
    alpha = reshape(params_array[ndx : ndx + n_alpha - 1], (M, M))
    ndx += n_alpha
    @assert ndx == expected_length + 1 "ndx should be equal to expected_length + 1"

    # Convert back to tracked arrays for the final QKModel construction
    return QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha)
end

"""
    model_to_params(model::QKModel{T, T2}) where {T<:Real, T2}

Convert a QKModel object into a vector of unconstrained parameters.

The ordering of the parameters is as follows:

1. **State parameters:**
   - `mu` (length N)
   - `Phi` (N×N, stored columnwise)
   - Unconstrained parameters for the state noise scaling factor Ω:
     For each row `i = 1,...,N` and column `j = 1,...,i` (i.e. the lower–triangular part):
       - If `i == j`: the parameter is `log(Ω[i,i])`
       - Else: the parameter is `Ω[i,j]`

2. **Measurement parameters:**
   - `A` (length M)
   - `B` (M×N, stored columnwise)
   - `C` (a vector of M matrices; each matrix is N×N and is flattened columnwise)
   - Unconstrained parameters for the measurement noise scaling factor D:
     For each row `i = 1,...,M` and column `j = 1,...,i`:
       - If `i == j`: the parameter is `log(D[i,i])`
       - Else: the parameter is `D[i,j]`
   - `alpha` (M×M, stored columnwise)

# Returns
A vector of unconstrained parameters that, when passed to `params_to_model`, reconstructs the original QKModel.
"""
function model_to_params(model::QKModel{T, T2}) where {T<:Real, T2}
  # Extract the dimensions from the model.
  N = model.state.N
  M = model.meas.M

  # Initialize an empty vector to accumulate the parameters.
  params = Vector{T}()

  # --------------------------------------------------
  # 1. State parameters
  # --------------------------------------------------
  # a. State mean: mu
  append!(params, model.state.mu)

  # b. State transition matrix: Phi (flattened columnwise)
  append!(params, vec(model.state.Phi))

  # c. Unconstrained parameters for the state noise scaling factor, Ω.
  #    For each (i,j) in the lower–triangular part of Ω:
  #      - if i == j, then parameter = log(Ω[i,i])
  #      - otherwise, parameter = Ω[i,j]
  for i in 1:N
    for j in 1:i
      if i == j
        push!(params, log(model.state.Omega[i, j]))
      else
        push!(params, model.state.Omega[i, j])
      end
    end
  end

  # --------------------------------------------------
  # 2. Measurement parameters
  # --------------------------------------------------
  # a. Measurement intercept: A
  append!(params, model.meas.A)

  # b. Measurement loading matrix: B (flattened columnwise)
  append!(params, vec(model.meas.B))

  # c. Quadratic term parameters: C
  #    C is stored as a vector of M matrices (each of size N×N).
  #    For each matrix, flatten it columnwise and append.
  for i in 1:M
    append!(params, vec(model.meas.C[i]))
  end

  # d. Unconstrained parameters for the measurement noise scaling factor, D.
  #    For each (i,j) in the lower–triangular part of D:
  #      - if i == j, then parameter = log(D[i,i])
  #      - otherwise, parameter = D[i,j]
  for i in 1:M
    for j in 1:i
      if i == j
        push!(params, log(model.meas.D[i, j]))
      else
        push!(params, model.meas.D[i, j])
      end
    end
  end

  # e. Measurement noise auto–regressive parameter: alpha (flattened columnwise)
  append!(params, vec(model.meas.alpha))

  return params
end
