"""
    StateParams{T<:Real}

A structure representing the parameters that govern the evolution of the state in a state-space model.
This type encapsulates the fundamental components required to describe the behavior of the state process,
including its initial mean, transition dynamics, noise characteristics, and the resulting covariance.

# Fields
- N::Int
    The dimensionality of the state vector. This parameter specifies the number of state variables.
- mu::AbstractVector{T}
    The initial state mean vector. It must have a length equal to N.
- Phi::AbstractMatrix{T}
    The state transition matrix, which models how the state evolves over time. This matrix should be of size N×N.
- Omega::AbstractMatrix{T}
    The state noise matrix, used to scale the impact of the stochastic noise on the state evolution. It must be of size N×N.
- Sigma::AbstractMatrix{T}
    The state covariance matrix, typically computed as Omega * Omega'. This matrix quantifies the uncertainty in the state.

# Details
The state evolution of a typical state-space model is represented by the equation:
    Xₜ = mu + Phi * Xₜ₋₁ + Omega * εₜ,
where εₜ represents a white noise process. This structure is a critical component in facilitating both the filtering
and smoothing processes within the QuadraticKalman framework, ensuring that the model's dynamics are accurately captured
and that its stability conditions can be properly validated.

This structure is also designed to integrate smoothly with automatic differentiation tools, taking advantage of Julia's
type system to provide both precision and performance in numerical computations.
"""
@with_kw struct StateParams{T<:Real}
    N::Int              
    mu::AbstractVector{T}       
    Phi::AbstractMatrix{T}      
    Omega::AbstractMatrix{T}    
    Sigma::AbstractMatrix{T}    
end

"""
    StateParams(N::Int, mu::AbstractVector{T}, Phi::Union{AbstractMatrix{T}, UniformScaling}, 
                Omega::Union{AbstractMatrix{T}, UniformScaling}; check_stability::Bool=false) where {T<:Real}

Constructs a StateParams instance encapsulating parameters for representing the state-space dynamics of a model.

This constructor accepts both explicit matrices and UniformScaling objects for the state transition component (Phi)
and the state noise component (Omega). It enforces that:
  • The state mean vector, mu, has length N.
  • When Phi is provided as a concrete matrix, it must be of dimensions N×N.
  • When Omega is provided as a concrete matrix, it must also be of dimensions N×N.
  • Optionally, if the check_stability flag is true, it verifies that the spectral radius of Phi is below 1 - 1e-6,
    ensuring the stability of the state transition dynamics.

UniformScaling inputs are converted to explicit N×N matrices to facilitate compatibility with automatic differentiation (AD)
and other numerical computations. The function then computes the state covariance matrix, Sigma, as Omega_final * Omega_final'.

# Parameters:
  - N::Int: The number of state variables (dimensionality of the state vector).
  - mu::AbstractVector{T}: The state mean vector; its length must be exactly N.
  - Phi::Union{AbstractMatrix{T}, UniformScaling}: The state transition matrix, provided either as a matrix or as a UniformScaling object.
  - Omega::Union{AbstractMatrix{T}, UniformScaling}: The state noise parameter, provided either as a matrix or as a UniformScaling object.

# Keyword Arguments:
  - check_stability::Bool=false: If set to true, checks that the spectral radius of Phi is less than 1 - 1e-6 to ensure stability.

# Returns:
  A StateParams{T} instance containing:
    • N: the state dimension.
    • mu: the state mean vector.
    • Phi_final: the explicit N×N state transition matrix.
    • Omega_final: the explicit N×N state noise matrix.
    • Sigma: the state covariance matrix computed as Omega_final * Omega_final'.
"""
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
"""
    MeasParams{T<:Real}

A container for the measurement equation parameters in a quadratic state-space model.

This structure holds all measurement-related parameters essential for specifying the measurement equation:

  Yₜ = A + B * Xₜ + α * Yₜ₋₁ + ∑₍ᵢ₌₁₎ᴹ (Xₜ' * Cᵢ * Xₜ) + noise

where:
  • M: The number of measurement variables.
  • A: A vector of intercept terms (length M).
  • B: An M×N matrix mapping the state vector (of dimension N) to the measurement space.
  • C: A vector of M matrices, each of size N×N, representing quadratic measurement parameters.
  • D: An M×M matrix scaling the measurement noise.
  • α: An M×M matrix capturing the autoregressive component in the measurement equation.
  • V: An auxiliary M×M matrix computed as V = D * D', representing the covariance structure of the measurement noise.

All fields should be provided as concrete matrices or vectors (or will be derived from UniformScaling objects as needed), ensuring consistency and compatibility with downstream filtering and smoothing routines. The use of the @with_kw macro facilitates clear initialization and automatic field assignment.
"""
@with_kw struct MeasParams{T<:Real}
    M::Int                                  
    A::AbstractVector{T}                            
    B::AbstractMatrix{T}                            
    C::Vector{<:AbstractMatrix{T}}                    
    D::AbstractMatrix{T}                            
    alpha::AbstractMatrix{T}                        
    V::AbstractMatrix{T}                            
end

"""
    MeasParams(M::Int, N::Int, A::AbstractVector{T}, 
               B::Union{AbstractMatrix{T}, UniformScaling},
               C::Vector{<:Union{AbstractMatrix{T}, UniformScaling}}, 
               D::Union{AbstractMatrix{T}, UniformScaling},
               alpha::Union{AbstractMatrix{T}, UniformScaling}) where {T<:Real}

Constructs a MeasParams object containing the parameters for the measurement equation in a quadratic state-space model.

# Arguments
- M::Int: The number of measurement variables.
- N::Int: The number of state variables.
- A::AbstractVector{T}: A vector of intercept terms in the measurement equation. Its length must equal M.
- B::Union{AbstractMatrix{T}, UniformScaling}: The matrix mapping the state vector to the measurement space. If provided as a UniformScaling, it is converted to an M×N matrix.
- C::Vector{<:Union{AbstractMatrix{T}, UniformScaling}}: A vector of matrices representing the quadratic measurement parameters. There must be M matrices, each of size N×N. UniformScaling inputs are converted accordingly.
- D::Union{AbstractMatrix{T}, UniformScaling}: A matrix scaling the measurement noise, required to be M×M. UniformScaling inputs are converted to a standard matrix.
- alpha::Union{AbstractMatrix{T}, UniformScaling}: A matrix capturing the autoregressive component in the measurement equation. If provided as UniformScaling, it is converted to an M×M matrix.

# Returns
A MeasParams object parameterized by type T, encapsulating all measurement model parameters, along with an auxiliary matrix V computed as V = D_final * D_final', which represents the covariance structure of the measurement noise.

# Details
This function ensures that all inputs are represented as concrete matrices, even when provided as UniformScaling objects. For B and each element in C, as well as D and alpha, if the input is a UniformScaling, it is converted into a full matrix with the appropriate dimensions (M×N for B, N×N for each Ci in C, and M×M for D and alpha). This conversion guarantees compatibility with downstream operations in filtering and smoothing routines, which require complete matrix representations for accurate linear algebra computations.
"""
function MeasParams(M::Int, N::Int, A::AbstractVector{T}, 
                    B::Union{AbstractMatrix{T}, UniformScaling},
                    C::Vector{<:Union{AbstractMatrix{T}, UniformScaling}}, 
                    D::Union{AbstractMatrix{T}, UniformScaling},
                    alpha::Union{AbstractMatrix{T}, UniformScaling}) where {T<:Real}
    @assert length(A) == M "A must have length M"
    
    # Handle UniformScaling for B: Convert to an M×N matrix if necessary.
    B_final = B isa UniformScaling ? Matrix(B, M, N) : B
    @assert size(B_final, 1) == M && size(B_final, 2) == N "B must be M×N"
    
    @assert length(C) == M "C must have length M"
    # Convert each element of C to an N×N matrix if given as UniformScaling, otherwise keep as is.
    C_final = [Ci isa UniformScaling ? Matrix(Ci, N, N) : Ci for Ci in C]
    @assert all(size(Ci) == (N, N) for Ci in C_final) "Each Ci must be N×N"
    
    # Handle UniformScaling for D: Convert to an M×M matrix if necessary.
    D_final = D isa UniformScaling ? Matrix(D, M, M) : D
    @assert size(D_final, 1) == M && size(D_final, 2) == M "D must be M×M"
    
    # Handle UniformScaling for alpha: Convert to an M×M matrix if necessary.
    alpha_final = alpha isa UniformScaling ? Matrix(alpha, M, M) : alpha
    
    # Compute the auxiliary measurement noise covariance matrix.
    V = D_final * D_final'
    
    return MeasParams{T}(M, A, B_final, C_final, D_final, alpha_final, V)
end

# Augmented state parameters
"""
    AugStateParams{T<:Real, T2<:Real}

An augmented state parameter container designed for quadratic measurement models. This type extends the conventional state-space representation to incorporate quadratic measurement features, enabling advanced filtering and smoothing algorithms to effectively handle non-linear measurement equations.

Fields:
  - mu_aug::AbstractVector{T}:
      The augmented state mean vector, which integrates the original state mean with additional terms arising from quadratic components.
  - Phi_aug::AbstractMatrix{T}:
      The augmented state transition matrix. It extends the traditional state transition dynamics to include quadratic interactions.
  - B_aug::AbstractMatrix{T}:
      The augmented measurement matrix that relates the extended state vector to the observed measurements, accounting for both linear and quadratic effects.
  - H_aug::AbstractMatrix{T}:
      The augmented selection matrix used in mapping the original state space to the augmented space, facilitating the extraction of relevant subcomponents.
  - G_aug::AbstractMatrix{T}:
      The augmented duplication matrix, which assists in preserving the symmetry properties of quadratic forms when processing covariance or moment adjustments.
  - Lambda::AbstractMatrix{T2}:
      A core structural matrix that captures the key quadratic interactions within the model. Its specific formulation supports the reconstruction of quadratic measures.
  - L1::AbstractMatrix{T}:
      An auxiliary matrix used for computing first-order moment corrections in the augmented state representation.
  - L2::AbstractMatrix{T}:
      An auxiliary matrix involved in the computation of second-order moment adjustments, contributing to the accurate determination of state covariances.
  - L3::AbstractMatrix{T}:
      An auxiliary matrix designed to support higher-order moment computations, often necessary for fine-tuning the filtering process.
  - P::Int:
      The total augmented state dimension, typically defined as the sum of the original state dimension and the square of the state dimension.

This structure is pivotal for models that incorporate quadratic measurement equations, allowing for the direct integration of quadratic terms into the state estimation process. It facilitates the derivation of augmented transition and measurement matrices, which are essential for achieving improved filtering and smoothing performance in non-linear state-space models.
"""
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

"""
    AugStateParams(N::Int, mu::AbstractVector{T}, Phi::AbstractMatrix{T}, 
                   Sigma::AbstractMatrix{T}, B::AbstractMatrix{T}, 
                   C::Vector{<:AbstractMatrix{T}}) where {T<:Real}

Construct an augmented state parameters object that encapsulates extended dynamics
for the quadratic measurement model. This function computes additional structures 
and matrices based on the traditional state-space parameters so that quadratic 
components in the measurement equation are properly addressed during filtering 
and smoothing.

# Arguments
- N::Int: The dimension of the state vector.
- mu::AbstractVector{T}: The initial state mean vector.
- Phi::AbstractMatrix{T}: The state transition matrix.
- Sigma::AbstractMatrix{T}: The covariance matrix for the state innovations.
- B::AbstractMatrix{T}: The measurement design matrix.
- C::Vector{<:AbstractMatrix{T}}: A vector of quadratic coefficient matrices for each 
  measurement element. Each element must be an N×N matrix representing the quadratic 
  contribution to the measurement equation.

# Returns
An instance of AugStateParams with the following fields:
- mu_aug: Augmented state mean vector computed from mu and Sigma.
- Phi_aug: Augmented state transition matrix derived from mu and Phi.
- B_aug: Augmented measurement matrix computed from B and the set of matrices in C.
- H_aug: Augmented selection matrix utilized in transforming state-space representations.
- G_aug: Augmented duplication matrix used to handle symmetry in quadratic forms.
- Lambda: A matrix computed to capture the structural features of the quadratic terms.
- L1, L2, L3: Auxiliary matrices derived from Sigma and Lambda, supporting moment computations.
- P: An integer indicating the total augmented state dimension, given by N + N^2.

# Details
The function performs the following steps:
1. Computes Lambda via a helper function, which encapsulates key aspects of the quadratic 
   measurement formulation.
2. Computes the augmented state mean (mu_aug) using the original mean and the covariance Sigma.
3. Computes the augmented state transition matrix (Phi_aug) by transforming Phi in accordance with 
   the augmented dynamics.
4. Determines the augmented measurement matrix (B_aug) by combining the original B with the quadratic 
   coefficient matrices C.
5. Calculates auxiliary matrices L1, L2, and L3 that are essential for subsequent moment and variance computations.
6. Generates selection and duplication matrices (H and G) using helper functions, and further refines them 
   to H_aug and G_aug to fit the augmented context.
7. Sets P, the overall augmented state dimension, to be the sum of the original state dimension and its square.

Ensure that all helper functions (e.g., compute_Lambda, compute_mu_aug, compute_Phi_aug, compute_B_aug, 
compute_L1, compute_L2, compute_L3, selection_matrix, duplication_matrix, compute_H_aug, and compute_G_aug) 
are defined and in scope prior to invoking this function.
"""
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
"""
    Moments{T<:Real}

Unconditional Moments Structure for the Quadratic Kalman Filter.

This structure encapsulates the long-run (stationary) moments of both the state
and the augmented state. These moments include the mean and covariance estimates,
which are critical for initializing the filter and for conducting diagnostic evaluations
of the model dynamics.

Fields:
  - state_mean::AbstractVector{T}: Unconditional (stationary) mean vector of the state.
  - state_cov::AbstractMatrix{T}: Unconditional covariance matrix of the state.
  - aug_mean::AbstractVector{T}: Unconditional mean vector of the augmented state.
  - aug_cov::AbstractMatrix{T}: Unconditional covariance matrix of the augmented state.
"""
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

This composite model encapsulates every component necessary to specify a quadratic state-space model. It bundles together the standard state evolution parameters, measurement parameters with quadratic terms, augmented state parameters for richer dynamics, and the unconditional moments that summarize the state behavior over time. This design enables integrated filtering and smoothing procedures within a unified framework.

# Fields
- `state::StateParams{T}`: Parameters governing the state process, defined by the equation
  X_t = μ + Φ X_{t-1} + Ω ε_t,
  where μ is the initial state mean, Φ is the state transition matrix, and Ω scales the process noise ε_t.
- `meas::MeasParams{T}`: Parameters for the measurement model, given by
  Y_t = A + B X_t + α Y_{t-1} + ∑_{i=1}^M (X_t' C_i X_t) + D ε_t,
  including the intercept A, linear loading B, autoregressive term α, quadratic terms involving matrices C_i, and the measurement noise scaling D.
- `aug_state::AugStateParams{T,T2}`: Parameters for the augmented state space which extend the state representation. This component includes transformed state means, transitions, and additional matrices that capture nonlinear or auxiliary features of the state process. The use of a secondary numeric type T2 facilitates compatibility with automatic differentiation.
- `moments::Moments{T}`: The unconditional (or stationary) moments of both the state and the augmented state. This includes the long-run mean and covariance for the state dynamics as well as the augmented state, which are critical for initialization and diagnostic evaluation of the model.

# Type Parameters
- `T`: The primary numeric type used for most parameters (e.g., Float64). It must be a subtype of Real, ensuring both numerical precision and compatibility with standard arithmetic operations.
- `T2`: A potentially different numeric type used specifically for parameters like Lambda in AugStateParams, often employed to leverage automatic differentiation (AD) techniques.
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
