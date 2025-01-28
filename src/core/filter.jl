"""
# Quadratic Kalman Filter (QKF) — File Overview

This file contains **all** of the core functions and utilities required 
to run a *Quadratic Kalman Filter (QKF)*. The QKF is an extension of the 
classical Kalman filter that tracks not just the mean state `xₜ` but also 
its second moment `(x xᵀ)ₜ`, effectively making the state vector *quadratic* 
in dimension. By doing so, it can capture certain nonlinear effects in the 
state evolution and measurement models.

## Contents

1. **State Prediction Functions**
   - [`predict_Zₜₜ₋₁!`](#predict_Zₜₜ₋₁!) / [`predict_Zₜₜ₋₁`](#predict_Zₜₜ₋₁)
   - [`predict_Pₜₜ₋₁!`](#predict_Pₜₜ₋₁!) / [`predict_Pₜₜ₋₁`](#predict_Pₜₜ₋₁)
   These update the *augmented* state and covariance one step ahead 
   (`Zₜₜ₋₁`, `Pₜₜ₋₁`).

2. **Measurement Prediction Functions**
   - [`predict_Yₜₜ₋₁!`](#predict_Yₜₜ₋₁!) / [`predict_Yₜₜ₋₁`](#predict_Yₜₜ₋₁)
   - [`predict_Mₜₜ₋₁!`](#predict_Mₜₜ₋₁!) / [`predict_Mₜₜ₋₁`](#predict_Mₜₜ₋₁)
   These produce the predicted observation `Yₜₜ₋₁` and its covariance 
   `Mₜₜ₋₁`.

3. **Kalman Gain Computation**
   - [`compute_Kₜ!`](#compute_Kₜ!) / [`compute_Kₜ`](#compute_Kₜ)
   Form the Kalman gain using the predicted covariance terms.

4. **State & Covariance Update Functions**
   - [`update_Zₜₜ!`](#update_Zₜₜ!) / [`update_Zₜₜ`](#update_Zₜₜ)
   - [`update_Pₜₜ!`](#update_Pₜₜ!) / [`update_Pₜₜ`](#update_Pₜₜ)
   Incorporate the measurement into the predicted state/covariance to get 
   the *posterior* (filtered) estimates `Zₜₜ`, `Pₜₜ`.

5. **Positive Semidefinite (PSD) Correction**
   - [`correct_Zₜₜ!`](#correct_Zₜₜ!) / [`correct_Zₜₜ`](#correct_Zₜₜ)
   The QKF state includes second-moment blocks that can become numerically 
   indefinite. These functions clamp any negative eigenvalues to enforce 
   PSD.

6. **Filtering Routines**
   - [`qkf_filter!`](#qkf_filter!) : In-place QKF over a time series.
   - [`qkf_filter`](#qkf_filter) : Out-of-place QKF, returning new arrays.
   They tie everything together by iterating from `t=1` to `t=T̄`, calling 
   the various predict, update, and correction routines.

7. **Auxiliary / Experimental**
   - [`qkf_filter_functional`](#qkf_filter_functional) : A more functional, 
     experimental approach using `foldl`.

## Usage

- **In-Place**: Call `qkf_filter!(data, params)` to run the filter in-place, 
  potentially avoiding unnecessary memory allocations.
- **Out-of-Place**: Call `qkf_filter(data, params)` if you prefer a 
  functional style that returns fresh arrays for each step’s results.

## Notes

- The `QKData` and `QKParams` types organize time series and model parameters.
- At every time step, the *predict* step forms `Zₜₜ₋₁` / `Pₜₜ₋₁` 
  and the *update* step forms `Zₜₜ` / `Pₜₜ`. 
- The "Quadratic" portion refers to tracking `(x xᵀ)ₜ` inside the state 
  to capture second-moment dynamics. 
- A PSD correction step (`correct_Zₜₜ!` or `correct_Zₜₜ`) is often necessary 
  to handle numerical or modeling approximations that might lead to 
  indefinite blocks in the augmented state.

For more detail, refer to each function’s docstring below.
"""

"""
    predict_Zₜₜ₋₁!(Ztt::AbstractMatrix{T}, 
                   Zttm1::AbstractMatrix{T},
                   params::QKParams{T,T2}, 
                   t::Int)

In-place computation of the **one-step-ahead predicted augmented state** 
`Zₜₜ₋₁[:, t]` from the current augmented state `Ztt[:, t]`.

# Arguments
- `Ztt::AbstractMatrix{T}`: (P×T̄) matrix storing the current augmented states, 
  with `Ztt[:, t]` as the time-`t` state.
- `Zttm1::AbstractMatrix{T}`: (P×T̄) matrix to store the new predicted state, 
  with the result in `Zttm1[:, t]`.
- `params::QKParams{T,T2}`: Parameter struct holding:
  - `μ̃::Vector{T}`: Augmented constant vector.
  - `Φ̃::Matrix{T}`: Augmented transition matrix (size P×P).
- `t::Int`: The current time index.

# Details
Computes `Zttm1[:, t] = μ̃ + Φ̃ * Ztt[:, t]`, 
corresponding to the QKF augmented state update.  
The result is written **in place** into `Zttm1[:, t]`.

"""
function predict_Zₜₜ₋₁!(Ztt::AbstractMatrix{T}, Zttm1::AbstractMatrix{T},
    params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    @unpack μ̃, Φ̃ = params
    Zttm1[:, t] = μ̃ .+ Φ̃ * Ztt[:, t]
end


"""
    predict_Zₜₜ₋₁(Ztt::AbstractVector{T}, params::QKParams{T,T2}) 
                  -> Vector{T}

Return a new **one-step-ahead predicted augmented state** given the current 
augmented state `Ztt`.

# Arguments
- `Ztt::AbstractVector{T}`: The current augmented state (dimension P).
- `params::QKParams{T,T2}`: Holds:
  - `μ̃::Vector{T}`: Augmented constant vector (length P).
  - `Φ̃::Matrix{T}`: P×P augmented transition matrix.

# Returns
- `Vector{T}`: A newly allocated vector for `Zₜₜ₋₁ = μ̃ + Φ̃ * Ztt`.

# Details
This purely functional approach does not modify `Ztt`, 
which is typically `(P×1)` in QKF contexts.  
Useful in AD frameworks that avoid in-place updates.

"""
function predict_Zₜₜ₋₁(Ztt::AbstractVector{T}, params::QKParams{T,T2}) where 
    {T <: Real, T2 <: Real}

    @unpack μ̃, Φ̃ = params
    return μ̃ .+ Φ̃ * Ztt
end


"""
    predict_Pₜₜ₋₁!(Pₜₜ::AbstractArray{Real,3}, 
                    Pₜₜ₋₁::AbstractArray{Real,3}, 
                    Σₜₜ₋₁::AbstractArray{Real,3},
                    Zₜₜ::AbstractMatrix{T}, 
                    tmpP::AbstractMatrix{T}, 
                    params::QKParams{T,T2}, 
                    t::Int)

In-place update of the **one-step-ahead predicted covariance** 
`Pₜₜ₋₁[:,:,t]` from the current filtered covariance `Pₜₜ[:,:,t]`. 
Also updates or uses `Σₜₜ₋₁[:,:,t]` as needed.

# Arguments
- `Pₜₜ::AbstractArray{Real,3}`: A 3D array `(P×P×T̄)` of filtered covariances; 
  we read `Pₜₜ[:,:,t]`.
- `Pₜₜ₋₁::AbstractArray{Real,3}`: A 3D array `(P×P×T̄)` to store the predicted 
  covariance in `Pₜₜ₋₁[:,:,t]`.
- `Σₜₜ₋₁::AbstractArray{Real,3}`: (P×P×T̄) array storing 
  state-dependent noise or extra terms computed by `compute_Σₜₜ₋₁!`.
- `Zₜₜ::AbstractMatrix{T}`: (P×T̄) the current augmented states. 
  Used by `compute_Σₜₜ₋₁!` to update `Σₜₜ₋₁`.
- `tmpP::AbstractMatrix{T}`: A scratch `(P×P)` buffer if needed 
  for multiplication (not fully used in the code snippet).
- `params::QKParams{T,T2}`: Must contain:
  - `Φ̃::Matrix{T}`: The P×P augmented transition matrix.
- `t::Int`: Time index.

# Details
1. Calls `compute_Σₜₜ₋₁!(...)` to update the noise/correction term in 
   `Σₜₜ₋₁[:,:,t]` based on `Zₜₜ[:,t]`.
2. Computes 
   `Pₜₜ₋₁[:,:,t] = ensure_positive_definite( Φ̃ * Pₜₜ[:,:,t] * Φ̃' + Σₜₜ₋₁[:,:,t] )`
   in-place, 
   ensuring numerical stability.

This is typically used in the predict step of the QKF for covariance.

"""
function predict_Pₜₜ₋₁!(Pₜₜ::AbstractArray{Real,3}, Pₜₜ₋₁::AbstractArray{Real,3},
    Σₜₜ₋₁::AbstractArray{Real,3}, Zₜₜ::AbstractMatrix{T}, tmpP::AbstractMatrix{T}, 
    params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    # 1) Update Σₜₜ₋₁ based on Zₜₜ (the code snippet calls compute_Σₜₜ₋₁!)
    compute_Σₜₜ₋₁!(Σₜₜ₋₁, Zₜₜ, params, t)

    # 2) Build predicted covariance
    @unpack Φ̃ = params
    Pₜₜ₋₁[:, :, t] = ensure_positive_definite(
        Φ̃ * Pₜₜ[:, :, t] * Φ̃' .+ Σₜₜ₋₁[:, :, t]
    )
end


"""
    predict_Pₜₜ₋₁(Pₜₜ::AbstractMatrix{T}, 
                   Zₜₜ::AbstractVecOrMat{<:Real}, 
                   params::QKParams{T,T2}, 
                   t::Int) -> Matrix{T}

Return a newly allocated **one-step-ahead predicted covariance** from 
the current covariance `Pₜₜ` and augmented state `Zₜₜ`, calling 
`compute_Σₜₜ₋₁` in the process.

# Arguments
- `Pₜₜ::AbstractMatrix{T}`: (P×P) the current filtered covariance.
- `Zₜₜ::AbstractVecOrMat{<:Real}`: The augmented state (dimension P) 
  or a matrix storing states if needed. 
- `params::QKParams{T,T2}`: Must contain:
  - `Φ̃::Matrix{T}`: The augmented transition matrix (P×P).
  - Possibly functions like `compute_Σₜₜ₋₁` to get extra noise terms.
- `t::Int`: Time index.

# Returns
- `Matrix{T}`: A new (P×P) matrix for the predicted covariance 
  `Φ̃ * Pₜₜ * Φ̃' + Σₜₜ₋₁`.

# Details
1. Calls `Σₜₜ₋₁ = compute_Σₜₜ₋₁(Zₜₜ, params, t)`.
2. Builds `P_tmp = Φ̃*Pₜₜ*Φ̃' + Σₜₜ₋₁`.
3. Checks `isposdef(P_tmp)`. If false, calls `make_positive_definite(P_tmp)`.
4. Returns the final matrix `P`.

Purely functional approach (allocates a new covariance) 
for AD or simpler code flow.

"""
function predict_Pₜₜ₋₁(Pₜₜ::AbstractMatrix{T}, Zₜₜ::AbstractVecOrMat{<:Real},
    params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    Σₜₜ₋₁ = compute_Σₜₜ₋₁(Zₜₜ, params, t)

    @unpack Φ̃ = params
    P_tmp = Φ̃ * Pₜₜ * Φ̃' .+ Σₜₜ₋₁

    if !isposdef(P_tmp)
        return make_positive_definite(P_tmp)
    else
        return P_tmp
    end
end


"""
    predict_Yₜₜ₋₁!(Yₜₜ₋₁::AbstractMatrix{T}, 
                   Zₜₜ₋₁::AbstractMatrix{T}, 
                   Y::AbstractMatrix{T}, 
                   params::QKParams{T,T2}, 
                   t::Int)

In-place update of the predicted measurement `Yₜₜ₋₁[:, t]` given:
- A *matrix* of state estimates `Zₜₜ₋₁`,
- A *matrix* of actual observations `Y`, and
- Parameter struct `params`.

# Arguments
- `Yₜₜ₋₁::AbstractMatrix{T}`: (M×T̄) array where the predicted measurement 
  at time `t` is stored in `Yₜₜ₋₁[:, t]`.
- `Zₜₜ₋₁::AbstractMatrix{T}`: (P×T̄) array of the one-step-ahead predicted states.
- `Y::AbstractMatrix{T}`: (M×T̄) array of actual observations.
- `params::QKParams{T,T2}`: Contains:
  - `A::Vector{T}`, `B̃::Matrix{T}`: The observation model terms.
  - `α::Matrix{T}`: Additional AR-like component for measurement.
- `t::Int`: Time index (1-based).

# Details
Computes `Yₜₜ₋₁[:, t] = A + B̃*Zₜₜ₋₁[:, t] + α * Y[:, t]`.
This is stored in-place in `Yₜₜ₋₁`. Purely a linear operation, 
so it is **AD-friendly**.

"""
function predict_Yₜₜ₋₁!(Yₜₜ₋₁::AbstractMatrix{T}, Zₜₜ₋₁::AbstractMatrix{T},
    Y::AbstractMatrix{T}, params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    @unpack A, B̃, α = params
    Yₜₜ₋₁[:, t] = A .+ B̃ * Zₜₜ₋₁[:, t] .+ α * Y[:, t]
end


"""
    predict_Yₜₜ₋₁!(Yₜₜ₋₁::AbstractVector{T}, 
                   Zₜₜ₋₁::AbstractMatrix{T}, 
                   Y::AbstractVector{T1}, 
                   params::QKParams{T,T2}, 
                   t::Int)

In-place update of the predicted measurement `Yₜₜ₋₁[t]` when both 
the measurement and observation are stored as vectors.

# Arguments
- `Yₜₜ₋₁::AbstractVector{T}`: A 1D array for predicted measurements, 
  storing the result at index `t`.
- `Zₜₜ₋₁::AbstractMatrix{T}`: (P×T̄) predicted states array.
- `Y::AbstractVector{T1}`: 1D array of actual measurements over time (size ≥ t).
- `params::QKParams{T,T2}`: Holds `A::Vector{T}`, `B̃::Matrix{T}`, 
  `α::Matrix{T}` or scalar, etc.  
- `t::Int`: The time index.

# Details
Computes `Yₜₜ₋₁[t] = (A + B̃ * Zₜₜ₋₁[:, t] + α * Y[t])[1]`, 
assuming `A` might be a 1-element vector, `α` is effectively 
(1×1), etc. Overwrites `Yₜₜ₋₁[t]` in place.

Useful for a univariate measurement scenario where measurement arrays 
are stored as vectors. AD-friendly as a single linear operation.

"""
function predict_Yₜₜ₋₁!(Yₜₜ₋₁::AbstractVector{T}, Zₜₜ₋₁::AbstractMatrix{T},
    Y::AbstractVector{T1}, params::QKParams{T,T2},
    t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

    @unpack A, B̃, α = params

    # Typically A is [scalar], B̃ is (1×P), α is (1×1?), 
    # so we do an indexing trick to retrieve a single scalar.
    Yₜₜ₋₁[t] = (A .+ B̃ * Zₜₜ₋₁[:, t] .+ α .* Y[t])[1]
end


"""
    predict_Yₜₜ₋₁!(Yₜₜ₋₁::AbstractMatrix{T}, 
                   Zₜₜ₋₁::AbstractMatrix{T}, 
                   Y::AbstractVector{T1}, 
                   params::QKParams{T,T2}, 
                   t::Int)

In-place update of `Yₜₜ₋₁[:, t]` using a vector `Y` of observations.

# Arguments
- `Yₜₜ₋₁::AbstractMatrix{T}`: (M×T̄) predicted measurement array; 
  we store the new column in `Yₜₜ₋₁[:, t]`.
- `Zₜₜ₋₋₁::AbstractMatrix{T}`: (P×T̄) predicted states.
- `Y::AbstractVector{T1}`: A 1D array of measurements over time (size ≥ t). 
  Typically used if `M==1`, or if `α` maps a scalar observation into M dims.
- `params::QKParams{T,T2}`: Model parameters, including 
  `A`, `B̃`, `α`.
- `t::Int`: Time index.

# Details
`Yₜₜ₋₁[:, t] = A + B̃*Zₜₜ₋₁[:, t] + α * Y[t]`.

If `M > 1`, we interpret `α` as an (M×1) or (M×M) matrix that 
broadcasts scalar Y[t] across M. 
AD-friendly, single linear expression.

"""
function predict_Yₜₜ₋₁!(Yₜₜ₋₁::AbstractMatrix{T}, Zₜₜ₋₁::AbstractMatrix{T},
    Y::AbstractVector{T1}, params::QKParams{T,T2}, 
    t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

    @unpack A, B̃, α = params
    Yₜₜ₋₁[:, t] = A .+ B̃ * Zₜₜ₋₁[:, t] .+ α .* Y[t]
end


"""
    predict_Yₜₜ₋₁(Zₜₜ₋₁::AbstractMatrix{T}, 
                  Y::AbstractMatrix{T1}, 
                  params::QKParams{T,T2}, 
                  t::Int) 
                  -> Vector{T}

Return a **new** predicted measurement vector for time `t`, 
given matrix-based `Zₜₜ₋₁` and `Y`.

# Arguments
- `Zₜₜ₋₁::AbstractMatrix{T}`: (P×T̄) predicted states.
- `Y::AbstractMatrix{T1}`: (M×T̄) actual observations.
- `params::QKParams{T,T2}`: Contains `A`, `B̃`, `α`.
- `t::Int`: Time index.

# Returns
- `Vector{T}`: The predicted measurement at time `t`, 
  computed as `A + B̃*Zₜₜ₋₁[:, t] + α*Y[:, t]`.

# Details
This version returns the result as a newly allocated vector. 
Use it if you want a purely functional style with no in-place modifications.

"""
function predict_Yₜₜ₋₁(Zₜₜ₋₁::AbstractMatrix{T}, Y::AbstractMatrix{T1},
    params::QKParams{T,T2}, t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack A, B̃, α = params
    return A .+ B̃ * Zₜₜ₋₁[:, t] .+ α * Y[:, t]
end


"""
    predict_Yₜₜ₋₁(Zₜₜ₋₁::AbstractVector{T}, 
                  Y::AbstractVector{T1}, 
                  params::QKParams{T,T2}, 
                  t::Int) 
                  -> Real (or 1-element vector)

Compute a new predicted measurement for univariate (or effectively scalar) 
observation, returning the result as a scalar.

# Arguments
- `Zₜₜ₋₁::AbstractVector{T}`: The predicted state vector of dimension P.
- `Y::AbstractVector{T1}`: The (univariate) actual observations over time.
- `params::QKParams{T,T2}`: Must hold `A::Vector{T}` (often length=1), 
  `B̃::Matrix{T}` (size 1×P?), `α::Matrix{T}` or scalar for the measurement update.
- `t::Int`: Time index.

# Returns
A single real number `predict_val`, computed as `(A + B̃*Zₜₜ₋₁ + α*Y[t])[1]`.

# Details
- If `A` has length > 1, or `B̃` is larger, you might effectively get the first element. 
- For truly univariate measurement, this is a simpler approach.
"""
function predict_Yₜₜ₋₁(Zₜₜ₋₁::AbstractVector{T}, Y::AbstractVector{T1},
    params::QKParams{T,T2}, t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack A, B̃, α = params
    return (A .+ B̃*Zₜₜ₋₁ .+ α .* Y[t])[1]
end


"""
    predict_Mₜₜ₋₁!(Mₜₜ₋₁, Pₜₜ₋₁, tmpB, params, t)

In-place computation of the predicted measurement covariance `Mₜₜ₋₁[:,:,t]` 
in a **constant** measurement noise scenario, storing the result into 
the 3D array `Mₜₜ₋₁`.

# Arguments
- `Mₜₜ₋₁::AbstractArray{T,3}`: A 3D array `(M×M×T̄)` for measurement covariances 
  over time. `Mₜₜ₋₁[:,:,t]` will be updated in place.
- `Pₜₜ₋₁::AbstractArray{T,3}`: The 3D array `(P×P×T̄)` of one-step-ahead 
  predicted state covariances. We read `Pₜₜ₋₁[:,:,t]`.
- `tmpB::AbstractMatrix{T}`: A working buffer of size `(M×P)` used for 
  intermediate matrix multiplication, e.g. `B̃ * Pₜₜ₋₁[:,:,t]`.
- `params::QKParams{T,T2}`: Parameter struct with:
  - `M::Int`: Measurement dimension.
  - `B̃::Matrix{T}`: Observation matrix in the QKF model (size `M×P`).
  - `V::Matrix{T}`: Constant measurement noise covariance (size `M×M`).
- `t::Int`: Time index (1-based).

# Details
1. We compute in place: `Mₜₜ₋₁[:,:,t] = B̃ * Pₜₜ₋₁[:,:,t] * B̃' + V`.
   - First do `mul!(tmpB, B̃, Pₜₜ₋₁[:,:,t])`.
   - Then do `mul!(Mₜₜ₋₁[:,:,t], tmpB, B̃', 1.0, 0.0)`.
   - Finally, add `V` to `Mₜₜ₋₁[:,:,t]`.
2. If `M==1` (univariate), we clamp any negative scalar 
   (`Mₜₜ₋₁[1,1,t] < 0 => 1e-4`).
3. Otherwise, if `M>1`, we call `isposdef(Mₜₜ₋₁[:,:,t])`. If false, we fix it 
   by `make_positive_definite` and overwrite `Mₜₜ₋₁[:,:,t]`.

This approach is **AD-friendly** since it keeps the final result 
in `Mₜₜ₋₁[:,:,t]` with minimal branching. 
Use this routine when `V` is a **precomputed constant** matrix, 
no longer time-varying.
"""
function predict_Mₜₜ₋₁!(Mₜₜ₋₁::AbstractArray{T,3}, Pₜₜ₋₁::AbstractArray{T,3},
    tmpB::AbstractMatrix{T}, params::QKParams{T,T2}, t::Int) where {T<:Real, T2<:Real}

    @unpack M, B̃, V = params

    # 1) In-place multiplication:
    #    tmpB = B̃ * Pₜₜ₋₁[:,:,t]
    mul!(tmpB, B̃, Pₜₜ₋₁[:,:,t])

    #    Mₜₜ₋₁[:,:,t] = tmpB * B̃'
    mul!(Mₜₜ₋₁[:,:,t], tmpB, B̃', 1.0, 0.0)

    # 2) Add V to Mₜₜ₋₁[:,:,t]
    @inbounds for i in 1:M
        for j in 1:M
            Mₜₜ₋₁[i,j,t] += V[i,j]
        end
    end

    # 3) Check for negativity in univariate case
    if M == 1
        if Mₜₜ₋₁[1,1,t] < 0
            Mₜₜ₋₁[1,1,t] = 1e-4
        end

    # 4) For multivariate, enforce positive-definiteness if needed
    else
        # `view` avoids copying, but you can index directly as well
        local slice_ref = @view Mₜₜ₋₁[:,:,t]
        if !isposdef(slice_ref)
            local corrected = make_positive_definite(slice_ref)
            slice_ref .= corrected
        end
    end
end


"""
    predict_Mₜₜ₋₁(Pₜₜ₋₁::AbstractMatrix{T}, params::QKParams{T,T2})
                  -> Matrix{T}

Compute the predicted measurement covariance `Mₜₜ₋₁` for the current time step,
assuming a **precomputed**, constant noise matrix `V` in `QKParams`.

# Arguments
- `Pₜₜ₋₁::AbstractMatrix{T}`: The one-step-ahead predicted covariance of the state.
- `params::QKParams{T,T2}`: A parameter struct holding:
  - `M::Int`: Measurement dimension.
  - `B̃::Matrix{T}`: Augmented observation matrix in the QKF model.
  - `V::Matrix{T}`: Precomputed, constant measurement noise/covariance matrix.

# Returns
- `Matrix{T}`: The predicted measurement covariance `Mₜₜ₋₁`. 
  - If `M=1` (univariate), the scalar case is checked for negativity; 
    if < 0, it's clamped to `[1e-4]`.  
  - If `M>1`, we check `isposdef`. If not positive definite, we call 
    `make_positive_definite` to fix it.

# Details
1. We compute `M_tmp = B̃*Pₜₜ₋₁*B̃' + V`.  
2. If univariate (`M==1`) and `M_tmp[1,1] < 0`, we clamp to a small positive `[1e-4]`.  
3. If multivariate (`M>1`), we call `isposdef(M_tmp)`. If it fails, 
   we fix it with `make_positive_definite(M_tmp)`.

This purely functional approach is **AD-friendly**, and the small clamp 
protects against numeric issues when `M=1`.

"""
function predict_Mₜₜ₋₁(Pₜₜ₋₁::AbstractMatrix{T}, params::QKParams{T,T2}) where 
    {T<:Real, T2<:Real}

    @unpack M, B̃, V = params
    
    # 1) Compute the "raw" predicted covariance for the measurement
    M_tmp = B̃ * Pₜₜ₋₁ * B̃' .+ V

    # 2) Handle univariate vs. multivariate
    if M == 1
        # Univariate => just a 1×1
        if M_tmp[1,1] < 0
            return reshape([1e-4], 1, 1)
        else
            return M_tmp
        end
    else
        # Multivariate => check positive definiteness
        return isposdef(M_tmp) ? M_tmp : make_positive_definite(M_tmp)
    end
end

"""
    compute_Kₜ!(Kₜ::AbstractArray{Real,3}, 
                Pₜₜ₋₁::AbstractArray{Real,3},
                Mₜₜ₋₁::AbstractArray{Real,3}, 
                tmpPB::AbstractMatrix{T}, 
                params::QKParams{T,T2}, 
                t::Int)

In-place computation of the **Kalman gain** `Kₜ[:, :, t]` for a **univariate** measurement 
scenario (i.e., `M=1`) within a Kalman or QKF step.

# Arguments
- `Kₜ::AbstractArray{Real,3}`: A `(P×M×T̄)` array (often `(P×1×T̄)`) storing the Kalman gain.
  We write the result into `Kₜ[:, :, t]`.
- `Pₜₜ₋₁::AbstractArray{Real,3}`: One-step-ahead predicted covariance, `(P×P×T̄)`.
- `Mₜₜ₋₁::AbstractArray{Real,3}`: Predicted measurement covariance, `(M×M×T̄)`. 
  For `M=1`, this is a `1×1×T̄`.
- `tmpPB::AbstractMatrix{T}`: A working buffer `(P×M)` if needed, but currently unused 
  in your snippet.
- `params::QKParams{T,T2}`: Must contain:
  - `B̃::Matrix{T}` (size `M×P`),
  - `M, P::Int` for dimensions.
- `t::Int`: Time index.

# Details
We compute:
```julia
Kₜ[:, :, t] = (Pₜₜ₋₁[:, :, t] * B̃') / Mₜₜ₋₁[:, :, t]
```
# Notes:
- If Mₜₜ₋₁[:, :, t] is 1×1, the division is just a scalar divide. For genuine multivariate M>1, one would typically do a matrix inverse or solve. This snippet is thus best suited for univariate measurement.
- The in-place assignment .= ... is AD-friendly if your AD can handle assignment to slices in a 3D array. If needed, define or use a custom adjoint.
"""
function compute_Kₜ!(Kₜ::AbstractArray{Real,3}, Pₜₜ₋₁::AbstractArray{Real,3},
    Mₜₜ₋₁::AbstractArray{Real,3}, tmpPB::AbstractMatrix{T}, params::QKParams{T,T2},
    t::Int) where {T <: Real, T2 <: Real}

    @unpack B̃ = params

    # For univariate M=1: (Pₜₜ₋₁ * B̃') is (P×1). Then we divide by Mₜₜ₋₁(1,1,t).
    # Directly do an elementwise assignment.
    Kₜ[:, :, t] .= (Pₜₜ₋₁[:, :, t] * B̃') ./ Mₜₜ₋₁[:, :, t]
end

"""
    compute_Kₜ(Pₜₜ₋₁::AbstractMatrix{T}, 
               Mₜₜ₋₁::AbstractMatrix{T},
               params::QKParams{T,T2}, 
               t::Int) -> Matrix{T}

Compute and **return** the Kalman gain `Kₜ` for a **univariate** measurement scenario, 
given `Pₜₜ₋₁` and `Mₜₜ₋₁`.

# Arguments
- `Pₜₜ₋₁::AbstractMatrix{T}`: `(P×P)` one-step-ahead predicted covariance for the state.
- `Mₜₜ₋₁::AbstractMatrix{T}`: `(M×M)` measurement covariance, presumably `1×1`. 
- `params::QKParams{T,T2}`: Must contain:
  - `B̃::Matrix{T}` (size `M×P`),
  - Possibly `M, P::Int` if needed.
- `t::Int`: Time index (not used in the snippet, but might be for consistency).

# Returns
- `Matrix{T}`: The Kalman gain of size `(P×M)`. 
  If `M=1`, then `(P×1)` is returned.

# Notes:
- If Mₜₜ₋₁ is 1×1, S is also a 1×1. The / operator is scalar division.
- For multi-dimensional S, you'd typically invert or solve. This snippet is thus best for univariate measurement.

# Details
We do:
```julia
S = B̃ * Pₜₜ₋₁ * B̃' + Mₜₜ₋₁
K = Pₜₜ₋₁ * B̃' / S
"""
function compute_Kₜ( Pₜₜ₋₁::AbstractMatrix{T}, Mₜₜ₋₁::AbstractMatrix{T},
    params::QKParams{T,T2}, t::Int ) where {T <: Real, T2 <: Real}
    
    @unpack B̃ = params
    
    # 1) S is a 1×1 if M=1. Then S[1,1] is the scalar denominator.
    S = B̃ * Pₜₜ₋₁ * B̃' .+ Mₜₜ₋₁
    
    # 2) The gain K is (P×M).
    return Pₜₜ₋₁ * B̃' ./ S
end

"""
    update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, 
                 Kₜ::AbstractArray{Real,3}, 
                 Yₜ::AbstractMatrix{T1}, 
                 Yₜₜ₋₁::AbstractMatrix{T}, 
                 Zₜₜ₋₁::AbstractMatrix{T}, 
                 tmpϵ::AbstractVector{T}, 
                 params::QKParams{T,T2}, 
                 t::Int)

In-place update of the augmented state `Zₜₜ[:, t+1]` given:
- A measurement vector `Yₜ[:, t]` stored in a matrix `(M×T̄)`,
- A predicted measurement `Yₜₜ₋₁[:, t]` also in `(M×T̄)`,
- The one-step-ahead state `Zₜₜ₋₁[:, t]`,
- and a Kalman-like gain `Kₜ[:, :, t]` of size `(P×M)`.

# Arguments
- `Zₜₜ::AbstractMatrix{T}`: (P×T̄) storing the **updated** states in column `t+1`.
- `Kₜ::AbstractArray{Real,3}`: A `(P×M×T̄)` array of gains.
- `Yₜ::AbstractMatrix{T1}`: Measurement data of size `(M×T̄)`.
- `Yₜₜ₋₁::AbstractMatrix{T}`: Predicted measurement `(M×T̄)`.
- `Zₜₜ₋₁::AbstractMatrix{T}`: One-step-ahead predicted state `(P×T̄)`.
- `tmpϵ::AbstractVector{T}`: A temporary buffer of length `M`, if needed for manual loops.
- `params::QKParams{T,T2}`: Contains state dimension `P`, measurement dimension `M`, etc.
- `t::Int`: The time index (1-based).

# Details
We perform: Zₜₜ[:, t+1] = Zₜₜ₋₁[:, t] + Kₜ[:, :, t] * (Yₜ[:, t] - Yₜₜ₋₁[:, t])
in place. `tmpϵ` can be used if you want a manual loop approach, but here we do 
the direct matrix expression. This is AD-friendly in many frameworks, though 
in-place writes can require custom adjoints depending on the AD library.

"""
function update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, Kₜ::AbstractArray{Real,3},
    Yₜ::AbstractMatrix{T1}, Yₜₜ₋₁::AbstractMatrix{T}, Zₜₜ₋₁::AbstractMatrix{T}, 
    tmpϵ::AbstractVector{T}, params::QKParams{T,T2}, t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack M, P = params
    Zₜₜ[:, t + 1] = Zₜₜ₋₁[:, t] .+ Kₜ[:, :, t] * (Yₜ[:, t] .- Yₜₜ₋₁[:, t])
end

"""
    update_Zₜₜ(Kₜ::AbstractArray{Real,3}, 
                Yₜ::AbstractMatrix{T1},
                Yₜₜ₋₁::AbstractMatrix{T1}, 
                Zₜₜ₋₁::AbstractMatrix{T1}, 
                params::QKParams{T,T2}, 
                t::Int)

Return a newly updated state vector for time `t+1` given matrix-based measurements.

# Arguments
- `Kₜ::AbstractArray{Real,3}`: (P×M×T̄) Kalman gains.
- `Yₜ::AbstractMatrix{T1}`, `Yₜₜ₋₁::AbstractMatrix{T1}`: (M×T̄) measurement and predicted measurement.
- `Zₜₜ₋₁::AbstractMatrix{T1}`: (P×T̄) the predicted state.
- `params::QKParams{T,T2}`: Contains `M, P`, etc.
- `t::Int`: Time index.

# Returns
- `Vector{T1}`: The updated state at time `t+1`, 
  computed as `Zₜₜ₋₁[:, t] + Kₜ[:, :, t]*(Yₜ[:, t] - Yₜₜ₋₁[:, t])`.

# Details
This is a purely functional approach, allocating a new vector. 
It's often simpler for AD, as it avoids in-place modifications.

"""
function update_Zₜₜ(Kₜ::AbstractArray{Real,3}, Yₜ::AbstractMatrix{T1},
    Yₜₜ₋₁::AbstractMatrix{T1}, Zₜₜ₋₁::AbstractMatrix{T1}, params::QKParams{T,T2}, 
    t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack M, P = params
    return Zₜₜ₋₁[:, t] .+ Kₜ[:, :, t] * (Yₜ[:, t] .- Yₜₜ₋₁[:, t])
end


"""
    update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, 
                 Kₜ::AbstractArray{Real,3},
                 Yₜ::AbstractVector{T1}, 
                 Yₜₜ₋₁::AbstractVector{T}, 
                 Zₜₜ₋₁::AbstractMatrix{T},
                 tmpϵ::AbstractVector{T}, 
                 params::QKParams{T,T2}, 
                 t::Int)

In-place update of `Zₜₜ[:, t+1]` when the measurement `Yₜ` and predicted measurement `Yₜₜ₋₁`
are stored as 1D arrays (univariate or effectively scalar per time).

# Arguments
- `Zₜₜ::AbstractMatrix{T}`: Updated states are stored in `Zₜₜ[:, t+1]`.
- `Kₜ::AbstractArray{Real,3}`: Gains `(P×M×T̄)`.
- `Yₜ::AbstractVector{T1}`, `Yₜₜ₋₁::AbstractVector{T}`: 1D arrays storing values at index `t`.
- `Zₜₜ₋₁::AbstractMatrix{T}`: Predicted state `(P×T̄)`.
- `tmpϵ::AbstractVector{T}`: Temp buffer for difference if needed.
- `params::QKParams{T,T2}`, `t::Int`: Model and time info.

# Details
We do Zₜₜ[:, t+1] = Zₜₜ₋₁[:, t] + Kₜ[:, :, t]*(Yₜ[t] - Yₜₜ₋₁[t])
in place. Typically `M=1` in this scenario.

"""
function update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, Kₜ::AbstractArray{Real,3},
    Yₜ::AbstractVector{T1}, Yₜₜ₋₁::AbstractVector{T}, Zₜₜ₋₁::AbstractMatrix{T},
    tmpϵ::AbstractVector{T}, params::QKParams{T,T2}, t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack M, P = params
    Zₜₜ[:, t + 1] = Zₜₜ₋₁[:, t] .+ Kₜ[:, :, t] * (Yₜ[t] - Yₜₜ₋₁[t])
end


"""
    update_Zₜₜ(Kₜ::AbstractArray{Real,3},
                Yₜ::AbstractVector{T1},
                Yₜₜ₋₁::AbstractVector{T},
                Zₜₜ₋₁::AbstractMatrix{T},
                params::QKParams{T,T2},
                t::Int)

Return a new updated state for time `t+1` in a univariate measurement scenario, 
where `Yₜ` and `Yₜₜ₋₁` are 1D arrays.

# Arguments
- `Kₜ::AbstractArray{Real,3}`: (P×M×T̄) gains.
- `Yₜ::AbstractVector{T1}`, `Yₜₜ₋₁::AbstractVector{T}`: 1D arrays with the observation at `Yₜ[t]`
  and predicted at `Yₜₜ₋₁[t]`.
- `Zₜₜ₋₁::AbstractMatrix{T}`: (P×T̄) predicted states.
- `params::QKParams{T,T2}`, `t::Int`: Model/time info.

# Returns
A newly allocated vector `(P×1)` for time `t+1`, computed as: 
`Zₜₜ₋₁[:, t] + Kₜ[:, :, t]*(Yₜ[t] - Yₜₜ₋₁[t])`.
This is a purely functional approach.

"""
function update_Zₜₜ(Kₜ::AbstractArray{Real,3}, Yₜ::AbstractVector{T1},
    Yₜₜ₋₁::AbstractVector{T}, Zₜₜ₋₁::AbstractMatrix{T}, params::QKParams{T,T2},
    t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack M, P = params
    return Zₜₜ₋₁[:, t] .+ Kₜ[:, :, t] * (Yₜ[t] - Yₜₜ₋₁[t])
end

"""
    update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, 
                 Kₜ::AbstractArray{Real,3},
                 Yₜ::AbstractMatrix{T1}, 
                 Yₜₜ₋₁::AbstractVector{T},
                 Zₜₜ₋₁::AbstractMatrix{T},
                 tmpϵ::AbstractVector{T},
                 params::QKParams{T,T2},
                 t::Int)

In-place update of `Zₜₜ[:, t+1]` with a **mixed** scenario:
- `Yₜ` is `(M×T̄)`,
- `Yₜₜ₋₁` is `(M×1)?` or a vector storing predicted measurement at index `t`.

# Arguments
- `Zₜₜ::AbstractMatrix{T}`, `Zₜₜ₋₁::AbstractMatrix{T}`: (P×T̄) augmented states.
- `Kₜ::AbstractArray{Real,3}`: (P×M×T̄) gains.
- `Yₜ::AbstractMatrix{T1}`: (M×T̄) measurement.
- `Yₜₜ₋₁::AbstractVector{T}`: (M) predicted measurement at time `t`.
- `tmpϵ::AbstractVector{T}`: temp buffer if needed.
- `params::QKParams{T,T2}`, `t::Int`.

# Details
Update rule:
`Zₜₜ[:, t+1] = Zₜₜ₋₁[:, t] + Kₜ[:, :, t]*(Yₜ[:, t] - Yₜₜ₋₁[t])`
Note that `Yₜₜ₋₁[t]` might be used if `Yₜₜ₋₁` has length `M`, 
and we broadcast or something similar. 
Keep track that dimensional consistency is correct for your usage.

"""
function update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, Kₜ::AbstractArray{Real,3}, 
    Yₜ::AbstractMatrix{T1}, Yₜₜ₋₁::AbstractVector{T}, Zₜₜ₋₁::AbstractMatrix{T},
    tmpϵ::AbstractVector{T}, params::QKParams{T,T2},
    t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack M, P = params
    Zₜₜ[:, t + 1] = Zₜₜ₋₁[:, t] .+ Kₜ[:, :, t] * (Yₜ[:, t] .- Yₜₜ₋₁[t])
end

"""
    update_Zₜₜ(Kₜ::AbstractArray{Real,3},
                Yₜ::AbstractMatrix{T1},
                Yₜₜ₋₁::AbstractVector{T},
                Zₜₜ₋₁::AbstractMatrix{T},
                params::QKParams{T,T2},
                t::Int)

Return a new state update with measurement `Yₜ` as (M×T̄), 
but `Yₜₜ₋₁` as an (M) vector indexing `t` for the predicted measurement. 
This is a somewhat "mixed" usage scenario.

# Arguments
- `Kₜ::AbstractArray{Real,3}`: Gains `(P×M×T̄)`.
- `Yₜ::AbstractMatrix{T1}`: `(M×T̄)` measurements.
- `Yₜₜ₋₁::AbstractVector{T}`: size `M` predicted measurement, used at index `t`.
- `Zₜₜ₋₁::AbstractMatrix{T}`: `(P×T̄)` predicted states.
- `params::QKParams{T,T2}`, `t::Int`.

# Returns
A `(P×1)` updated state vector for time `t+1`, 
via `Zₜₜ₋₁[:, t] + Kₜ[:, :, t] * (Yₜ[:, t] - Yₜₜ₋₁[t])`.

"""
function update_Zₜₜ(Kₜ::AbstractArray{Real,3}, Yₜ::AbstractMatrix{T1},
    Yₜₜ₋₁::AbstractVector{T}, Zₜₜ₋₁::AbstractMatrix{T}, params::QKParams{T,T2},
    t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack M, P = params
    return Zₜₜ₋₁[:, t] .+ Kₜ[:, :, t] * (Yₜ[:, t] .- Yₜₜ₋₁[t])
end


"""
    update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, 
                 Kₜ::AbstractArray{Real,3},
                 Yₜ::AbstractVector{T1}, 
                 Yₜₜ₋₁::AbstractMatrix{T},
                 Zₜₜ₋₁::AbstractMatrix{T},
                 tmpϵ::AbstractVector{T}, 
                 params::QKParams{T,T2}, 
                 t::Int)

In-place update of the state with a "mixed" scenario:
- `Yₜ` is 1D (size `M×1`?), 
- `Yₜₜ₋₁` is `(M×T̄)`?

# Arguments
- `Zₜₜ::AbstractMatrix{T}`, `Zₜₜ₋₁::AbstractMatrix{T}`: State arrays `(P×T̄)`.
- `Kₜ::AbstractArray{Real,3}`: Gains `(P×M×T̄)`.
- `Yₜ::AbstractVector{T1}`, `Yₜₜ₋₁::AbstractMatrix{T}`: 
  The actual measurement at `Yₜ[t]`, predicted measurement at `Yₜₜ₋₁[:, t]`.
- `tmpϵ::AbstractVector{T}`: buffer if needed.
- `params::QKParams{T,T2}`, `t::Int`.

# Implementation
`Zₜₜ[:, t+1] = Zₜₜ₋₁[:, t] + Kₜ[:, :, t]*(Yₜ[t] - Yₜₜ₋₁[:, t])`
Dimension checks are up to the user; ensure consistent shapes for broadcast.

"""
function update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, Kₜ::AbstractArray{Real,3},
    Yₜ::AbstractVector{T1}, Yₜₜ₋₁::AbstractMatrix{T}, Zₜₜ₋₁::AbstractMatrix{T},
    tmpϵ::AbstractVector{T}, params::QKParams{T,T2}, 
    t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack M, P = params
    Zₜₜ[:, t + 1] = Zₜₜ₋₁[:, t] .+ Kₜ[:, :, t] * (Yₜ[t] - Yₜₜ₋₁[:, t])
end

"""
    update_Zₜₜ(Kₜ::AbstractMatrix{T}, 
                Yₜ::Real, 
                Yₜₜ₋₁::Real, 
                Zₜₜ₋₁::AbstractVector{T}, 
                params::QKParams{T,T2}, 
                t::Int)

Return a new updated state when:
- `Kₜ` is `(P×1)`,
- `Yₜ` and `Yₜₜ₋₁` are scalar reals,
- `Zₜₜ₋₁` is `(P×1)` or `Vector{T}`.

# Arguments
- `Kₜ::AbstractMatrix{T}`: The gain (P×1).
- `Yₜ::Real`: The scalar measurement at time `t`.
- `Yₜₜ₋₁::Real`: The predicted scalar measurement.
- `Zₜₜ₋₁::AbstractVector{T}`: The predicted state (P).
- `params::QKParams{T,T2}`, `t::Int`: Possibly used for dimension checks or in a bigger loop.

# Returns
- `Vector{T}`: The new updated state: `Zₜₜ₋₁ + Kₜ*(Yₜ - Yₜₜ₋₁)`.

# Description
This is the simplest possible scenario: 
**univariate** measurement (`Yₜ` is a real) 
and a vector state `(P×1)`. We do a purely functional 
update returning a brand-new `Vector{T}`.

"""
function update_Zₜₜ(Kₜ::AbstractMatrix{T}, Yₜ::Real, Yₜₜ₋₁::Real, Zₜₜ₋₁::AbstractVector{T},
    params::QKParams{T,T2}, t::Int
) where {T <: Real, T2 <: Real}

    return Zₜₜ₋₁ .+ Kₜ * (Yₜ - Yₜₜ₋₁)
end

"""
    update_Pₜₜ!(Pₜₜ, Kₜ, Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, tmpKM, tmpKMK, params, t)

In-place update of the **filtered covariance** `Pₜₜ[:, :, t+1]` given the one-step-ahead 
covariance `Pₜₜ₋₁[:, :, t]`, and the Kalman gain `Kₜ[:, :, t]`.

# Arguments
- `Pₜₜ::AbstractArray{<:Real, 3}`: A 3D array `(P×P×T̄)` storing filtered covariances.  
  We write the new filtered covariance into `Pₜₜ[:, :, t+1]`.
- `Kₜ::AbstractArray{<:Real, 3}`: A 3D array `(P×M×T̄)` of Kalman gains, 
  using `Kₜ[:, :, t]`.
- `Mₜₜ₋₁::AbstractArray{<:Real, 3}`: *(Currently unused)* or was presumably
  the predicted measurement covariance `(M×M×T̄)`. Not directly used in the final expression.
- `Pₜₜ₋₁::AbstractArray{<:Real, 3}`: The one-step-ahead covariance array `(P×P×T̄)`, 
  from which `Pₜₜ₋₁[:, :, t]` is read.
- `Zₜₜ₋₁::AbstractArray{<:Real, 2}`: `(P×T̄)` storing the predicted augmented state. 
- `tmpKM, tmpKMK::AbstractMatrix{<:Real}`: Temporary buffers `(P×M, P×P)` 
  if you want manual multiplication. In the final code, we do not use them, 
  but they can be placeholders for expansions.
- `params::QKParams{T,T2}`: Must contain:
  - `B̃::Matrix{T}` (size `M×P`),
  - `P::Int, M::Int`, etc. 
- `t::Int`: The time index (1-based).

# Computation
1. Let `A = I - Kₜ[:, :, t]*B̃`.
2. Then
    `Pₜₜ[:, :, t+1] = make_positive_definite(A * Pₜₜ₋₁[:, :, t] * A' + Kₜ[:, :, t]*V_tmp*Kₜ[:, :, t])'
3. We wrap the result with make_positive_definite to ensure no negative eigenvalues from the update.

# Notes
- This is an in-place update: we store the new covariance in Pₜₜ[:, :, t+1].
- AD-Friendliness: The final expression is a typical linear+outer-product operation plus
   make_positive_definite. If your AD can handle in-place modifications, or you define a
   custom adjoint, it should be fine. Otherwise, consider a purely functional approach.
"""
function update_Pₜₜ!( Pₜₜ::AbstractArray{Real,3}, Kₜ::AbstractArray{Real,3},
    Mₜₜ₋₁::AbstractArray{Real,3}, Pₜₜ₋₁::AbstractArray{Real,3},
    Zₜₜ₋₁::AbstractArray{Real,2}, tmpKM::AbstractMatrix{Real},
    tmpKMK::AbstractMatrix{Real}, params::QKParams{T,T2}, t::Int ) where {T<:Real, T2<:Real}
    @unpack B̃, V = params
    
    # 1) Form A = I - Kₜ[:, :, t]*B̃
    local A = I - Kₜ[:, :, t]*B̃
    
    # 2) In-place assign the final updated covariance
    Pₜₜ[:, :, t+1] = make_positive_definite(
        A * Pₜₜ₋₁[:, :, t] * A' .+ (Kₜ[:, :, t] * V * Kₜ[:, :, t]')
    )
end

"""
    update_Pₜₜ(Kₜ, Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, params, t)

Compute and return a **new** filtered covariance matrix, 
given a predicted covariance `Pₜₜ₋₁`, a Kalman gain `Kₜ`.

# Arguments
- `Kₜ::AbstractMatrix{T5}`: (P×M) or (P×1) Kalman gain for time `t`.
- `Mₜₜ₋₁::AbstractMatrix{T6}`: *(Potentially unused here or if you only 
  rely on `V_tmp`?),*
- `Pₜₜ₋₁::AbstractMatrix{T3}`: The one-step-ahead covariance `(P×P)`.
- `Zₜₜ₋₁::AbstractVector{T4}`: The predicted augmented state (length P).
- `t::Int`: The time index (not strictly used, but might be for logging or consistency).

# Returns
- `Matrix{T}`: A newly allocated `(P×P)` covariance matrix 
  after the update step, guaranteed to be positive-definite if `make_positive_definite` 
  fixes any negative eigenvalues.

# Details
1. Let `A = I - Kₜ*B̃`.
2. Build `P = A * Pₜₜ₋₁ * A' + Kₜ*V_tmp*Kₜ'`
3. This version is purely functional (returns a new matrix). It's often simpler for AD if
   you want a direct forward pass without in-place modifications.

"""
function update_Pₜₜ( Kₜ::AbstractMatrix{T5}, Mₜₜ₋₁::AbstractMatrix{T6},
    Pₜₜ₋₁::AbstractMatrix{T3}, Zₜₜ₋₁::AbstractVector{T4}, params::QKParams{T,T2},
    t::Int ) where {T<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real}
    @unpack B̃, V = params


    # 1) A = I - Kₜ*B̃
    local A = I - Kₜ*B̃

    # 2) Build the new covariance
    local P = A * Pₜₜ₋₁ * A' .+ (Kₜ * V * Kₜ')

    # 4) Ensure positivity
    if !isposdef(P)
        return make_positive_definite(P)
    else
        return P
    end
end

"""
    correct_Zₜₜ!(Zₜₜ, params, t)

In-place correction of the sub-block in `Zₜₜ` that corresponds to the implied
covariance `[XX']ₜ|ₜ - Xₜ|ₜ Xₜ|ₜ'`. This function ensures that sub-block is 
positive semidefinite (PSD).

# Arguments
- `Zₜₜ::AbstractMatrix{<:Real}`:
  A matrix where each column is a state vector. The `(t+1)`th column contains:
    - The first `N` entries = `X_t|t`
    - The next `N*N` entries (reshaped to `N×N`) = `[X X']_t|t`.
- `params::QKParams`: A parameter struct holding (at least) `N`, the state dimension.
- `t::Int`: The time index whose data we are correcting. 

# Details
1. Extract the current column `Ztt = Zₜₜ[:, t+1]`.
2. Let `x_t = Ztt[1:N]`.
3. Implied covariance = `reshape(Ztt[N+1:end], N, N) - x_t * x_t'`.
4. We compute its eigen-decomposition and clamp negative eigenvalues to 0. 
5. Reconstruct the PSD version and store it back into `Zₜₜ[N+1:end, t+1]`.

This follows the idea in the QKF algorithm to keep the implied covariance valid.
"""
function correct_Zₜₜ!(Zₜₜ::AbstractMatrix{T1}, params::QKParams{T, T2}, t::Int) where 
    {T1 <: Real, T <: Real, T2 <: Real}

    @unpack N = params
    
    # 1) Extract relevant piece
    Ztt = Zₜₜ[:, t + 1] # Zₜₜ[:, t+1] is the "current" column in the filter
    
    # 2) Extract x_t and implied (XX')ₜ
    xt = Ztt[1:N]                              # the first N entries are xₜ|ₜ
    XtXt_prime = reshape(Ztt[N+1:end], N, N)    # the next N*N entries are XX' in vectorized form
    
    # 3) implied_cov = (XX') - xₜ xₜ'
    implied_cov = XtXt_prime - xt*xt'
    
    # 4) Eigen-decomposition to clamp negative eigenvalues
    #    Use `Symmetric` or `Hermitian` to ensure real sym decomposition
    eig_vals, eig_vecs = eigen(Symmetric(implied_cov))
    
    # 5) Replace negative eigenvalues with 0 => PSD
    eig_vals .= max.(eig_vals, 0.0)
    
    # 6) Reassemble the corrected block = V * diag(λ⁺) * V' + xₜ xₜ'
    corrected = eig_vecs * Diagonal(eig_vals) * eig_vecs' + x_t*x_t'
    
    # 7) Write it back into the corresponding part of Zₜₜ
    Zₜₜ[N+1:end, t + 1] = vec(corrected)
    
    return nothing
end


"""
    correct_Zₜₜ(Zₜₜ, params, t) -> Vector

Non-in-place version returning a new corrected state vector, 
using a simple ε-shift instead of eigenvalue truncation.

# Arguments
- `Zₜₜ::AbstractVector{<:Real}`:
  A single "flattened" state vector, containing:
    - The first `N` entries = `x_t|t`
    - The next `N*N` entries (reshaped to `N×N`) = `[X X']_t|t`.
- `params::QKParams`: A parameter struct holding (at least) `N`, the state dimension.
- `t::Int`: (Unused here, but consistent with the signature of the in-place version.)

# Details
1. Extract `x_t` from the front.
2. Compute implied_cov = (XX') - xₜ xₜ'.
3. "Correct" it by adding `ε*I` to ensure positivity (a cheap fix).
4. Recombine into a single vector: [x_t; vec(corrected_cov + xₜ xₜ')].

Returns the newly constructed vector, leaving the original unchanged.
"""
function correct_Zₜₜ(Zₜₜ::AbstractVector{T1}, params::QKParams{T,T2}, t::Int) where 
    {T1 <: Real, T <: Real, T2 <: Real}

    @unpack N = params
    
    # 1) Extract xₜ
    xt = Zₜₜ[1:N]
    
    # 2) Extract XX'
    XtXt_prime = reshape(Zₜₜ[N+1:end], N, N)
    
    # 3) implied_cov = XX' - xₜ xₜ'
    implied_cov = XtXt_prime - xt*xt'
    
    # 4) Simple "ε-shift" to ensure PSD
    ε = sqrt(eps(T))
    corrected_cov = implied_cov + ε*I(N)
    
    # 5) Recombine => xₜ and (corrected_cov + xₜ xₜ')
    return vcat(xt, vec(corrected_cov + xt*xt'))
end


#Quadratic Kalman Filter
"""
    qkf_filter!(data::QKData{T1,1}, params::QKParams{T,T2})

Run the **Quadratic Kalman Filter (QKF)** on a time series of length `T̄`, modifying
the result in-place.

# Description

This function implements a Kalman-like recursive filter where the state 
vector `Zₜ` includes not only the usual mean component `xₜ` but also 
terms for the second-moment `(x xᵀ)ₜ`, making it a *quadratic* extension. 
At each time step, it performs:

1. **State Prediction** (`predict_Zₜₜ₋₁!` / `predict_Pₜₜ₋₁!`)
2. **Measurement Prediction** (`predict_Yₜₜ₋₁!` / `predict_Mₜₜ₋₁!`)
3. **Kalman Gain Computation** (`compute_Kₜ!`)
4. **State & Covariance Update** (`update_Zₜₜ!`, `update_Pₜₜ!`)
5. **PSD Correction** (`correct_Zₜₜ!`)
6. **Log-Likelihood** computation for the current innovation.

Unlike the non-mutating version (`qkf_filter`), this function reuses 
and mutates internal arrays and data structures in-place, which can 
improve performance and reduce memory allocations.

# Arguments

- `data::QKData{T1,1}`  
  A structure holding:
  - `Y::Vector{T1}` of length `T̄+1`, containing observations. 
    Typically, `Y[1]` is an initial placeholder and `Y[2..end]` 
    are the actual measurements.
  - `T̄::Int` the total number of time steps (excluding index 0).
  - `M::Int` the dimension of the measurement at each time step.

- `params::QKParams{T,T2}`  
  A parameter structure holding:
  - `N::Int`: State dimension (for the mean part).
  - `P::Int`: Dimension of the augmented "quadratic" state 
    (`P = N + N(N+1)/2`).
  - `μ̃ᵘ, Σ̃ᵘ`: The unconditional mean and covariance used for 
    initialization.
  - Additional model matrices or functions (e.g., `Φ̃`, `B̃`, `A`, `V`) 
    accessed via subroutines.

# Return

A named tuple with fields:

- `llₜ::Vector{Float64}`  
  The per-time-step log-likelihoods (size = `T̄`).
- `Zₜₜ::Array{T,3}`  
  The filtered state at each time step. Dimensions: `(T, P, T̄+1)` in your 
  specific code (or `(P, T̄+1)` in a more generic version).
- `Pₜₜ::Array{T,4}`  
  The filtered state covariance array. Dimensions often `(T, P, P, T̄+1)` 
  in your code.
- `Yₜₜ₋₁::Vector{T}`  
  The predicted measurement at each time step (size = `T̄`).
- `Mₜₜ₋₁::Array{T,4}`  
  The predicted measurement covariance, dimensions `(T, M, M, T̄)`.
- `Kₜ::Array{T,4}`  
  The Kalman gain for each time step, `(T, P, M, T̄)`.
- `Zₜₜ₋₁::Array{T,3}`  
  One-step-ahead predicted states.
- `Pₜₜ₋₁::Array{T,4}`  
  One-step-ahead predicted covariances.
- `Σₜₜ₋₁::Array{T,4}`  
  Any intermediate covariance terms used for prediction.

# Details

1. **Initialization**: 
   - `Zₜₜ[:, 1] .= μ̃ᵘ` and `Pₜₜ[:,:,1] .= Σ̃ᵘ`.
2. **Recursive Steps** (`for t in 1:T̄`):
   - **Prediction**: `predict_Zₜₜ₋₁!` / `predict_Pₜₜ₋₁!`.
   - **Measurement**: `predict_Yₜₜ₋₁!` / `predict_Mₜₜ₋₁!`.
   - **Gain & Update**: `compute_Kₜ!`, then `update_Zₜₜ!` / `update_Pₜₜ!`.
   - **Correction**: `correct_Zₜₜ!` clamps negative eigenvalues 
     for PSD.
   - **Likelihood**: `compute_loglik!` appends the log-likelihood.
3. **Positive Semidefinite Correction**: 
   - Negative eigenvalues introduced by approximation are set to zero.

# Example

```julia
data = QKData(Y, M=measurement_dim, T̄=length(Y)-1)
params = QKParams(...)
result = qkf_filter!(data, params)

@show result.llₜ
@show result.Zₜₜ[:, end]   # final state
```
"""
function qkf_filter!(data::QKData{T1, 1},
    params::QKParams{T,T2}) where {T1 <: Real, T <: Real, T2 <: Real}

    @unpack T̄, Y, M = data
    @unpack N, μ̃ᵘ, Σ̃ᵘ, P = params

    # Predfine Matrix
    Zₜₜ =  zeros(T, P, T̄ + 1)
    Zₜₜ₋₁ = zeros(T, P, T̄)
    Pₜₜ = zeros(T, P, P, T̄ + 1)
    Pₜₜ₋₁ = zeros(T, P, P, T̄)
    Σₜₜ₋₁ = zeros(T, P, P, T̄)
    #vecΣₜₜ₋₁ = zeros(T, P^2, T̄)
    Kₜ = zeros(T, P, M, T̄)
    tmpP = zeros(T, P, P)
    tmpB = zeros(T, M, P)
    Yₜₜ₋₁ = zeros(T, T̄)
    Mₜₜ₋₁ = zeros(T, M, M, T̄)
    tmpPB = zeros(T, P, M)
    llₜ = zeros(Float64, T̄)
    tmpϵ = zeros(T, M)
    tmpKM = zeros(T, P, M)
    tmpKMK = zeros(T, P, P)

    Yₜ = Vector{T}(undef, T̄)
    copyto!(Yₜ, 1, Y, 2, T̄)

    #Initalize: Z₀₀ = μ̃ᵘ, P₀₀ = Σ̃ᵘ
    Zₜₜ[:, 1] .= μ̃ᵘ
    Pₜₜ[:, :, 1] .= Σ̃ᵘ
    
    # Loop over time
    for t in 1:T̄

        # State Prediction: Zₜₜ₋₁ = μ̃ + Φ̃Zₜ₋₁ₜ₋₁, Pₜₜ₋₁ = Φ̃Pₜ₋₁ₜ₋₁Φ̃' + Σ̃(Zₜ₋₁ₜ₋₁)
        predict_Zₜₜ₋₁!(Zₜₜ, Zₜₜ₋₁, params, t)
        predict_Pₜₜ₋₁!(Pₜₜ, Pₜₜ₋₁, Σₜₜ₋₁, Zₜₜ, tmpP, params, t)

        # Observation Prediction: Yₜₜ₋₁ = A + B̃Zₜₜ₋₁, Mₜₜ₋₁ = B̃Pₜₜ₋₁B̃' + V
        predict_Yₜₜ₋₁!(Yₜₜ₋₁, Zₜₜ₋₁, Y, params, t)
        predict_Mₜₜ₋₁!(Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, tmpB, params, t)

        # Kalman Gain: Kₜ = Pₜₜ₋₁B̃′/Mₜₜ₋₁
        compute_Kₜ!(Kₜ, Pₜₜ₋₁, Mₜₜ₋₁, tmpPB, params, t)

        # Update States: Zₜₜ = Zₜₜ₋₁ + Kₜ(Yₜ - Yₜₜ₋₁); Pₜₜ = Pₜₜ₋₁ - KₜMₜₜ₋₁Kₜ'
        update_Zₜₜ!(Zₜₜ, Kₜ, Yₜ, Yₜₜ₋₁, Zₜₜ₋₁, tmpϵ, params, t)
        update_Pₜₜ!(Pₜₜ, Kₜ, Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, tmpKM, tmpKMK, params, t)

        #Correct for update
        correct_Zₜₜ!(Zₜₜ, params, t)

        #Compute Log Likelihood
        compute_loglik!(llₜ, Yₜ, Yₜₜ₋₁, Mₜₜ₋₁, t)
    end

    return (llₜ = llₜ, Zₜₜ = Zₜₜ, Pₜₜ = Pₜₜ,  Yₜₜ₋₁ = Yₜₜ₋₁, Mₜₜ₋₁ = Mₜₜ₋₁, Kₜ = Kₜ, Zₜₜ₋₁ = Zₜₜ₋₁,
            Pₜₜ₋₁ = Pₜₜ₋₁, Σₜₜ₋₁ = Σₜₜ₋₁)

end

"""
    qkf_filter(data::QKData{T1,1}, params::QKParams{T,T2})

Run the **Quadratic Kalman Filter (QKF)** on a time series of length `T̄`, 
returning a new set of result arrays (out-of-place).

# Description

This function implements the same *quadratic* Kalman filter recursion 
as `qkf_filter!`, but instead of updating arrays in-place, it allocates 
new arrays for predictions, updates, and outputs. This can be simpler to 
use in contexts where you don’t want to mutate or reuse `data` and `params`, 
but it may be less memory-efficient for large-scale problems.

At each time step, it performs:

1. **State Prediction** (`predict_Zₜₜ₋₁` / `predict_Pₜₜ₋₁`)
2. **Measurement Prediction** (`predict_Yₜₜ₋₁` / `predict_Mₜₜ₋₁`)
3. **Kalman Gain Computation** (`compute_Kₜ`)
4. **State & Covariance Update** (`update_Zₜₜ`, `update_Pₜₜ`)
5. **PSD Correction** (`correct_Zₜₜ`)
6. **Log-Likelihood** computation.

# Arguments

- `data::QKData{T1,1}`  
  Same structure as in `qkf_filter!`, with fields:
  - `Y::Vector{T1}`, `T̄::Int`, `M::Int`.
- `params::QKParams{T,T2}`  
  Same parameter structure as in `qkf_filter!`, with fields:
  - `N::Int`, `P::Int`, `μ̃ᵘ, Σ̃ᵘ`, etc.

# Return

A named tuple with fields:

- `llₜ::Vector{Float64}`  
  Per-time-step log-likelihoods (size = `T̄`).
- `Zₜₜ::Array{T,3}`  
  The filtered state at each time step (dimensions `(T, P, T̄+1)` in your usage).
- `Pₜₜ::Array{T,4}`  
  The filtered state covariance array.
- `Yₜₜ₋₁::Vector{T}`  
  Predicted measurement for each step.
- `Mₜₜ₋₁::Array{T,4}`  
  Predicted measurement covariances.
- `Kₜ::Array{T,4}`  
  The Kalman gain for each time step.
- `Zₜₜ₋₁::Array{T,3}`  
  One-step-ahead predicted states.
- `Pₜₜ₋₁::Array{T,4}`  
  One-step-ahead predicted covariances.

# Details

1. **Initialization**:
   - Creates new arrays for `Zₜₜ` and `Pₜₜ` and sets the initial state 
     to `μ̃ᵘ` and `Σ̃ᵘ`.
2. **Time Loop**: 
   - **Prediction**: `predict_Zₜₜ₋₁`, `predict_Pₜₜ₋₁`.
   - **Measurement**: `predict_Yₜₜ₋₁`, `predict_Mₜₜ₋₁`.
   - **Gain & Update**: `compute_Kₜ`, `update_Zₜₜ`, `update_Pₜₜ`.
   - **Correction**: `correct_Zₜₜ` for PSD.
   - **Likelihood**: `compute_loglik`.
3. **No In-Place Mutation**:
   - Each step returns fresh arrays; original inputs are not modified.

# Example

```julia
data = QKData(Y, M=measurement_dim, T̄=length(Y)-1)
params = QKParams(...)
result = qkf_filter(data, params)

@show result.llₜ
@show result.Zₜₜ[:, end]   # final state
'''
"""
function qkf_filter(data::QKData{T1, 1},
    params::QKParams{T,T2}) where {T1 <: Real, T <: Real, T2 <: Real}
    @unpack T̄, Y, M = data
    @unpack N, μ̃ᵘ, Σ̃ᵘ, P = params
    Y_concrete = Vector{T1}(vec(Y))  # Convert to vector if it's not already
    Yₜ = @view Y_concrete[2:end]

    Zₜₜ = zeros(T, P, T̄ + 1)
    Pₜₜ = zeros(T, P, P, T̄ + 1)
    Zₜₜ₋₁ = zeros(T, P, T̄)
    Pₜₜ₋₁ = zeros(T, P, P, T̄)
    Kₜ = zeros(T, P, M, T̄)
    Yₜₜ₋₁ = zeros(T, T̄)
    Mₜₜ₋₁ = zeros(T, M, M, T̄)
    llₜ = zeros(T, T̄)

    # Initialize
    Zₜₜ[:, 1] = μ̃ᵘ
    Pₜₜ[:, :, 1] = Σ̃ᵘ

    for t in 1:T̄
        Zₜₜ₋₁[:, t] = predict_Zₜₜ₋₁(Zₜₜ[:, t], params)
        Pₜₜ₋₁[:, :, t] = predict_Pₜₜ₋₁(Pₜₜ[:, :, t], Zₜₜ[:, t], params, t)

        Yₜₜ₋₁[t] = predict_Yₜₜ₋₁(Zₜₜ₋₁[:, t], Y, params, t)
        Mₜₜ₋₁[:, :, t] = predict_Mₜₜ₋₁(Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)

        Kₜ[:, :, t] = compute_Kₜ(Pₜₜ₋₁[:, :, t], Mₜₜ₋₁[:, :, t], params, t)

        Zₜₜ[:, t + 1] = update_Zₜₜ(Kₜ[:, :, t], Yₜ[t], Yₜₜ₋₁[t], Zₜₜ₋₁[:, t], params, t)
        Pₜₜ[:, :, t + 1] = update_Pₜₜ(Kₜ[:, :, t], Mₜₜ₋₁[:, :, t], Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)

        Zₜₜ[:, t + 1] = correct_Zₜₜ(Zₜₜ[:, t + 1], params, t)

        llₜ[t] = compute_loglik(Yₜ[t], Yₜₜ₋₁[t], Mₜₜ₋₁[:, :, t])
    end

    return (llₜ = copy(llₜ), Zₜₜ = copy(Zₜₜ), Pₜₜ = copy(Pₜₜ), Yₜₜ₋₁ = copy(Yₜₜ₋₁),
        Mₜₜ₋₁ = copy(Mₜₜ₋₁), Kₜ = copy(Kₜ), Zₜₜ₋₁ = copy(Zₜₜ₋₁), Pₜₜ₋₁ = copy(Pₜₜ₋₁))
end

"""
    qkf_filter!(data::QKData{T1,1}, params::QKParams{T,T2})

Run the **Quadratic Kalman Filter (QKF)** on a time series of length `T̄`.

# Description

This function implements a Kalman-like recursive filter where the state 
vector `Zₜ` includes not only the usual mean component `xₜ` but also 
terms for the second-moment `(x xᵀ)ₜ`, making it a *quadratic* extension. 
At each time step, it performs:

1. **State Prediction** (`predict_Zₜₜ₋₁!` / `predict_Pₜₜ₋₁!`)
2. **Measurement Prediction** (`predict_Yₜₜ₋₁!` / `predict_Mₜₜ₋₁!`)
3. **Kalman Gain Computation** (`compute_Kₜ!`)
4. **State & Covariance Update** (`update_Zₜₜ!`, `update_Pₜₜ!`)
5. **PSD Correction**: Ensures the implied covariance is positive semidefinite 
   by clamping negative eigenvalues via `correct_Zₜₜ!`.
6. **Log-Likelihood** computation for the current innovation.

The filter stores and returns the entire history of filtered states, covariances,
predicted measurements, and related arrays. 

# Arguments

- `data::QKData{T1,1}`  
  A structure holding:
  - `Y::Vector{T1}` of length `T̄+1`, which contains observations 
    (`Y[1]` is unused or some initial placeholder, and `Y[2..end]` are the actual measurements).
  - `T̄::Int` the total number of time steps (excluding index 0).
  - `M::Int` the dimension of the measurement at each time step.

- `params::QKParams{T,T2}`  
  A parameter structure holding:
  - `N::Int`: State dimension (for the mean part).
  - `P::Int`: Dimension of the augmented "quadratic" state vector 
    (`P = N + N(N+1)/2`).
  - `μ̃ᵘ, Σ̃ᵘ`: The unconditional mean and covariance used for initialization.
  - Additional model matrices or functions (e.g., `Φ̃`, `B̃`, `A`, `V`), 
    typically accessed via separate predict/update subroutines.

# Return

A named tuple with fields:

- `llₜ::Vector{Float64}`  
  The per-time-step log-likelihoods of the innovations (size = `T̄`).

- `Zₜₜ::Matrix{T}`  
  The updated ("filtered") state at each time step. Dimensions: `(P, T̄+1)`, 
  where column `k` corresponds to time index `k-1`.

- `Pₜₜ::Array{T,3}`  
  The updated ("filtered") state covariance array (or the augmented second-moment 
  representation) at each time step. Dimensions: `(P, P, T̄+1)`.

- `Yₜₜ₋₁::Vector{T}`  
  The predicted measurement at each time step (size = `T̄`).

- `Mₜₜ₋₁::Array{T,3}`  
  The predicted measurement covariance at each time step (dimensions: `(M, M, T̄)`).

- `Kₜ::Array{T,3}`  
  The Kalman gain at each time step `(P, M, T̄)`.

- `Zₜₜ₋₁::Matrix{T}`  
  The one-step-ahead predicted state (dimensions: `(P, T̄)`).

- `Pₜₜ₋₁::Array{T,4}`  
  The one-step-ahead predicted covariance (dimensions: `(P, P, T̄)`).

- `Σₜₜ₋₁::Array{T,4}`  
  Any intermediate state-dependent covariance terms added during prediction 
  (dimensions: `(P, P, T̄)`).

# Details

1. **Initialization**:  
   - `Zₜₜ[:,1] .= μ̃ᵘ` and `Pₜₜ[:,:,1] .= Σ̃ᵘ`, representing the state at time 0.

2. **Time Loop** (`for t in 1:T̄`):  
   - **Predict** step: 
     - `predict_Zₜₜ₋₁!` sets `Zₜₜ₋₁[:,t]` from `Zₜₜ[:,t]`.
     - `predict_Pₜₜ₋₁!` sets `Pₜₜ₋₁[:,:,t]` from `Pₜₜ[:,:,t]`.
   - **Measurement** step: 
     - `predict_Yₜₜ₋₁!` computes `Yₜₜ₋₁[t]` from `Zₜₜ₋₁[:,t]`.
     - `predict_Mₜₜ₋₁!` updates `Mₜₜ₋₁[:,:,t]`.
   - **Gain & Update**:
     - `compute_Kₜ!` obtains the Kalman gain `Kₜ[:,:,t]`.
     - `update_Zₜₜ!` and `update_Pₜₜ!` yield the next `Zₜₜ[:,t+1]` and `Pₜₜ[:,:,t+1]`.
   - **Correction**:  
     - `correct_Zₜₜ!` clamps negative eigenvalues in the implied covariance portion 
       of `Zₜₜ` to ensure PSD. 
   - **Likelihood**:
     - `compute_loglik!` appends/logs the innovation-based log-likelihood 
       in `llₜ[t]`.

3. **Positive Semidefinite Correction**:
   - Because the filter tracks `[X Xᵀ] - X Xᵀᵀ` as a covariance block, it can 
     become indefinite due to linearization or numerical issues. We enforce 
     PSD by zeroing out negative eigenvalues in `correct_Zₜₜ!`.
   - This step is not strictly differentiable at eigenvalues crossing zero. 
     If you need AD through the filter, consider an alternative correction 
     or a custom adjoint.

# Example

```julia
data = QKData(Y, M=measurement_dim, T̄=length(Y)-1)
params = QKParams( ... )   # set up your model

result = qkf_filter!(data, params)

@show result.llₜ
@show result.Zₜₜ[:, end]   # final state vector
"""
function qkf_filter!(data::QKData{T, 2}, params::QKParams{T,T2}) where {T <: Real, T2 <: Real}

    @unpack T̄, Y, M = data
    @unpack N, μ̃ᵘ, Σ̃ᵘ, P = params

    # Predfine Matrix
    Zₜₜ = Matrix{<:Real}(undef, P, T̄ + 1)
    Pₜₜ = Array{T, 3}(undef, P, P, T̄ + 1)
    Zₜₜ₋₁ = Matrix{<:Real}(undef, P, T̄)
    Pₜₜ₋₁ = Array{T, 3}(undef, P, P, T̄)
    Kₜ = Array{T, 3}(undef, P, M, T̄)
    Yₜₜ₋₁ = Vector{<:Real}(undef, T̄)
    Mₜₜ₋₁ = Array{T, 3}(undef, M, M, T̄)
    llₜ = Vector{<:Real}(undef, T̄)

    #Initalize: Z₀₀ = μ̃ᵘ, P₀₀ = Σ̃ᵘ
    Zₜₜ[:, 1] = μ̃ᵘ
    Pₜₜ[:, :, 1] = Σ̃ᵘ
    
    # Loop over time
    for t in 1:T̄
        Zₜₜ₋₁[:, t] = predict_Zₜₜ₋₁(Zₜₜ[:, t], params)
        Pₜₜ₋₁[:, :, t] = predict_Pₜₜ₋₁(Pₜₜ[:, :, t], Zₜₜ[:, t], params, t)

        Yₜₜ₋₁[t] = predict_Yₜₜ₋₁(Zₜₜ₋₁[:, t], Y, params, t)
        Mₜₜ₋₁[:, :, t] = predict_Mₜₜ₋₁(Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)

        Kₜ[:, :, t] = compute_Kₜ(Pₜₜ₋₁[:, :, t], Mₜₜ₋₁[:, :, t], params, t)

        Zₜₜ[:, t + 1] = update_Zₜₜ(Kₜ[:, :, t], Yₜ[t], Yₜₜ₋₁[t], Zₜₜ₋₁[:, t], params, t)
        Pₜₜ[:, :, t + 1] = update_Pₜₜ(Kₜ[:, :, t], Mₜₜ₋₁[:, :, t], Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)

        Zₜₜ[:, t + 1] = correct_Zₜₜ(Zₜₜ[:, t + 1], params, t)

        llₜ[t] = compute_loglik(Yₜ[t], Yₜₜ₋₁[t], Mₜₜ₋₁[:, :, t])
    end

    return (llₜ = llₜ, Zₜₜ = Zₜₜ, Pₜₜ = Pₜₜ,  Yₜₜ₋₁ = Yₜₜ₋₁, Mₜₜ₋₁ = Mₜₜ₋₁, Kₜ = Kₜ, Zₜₜ₋₁ = Zₜₜ₋₁,
            Pₜₜ₋₁ = Pₜₜ₋₁, Σₜₜ₋₁ = Σₜₜ₋₁)

end

"""
    qkf_filter_functional(data, params)

!!! warning
    **Experimental** function, not fully integrated with main code.
    Use at your own risk; API may change.
"""
function qkf_filter_functional(data::QKData{T, 2}, params::QKParams{T,T2})
    @unpack T̄, Y, M = data
    @unpack N, μ̃ᵘ, Σ̃ᵘ, P = params

    Yₜ = Y[:, 2:end]

    # Initialize: Z₀₀ = μ̃ᵘ, P₀₀ = Σ̃ᵘ
    Zₜₜ_init = hcat(μ̃ᵘ, zeros(T, P, T̄))
    Pₜₜ_init = cat(Σ̃ᵘ, zeros(T, P, P, T̄), dims=3)

    function step(t, state)
        (Zₜₜ, Pₜₜ, Zₜₜ₋₁, Pₜₜ₋₁, Σₜₜ₋₁, Kₜ, Yₜₜ₋₁, Mₜₜ₋₁, llₜ) = state

        # State Prediction
        Zₜₜ₋₁_t = predict_Zₜₜ₋₁(Zₜₜ[:, t], params)
        Pₜₜ₋₁_t = predict_Pₜₜ₋₁(Pₜₜ[:, :, t], Zₜₜ[:, t], params, t)
        
        # Compute Σₜₜ₋₁
        Σₜₜ₋₁_t = compute_Σₜₜ₋₁(Zₜₜ, params, t)

        # Observation Prediction
        Yₜₜ₋₁_t = predict_Yₜₜ₋₁(Zₜₜ₋₁_t, Y, params, t)
        Mₜₜ₋₁_t = predict_Mₜₜ₋₁(Pₜₜ₋₁_t, Zₜₜ₋₁_t, params, t)

        # Kalman Gain
        Kₜ_t = compute_Kₜ(Pₜₜ₋₁_t, Mₜₜ₋₁_t, params, t)

        # Update States
        Zₜₜ_next = update_Zₜₜ(Kₜ_t, Yₜ[:, t], Yₜₜ₋₁_t, Zₜₜ₋₁_t, params, t)
        Pₜₜ_next = update_Pₜₜ(Kₜ_t, Mₜₜ₋₁_t, Pₜₜ₋₁_t, Zₜₜ₋₁_t, params, t)

        # Correct for update
        Zₜₜ_next = correct_Zₜₜ(Zₜₜ_next, params, t)

        # Compute Log Likelihood
        llₜ_t = compute_loglik(Yₜ[:, t], Yₜₜ₋₁_t, Mₜₜ₋₁_t, t)

        Zₜₜ_new = hcat(Zₜₜ[:, 1:t], Zₜₜ_next)
        Pₜₜ_new = cat(Pₜₜ[:, :, 1:t], Pₜₜ_next, dims=3)
        Zₜₜ₋₁_new = hcat(Zₜₜ₋₁, Zₜₜ₋₁_t)
        Pₜₜ₋₁_new = cat(Pₜₜ₋₁, Pₜₜ₋₁_t, dims=3)
        Σₜₜ₋₁_new = cat(Σₜₜ₋₁, Σₜₜ₋₁_t, dims=3)
        Kₜ_new = cat(Kₜ, Kₜ_t, dims=3)
        Yₜₜ₋₁_new = hcat(Yₜₜ₋₁, Yₜₜ₋₁_t)
        Mₜₜ₋₁_new = cat(Mₜₜ₋₁, Mₜₜ₋₁_t, dims=3)
        llₜ_new = vcat(llₜ, llₜ_t)

        (Zₜₜ_new, Pₜₜ_new, Zₜₜ₋₁_new, Pₜₜ₋₁_new, Σₜₜ₋₁_new, Kₜ_new, Yₜₜ₋₁_new, Mₜₜ₋₁_new, llₜ_new)
    end

    init_state = (Zₜₜ_init, Pₜₜ_init, zeros(T, P, 0), zeros(T, P, P, 0), zeros(T, P, P, 0), 
                    zeros(T, P, M, 0), zeros(T, M, 0), zeros(T, M, M, 0), Float64[])

    final_state = foldl(step, 1:T̄, init=init_state)

    (Zₜₜ, Pₜₜ, Zₜₜ₋₁, Pₜₜ₋₁, Σₜₜ₋₁, Kₜ, Yₜₜ₋₁, Mₜₜ₋₁, llₜ) = final_state

    return (llₜ = llₜ, Zₜₜ = Zₜₜ, Pₜₜ = Pₜₜ, Yₜₜ₋₁ = Yₜₜ₋₁, Mₜₜ₋₁ = Mₜₜ₋₁, Kₜ = Kₜ, Zₜₜ₋₁ = Zₜₜ₋₁,
            Pₜₜ₋₁ = Pₜₜ₋₁, Σₜₜ₋₁ = Σₜₜ₋₁)
end

"""
    qkf_filter(data::QKData{T1,N}, params::QKParams{T,T2})

Run a *Quadratic Kalman Filter (QKF)* over `T̄` timesteps, returning the filtered 
states, covariances, and per-step log-likelihoods in a purely functional manner.

# Description

- At each time `t = 1..T̄`, the filter performs:
  1. **State Prediction** (`Zₜₜ₋₁`) from the last filtered state (`last(Zₜₜ)`).
  2. **Covariance Prediction** (`Pₜₜ₋₁`) from the last filtered covariance (`last(Pₜₜ)`).
  3. **Measurement Prediction** (`Yₜₜ₋₁`, `Mₜₜ₋₁`) from `Zₜₜ₋₁`, `Pₜₜ₋₁`.
  4. **Kalman Gain** (`Kₜ`) from `Mₜₜ₋₁` and `Pₜₜ₋₁`.
  5. **State & Covariance Update** (`Zₜₜ_new`, `Pₜₜ_new`) using the incoming observation `Y[t+1]`.
  6. **PSD Correction** (`Zₜₜ_corrected`) ensuring the implied second-moment portion is positive semidefinite.
  7. **Log-Likelihood** computation using the innovation `(Y[t+1] - Yₜₜ₋₁)` and its covariance `Mₜₜ₋₁`.

- Internal arrays `Zₜₜ`, `Pₜₜ`, and the log-likelihood vector `llₜ` are built incrementally 
  using `push!`. The final outputs are concatenated into a single matrix or array 
  along the time dimension.

# Arguments

- `data::QKData{T1,N}`
  - A structure holding:
    - `Y::AbstractArray{T1,N}`: Observations over `T̄+1` time points (with dimension 
      `N=1` or `N=2` depending on your setup).
    - `T̄::Int`: Number of filter steps (i.e. `length(Y)-1`).
    - `M::Int`: Measurement dimension (if the data is vector- or matrix-valued).
- `params::QKParams{T,T2}`
  - A parameter set containing:
    - `μ̃ᵘ::AbstractVector{T}`: The unconditional mean for initialization.
    - `Σ̃ᵘ::AbstractMatrix{T}`: The unconditional covariance for initialization.
    - Possibly other necessary model parameters (like `Φ̃`, `B̃`, etc.).
  - The function calls the subroutines `predict_Zₜₜ₋₁`, `predict_Pₜₜ₋₁`, `predict_Yₜₜ₋₁`, 
    `predict_Mₜₜ₋₁`, `compute_Kₜ`, `update_Zₜₜ`, `update_Pₜₜ`, `correct_Zₜₜ`, 
    `compute_loglik`, which should be defined and tailored to your specific QKF model.

# Returns

A named tuple with:
  - `llₜ::Vector{T}`: The log-likelihood at each time step `t = 1..T̄`.
  - `Zₜₜ::Matrix{T}`: The final filtered states stacked horizontally; 
    size `(dimension_of_state) × (T̄+1)`.
  - `Pₜₜ::Array{T,3}`: The final filtered covariances (or augmented quadratic covariance) 
    stacked along `dims=3`; size `(dimension_of_state, dimension_of_state, T̄+1)`.

# Notes

- The function stores the result of each iteration in dynamic arrays (`Zₜₜ`, `Pₜₜ`, `llₜ`) 
  via `push!`. This is convenient for clarity, but it may be less efficient for large `T̄`. 
  For high-performance filtering, consider preallocating arrays or using an in-place approach.
- The *Quadratic* Kalman Filter implies that the state vector may contain both the 
  mean `x_t` and second-moment terms `(x xᵀ)_t`. Therefore, `Zₜₜ` might be larger 
  in dimension than a standard Kalman filter state vector.
- The `correct_Zₜₜ` step enforces positive semidefiniteness by clamping any negative 
  eigenvalues in the implied covariance portion of the state vector. This is necessary 
  for numeric stability when the linearization or updates cause indefiniteness. 
  (Be aware it is not differentiable at eigenvalues crossing zero.)

# Example

```julia
# Suppose Y is a vector of length T̄+1
data = QKData(Y, M=1, T̄=length(Y)-1)
params = QKParams(μ̃ᵘ=..., Σ̃ᵘ=..., ...)

results = qkf_filter(data, params)

@show results.llₜ
@show size(results.Zₜₜ)
@show size(results.Pₜₜ)
"""
function qkf_filter(data::QKData{T1, N}, params::QKParams{T,T2}) where {T1 <:Real, T <: Real, T2 <:Real, N}
    @unpack T̄, Y, M = data
    @assert length(Y) == T̄ + 1 "Y should have T̄ + 1 observations"

    # Initialize (use Y[1] here if needed)
    Zₜₜ = [params.μ̃ᵘ]
    Pₜₜ = [params.Σ̃ᵘ]
    llₜ = T[]

    for t in 1:T̄
        # Prediction step
        Zₜₜ₋₁ = predict_Zₜₜ₋₁(last(Zₜₜ), params)
        Pₜₜ₋₁ = predict_Pₜₜ₋₁(last(Pₜₜ), last(Zₜₜ), params, t)

        # Observation prediction
        Yₜₜ₋₁ = predict_Yₜₜ₋₁(Zₜₜ₋₁, Y, params, t)
        Mₜₜ₋₁ = predict_Mₜₜ₋₁(Pₜₜ₋₁, Zₜₜ₋₁, params, t)

        # Kalman gain
        Kₜ = compute_Kₜ(Pₜₜ₋₁, Mₜₜ₋₁, params, t)

        # Update step
        Zₜₜ_new = update_Zₜₜ(Kₜ, Y[t+1], Yₜₜ₋₁, Zₜₜ₋₁, params, t)
        Pₜₜ_new = update_Pₜₜ(Kₜ, Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, params, t)

        # Correct for update
        Zₜₜ_corrected = correct_Zₜₜ(Zₜₜ_new[:,1], params, t)

        # Compute log-likelihood
        ll = compute_loglik(Y[t+1], Yₜₜ₋₁, Mₜₜ₋₁)

        push!(Zₜₜ, Zₜₜ_corrected)
        push!(Pₜₜ, Pₜₜ_new)
        push!(llₜ, ll)

    end

    return (llₜ = llₜ, Zₜₜ = hcat(Zₜₜ...), Pₜₜ = cat(Pₜₜ..., dims=3))
end

export qkf_filter, qkf_filter!