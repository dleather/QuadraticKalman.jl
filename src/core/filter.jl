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

- **In-Place**: Call `qkf_filter!(data, model)` to run the filter in-place, 
  potentially avoiding unnecessary memory allocations.
- **Out-of-Place**: Call `qkf_filter(data, model)` if you prefer a 
  functional style that returns fresh arrays for each step's results.

## Notes

- The `QKData` and `QKModel` types organize time series and model parameters.
- At every time step, the *predict* step forms `Zₜₜ₋₁` / `Pₜₜ₋₁` 
  and the *update* step forms `Zₜₜ` / `Pₜₜ`. 
- The "Quadratic" portion refers to tracking `(x xᵀ)ₜ` inside the state 
  to capture second-moment dynamics. 
- A PSD correction step (`correct_Zₜₜ!` or `correct_Zₜₜ`) is often necessary 
  to handle numerical or modeling approximations that might lead to 
  indefinite blocks in the augmented state.

For more detail, refer to each function's docstring below.
"""

"""
    predict_Z_ttm1!(Z_tt::AbstractMatrix{T}, 
                   Z_ttm1::AbstractMatrix{T},
                   model::QKModel{T,T2}, 
                   t::Int)

In-place computation of the **one-step-ahead predicted augmented state** 
`Z_ttm1[:, t]` from the current augmented state `Z_tt[:, t]`.

# Arguments
- `Z_tt::AbstractMatrix{T}`: (P×T̄) matrix storing the current augmented states, 
  with `Z_tt[:, t]` as the time-`t` state.
- `Z_ttm1::AbstractMatrix{T}`: (P×T̄) matrix to store the new predicted state, 
  with the result in `Z_ttm1[:, t]`.
- `model::QKModel{T,T2}`: Parameter struct holding:
  - `aug_mean::Vector{T}`: Augmented constant vector.
  - `Phi_aug::Matrix{T}`: Augmented transition matrix (size P×P).
- `t::Int`: The current time index.

# Details
Computes `Z_ttm1[:, t] = aug_mean + Phi_aug * Z_tt[:, t]`, 
corresponding to the QKF augmented state update.  
The result is written **in place** into `Z_ttm1[:, t]`.

"""
function predict_Z_ttm1!(Z_tt::AbstractMatrix{T}, Z_ttm1::AbstractMatrix{T},
    model::QKModel{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    @unpack Phi_aug, mu_aug = model.aug_state
    Z_ttm1_view = @view Z_ttm1[:, t]
    Z_tt_view = @view Z_tt[:, t]
    Z_ttm1_view .= mu_aug .+ Phi_aug * Z_tt_view
end

"""
    predict_Z_ttm1(Z_tt::AbstractVector{T}, model::QKModel{T,T2}) 
                  -> Vector{T}

Return a new **one-step-ahead predicted augmented state** given the current 
augmented state `Z_tt`.

# Arguments
- `Z_tt::AbstractVector{T}`: The current augmented state (dimension P).
- `model::QKModel{T,T2}`: Holds:
  - `aug_mean::Vector{T}`: Augmented constant vector (length P).
  - `Phi_aug::Matrix{T}`: P×P augmented transition matrix.

# Returns
- `Vector{T}`: A newly allocated vector for `Z_ttm1 = aug_mean + Phi_aug * Z_tt`.

# Details
This purely functional approach does not modify `Z_tt`, 
which is typically `(P×1)` in QKF contexts.  
Useful in AD frameworks that avoid in-place updates.

"""
function predict_Z_ttm1(Z_tt::AbstractVector{T}, model::QKModel{T,T2}) where {T <: Real, T2 <: Real}

    @unpack Phi_aug, mu_aug = model.aug_state
    return mu_aug .+ Phi_aug * Z_tt
end

"""
    predict_P_ttm1!(P_tt::AbstractArray{Real,3}, 
                    P_ttm1::AbstractArray{Real,3}, 
                    Σ_ttm1::AbstractArray{Real,3},
                    Z_tt::AbstractMatrix{T}, 
                    tmpP::AbstractMatrix{T}, 
                    model::QKModel{T,T2}, 
                    t::Int)

In-place update of the **one-step-ahead predicted covariance** 
`P_ttm1[:,:,t]` from the current filtered covariance `P_tt[:,:,t]`. 
Also updates or uses `Σ_ttm1[:,:,t]` as needed.

# Arguments
- `P_tt::AbstractArray{Real,3}`: A 3D array `(P×P×T̄)` of filtered covariances; 
  we read `P_tt[:,:,t]`.
- `P_ttm1::AbstractArray{Real,3}`: A 3D array `(P×P×T̄)` to store the predicted 
  covariance in `P_ttm1[:,:,t]`.
- `Σ_ttm1::AbstractArray{Real,3}`: (P×P×T̄) array storing 
  state-dependent noise or extra terms computed by `compute_Σ_ttm1!`.
- `Z_tt::AbstractMatrix{T}`: (P×T̄) the current augmented states. 
  Used by `compute_Sigma_ttm1!` to update `Σ_ttm1`.
- `tmpP::AbstractMatrix{T}`: A scratch `(P×P)` buffer if needed 
  for multiplication (not fully used in the code snippet).
- `model::QKModel{T,T2}`: Must contain:
  - `Phi_aug::Matrix{T}`: The P×P augmented transition matrix.
- `t::Int`: Time index.

# Details
1. Calls `compute_Sigma_ttm1!(...)` to update the noise/correction term in 
   `Σ_ttm1[:,:,t]` based on `Z_tt[:,t]`.
2. Computes 
   `P_ttm1[:,:,t] = ensure_positive_definite( Phi_aug * P_tt[:,:,t] * Phi_aug' + Σ_ttm1[:,:,t] )`
   in-place, 
   ensuring numerical stability.

This is typically used in the predict step of the QKF for covariance.

"""
function predict_P_ttm1!(P_tt::AbstractArray{T,3}, P_ttm1::AbstractArray{T,3},
    Σ_ttm1::AbstractArray{T,3}, Z_tt::AbstractMatrix{T}, tmpP::AbstractMatrix{T}, 
    model::QKModel{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    # 1) Update Σₜₜ₋₁ based on Zₜₜ (call the function with capital Sigma)
    compute_Sigma_ttm1!(Σ_ttm1, Z_tt, model, t)

    # 2) Build predicted covariance using views
    @unpack Phi_aug = model.aug_state
    local P_tt_view = @view P_tt[:, :, t]
    local Σ_ttm1_view = @view Σ_ttm1[:, :, t]
    local P_ttm1_view = @view P_ttm1[:, :, t]
    
    P_ttm1_view .= ensure_positive_definite(
        Phi_aug * P_tt_view * Phi_aug' .+ Σ_ttm1_view
    )
end

"""
    predict_P_ttm1(P_tt::AbstractMatrix{T}, 
                   Z_tt::AbstractVecOrMat{<:Real}, 
                   model::QKModel{T,T2}, 
                   t::Int) -> Matrix{T}

Return a newly allocated **one-step-ahead predicted covariance** from 
the current covariance `P_tt` and augmented state `Z_tt`, calling 
`compute_Sigma_ttm1` in the process.

# Arguments
- `P_tt::AbstractMatrix{T}`: (P×P) the current filtered covariance.
- `Z_tt::AbstractVecOrMat{<:Real}`: The augmented state (dimension P) 
  or a matrix storing states if needed. 
- `model::QKModel{T,T2}`: Must contain:
  - `Phi_aug::Matrix{T}`: The augmented transition matrix (P×P).
  - Possibly functions like `compute_Sigma_ttm1` to get extra noise terms.
- `t::Int`: Time index.

# Returns
- `Matrix{T}`: A new (P×P) matrix for the predicted covariance 
  `Φ̃ * Pₜₜ * Φ̃' + Σₜₜ₋₁`.

# Details
1. Calls `Σ_ttm1 = compute_Sigma_ttm1(Z_tt, model, t)`.
2. Builds `P_tmp = Phi_aug*P_tt*Phi_aug' + Σ_ttm1`.
3. Checks `isposdef(P_tmp)`. If false, calls `make_positive_definite(P_tmp)`.
4. Returns the final matrix `P`.

Purely functional approach (allocates a new covariance) 
for AD or simpler code flow.

"""
function predict_P_ttm1(P_tt::AbstractMatrix{T}, Z_tt::AbstractVecOrMat{<:Real},
    model::QKModel{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    Sigma_ttm1 = compute_Sigma_ttm1(Z_tt, model)

    @unpack Phi_aug = model.aug_state
    P_tmp = Phi_aug * P_tt * Phi_aug' .+ Sigma_ttm1

    if !isposdef(P_tmp)
        return make_positive_definite(P_tmp)
    else
        return P_tmp
    end
end

"""
    predict_Y_ttm1!(Y_ttm1, Z_ttm1, Y, model, t)

In-place update of the **predicted measurement** at time `t` in the range `1..T̄`.

# Arguments

- `Y_ttm1::AbstractMatrix{T}`:  
  A matrix of size `(M, T̄)` where we store the predicted measurement 
  at column `t`.  
  - After the call, `Y_ttm1[:, t]` is overwritten with the new predicted values.

- `Z_ttm1::AbstractMatrix{T}`:  
  A `(P, T̄)` matrix of one-step-ahead predicted states. We read `Z_ttm1[:, t]`.

- `Y::AbstractMatrix{T}`:  
  The **original** measurements of size `(M, T̄)`. If your data is univariate,
  reshape a length-`T̄` vector into `(1, T̄)`.

- `model::QKModel{T,T2}`:  
  Contains fields:
  - `A::Vector{T}` of length `M`
  - `B_aug::Matrix{T}` of size `(M, P)`
  - `alpha::Matrix{T}` of size `(M, M)` or `(M, 1)` for AR-like terms
  and so on.

- `t::Int`:  
  The time index (1-based).

# Behavior
Sets  
Y_ttm1[:, t] = A + B_aug * Z_ttm1[:, t] + alpha * Y[:, t]``
in place.  
This operation is purely linear, hence **AD-friendly** in most Julia AD frameworks.

# Notes
- For **univariate** usage (M=1), store your data as `(1, T̄)`. Then this exact same function applies; you'll just be dealing with 1×T columns.
- If you need a scalar at the end, you can index `Y_ttm1[1, t]`.

"""
function predict_Y_ttm1!(
    Y_ttm1::AbstractMatrix{T},
    Z_ttm1::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    model::QKModel{T,T2},
    t::Int) where {T<:Real, T2<:Real}

    @unpack A, alpha = model.meas
    @unpack B_aug = model.aug_state

    # Overwrite the t-th column in-place using views
    Y_ttm1_view = @view Y_ttm1[:, t]
    Z_ttm1_view = @view Z_ttm1[:, t]
    Y_view = @view Y[:, t]

    Y_ttm1_view .= A .+ B_aug * Z_ttm1_view .+ alpha * Y_view
end

"""
    predict_Y_ttm1(Z_ttm1, Y, model, t) -> Vector{T}

Return a **newly allocated** predicted measurement for time `t`.

# Arguments

- `Z_ttm1::AbstractMatrix{T}`:  
  `(P, T̄)` matrix of one-step-ahead predicted states. We read `Z_ttm1[:, t]`.

- `Y::AbstractMatrix{T}`:  
  `(M, T̄)` matrix of actual observations.
  - For univariate data, you can reshape from a vector to `(1, T̄)`.

- `model::QKModel{T,T2}`:  
  Holds:
  - `A::Vector{T}` (length M)
  - `B_aug::Matrix{T}` (size M×P)
  - `alpha::Matrix{T}` (size M×M or M×1)
  etc.

- `t::Int`:  
  Time index.

# Returns
A `Vector{T}` of length `M`, computed via
`A + B_aug * Z_ttm1[:, t] + alpha * Y[:, t]`

# Details
- This does **not** modify any arrays in place; it creates a fresh vector.  
- If `M=1`, you'll get a length-1 vector containing the predicted scalar.

"""
function predict_Y_ttm1(
    Z_ttm1::AbstractMatrix{T1},
    Y::AbstractMatrix{T3},
    model::QKModel{T,T2},
    t::Int) where {T<:Real, T1<:Real, T3<:Real, T2<:Real}

    @unpack A, alpha = model.meas
    @unpack B_aug = model.aug_state

    # Allocate a new vector as the sum
    return A .+ B_aug * Z_ttm1[:, t] .+ alpha * Y[:, t]
end

"""
    predict_M_ttm1!(M_ttm1, P_ttm1, tmpB, model, t)

In-place computation of the predicted measurement covariance `M_ttm1[:,:,t]` 
in a **constant** measurement noise scenario, storing the result into 
the 3D array `M_ttm1`.

# Arguments
- `M_ttm1::AbstractArray{T,3}`: A 3D array `(M×M×T̄)` for measurement covariances 
  over time. `M_ttm1[:,:,t]` will be updated in place.
- `Pₜₜ₋₁::AbstractArray{T,3}`: The 3D array `(P×P×T̄)` of one-step-ahead 
  predicted state covariances. We read `Pₜₜ₋₁[:,:,t]`.
- `tmpB::AbstractMatrix{T}`: A working buffer of size `(M×P)` used for 
  intermediate matrix multiplication, e.g. `B̃ * Pₜₜ₋₁[:,:,t]`.
- `model::QKModel{T,T2}`: Parameter struct with:
  - `M::Int`: Measurement dimension.
  - `B_aug::Matrix{T}`: Observation matrix in the QKF model (size `M×P`).
  - `V::Matrix{T}`: Constant measurement noise covariance (size `M×M`).
- `t::Int`: Time index (1-based).

# Details
1. We compute in place: `M_ttm1[:,:,t] = B_aug * P_ttm1[:,:,t] * B_aug' + V`.
   - First do `mul!(tmpB, B_aug, P_ttm1[:,:,t])`.
   - Then do `mul!(Mₜₜ₋₁[:,:,t], tmpB, B̃', 1.0, 0.0)`.
   - Finally, add `V` to `M_ttm1[:,:,t]`.
2. If `M==1` (univariate), we clamp any negative scalar 
   (`M_ttm1[1,1,t] < 0 => 1e-4`).
3. Otherwise, if `M>1`, we call `isposdef(M_ttm1[:,:,t])`. If false, we fix it 
   by `make_positive_definite` and overwrite `M_ttm1[:,:,t]`.

This approach is **AD-friendly** since it keeps the final result 
in `M_ttm1[:,:,t]` with minimal branching. 
Use this routine when `V` is a **precomputed constant** matrix, 
no longer time-varying.
"""
function predict_M_ttm1!(M_ttm1::AbstractArray{T,3}, P_ttm1::AbstractArray{T,3},
    tmpB::AbstractMatrix{T}, model::QKModel{T,T2}, t::Int) where {T<:Real, T2<:Real}

    @unpack M, V = model.meas
    @unpack B_aug = model.aug_state
    
    # 1) In-place multiplication:
    #    tmpB = B̃ * Pₜₜ₋₁[:,:,t]
    mul!(tmpB, B_aug, @view P_ttm1[:,:,t])

    #    Mₜₜ₋₁[:,:,t] = tmpB * B̃'
    local M_slice = @view M_ttm1[:,:,t]
    mul!(M_slice, tmpB, B_aug', 1.0, 0.0)

    # 2) Add V to Mₜₜ₋₁[:,:,t]
    @inbounds for i in 1:M
        for j in 1:M
            M_slice[i,j] += V[i,j]
        end
    end

    # 3) Check for negativity in univariate case
    if M == 1
        if M_slice[1,1] < 0
            M_slice[1,1] = 1e-4
        end

    # 4) For multivariate, enforce positive-definiteness if needed
    else
        if !isposdef(M_slice)
            local corrected = make_positive_definite(M_slice)
            M_slice .= corrected
        end
    end
end

"""
    predict_M_ttm1(P_ttm1::AbstractMatrix{T}, model::QKModel{T,T2})
                  -> Matrix{T}

Compute the predicted measurement covariance `Mₜₜ₋₁` for the current time step,
assuming a **precomputed**, constant noise matrix `V` in `QKModel`.

# Arguments
- `P_ttm1::AbstractMatrix{T}`: The one-step-ahead predicted covariance of the state.
- `model::QKModel{T,T2}`: A parameter struct holding:
  - `M::Int`: Measurement dimension.
  - `B̃::Matrix{T}`: Augmented observation matrix in the QKF model.
  - `V::Matrix{T}`: Precomputed, constant measurement noise/covariance matrix.

# Returns
- `Matrix{T}`: The predicted measurement covariance `M_ttm1`. 
  - If `M=1` (univariate), the scalar case is checked for negativity; 
    if < 0, it's clamped to `[1e-4]`.  
  - If `M>1`, we check `isposdef`. If not positive definite, we call 
    `make_positive_definite` to fix it.

# Details
1. We compute `M_tmp = B_aug*P_ttm1*B_aug' + V`.  
2. If univariate (`M==1`) and `M_tmp[1,1] < 0`, we clamp to a small positive `[1e-4]`.  
3. If multivariate (`M>1`), we call `isposdef(M_tmp)`. If it fails, 
   we fix it with `make_positive_definite(M_tmp)`.

This purely functional approach is **AD-friendly**, and the small clamp 
protects against numeric issues when `M=1`.

"""
function predict_M_ttm1(P_ttm1::AbstractMatrix{T}, model::QKModel{T,T2}) where 
    {T<:Real, T2<:Real}

    @unpack M, V = model.meas
    @unpack B_aug = model.aug_state
    # 1) Compute the "raw" predicted covariance for the measurement
    M_tmp = B_aug * P_ttm1 * B_aug' .+ V

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
    compute_K_t!(K_t, P_ttm1, M_ttm1, tmpB, model, t)

In-place computation of the Kalman Gain `K_t[:, :, t]` for **any** measurement dimension `M ≥ 1`.

# Arguments

- `K_t::AbstractArray{T,3}`:  
  A 3D array of size `(P, M, T̄)` storing Kalman gains over time.  
  We will overwrite the slice `K_t[:, :, t]`.

- `P_ttm1::AbstractArray{T,3}`:  
  The one-step-ahead predicted state covariance `(P, P, T̄)`.  
  We read `P_ttm1[:, :, t]`.

- `M_ttm1::AbstractArray{T,3}`:  
  The measurement prediction covariance `(M, M, T̄)`.  
  We read `M_ttm1[:, :, t]`.

- `tmpB::AbstractMatrix{T}`:  
  A working buffer `(M, P)` or `(P, M)` used for multiplication or solves. 
  Size depends on your approach. This is optional if you do not strictly need it.

- `model::QKModel{T,T2}`:  
  Must hold:
  - `B_aug::Matrix{T}` of size `(M, P)`
  - Possibly `M, P::Int`
  - etc.

- `t::Int`:  
  Time index (1-based).

# Details

We compute the Kalman gain **in place**:
`S = B_aug * P_ttm1[:,:,t] * B_aug' + M_ttm1[:,:,t] K = P_ttm1[:,:,t] * B_aug' * inv(S)`
storing `K` into `K_t[:, :, t]`.

- If `M == 1`, `S` is `1×1`, so we do a simple scalar division.
- If `M > 1`, we do a matrix factorization/solve.

**AD-Friendliness**: Using standard `inv`, `cholesky`, or `lu` is typically well-supported. 
For big `M`, prefer a `\` solve instead of `inv`.

"""
function compute_K_t!(
  K_t::AbstractArray{T,3},
  P_ttm1::AbstractArray{T,3},
  M_ttm1::AbstractArray{T,3},
  tmpB::AbstractMatrix{T},
  model::QKModel{T,T2},
  t::Int) where {T<:Real, T2<:Real}

  @unpack B_aug = model.aug_state

  # 1) Extract slices at time t
  local P_pred = @view P_ttm1[:, :, t]  # (P×P)
  local M_pred = @view M_ttm1[:, :, t]  # (M×M)
  local Kslice = @view K_t[:, :, t]     # (P×M)

  # 2) Compute K_pre = P_pred * B_aug' (no redundant S term)
  mul!(Kslice, P_pred, B_aug', 1.0, 0.0)  # Kslice = P_pred * B_aug'

  # 3) Compute the Kalman gain using M_pred directly
  if size(M_pred,1) == 1
      # Scalar case
      @. Kslice = Kslice / M_pred[1, 1]
  else
      # Matrix solve: Kslice = (P_pred * B_aug') * inv(M_pred)
      F = lu(M_pred)  # Use M_pred instead of S
      for col in 1:size(Kslice,2)
          view(Kslice, :, col) .= F \ view(Kslice, :, col)
      end
  end
end

"""
    compute_K_t(P_ttm1, M_ttm1, model, t) -> Matrix{T}

Return the Kalman gain `K_t` for measurement dimension `M ≥ 1`, 
using the one-step-ahead state covariance and measurement covariance.

# Arguments
- `P_ttm1::AbstractMatrix{T}`: `(P×P)` the one-step-ahead predicted covariance.
- `M_ttm1::AbstractMatrix{T}`: `(M×M)` the measurement covariance for the current step.
- `model::QKModel{T,T2}`: Must hold:
  - `B_aug::Matrix{T}` of size `(M, P)`,
  - Possibly `M, P::Int`,
  - etc.
- `t::Int`: Time index (not strictly needed, but included for consistency).

# Returns
A newly allocated `(P×M)` Kalman Gain matrix, computed by:
`S = B_augP_ttm1B_aug' + M_ttm1 K = (P_ttm1*B_aug') * inv(S)`
- If `M=1`, S is `1×1`, so we do a scalar division.
- If `M>1`, we do a matrix solve or factorization-based inverse.

# Notes
- This function does **not** modify its inputs in place.
- For large `M`, prefer a factorization-based approach 
  (e.g. `lu(S); K = K / S`) instead of `inv(S)` for better performance.

"""
function compute_K_t(
    P_ttm1::AbstractMatrix{T},
    M_ttm1::AbstractMatrix{T},
    model::QKModel{T,T2},
    t::Int) :: Matrix{T} where {T<:Real, T2<:Real}

    @unpack B_aug = model.aug_state

    # 1) K_pre = P_ttm1 * B_aug'
    K_pre = P_ttm1 * B_aug'

    # 2) Create K
    local K::Matrix{T}
    if size(M_ttm1,1) == 1
        # M=1 => scalar
        denom = M_ttm1[1,1]
        K = K_pre ./ denom  # elementwise
    else
        # M>1 => do a solve
        F = lu(M_ttm1)
        # We'll solve S * K' = K_pre' to get K
        K = similar(K_pre)
        # Solve the transposed system and transpose back
        K .= (F \ K_pre')'
    end

    return K
end

"""
    update_Z_tt!(Z_tt, K_t, Y, Y_ttm1, Z_ttm1, t)

In-place update of the augmented state `Z_tt[:, t+1]` at time `t+1` 
using the current measurements and the Kalman gain.

# Arguments
- `Z_tt::AbstractMatrix{T}`: 
  A `(P×T̄)` array holding the **filtered** or updated states. We will overwrite 
  column `t+1`.
- `K_t::AbstractArray{T,3}`: 
  A `(P×M×T̄)` 3D array of Kalman gains. We read `K_t[:, :, t]` (size `(P×M)`).
- `Y::AbstractMatrix{T}`: 
  The actual measurement data, size `(M×T̄)`. If univariate, shape is `(1×T̄)`.
- `Y_ttm1::AbstractMatrix{T}`: 
  The predicted measurement, also `(M×T̄)`. We read column `t` for time step.
- `Z_ttm1::AbstractMatrix{T}`: 
  The one-step-ahead predicted state `(P×T̄)`. We read column `t`.
- `t::Int`: 
  The (1-based) time index.

# Returns
Nothing. It modifies `Z_tt[:, t+1]` **in place** to:
`Z_tt[:, t+1] = Z_ttm1[:, t] + K_t[:, :, t] * ( Y[:, t] - Y_ttm1[:, t] )`

# Notes
- For univariate data (`M=1`), both `Y` and `Y_ttm1` are `(1×T̄)`. 
- No separate overload needed; the same code covers scalar or vector measurements, 
  because it's all matrix-based with a single row if univariate.
- This is **AD-friendly** for typical Julia AD frameworks, 
  though in-place mutation sometimes needs a custom adjoint.

"""
function update_Z_tt!(
    Z_tt::AbstractMatrix{T},
    K_t::AbstractArray{T,3},
    Y::AbstractMatrix{T},
    Y_ttm1::AbstractMatrix{T},
    Z_ttm1::AbstractMatrix{T},
    t::Int) where {T<:Real}

    # Core linear update
    view(Z_tt, :, t+1) .= view(Z_ttm1, :, t) .+ view(K_t, :, :, t) * (view(Y, :, t + 1) .- view(Y_ttm1, :, t))
end

"""
    update_Z_tt(K_t, Y, Y_ttm1, Z_ttm1, t) -> Vector{T}

Return a **newly allocated** updated state vector for time `t+1`, using the Kalman gain 
and measurement data.

# Arguments
- `K_t::AbstractArray{T,3}`: 
  A 3D array of Kalman gains `(P×M×T̄)`. 
  We read `K_t[:, :, t]`, which is `(P×M)`.

- `Y::AbstractMatrix{T}`: 
  The measurement array `(M×T̄)`. 
  For univariate data, shape `(1×T̄)`.

- `Y_ttm1::AbstractMatrix{T}`: 
  The predicted measurement `(M×T̄)`. 
  We read column `t` for time step.

- `Z_ttm1::AbstractMatrix{T}`: 
  The one-step-ahead predicted state `(P×T̄)`, 
  from which we read column `t`.

- `t::Int`: 
  Time index (1-based).

# Returns
A newly allocated vector of length `P`, computed via:
`Z_next = Z_ttm1[:, t] + K_t[:, :, t] * ( Y[:, t] - Y_ttm1[:, t] )`

# Notes
- This is the **out-of-place** version, returning a fresh vector each time.
- Useful if you prefer a purely functional style or if you want to keep 
  your existing arrays immutable for AD.

"""
function update_Z_tt(
    K_t::AbstractArray{T1,3},
    Y::AbstractVecOrMat{T},
    Y_ttm1::AbstractMatrix{T1},
    Z_ttm1::AbstractMatrix{T1},
    t::Int) where {T<:Real, T1 <: Real}

    # Handle both vector and matrix Y inputs
    current_Y = Y isa AbstractVector ? Y[t+1] : Y[:, t+1]
    
    # Compute innovation
    innovation = current_Y .- Y_ttm1[:, t]
    
    # Return updated state
    return Z_ttm1[:, t] .+ K_t[:, :, t] * innovation
end

"""
    update_P_tt!(P_tt, K_t, P_ttm1, Z_ttm1, tmpKM, tmpKMK, model, t)

In-place update of the **filtered covariance** `P_tt[:, :, t+1]` given the one-step-ahead 
covariance `P_ttm1[:, :, t]`, and the Kalman gain `K_t[:, :, t]`.

# Arguments
- `P_tt::AbstractArray{<:Real, 3}`: A 3D array `(P×P×T̄)` storing filtered covariances.  
  We write the new filtered covariance into `P_tt[:, :, t+1]`.
- `K_t::AbstractArray{<:Real, 3}`: A 3D array `(P×M×T̄)` of Kalman gains, 
  using `K_t[:, :, t]`.
- `P_ttm1::AbstractArray{<:Real, 3}`: The one-step-ahead covariance array `(P×P×T̄)`, 
  from which `P_ttm1[:, :, t]` is read.
- `Z_ttm1::AbstractArray{<:Real, 2}`: `(P×T̄)` storing the predicted augmented state. 
- `tmpKM, tmpKMK::AbstractMatrix{<:Real}`: Temporary buffers `(P×M, P×P)` 
  if you want manual multiplication. In the final code, we do not use them, 
  but they can be placeholders for expansions.
- `model::QKModel{T,T2}`: Must contain:
  - `B_aug::Matrix{T}` (size `M×P`),
  - `V::Matrix{T}` (size `M×M`),
  - `P::Int, M::Int`, etc. 
- `t::Int`: The time index (1-based).

# Computation
1. Let `A = I - K_t[:, :, t]*B_aug`.
2. Then
    `P_tt[:, :, t+1] = make_positive_definite(A * P_ttm1[:, :, t] * A' + K_t[:, :, t]*V*K_t[:, :, t])'
3. We wrap the result with make_positive_definite to ensure no negative eigenvalues from the update.

# Notes
- This is an in-place update: we store the new covariance in Pₜₜ[:, :, t+1].
- AD-Friendliness: The final expression is a typical linear+outer-product operation plus
   make_positive_definite. If your AD can handle in-place modifications, or you define a
   custom adjoint, it should be fine. Otherwise, consider a purely functional approach.
"""
function update_P_tt!(P_tt::AbstractArray{T1,3}, K_t::AbstractArray{T1,3},
    P_ttm1::AbstractArray{T1,3},
    Z_ttm1::AbstractMatrix{T1}, tmpKM::AbstractMatrix{T1},
    tmpKMK::AbstractMatrix{T1}, model::QKModel{T,T2}, t::Int) where {T1<:Real, T<:Real, T2<:Real}
    @unpack V = model.meas
    @unpack B_aug = model.aug_state
    
    # 1) Extract views
    local K_slice = view(K_t, :, :, t)
    local P_ttm1_view = view(P_ttm1, :, :, t)
    local P_tt_dest = view(P_tt, :, :, t+1)
    
    # 2) Form A = I - K_slice*B_aug
    # Create identity matrix of appropriate size
    P = size(K_slice, 1)
    A = Matrix{T1}(I, P, P)
    mul!(tmpKM, K_slice, B_aug)  # tmpKM = K_slice * B_aug
    A .-= tmpKM  # A = I - tmpKM
    
    # 3) Compute A * P_ttm1_view * A' efficiently
    mul!(tmpKMK, A, P_ttm1_view)  # tmpKMK = A * P_ttm1_view
    mul!(P_tt_dest, tmpKMK, A')   # P_tt_dest = tmpKMK * A'
    
    # 4) Add the K*V*K' term - MODIFIED TO FIX DIMENSION MISMATCH
    # First compute KV directly - K_slice is P×M, V is M×M, result is P×M
    KV = K_slice * V  # Temporary allocation, but avoids dimension issues
    
    # Then compute KV*K' to get P×P result
    mul!(tmpKM, KV, K_slice')  # tmpKM = KV * K_slice'
    P_tt_dest .+= tmpKM        # Add to result
    
    # 5) Ensure positive-definiteness
    P_tt_dest .= make_positive_definite(P_tt_dest)
end

"""
    update_P_tt(K_t::AbstractMatrix{T1}, P_ttm1::AbstractMatrix{T1}, Z_ttm1::AbstractVector{T1}, 
                model::QKModel{T3, T2}, t::Int64) where {T1<:Real, T2<:Real, T3<:Real}

Compute and return a **new** filtered covariance matrix, 
given a predicted covariance `P_ttm1`, a Kalman gain `K_t`.

# Arguments
- `K_t::AbstractMatrix{T1}`: (P×M) or (P×1) Kalman gain for time `t`.
- `P_ttm1::AbstractMatrix{T1}`: The one-step-ahead covariance `(P×P)`.
- `Z_ttm1::AbstractVector{T1}`: The predicted augmented state (length P).
- `model::QKModel{T3, T2}`: Must hold:
  - `B_aug::Matrix{T}` (size `M×P`),
  - `V::Matrix{T}` (size `M×M`),
  - `P::Int`, `M::Int`, etc.
- `t::Int`: The time index (not strictly used, but might be for logging or consistency).

# Returns
- `Matrix{T}`: A newly allocated `(P×P)` covariance matrix 
  after the update step, guaranteed to be positive-definite if `make_positive_definite` 
  fixes any negative eigenvalues.

# Details
1. Let `A = I - K_t*B_aug`.
2. Build `P = A * P_ttm1 * A' + K_t*V*K_t'`
3. This version is purely functional (returns a new matrix). It's often simpler for AD if
   you want a direct forward pass without in-place modifications.

"""
function update_P_tt(K_t::AbstractMatrix{T1}, P_ttm1::AbstractMatrix{T1}, Z_ttm1::AbstractVector{T1}, 
                    model::QKModel{T3, T2}, t::Int64) where {T1<:Real, T2<:Real, T3<:Real}
    @unpack V = model.meas
    @unpack B_aug = model.aug_state

    # 1) A = I - K_t*B_aug
    A = I - K_t*B_aug

    # 2) Build the new covariance
    P = A * P_ttm1 * A' .+ (K_t * V * K_t')

    # 4) Ensure positivity
    if !isposdef(P)
        return make_positive_definite(P)
    else
        return P
    end
end

"""
    correct_Z_tt!(Z_tt, model, t)

In-place correction of the sub-block in `Z_tt` that corresponds to the implied
covariance `[XX']ₜ|ₜ - Xₜ|ₜ Xₜ|ₜ'`. This function ensures that sub-block is 
positive semidefinite (PSD).

# Arguments
- `Z_tt::AbstractMatrix{<:Real}`:
  A matrix where each column is a state vector. The `(t+1)`th column contains:
    - The first `N` entries = `X_t|t`
    - The next `N*N` entries (reshaped to `N×N`) = `[X X']_t|t`.
- `model::QKModel`: A parameter struct holding (at least) `N`, the state dimension.
- `t::Int`: The time index whose data we are correcting. 

# Details
1. Extract the current column `Z_tt = Z_tt[:, t+1]`.
2. Let `x_t = Z_tt[1:N]`.
3. Implied covariance = `reshape(Z_tt[N+1:end], N, N) - x_t * x_t'`.
4. We compute its eigen-decomposition and clamp negative eigenvalues to 0. 
5. Reconstruct the PSD version and store it back into `Z_tt[N+1:end, t+1]`.

This follows the idea in the QKF algorithm to keep the implied covariance valid.
"""
function correct_Z_tt!(Z_tt::AbstractMatrix{T1}, model::QKModel{T, T2}, t::Int) where 
    {T1 <: Real, T <: Real, T2 <: Real}

    @unpack N = model.state
    
    # 1) Extract relevant piece with view
    Z_current = @view Z_tt[:, t + 1] # Z_tt[:, t+1] is the "current" column in the filter
    
    # 2) Extract x_t and implied (XX')ₜ with views
    xt_view = @view Z_current[1:N]                    # view of the first N entries (xₜ|ₜ)
    XtXt_prime_data = @view Z_current[N+1:end]        # view of the XX' data
    XtXt_prime = reshape(XtXt_prime_data, N, N)       # reshape the view (no copy)
    
    # 3) implied_cov = (XX') - xₜ xₜ'
    # Need to materialize this for eigen decomposition
    implied_cov = XtXt_prime - xt_view*xt_view'
    
    # 4) Eigen-decomposition to clamp negative eigenvalues
    #    Use `Symmetric` to ensure real sym decomposition
    F = LinearAlgebra.eigen(Symmetric(implied_cov))
    eig_vals, eig_vecs = F.values, F.vectors
    
    # 5) Replace negative eigenvalues with 0 => PSD
    eig_vals .= max.(eig_vals, 0.0)
    
    # 6) Reassemble the corrected block more efficiently
    corrected = similar(implied_cov)
    mul!(corrected, eig_vecs, Diagonal(eig_vals))
    mul!(XtXt_prime, corrected, eig_vecs')         
    
    # 7) Add the outer product and write back
    XtXt_prime .+= xt_view * xt_view'
    
    return nothing
end

"""
    correct_Z_tt(Z_tt, model, t) -> Vector

Non-in-place version returning a new corrected state vector, 
using a simple ε-shift instead of eigenvalue truncation.

# Arguments
- `Z_tt::AbstractVector{<:Real}`:
  A single "flattened" state vector, containing:
    - The first `N` entries = `x_t|t`
    - The next `N*N` entries (reshaped to `N×N`) = `[X X']_t|t`.
- `model::QKModel`: A parameter struct holding (at least) `N`, the state dimension.
- `t::Int`: (Unused here, but consistent with the signature of the in-place version.)

# Details
1. Extract `x_t` from the front.
2. Compute implied_cov = (XX') - xₜ xₜ'.
3. "Correct" it by adding `ε*I` to ensure positivity (a cheap fix).
4. Recombine into a single vector: [x_t; vec(corrected_cov + xₜ xₜ')].

Returns the newly constructed vector, leaving the original unchanged.
"""
function correct_Z_tt(Z_tt::AbstractVector{T1}, model::QKModel{T,T2}, t::Int) where 
  {T1 <: Real, T <: Real, T2 <: Real}

  @unpack N = model.state
  
  # 1) Extract xₜ
  xt = Z_tt[1:N]
  
  # 2) Extract XX'
  XtXt_prime = reshape(Z_tt[N+1:end], N, N)
  
  # 3) Compute implied covariance matrix
  implied_cov = Symmetric(XtXt_prime - xt * xt')  # Enforce symmetry for eigen
  
  # 4) Eigenvalue-based correction (AD-friendly)
  F = LinearAlgebra.eigen(Symmetric(implied_cov))
  eigs_corrected = max.(F.values, zero(T))  # Clamp negative eigenvalues to 0
  corrected_cov = F.vectors * Diagonal(eigs_corrected) * F.vectors'
  
  # 5) Ensure symmetry (due to potential floating point errors)
  corrected_cov = Symmetric(corrected_cov)
  
  # 6) Recombine xₜ with corrected XX'
  corrected_XX = corrected_cov + xt * xt'
  
  return vcat(xt, vec(corrected_XX))
end

"""
    qkf_filter(data::QKData{T1,N}, model::QKModel{T,T2}) -> FilterOutput{T}

Perform the Quadratic Kalman Filter (QKF) in a purely functional (non-mutating) manner.

# Arguments
- `data::QKData{T1,N}`: Observed data container.
- `model::QKModel{T,T2}`: Model parameters and matrices.

# Returns
- `FilterOutput{T}`: Struct containing filtered states (`Z_tt`), covariances (`P_tt`), predicted states (`Z_ttm1`), predicted covariances (`P_ttm1`), predicted measurements (`Y_ttm1`), measurement covariances (`M_ttm1`), Kalman gains (`K_t`), and log-likelihoods (`ll_t`).

# Details
This function allocates new arrays at each step and does not mutate any input arrays. It is suitable for use with automatic differentiation frameworks or when immutability is desired.
"""
function qkf_filter(data::QKData{T1,N}, model::QKModel{T,T2}) where {T1<:Real, T<:Real, T2<:Real, N}
    @unpack T_bar, Y, M = data
    @unpack aug_mean, aug_cov = model.moments
    @unpack P = model.aug_state

    # Handle both vector and matrix Y inputs
    Y_concrete = if N == 1
      Vector{T1}(vec(Y))  # Convert to vector if it's not already
    else
        Y  # Keep as matrix for N == 2
    end
    Y_t = if N == 1
        @view Y_concrete[1:end]
    else
        @view Y[:, 1:end]
    end
    # Allocate arrays
    Z_tt = zeros(T, P, T_bar + 1)
    P_tt = zeros(T, P, P, T_bar + 1)
    Z_ttm1 = zeros(T, P, T_bar)
    P_ttm1 = zeros(T, P, P, T_bar)
    K_t = zeros(T, P, M, T_bar)
    Y_ttm1 = zeros(T, M, T_bar)
    M_ttm1 = zeros(T, M, M, T_bar)
    ll_t = zeros(T, T_bar)

    # Initialize
    Z_tt[:, 1] = aug_mean
    P_tt[:, :, 1] = aug_cov

    # Main filtering loop
    for t in 1:T_bar
        # Predict state
        Z_ttm1[:, t] = predict_Z_ttm1(Z_tt[:, t], model)
        
        # Predict covariance
        P_ttm1[:, :, t] = predict_P_ttm1(P_tt[:, :, t], Z_tt[:, t], model, t)

        # Predict measurement
        Y_ttm1[:, t] = predict_Y_ttm1(Z_ttm1, Y_t, model, t)
        
        # Predict measurement covariance
        M_ttm1[:, :, t] = predict_M_ttm1(P_ttm1[:, :, t], model)

        # Compute Kalman gain
        K_t[:, :, t] = compute_K_t(P_ttm1[:, :, t], M_ttm1[:, :, t], model, t)

        # Handle different Y formats based on N
        Z_tt[:, t + 1] = update_Z_tt(K_t, Y_t, Y_ttm1, Z_ttm1, t)
        
        # Update covariance
        P_tt[:, :, t + 1] = update_P_tt(K_t[:, :, t], P_ttm1[:, :, t], Z_ttm1[:, t], model, t)

        # PSD correction
        Z_tt[:, t + 1] = correct_Z_tt(Z_tt[:, t + 1], model, t)

        # Compute log-likelihood
        ll_t[t] = compute_loglik(Y_t, Y_ttm1, M_ttm1, t)
    end

    # Final log-likelihood at T̄
    # Y_ttm1[:, T̄] = predict_Y_ttm1(Z_ttm1, Y, model, T̄)
    # M_ttm1[:, :, T̄] = predict_M_ttm1(P_ttm1[:, :, T̄], model)
    # ll_t[T̄] = compute_loglik(Y, Y_ttm1, M_ttm1, T̄)

    return FilterOutput(ll_t, Z_tt, P_tt, Y_ttm1, M_ttm1, K_t, Z_ttm1, P_ttm1)
end

"""
    _qkf_filter_impl!(
        Z_tt, P_tt, Z_ttm1, P_ttm1, 
        Y_ttm1, M_ttm1, K_t, ll_t,
        Sigma_ttm1, tmpP, tmpKM, tmpKMK, tmpB,
        data, model
    ) -> Nothing

Low-level implementation of the Quadratic Kalman Filter (QKF) that performs in-place operations
with user-provided storage arrays.

# Arguments
- `Z_tt::AbstractMatrix{T}`: Preallocated (P×T̄) matrix for filtered states.
- `P_tt::AbstractArray{T,3}`: Preallocated (P×P×T̄) array for filtered covariances.
- `Z_ttm1::AbstractMatrix{T}`: Preallocated (P×(T̄-1)) matrix for predicted states.
- `P_ttm1::AbstractArray{T,3}`: Preallocated (P×P×(T̄-1)) array for predicted covariances.
- `Y_ttm1::AbstractMatrix{T}`: Preallocated (M×(T̄-1)) matrix for predicted measurements.
- `M_ttm1::AbstractArray{T,3}`: Preallocated (M×M×(T̄-1)) array for predicted measurement covariances.
- `K_t::AbstractArray{T,3}`: Preallocated (P×M×(T̄-1)) array for Kalman gains.
- `ll_t::AbstractVector{T}`: Preallocated vector of length (T̄-1) for log-likelihoods.
- `Sigma_ttm1::AbstractArray{T,3}`: Preallocated (P×P×(T̄-1)) array for state-dependent noise terms.
- `tmpP::AbstractMatrix{T}`: Scratch buffer of size (P×P).
- `tmpKM::AbstractMatrix{T}`: Scratch buffer of size (P×P).
- `tmpKMK::AbstractMatrix{T}`: Scratch buffer of size (P×P).
- `tmpB::AbstractMatrix{T}`: Scratch buffer of size (M×P).
- `data::QKData{T}`: Contains the observation data `Y` (M×T̄).
- `model::QKModel{T,T2}`: Contains all model parameters.

# Details
This is an internal, low-level function that performs the QKF algorithm with zero allocations.
All storage arrays must be preallocated and passed in by the caller. This allows for maximum 
performance in repeated calls or integration with optimization routines.

The function writes results directly into the provided arrays and returns nothing.
"""
function _qkf_filter_impl!(
    Z_tt::AbstractMatrix{T}, 
    P_tt::AbstractArray{T,3}, 
    Z_ttm1::AbstractMatrix{T}, 
    P_ttm1::AbstractArray{T,3},
    Y_ttm1::AbstractMatrix{T}, 
    M_ttm1::AbstractArray{T,3}, 
    K_t::AbstractArray{T,3}, 
    ll_t::AbstractVector{T},
    Sigma_ttm1::AbstractArray{T,3}, 
    tmpP::AbstractMatrix{T}, 
    tmpKM::AbstractMatrix{T}, 
    tmpKMK::AbstractMatrix{T}, 
    tmpB::AbstractMatrix{T},
    data::QKData{T}, 
    model::QKModel{T,T2}
) where {T <: Real, T2 <: Real}
    
    @unpack Y = data
    T̄ = size(Y, 2)  # Number of time steps

    # Initialize state and covariance
    Z_tt[:, 1] .= model.moments.aug_mean
    P_tt[:, :, 1] .= model.moments.aug_cov

    # Main filtering loop
    for t in 1:(T̄-1)
        # (1) Predict state
        predict_Z_ttm1!(Z_tt, Z_ttm1, model, t)

        # (2) Predict covariance
        predict_P_ttm1!(P_tt, P_ttm1, Sigma_ttm1, Z_tt, tmpP, model, t)

        # (3) Predict measurement
        predict_Y_ttm1!(Y_ttm1, Z_ttm1, Y, model, t)

        # (4) Predict measurement covariance
        predict_M_ttm1!(M_ttm1, P_ttm1, tmpB, model, t)

        # (5) Compute Kalman gain
        compute_K_t!(K_t, P_ttm1, M_ttm1, tmpB, model, t)

        # (6) Update state - updates at t+1
        update_Z_tt!(Z_tt, K_t, Y, Y_ttm1, Z_ttm1, t)

        # (7) Update covariance (using extra scratch buffers) - updates at t+1
        update_P_tt!(P_tt, K_t, P_ttm1, Z_ttm1, tmpKM, tmpKMK, model, t)

        # (8) PSD correction on the augmented state - updates at t+1
        correct_Z_tt!(Z_tt, model, t)

        # (9) Compute log-likelihood for this step
        compute_loglik!(ll_t, Y, Y_ttm1, M_ttm1, t)
    end
    
    return nothing
end

"""
    qkf_filter!(data::QKData{T}, model::QKModel{T,T2}) -> FilterOutput{T}

Run the Quadratic Kalman Filter in-place on the given data and model.

This is the fully mutating version of `qkf_filter` that uses in-place operations
to minimize memory allocations.

# Arguments
- `data::QKData{T}`: Contains the observation data `Y` (M×T̄).
- `model::QKModel{T,T2}`: Contains all model parameters.

# Returns
- `FilterOutput{T}`: A struct containing all filter results:
  - `Z_tt`: Filtered state estimates (P×T̄)
  - `P_tt`: Filtered covariance matrices (P×P×T̄)
  - `Z_ttm1`: Predicted state estimates (P×T̄)
  - `P_ttm1`: Predicted covariance matrices (P×P×T̄)
  - `Y_ttm1`: Predicted observations (M×T̄)
  - `M_ttm1`: Predicted observation covariances (M×M×T̄)
  - `K_t`: Kalman gain matrices (P×M×T̄)
  - `ll_t`: Log-likelihood at each time step (T̄)

# Details
This function performs the full QKF algorithm using in-place operations:
1. Initialize state and covariance
2. For each time step:
   - Predict state and covariance
   - Predict measurement and its covariance
   - Compute Kalman gain
   - Update state and covariance
   - Apply PSD correction
   - Compute log-likelihood
3. Return all results

This version is optimized for performance by using in-place operations.

For power users who need full control over memory allocation, see the internal
`_qkf_filter_impl!` function.
"""
function qkf_filter!(data::QKData{T}, model::QKModel{T,T2}) where {T <: Real, T2 <: Real}
    @unpack Y = data
    T̄ = size(Y, 2)                # Number of time steps
    P = size(model.aug_state.Phi_aug, 1)  # Augmented state dimension
    M = size(Y, 1)                # Measurement dimension

    # Allocate arrays for filter results
    Z_tt = zeros(T, P, T̄)
    P_tt = zeros(T, P, P, T̄)
    Z_ttm1 = zeros(T, P, T̄ - 1)
    P_ttm1 = zeros(T, P, P, T̄ - 1)
    Y_ttm1 = zeros(T, M, T̄ - 1)
    M_ttm1 = zeros(T, M, M, T̄ - 1)
    K_t = zeros(T, P, M, T̄ - 1)
    ll_t = zeros(T, T̄ - 1)
    
    # Temporary storage for computations
    Sigma_ttm1 = zeros(T, P, P, T̄ - 1)
    tmpP = zeros(T, P, P)      # Used by predict_P_ttm1!
    tmpKM = zeros(T, P, P)     # Used by update_P_tt!
    tmpKMK = zeros(T, P, P)    # Used by update_P_tt!
    tmpB = zeros(T, M, P)      # Used by predict_M_ttm1! and compute_K_t!

    # Call the low-level implementation function
    _qkf_filter_impl!(
        Z_tt, P_tt, Z_ttm1, P_ttm1, 
        Y_ttm1, M_ttm1, K_t, ll_t,
        Sigma_ttm1, tmpP, tmpKM, tmpKMK, tmpB,
        data, model
    )

    # Return a FilterOutput with the results
    return FilterOutput(ll_t, Z_tt, P_tt, Y_ttm1, M_ttm1, K_t, Z_ttm1, P_ttm1)
end

