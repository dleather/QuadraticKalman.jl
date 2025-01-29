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

    @unpack aug_mean, Phi_aug = model.aug_state
    Z_ttm1[:, t] = aug_mean .+ Phi_aug * Z_tt[:, t]
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

    @unpack Phi_aug = model.aug_state
    @unpack aug_mean = model.moments
    return aug_mean .+ Phi_aug * Z_tt
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
function predict_P_ttm1!(P_tt::AbstractArray{Real,3}, P_ttm1::AbstractArray{Real,3},
    Σ_ttm1::AbstractArray{Real,3}, Z_tt::AbstractMatrix{T}, tmpP::AbstractMatrix{T}, 
    model::QKModel{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    # 1) Update Σₜₜ₋₁ based on Zₜₜ (the code snippet calls compute_Σₜₜ₋₁!)
    compute_Sigma_ttm1!(Σ_ttm1, Z_tt, model, t)

    # 2) Build predicted covariance
    @unpack Phi_aug = model.aug_state
    P_ttm1[:, :, t] = ensure_positive_definite(
        Phi_aug * P_tt[:, :, t] * Phi_aug' .+ Σ_ttm1[:, :, t]
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

    Σ_ttm1 = compute_Sigma_ttm1(Z_tt, model)

    @unpack Phi_aug = model.aug_state
    P_tmp = Phi_aug * P_tt * Phi_aug' .+ Σ_ttm1

    if !isposdef(P_tmp)
        return make_positive_definite(P_tmp)
    else
        return P_tmp
    end
end

"""
    predict_Y_ttm1!(Y_ttm1::AbstractMatrix{T}, 
                   Z_ttm1::AbstractMatrix{T}, 
                   Y::AbstractMatrix{T}, 
                   model::QKModel{T,T2}, 
                   t::Int)

In-place update of the predicted measurement `Y_ttm1[:, t]` given:
- A *matrix* of state estimates `Z_ttm1`,
- A *matrix* of actual observations `Y`, and
- Parameter struct `model`.

# Arguments
- `Y_ttm1::AbstractMatrix{T}`: (M×T̄) array where the predicted measurement 
  at time `t` is stored in `Y_ttm1[:, t]`.
- `Z_ttm1::AbstractMatrix{T}`: (P×T̄) array of the one-step-ahead predicted states.
- `Y::AbstractMatrix{T}`: (M×T̄) array of actual observations.
- `model::QKModel{T,T2}`: Contains:
  - `A::Vector{T}`, `B̃::Matrix{T}`: The observation model terms.
  - `α::Matrix{T}`: Additional AR-like component for measurement.
- `t::Int`: Time index (1-based).

# Details
Computes `Y_ttm1[:, t] = A + B̃*Z_ttm1[:, t] + α * Y[:, t]`.
This is stored in-place in `Y_ttm1`. Purely a linear operation, 
so it is **AD-friendly**.

"""
function predict_Y_ttm1!(Y_ttm1::AbstractMatrix{T}, Z_ttm1::AbstractMatrix{T},
    Y::AbstractMatrix{T}, model::QKModel{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    @unpack A, α = model.meas
    @unpack B_aug = model.aug_state
    Y_ttm1[:, t] = A .+ B_aug * Z_ttm1[:, t] .+ α * Y[:, t]
end


"""
    predict_Y_ttm1!(Y_ttm1::AbstractVector{T}, 
                   Z_ttm1::AbstractMatrix{T}, 
                   Y::AbstractVector{T1}, 
                   model::QKModel{T,T2}, 
                   t::Int)

In-place update of the predicted measurement `Y_ttm1[t]` when both 
the measurement and observation are stored as vectors.

# Arguments
- `Y_ttm1::AbstractVector{T}`: A 1D array for predicted measurements, 
  storing the result at index `t`.
- `Z_ttm1::AbstractMatrix{T}`: (P×T̄) predicted states array.
- `Y::AbstractVector{T1}`: 1D array of actual measurements over time (size ≥ t).
- `model::QKModel{T,T2}`: Holds `A::Vector{T}`, `B̃::Matrix{T}`, 
  `alpha::Matrix{T}` or scalar, etc.  
- `t::Int`: The time index.

# Details
Computes `Y_ttm1[t] = (A + B̃ * Z_ttm1[:, t] + alpha * Y[t])[1]`, 
assuming `A` might be a 1-element vector, `alpha` is effectively 
(1×1), etc. Overwrites `Y_ttm1[t]` in place.

Useful for a univariate measurement scenario where measurement arrays 
are stored as vectors. AD-friendly as a single linear operation.

"""
function predict_Y_ttm1!(Y_ttm1::AbstractVector{T}, Z_ttm1::AbstractMatrix{T},
    Y::AbstractVector{T1}, model::QKModel{T,T2},
    t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

    @unpack A, alpha = model.meas
    @unpack B_aug = model.aug_state
    # Typically A is [scalar], B̃ is (1×P), α is (1×1?), 
    # so we do an indexing trick to retrieve a single scalar.
    Y_ttm1[t] = (A .+ B_aug * Z_ttm1[:, t] .+ alpha .* Y[t])[1]
end


"""
    predict_Y_ttm1!(Y_ttm1::AbstractMatrix{T}, 
                   Z_ttm1::AbstractMatrix{T}, 
                   Y::AbstractVector{T1}, 
                   model::QKModel{T,T2}, 
                   t::Int)

In-place update of `Y_ttm1[:, t]` using a vector `Y` of observations.

# Arguments
- `Y_ttm1::AbstractMatrix{T}`: (M×T̄) predicted measurement array; 
  we store the new column in `Y_ttm1[:, t]`.
- `Z_ttm1::AbstractMatrix{T}`: (P×T̄) predicted states.
- `Y::AbstractVector{T1}`: A 1D array of measurements over time (size ≥ t). 
  Typically used if `M==1`, or if `alpha` maps a scalar observation into M dims.
- `model::QKModel{T,T2}`: Model parameters, including 
  `A`, `B_aug`, `alpha`.
- `t::Int`: Time index.

# Details
`Y_ttm1[:, t] = A + B_aug*Z_ttm1[:, t] + alpha * Y[t]`.

If `M > 1`, we interpret `alpha` as an (M×1) or (M×M) matrix that 
broadcasts scalar Y[t] across M. 
AD-friendly, single linear expression.

"""
function predict_Y_ttm1!(Y_ttm1::AbstractMatrix{T}, Z_ttm1::AbstractMatrix{T},
    Y::AbstractVector{T1}, model::QKModel{T,T2}, 
    t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

    @unpack A, alpha = model.meas
    @unpack B_aug = model.aug_state
    Y_ttm1[:, t] = A .+ B_aug * Z_ttm1[:, t] .+ alpha .* Y[t]
end

"""
    predict_Y_ttm1(Z_ttm1::AbstractMatrix{T}, 
                   Y::AbstractMatrix{T1}, 
                   model::QKModel{T,T2}, 
                   t::Int) -> Vector{T}

Return a **new** predicted measurement vector for time `t`, 
given matrix-based `Z_ttm1` and `Y`.

# Arguments
- `Z_ttm1::AbstractMatrix{T}`: (P×T̄) predicted states.
- `Y::AbstractMatrix{T1}`: (M×T̄) actual observations.
- `model::QKModel{T,T2}`: Contains `A`, `B_aug`, `alpha`.
- `t::Int`: Time index.

# Returns
- `Vector{T}`: The predicted measurement at time `t`, 
  computed as `A + B_aug*Z_ttm1[:, t] + alpha*Y[:, t]`.

# Details
This version returns the result as a newly allocated vector. 
Use it if you want a purely functional style with no in-place modifications.

"""
function predict_Y_ttm1(Z_ttm1::AbstractMatrix{T}, Y::AbstractMatrix{T1},
    model::QKModel{T,T2}, t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack A, alpha = model.meas
    @unpack B_aug = model.aug_state
    return A .+ B_aug * Z_ttm1[:, t] .+ alpha * Y[:, t]
end

"""
    predict_Y_ttm1(Z_ttm1::AbstractVector{T}, 
                   Y::AbstractVector{T1}, 
                   model::QKModel{T,T2}, 
                   t::Int) -> Real (or 1-element vector)

Compute a new predicted measurement for univariate (or effectively scalar) 
observation, returning the result as a scalar.

# Arguments
- `Z_ttm1::AbstractVector{T}`: The predicted state vector of dimension P.
- `Y::AbstractVector{T1}`: The (univariate) actual observations over time.
- `model::QKModel{T,T2}`: Must hold `A::Vector{T}` (often length=1), 
  `B̃::Matrix{T}` (size 1×P?), `α::Matrix{T}` or scalar for the measurement update.
- `t::Int`: Time index.

# Returns
A single real number `predict_val`, computed as `(A + B_aug*Z_ttm1 + alpha*Y[t])[1]`.

# Details
- If `A` has length > 1, or `B_aug` is larger, you might effectively get the first element. 
- For truly univariate measurement, this is a simpler approach.
"""
function predict_Y_ttm1(Z_ttm1::AbstractVector{T}, Y::AbstractVector{T1},
    model::QKModel{T,T2}, t::Int) where {T1<:Real, T<:Real, T2<:Real}

    @unpack A, alpha = model.meas
    @unpack B_aug = model.aug_state
    return (A .+ B_aug*Z_ttm1 .+ alpha .* Y[t])[1]
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
    mul!(tmpB, B_aug, P_ttm1[:,:,t])

    #    Mₜₜ₋₁[:,:,t] = tmpB * B̃'
    mul!(M_ttm1[:,:,t], tmpB, B̃', 1.0, 0.0)

    # 2) Add V to Mₜₜ₋₁[:,:,t]
    @inbounds for i in 1:M
        for j in 1:M
            M_ttm1[i,j,t] += V[i,j]
        end
    end

    # 3) Check for negativity in univariate case
    if M == 1
        if M_ttm1[1,1,t] < 0
            M_ttm1[1,1,t] = 1e-4
        end

    # 4) For multivariate, enforce positive-definiteness if needed
    else
        # `view` avoids copying, but you can index directly as well
        local slice_ref = @view M_ttm1[:,:,t]
        if !isposdef(slice_ref)
            local corrected = make_positive_definite(slice_ref)
            slice_ref .= corrected
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
    compute_K_t!(K_t::AbstractArray{Real,3}, 
                P_ttm1::AbstractArray{Real,3},
                M_ttm1::AbstractArray{Real,3}, 
                tmpPB::AbstractMatrix{T}, 
                model::QKModel{T,T2}, 
                t::Int)

In-place computation of the **Kalman gain** `K_ttm1[:, :, t]` for a **univariate** measurement 
scenario (i.e., `M=1`) within a Kalman or QKF step.

# Arguments
- `K_ttm1::AbstractArray{Real,3}`: A `(P×M×T̄)` array (often `(P×1×T̄)`) storing the Kalman gain.
  We write the result into `K_ttm1[:, :, t]`.
- `P_ttm1::AbstractArray{Real,3}`: One-step-ahead predicted covariance, `(P×P×T̄)`.
- `M_ttm1::AbstractArray{Real,3}`: Predicted measurement covariance, `(M×M×T̄)`. 
  For `M=1`, this is a `1×1×T̄`.
- `tmpPB::AbstractMatrix{T}`: A working buffer `(P×M)` if needed, but currently unused 
  in your snippet.
- `model::QKModel{T,T2}`: Must contain:
  - `B_aug::Matrix{T}` (size `M×P`),
  - `M, P::Int` for dimensions.
- `t::Int`: Time index.

# Details
We compute:
```julia
K_t[:, :, t] = (P_ttm1[:, :, t] * B_aug') / M_ttm1[:, :, t]
```
# Notes:
- If M_ttm1[:, :, t] is 1×1, the division is just a scalar divide. For genuine multivariate M>1, one would typically do a matrix inverse or solve. This snippet is thus best suited for univariate measurement.
- The in-place assignment .= ... is AD-friendly if your AD can handle assignment to slices in a 3D array. If needed, define or use a custom adjoint.
"""
function compute_K_t!(K_t::AbstractArray{Real,3}, P_ttm1::AbstractArray{Real,3},
    M_ttm1::AbstractArray{Real,3}, tmpPB::AbstractMatrix{T}, model::QKModel{T,T2},
    t::Int) where {T <: Real, T2 <: Real}

    @unpack B_aug = model.aug_state

    # For univariate M=1: (Pₜₜ₋₁ * B̃') is (P×1). Then we divide by Mₜₜ₋₁(1,1,t).
    # Directly do an elementwise assignment.
    K_t[:, :, t] .= (P_ttm1[:, :, t] * B_aug') ./ M_ttm1[:, :, t]
end

"""
    compute_K_t(P_ttm1::AbstractMatrix{T}, 
               M_ttm1::AbstractMatrix{T},
               model::QKModel{T,T2}, 
               t::Int) -> Matrix{T}

Compute and **return** the Kalman gain `K_ttm1` for a **univariate** measurement scenario, 
given `P_ttm1` and `M_ttm1`.

# Arguments
- `P_ttm1::AbstractMatrix{T}`: `(P×P)` one-step-ahead predicted covariance for the state.
- `M_ttm1::AbstractMatrix{T}`: `(M×M)` measurement covariance. 
- `model::QKModel{T,T2}`: Must contain:
  - `B_aug::Matrix{T}` (size `M×P`),
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
S = B_aug * P_ttm1 * B_aug' + M_ttm1
K = P_ttm1 * B_aug' / S
```
"""
function compute_K_t(P_ttm1::AbstractMatrix{T}, M_ttm1::AbstractMatrix{T},
    model::QKModel{T,T2}, t::Int ) where {T <: Real, T2 <: Real}
    
    @unpack B_aug = model.aug_state
    
    # 1) S is a 1×1 if M=1. Then S[1,1] is the scalar denominator.
    S = B_aug * P_ttm1 * B_aug' .+ M_ttm1
    
    # 2) The gain K is (P×M).
    return P_ttm1 * B_aug' ./ S
end

"""
    update_Z_tt!(Z_tt::AbstractMatrix{T}, 
                 K_t::AbstractArray{Real,3}, 
                 Y_t::AbstractMatrix{T1}, 
                 Y_ttm1::AbstractMatrix{T}, 
                 Z_ttm1::AbstractMatrix{T}, 
                 tmpϵ::AbstractVector{T}, 
                 t::Int)

In-place update of the augmented state `Z_tt[:, t+1]` given:
- A measurement vector `Y_t[:, t]` stored in a matrix `(M×T̄)`,
- A predicted measurement `Y_ttm1[:, t]` also in `(M×T̄)`,
- The one-step-ahead state `Z_ttm1[:, t]`,
- and a Kalman-like gain `K_t[:, :, t]` of size `(P×M)`.

# Arguments
- `Z_tt::AbstractMatrix{T}`: (P×T̄) storing the **updated** states in column `t+1`.
- `K_t::AbstractArray{Real,3}`: A `(P×M×T̄)` array of gains.
- `Y_t::AbstractMatrix{T1}`: Measurement data of size `(M×T̄)`.
- `Y_ttm1::AbstractMatrix{T}`: Predicted measurement `(M×T̄)`.
- `Z_ttm1::AbstractMatrix{T}`: One-step-ahead predicted state `(P×T̄)`.
- `tmp_eps::AbstractVector{T}`: A temporary buffer of length `M`, if needed for manual loops.
- `t::Int`: The time index (1-based).

# Details
We perform: Z_tt[:, t+1] = Z_tt[:, t] + K_t[:, :, t] * (Y_t[:, t] - Y_ttm1[:, t])
in place. `tmp_eps` can be used if you want a manual loop approach, but here we do 
the direct matrix expression. This is AD-friendly in many frameworks, though 
in-place writes can require custom adjoints depending on the AD library.

"""
function update_Z_tt!(Z_tt::AbstractMatrix{T}, K_t::AbstractArray{Real,3},
    Y_t::AbstractMatrix{T1}, Y_ttm1::AbstractMatrix{T}, Z_ttm1::AbstractMatrix{T}, 
    tmp_eps::AbstractVector{T}, t::Int) where {T1<:Real, T<:Real}

    Z_tt[:, t + 1] = Z_tt[:, t] .+ K_t[:, :, t] * (Y_t[:, t] .- Y_ttm1[:, t])
end

"""
    update_Z_tt(K_t::AbstractArray{Real,3}, 
                Y_t::AbstractMatrix{T1},
                Y_ttm1::AbstractMatrix{T1}, 
                Z_ttm1::AbstractMatrix{T1}, 
                t::Int)

Return a newly updated state vector for time `t+1` given matrix-based measurements.

# Arguments
- `K_t::AbstractArray{Real,3}`: (P×M×T̄) Kalman gains.
- `Y_t::AbstractMatrix{T1}`, `Y_ttm1::AbstractMatrix{T1}`: (M×T̄) measurement and predicted measurement.
- `Z_ttm1::AbstractMatrix{T1}`: (P×T̄) the predicted state.
- `t::Int`: Time index.

# Returns
- `Vector{T1}`: The updated state at time `t+1`, 
  computed as `Z_ttm1[:, t] + K_t[:, :, t]*(Y_t[:, t] - Y_ttm1[:, t])`.

# Details
This is a purely functional approach, allocating a new vector. 
It's often simpler for AD, as it avoids in-place modifications.

"""
function update_Z_tt(K_t::AbstractArray{T,3}, Y_t::AbstractMatrix{T1},
    Y_ttm1::AbstractMatrix{T1}, Z_ttm1::AbstractMatrix{T1}, 
    t::Int) where {T1<:Real, T<:Real}

    return Z_ttm1[:, t] .+ K_t[:, :, t] * (Y_t[:, t] .- Y_ttm1[:, t])
end


"""
    update_Z_tt!(Z_tt::AbstractMatrix{T}, 
                 K_t::AbstractArray{Real,3},
                 Y_t::AbstractVector{T1}, 
                 Y_ttm1::AbstractVector{T}, 
                 Z_ttm1::AbstractMatrix{T},
                 tmpϵ::AbstractVector{T}, 
                 t::Int)

In-place update of `Z_tt[:, t+1]` when the measurement `Y_t` and predicted measurement `Y_ttm1`
are stored as 1D arrays (univariate or effectively scalar per time).

# Arguments
- `Z_tt::AbstractMatrix{T}`: Updated states are stored in `Z_tt[:, t+1]`.
- `K_t::AbstractArray{Real,3}`: Gains `(P×M×T̄)`.
- `Y_t::AbstractVector{T1}`, `Y_ttm1::AbstractVector{T}`: 1D arrays storing values at index `t`.
- `Zₜₜ₋₁::AbstractMatrix{T}`: Predicted state `(P×T̄)`.
- `tmpϵ::AbstractVector{T}`: Temp buffer for difference if needed.

# Details
We do Z_tt[:, t+1] = Z_tt[:, t] + K_t[:, :, t]*(Y_t[t] - Y_ttm1[t])
in place. Typically `M=1` in this scenario.

"""
function update_Z_tt!(Z_tt::AbstractMatrix{T}, K_t::AbstractArray{Real,3},
    Y_t::AbstractVector{T1}, Y_ttm1::AbstractVector{T}, Z_ttm1::AbstractMatrix{T},
    tmp_eps::AbstractVector{T}, t::Int) where {T1<:Real, T<:Real}

    Z_tt[:, t + 1] = Z_tt[:, t] .+ K_t[:, :, t] * (Y_t[t] - Y_ttm1[t])
end


"""
    update_Z_tt(K_t::AbstractArray{Real,3},
                Y_t::AbstractVector{T1},
                Y_ttm1::AbstractVector{T},
                Z_ttm1::AbstractMatrix{T},
                t::Int)

Return a new updated state for time `t+1` in a univariate measurement scenario, 
where `Y_t` and `Y_ttm1` are 1D arrays.

# Arguments
- `K_t::AbstractArray{Real,3}`: (P×M×T̄) gains.
- `Y_t::AbstractVector{T1}`, `Y_ttm1::AbstractVector{T}`: 1D arrays with the observation at `Y_t[t]`
  and predicted at `Y_ttm1[t]`.
- `Zₜₜ₋₁::AbstractMatrix{T}`: (P×T̄) predicted states.

# Returns
A newly allocated vector `(P×1)` for time `t+1`, computed as: 
`Zₜₜ₋₁[:, t] + Kₜ[:, :, t]*(Yₜ[t] - Yₜₜ₋₁[t])`.
This is a purely functional approach.

"""
function update_Z_tt(K_t::AbstractArray{Real,3}, Y_t::AbstractVector{T1},
    Y_ttm1::AbstractVector{T}, Z_ttm1::AbstractMatrix{T}, t::Int) where {T1<:Real, T<:Real}

    return Z_ttm1[:, t] .+ K_t[:, :, t] * (Y_t[t] - Y_ttm1[t])
end

"""
    update_Z_tt!(Z_tt::AbstractMatrix{T}, 
                 K_t::AbstractArray{Real,3},
                 Y_t::AbstractMatrix{T1}, 
                 Y_ttm1::AbstractVector{T},
                 Z_ttm1::AbstractMatrix{T},
                 tmpϵ::AbstractVector{T},
                 t::Int)

In-place update of `Z_tt[:, t+1]` with a **mixed** scenario:
- `Y_t` is `(M×T̄)`,
- `Y_ttm1` is `(M×1)?` or a vector storing predicted measurement at index `t`.

# Arguments
- `Z_tt::AbstractMatrix{T}`, `Z_ttm1::AbstractMatrix{T}`: (P×T̄) augmented states.
- `K_t::AbstractArray{Real,3}`: (P×M×T̄) gains.
- `Y_t::AbstractMatrix{T1}`: (M×T̄) measurement.
- `Yₜₜ₋₁::AbstractVector{T}`: (M) predicted measurement at time `t`.
- `tmpϵ::AbstractVector{T}`: temp buffer if needed.

# Details
Update rule:
`Z_tt[:, t+1] = Z_tt[:, t] + K_t[:, :, t]*(Y_t[:, t] - Y_ttm1[t])`
Note that `Y_ttm1[t]` might be used if `Y_ttm1` has length `M`, 
and we broadcast or something similar. 
Keep track that dimensional consistency is correct for your usage.

"""
function update_Z_tt!(Z_tt::AbstractMatrix{T}, K_t::AbstractArray{Real,3}, 
    Y_t::AbstractMatrix{T1}, Y_ttm1::AbstractVector{T}, Z_ttm1::AbstractMatrix{T},
    tmp_eps::AbstractVector{T}, t::Int) where {T1<:Real, T<:Real}

    Z_tt[:, t + 1] = Z_ttm1[:, t] .+ K_t[:, :, t] * (Y_t[:, t] .- Y_ttm1[t])
end

"""
    update_Z_tt(K_t::AbstractArray{Real,3},
                Y_t::AbstractMatrix{T1},
                Y_ttm1::AbstractVector{T},
                Z_ttm1::AbstractMatrix{T},
                t::Int)

Return a new state update with measurement `Y_t` as (M×T̄), 
but `Y_ttm1` as an (M) vector indexing `t` for the predicted measurement. 
This is a somewhat "mixed" usage scenario.

# Arguments
- `K_t::AbstractArray{Real,3}`: Gains `(P×M×T̄)`.
- `Y_t::AbstractMatrix{T1}`: `(M×T̄)` measurements.
- `Y_ttm1::AbstractVector{T}`: size `M` predicted measurement, used at index `t`.
- `Zₜₜ₋₁::AbstractMatrix{T}`: `(P×T̄)` predicted states.

# Returns
A `(P×1)` updated state vector for time `t+1`, 
via `Z_ttm1[:, t] + K_t[:, :, t] * (Y_t[:, t] - Y_ttm1[t])`.

"""
function update_Z_tt(K_t::AbstractArray{Real,3}, Y_t::AbstractMatrix{T1},
    Y_ttm1::AbstractVector{T}, Z_ttm1::AbstractMatrix{T}, t::Int) where {T1<:Real, T<:Real}

    return Z_ttm1[:, t] .+ K_t[:, :, t] * (Y_t[:, t] .- Y_ttm1[t])
end


"""
    update_Z_tt!(Z_tt::AbstractMatrix{T}, 
                 K_t::AbstractArray{Real,3},
                 Y_t::AbstractVector{T1}, 
                 Y_ttm1::AbstractMatrix{T},
                 Z_ttm1::AbstractMatrix{T},
                 tmp_eps::AbstractVector{T}, 
                 t::Int)

In-place update of the state with a "mixed" scenario:
- `Y_t` is 1D (size `M×1`?), 
- `Y_ttm1` is `(M×T̄)`?

# Arguments
- `Z_tt::AbstractMatrix{T}`, `Z_ttm1::AbstractMatrix{T}`: State arrays `(P×T̄)`.
- `K_t::AbstractArray{Real,3}`: Gains `(P×M×T̄)`.
- `Y_t::AbstractVector{T1}`, `Y_ttm1::AbstractMatrix{T}`: 
  The actual measurement at `Yₜ[t]`, predicted measurement at `Yₜₜ₋₁[:, t]`.
- `tmp_eps::AbstractVector{T}`: buffer if needed.

# Implementation
`Z_tt[:, t+1] = Z_ttm1[:, t] + K_t[:, :, t]*(Y_t[t] - Y_ttm1[:, t])`
Dimension checks are up to the user; ensure consistent shapes for broadcast.

"""
function update_Z_tt!(Z_tt::AbstractMatrix{T}, K_t::AbstractArray{Real,3},
    Y_t::AbstractVector{T1}, Y_ttm1::AbstractMatrix{T}, Z_ttm1::AbstractMatrix{T},
    tmp_eps::AbstractVector{T}, t::Int) where {T1<:Real, T<:Real}

    Z_tt[:, t + 1] = Z_ttm1[:, t] .+ K_t[:, :, t] * (Y_t[t] - Y_ttm1[:, t])
end

"""
    update_Z_tt(K_t::AbstractMatrix{T}, 
                Y_t::Real, 
                Y_ttm1::Real, 
                Z_ttm1::AbstractVector{T}, 
                t::Int)

Return a new updated state when:
- `K_t` is `(P×1)`,
- `Y_t` and `Y_ttm1` are scalar reals,
- `Z_ttm1` is `(P×1)` or `Vector{T}`.

# Arguments
- `K_t::AbstractMatrix{T}`: The gain (P×1).
- `Y_t::Real`: The scalar measurement at time `t`.
- `Y_ttm1::Real`: The predicted scalar measurement.
- `Z_ttm1::AbstractVector{T}`: The predicted state (P).

# Returns
- `Vector{T}`: The new updated state: `Zₜₜ₋₁ + Kₜ*(Yₜ - Yₜₜ₋₁)`.

# Description
This is the simplest possible scenario: 
**univariate** measurement (`Y_t` is a real) 
and a vector state `(P×1)`. We do a purely functional 
update returning a brand-new `Vector{T}`.

"""
function update_Z_tt(K_t::AbstractMatrix{T}, Y_t::Real, Y_ttm1::Real, Z_ttm1::AbstractVector{T}, t::Int) where {T <: Real}

    return Z_ttm1 .+ K_t * (Y_t - Y_ttm1)
end

"""
    update_P_tt!(P_tt, K_t, M_ttm1, P_ttm1, Z_ttm1, tmpKM, tmpKMK, model, t)

In-place update of the **filtered covariance** `P_tt[:, :, t+1]` given the one-step-ahead 
covariance `P_ttm1[:, :, t]`, and the Kalman gain `K_t[:, :, t]`.

# Arguments
- `P_tt::AbstractArray{<:Real, 3}`: A 3D array `(P×P×T̄)` storing filtered covariances.  
  We write the new filtered covariance into `P_tt[:, :, t+1]`.
- `K_t::AbstractArray{<:Real, 3}`: A 3D array `(P×M×T̄)` of Kalman gains, 
  using `K_t[:, :, t]`.
- `M_ttm1::AbstractArray{<:Real, 3}`: *(Currently unused)* or was presumably
  the predicted measurement covariance `(M×M×T̄)`. Not directly used in the final expression.
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
function update_P_tt!( P_tt::AbstractArray{Real,3}, K_t::AbstractArray{Real,3},
    M_ttm1::AbstractArray{Real,3}, P_ttm1::AbstractArray{Real,3},
    Z_ttm1::AbstractArray{Real,2}, tmpKM::AbstractMatrix{Real},
    tmpKMK::AbstractMatrix{Real}, model::QKModel{T,T2}, t::Int ) where {T<:Real, T2<:Real}
    @unpack V = model.meas
    @unpack B_aug = model.aug_state
    
    # 1) Form A = I - K_t[:, :, t]*B_aug
    local A = I - K_t[:, :, t]*B_aug
    
    # 2) In-place assign the final updated covariance
    P_tt[:, :, t+1] = make_positive_definite(
        A * P_ttm1[:, :, t] * A' .+ (K_t[:, :, t] * V * K_t[:, :, t]')
    )
end

"""
    update_P_tt(K_t, M_ttm1, P_ttm1, Z_ttm1, model, t)

Compute and return a **new** filtered covariance matrix, 
given a predicted covariance `P_ttm1`, a Kalman gain `K_t`.

# Arguments
- `K_t::AbstractMatrix{T5}`: (P×M) or (P×1) Kalman gain for time `t`.
- `M_ttm1::AbstractMatrix{T6}`: *(Potentially unused here or if you only 
  rely on `V_tmp`?),*
- `P_ttm1::AbstractMatrix{T3}`: The one-step-ahead covariance `(P×P)`.
- `Z_ttm1::AbstractVector{T4}`: The predicted augmented state (length P).
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
function update_P_tt( K_t::AbstractMatrix{T5}, M_ttm1::AbstractMatrix{T6},
    P_ttm1::AbstractMatrix{T3}, Z_ttm1::AbstractVector{T4}, model::QKModel{T,T2},
    t::Int ) where {T<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real}
    @unpack V = model.meas
    @unpack B_aug = model.aug_state


    # 1) A = I - K_t*B_aug
    local A = I - K_t*B_aug

    # 2) Build the new covariance
    local P = A * P_ttm1 * A' .+ (K_t * V * K_t')

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
    
    # 1) Extract relevant piece
    Z_tt = Z_tt[:, t + 1] # Z_tt[:, t+1] is the "current" column in the filter
    
    # 2) Extract x_t and implied (XX')ₜ
    xt = Z_tt[1:N]                              # the first N entries are xₜ|ₜ
    XtXt_prime = reshape(Z_tt[N+1:end], N, N)    # the next N*N entries are XX' in vectorized form
    
    # 3) implied_cov = (XX') - xₜ xₜ'
    implied_cov = XtXt_prime - xt*xt'
    
    # 4) Eigen-decomposition to clamp negative eigenvalues
    #    Use `Symmetric` or `Hermitian` to ensure real sym decomposition
    eig_vals, eig_vecs = eigen(Symmetric(implied_cov))
    
    # 5) Replace negative eigenvalues with 0 => PSD
    eig_vals .= max.(eig_vals, 0.0)
    
    # 6) Reassemble the corrected block = V * diag(λ⁺) * V' + xₜ xₜ'
    corrected = eig_vecs * Diagonal(eig_vals) * eig_vecs' + x_t*x_t'
    
    # 7) Write it back into the corresponding part of Z_tt
    Z_tt[N+1:end, t + 1] = vec(corrected)
    
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
    qkf_filter!(data::QKData{T1,1}, model::QKModel{T,T2})

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
  - `T_bar::Int` the total number of time steps (excluding index 0).
  - `M::Int` the dimension of the measurement at each time step.

- `model::QKModel{T,T2}`  
  A parameter structure holding:
  - `N::Int`: State dimension (for the mean part).
  - `P::Int`: Dimension of the augmented "quadratic" state 
    (`P = N + N(N+1)/2`).
  - `μ_aug, Σ_aug`: The unconditional mean and covariance used for 
    initialization.
  - Additional model matrices or functions (e.g., `Φ_aug`, `B_aug`, `A`, `V`) 
    accessed via subroutines.

# Return

A named tuple with fields:

- `ll_t::Vector{Float64}`  
  The per-time-step log-likelihoods (size = `T̄`).
- `Z_tt::Array{T,3}`  
  The filtered state at each time step. Dimensions: `(T, P, T̄+1)` in your 
  specific code (or `(P, T̄+1)` in a more generic version).
- `Pₜₜ::Array{T,4}`  
  The filtered state covariance array. Dimensions often `(T, P, P, T̄+1)` 
  in your code.
- `Yₜₜ₋₁::Vector{T}`  
  The predicted measurement at each time step (size = `T̄`).
- `M_ttm1::Array{T,4}`  
  The predicted measurement covariance, dimensions `(T, M, M, T̄)`.
- `K_t::Array{T,4}`  
  The Kalman gain for each time step, `(T, P, M, T̄)`.
- `Zₜₜ₋₁::Array{T,3}`  
  One-step-ahead predicted states.
- `Pₜₜ₋₁::Array{T,4}`  
  One-step-ahead predicted covariances.
- `Σ_ttm1::Array{T,4}`  
  Any intermediate covariance terms used for prediction.

# Details

1. **Initialization**: 
   - `Zₜₜ[:, 1] .= μ̃ᵘ` and `Pₜₜ[:,:,1] .= Σ̃ᵘ`.
2. **Recursive Steps** (`for t in 1:T̄`):
   - **Prediction**: `predict_Zₜₜ₋₁!` / `predict_Pₜₜ₋₁!`.
   - **Measurement**: `predict_Yₜₜ₋₁!` / `predict_Mₜₜ₋₁!`.
   - **Gain & Update**: `compute_Kₜ!`, then `update_Zₜₜ!` / `update_Pₜₜ!`.
   - **Correction**: `correct_Z_tt!` clamps negative eigenvalues 
     for PSD.
   - **Likelihood**: `compute_loglik!` appends the log-likelihood.
3. **Positive Semidefinite Correction**: 
   - Negative eigenvalues introduced by approximation are set to zero.

# Example

```julia
data = QKData(Y, M=measurement_dim, T̄=length(Y)-1)
model = QKModel(...)
result = qkf_filter!(data, model)

@show result.ll_t
@show result.Z_tt[:, end]   # final state
```
"""
function qkf_filter!(data::QKData{T1, 1},
    model::QKModel{T,T2}) where {T1 <: Real, T <: Real, T2 <: Real}

    @unpack T_bar, Y, M = data
    @unpack N = model.state
    @unpack aug_mean, aug_cov = model.moments
    @unpack P = model.aug_state

    # Predfine Matrix
    Z_tt =  zeros(T, P, T_bar + 1)
    Z_ttm1 = zeros(T, P, T_bar)
    P_tt = zeros(T, P, P, T_bar + 1)
    P_ttm1 = zeros(T, P, P, T_bar)
    Sigma_ttm1 = zeros(T, P, P, T_bar)
    #vecΣ_ttm1 = zeros(T, P^2, T_bar)
    K_t = zeros(T, P, M, T_bar)
    tmpP = zeros(T, P, P)
    tmpB = zeros(T, M, P)
    Y_ttm1 = zeros(T, T_bar)
    M_ttm1 = zeros(T, M, M, T_bar)
    tmpPB = zeros(T, P, M)
    ll_t = zeros(Float64, T_bar)
    tmp_eps = zeros(T, M)
    tmpKM = zeros(T, P, M)
    tmpKMK = zeros(T, P, P)

    Y_t = Vector{T}(undef, T_bar)
    copyto!(Y_t, 1, Y, 2, T_bar)

    #Initalize: Z₀₀ = μ̃ᵘ, P₀₀ = Σ̃ᵘ
    Z_tt[:, 1] .= aug_mean
    P_tt[:, :, 1] .= aug_cov
    
    # Loop over time
    for t in 1:T_bar

        # State Prediction: Zₜₜ₋₁ = μ̃ + Φ̃Zₜ₋₁ₜ₋₁, Pₜₜ₋₁ = Φ̃Pₜ₋₁ₜ₋₁Φ̃' + Σ̃(Zₜ₋₁ₜ₋₁)
        predict_Z_tt!(Z_tt, Z_ttm1, model, t)
        predict_P_tt!(P_tt, P_ttm1, Sigma_ttm1, Z_tt, tmpP, model, t)

        # Observation Prediction: Yₜₜ₋₁ = A + B̃Zₜₜ₋₁, Mₜₜ₋₁ = B̃Pₜₜ₋₁B̃' + V
        predict_Y_tt!(Y_ttm1, Z_ttm1, Y, model, t)
        predict_M_ttm1!(M_ttm1, P_ttm1, tmpB, model, t)

        # Kalman Gain: Kₜ = Pₜₜ₋₁B̃′/Mₜₜ₋₁
        compute_K_t!(K_t, P_ttm1, M_ttm1, tmpPB, model, t)

        # Update States: Zₜₜ = Zₜₜ₋₁ + Kₜ(Yₜ - Yₜₜ₋₁); Pₜₜ = Pₜₜ₋₁ - KₜMₜₜ₋₁Kₜ'
        update_Z_tt!(Z_tt, K_t, Y_t, Y_ttm1, Z_ttm1, tmp_eps, model, t)
        update_P_tt!(P_tt, K_t, M_ttm1, P_ttm1, Z_ttm1, tmpKM, tmpKMK, model, t)

        #Correct for update
        correct_Z_tt!(Z_tt, model, t)

        #Compute Log Likelihood
        compute_loglik!(ll_t, Y_t, Y_ttm1, M_ttm1, t)
    end

    return FilterOutput(ll_t = ll_t, Z_tt = Z_tt, P_tt = P_tt,  Y_ttm1 = Y_ttm1, M_ttm1 = M_ttm1,
        K_t = K_t, Z_ttm1 = Z_ttm1, P_ttm1 = P_ttm1, Sigma_ttm1 = Sigma_ttm1)

end

"""
    qkf_filter(data::QKData{T1,1}, model::QKModel{T,T2})

Run the **Quadratic Kalman Filter (QKF)** on a time series of length `T̄`, 
returning a new set of result arrays (out-of-place).

# Description

This function implements the same *quadratic* Kalman filter recursion 
as `qkf_filter!`, but instead of updating arrays in-place, it allocates 
new arrays for predictions, updates, and outputs. This can be simpler to 
use in contexts where you don't want to mutate or reuse `data` and `model`, 
but it may be less memory-efficient for large-scale problems.

At each time step, it performs:

1. **State Prediction** (`predict_Z_tt` / `predict_P_tt`)
2. **Measurement Prediction** (`predict_Y_tt` / `predict_M_ttm1`)
3. **Kalman Gain Computation** (`compute_K_t`)
4. **State & Covariance Update** (`update_Z_tt`, `update_P_tt`)
5. **PSD Correction** (`correct_Zₜₜ`)
6. **Log-Likelihood** computation.

# Arguments

- `data::QKData{T1,1}`  
  Same structure as in `qkf_filter!`, with fields:
  - `Y::Vector{T1}`, `T̄::Int`, `M::Int`.
- `model::QKModel{T,T2}`  
  Same parameter structure as in `qkf_filter!`, with fields:
  - `N::Int`, `P::Int`, `μ̃ᵘ, Σ̃ᵘ`, etc.

# Return

A named tuple with fields:

- `ll_t::Vector{Float64}`  
  Per-time-step log-likelihoods (size = `T̄`).
- `Z_tt::Array{T,3}`  
  The filtered state at each time step (dimensions `(T, P, T̄+1)` in your usage).
- `P_tt::Array{T,4}`  
  The filtered state covariance array.
- `Y_ttm1::Vector{T}`  
  Predicted measurement for each step.
- `M_ttm1::Array{T,4}`  
  Predicted measurement covariances.
- `K_t::Array{T,4}`  
  The Kalman gain for each time step.
- `Z_ttm1::Array{T,3}`  
  One-step-ahead predicted states.
- `P_ttm1::Array{T,4}`  
  One-step-ahead predicted covariances.

# Details

1. **Initialization**:
   - Creates new arrays for `Z_tt` and `P_tt` and sets the initial state 
     to `aug_mean` and `aug_cov`.
2. **Time Loop**: 
   - **Prediction**: `predict_Z_tt`, `predict_P_tt`.
   - **Measurement**: `predict_Y_tt`, `predict_M_ttm1`.
   - **Gain & Update**: `compute_K_t`, `update_Z_tt`, `update_P_tt`.
   - **Correction**: `correct_Z_tt` for PSD.
   - **Likelihood**: `compute_loglik`.
3. **No In-Place Mutation**:
   - Each step returns fresh arrays; original inputs are not modified.

# Example

```julia
data = QKData(Y, M=measurement_dim, T̄=length(Y)-1)
model = QKModel(...)
result = qkf_filter(data, model)

@show result.ll_t
@show result.Z_tt[:, end]   # final state
'''
"""
function qkf_filter(data::QKData{T1, 1},
    model::QKModel{T,T2}) where {T1 <: Real, T <: Real, T2 <: Real}
    @unpack T_bar, Y, M = data
    @unpack N = model.state
    @unpack aug_mean, aug_cov = model.moments
    @unpack P = model.aug_state
    Y_concrete = Vector{T1}(vec(Y))  # Convert to vector if it's not already
    Y_t = @view Y_concrete[2:end]

    Z_tt = zeros(T, P, T_bar + 1)
    P_tt = zeros(T, P, P, T_bar + 1)
    Z_ttm1 = zeros(T, P, T_bar)
    P_ttm1 = zeros(T, P, P, T_bar)
    K_t = zeros(T, P, M, T_bar)
    Y_ttm1 = zeros(T, T_bar)
    M_ttm1 = zeros(T, M, M, T_bar)
    ll_t = zeros(T, T_bar)

    # Initialize
    Z_tt[:, 1] = aug_mean
    P_tt[:, :, 1] = aug_cov

    for t in 1:T_bar
        Z_ttm1[:, t] = predict_Z_ttm1(Z_tt[:, t], model)
        P_ttm1[:, :, t] = predict_P_ttm1(P_tt[:, :, t], Z_tt[:, t], model, t)

        Y_ttm1[t] = predict_Y_ttm1(Z_ttm1[:, t], Y, model, t)
        M_ttm1[:, :, t] = predict_M_ttm1(P_ttm1[:, :, t], model)

        K_t[:, :, t] = compute_K_t(P_ttm1[:, :, t], M_ttm1[:, :, t], model, t)

        Z_tt[:, t + 1] = update_Z_tt(K_t[:, :, t], Y_t[t], Y_ttm1[t], Z_ttm1[:, t], t)
        P_tt[:, :, t + 1] = update_P_tt(K_t[:, :, t], M_ttm1[:, :, t], P_ttm1[:, :, t], Z_ttm1[:, t], model, t)

        Z_tt[:, t + 1] = correct_Z_tt(Z_tt[:, t + 1], model, t)

        ll_t[t] = compute_loglik(Y_t[t], Y_ttm1[t], M_ttm1[:, :, t])
    end

    return FilterOutput(ll_t, Z_tt, P_tt, Y_ttm1, M_ttm1, K_t, Z_ttm1, P_ttm1)
end


"""
    qkf_filter!(data::QKData{T1,1}, model::QKModel{T,T2})

Run the **Quadratic Kalman Filter (QKF)** on a time series of length `T̄`.

# Description

This function implements a Kalman-like recursive filter where the state 
vector `Z_t` includes not only the usual mean component `xₜ` but also 
terms for the second-moment `(x xᵀ)ₜ`, making it a *quadratic* extension. 
At each time step, it performs:

1. **State Prediction** (`predict_Z_ttm1` / `predict_P_ttm1`)
2. **Measurement Prediction** (`predict_Y_ttm1` / `predict_M_ttm1`)
3. **Kalman Gain Computation** (`compute_K_t`)
4. **State & Covariance Update** (`update_Zₜₜ!`, `update_Pₜₜ!`)
5. **PSD Correction**: Ensures the implied covariance is positive semidefinite 
   by clamping negative eigenvalues via `correct_Zₜₜ!`.
6. **Log-Likelihood** computation for the current innovation.

The filter stores and returns the entire history of filtered states, covariances,
predicted measurements, and related arrays. 

# Arguments

- `data::QKData{T1,1}`  
  A structure holding:
  - `Y::Vector{T1}` of length `T_bar+1`, which contains observations 
    (`Y[1]` is unused or some initial placeholder, and `Y[2..end]` are the actual measurements).
  - `T_bar::Int` the total number of time steps (excluding index 0).
  - `M::Int` the dimension of the measurement at each time step.

- `model::QKModel{T,T2}`  
  A parameter structure holding:
  - `N::Int`: State dimension (for the mean part).
  - `P::Int`: Dimension of the augmented "quadratic" state vector 
    (`P = N + N(N+1)/2`).
  - `aug_mean, aug_cov`: The unconditional mean and covariance used for initialization.
  - Additional model matrices or functions (e.g., `Phi_aug`, `B_aug`, `A`, `V`), 
    typically accessed via separate predict/update subroutines.

# Return

A named tuple with fields:

- `ll_t::Vector{Float64}`  
  The per-time-step log-likelihoods of the innovations (size = `T_bar`).

- `Z_tt::Matrix{T}`  
  The updated ("filtered") state at each time step. Dimensions: `(P, T_bar+1)`, 
  where column `k` corresponds to time index `k-1`.

- `P_tt::Array{T,3}`  
  The updated ("filtered") state covariance array (or the augmented second-moment 
  representation) at each time step. Dimensions: `(P, P, T_bar+1)`.

- `Y_ttm1::Vector{T}`  
  The predicted measurement at each time step (size = `T_bar`).

- `M_ttm1::Array{T,3}`  
  The predicted measurement covariance at each time step (dimensions: `(M, M, T_bar)`).

- `K_t::Array{T,3}`  
  The Kalman gain at each time step `(P, M, T_bar)`.

- `Z_ttm1::Matrix{T}`  
  The one-step-ahead predicted state (dimensions: `(P, T_bar)`).

- `P_ttm1::Array{T,4}`  
  The one-step-ahead predicted covariance (dimensions: `(P, P, T_bar)`).

- `Σ_ttm1::Array{T,4}`  
  Any intermediate state-dependent covariance terms added during prediction 
  (dimensions: `(P, P, T_bar)`).

# Details

1. **Initialization**:  
   - `Z_tt[:,1] .= aug_mean` and `P_tt[:,:,1] .= aug_cov`, representing the state at time 0.

2. **Time Loop** (`for t in 1:T_bar`):  
   - **Predict** step: 
     - `predict_Z_ttm1` sets `Z_ttm1[:,t]` from `Z_tt[:,t]`.
     - `predict_P_ttm1` sets `P_ttm1[:,:,t]` from `P_tt[:,:,t]`.
   - **Measurement** step: 
     - `predict_Y_ttm1` computes `Y_ttm1[t]` from `Z_ttm1[:,t]`.
     - `predict_M_ttm1` updates `M_ttm1[:,:,t]`.
   - **Gain & Update**:
     - `compute_K_t` obtains the Kalman gain `K_t[:,:,t]`.
     - `update_Z_tt` and `update_P_tt` yield the next `Z_tt[:,t+1]` and `P_tt[:,:,t+1]`.
   - **Correction**:  
     - `correct_Z_tt` clamps negative eigenvalues in the implied covariance portion 
       of `Z_tt` to ensure PSD. 
   - **Likelihood**:
     - `compute_loglik` appends/logs the innovation-based log-likelihood 
       in `ll_t[t]`.

3. **Positive Semidefinite Correction**:
   - Because the filter tracks `[X Xᵀ] - X Xᵀᵀ` as a covariance block, it can 
     become indefinite due to linearization or numerical issues. We enforce 
     PSD by zeroing out negative eigenvalues in `correct_Zₜₜ!`.
   - This step is not strictly differentiable at eigenvalues crossing zero. 
     If you need AD through the filter, consider an alternative correction 
     or a custom adjoint.

# Example

```julia
data = QKData(Y, M=measurement_dim, T_bar=length(Y)-1)
model = QKModel( ... )   # set up your model

result = qkf_filter!(data, model)

@show result.ll_t
@show result.Z_tt[:, end]   # final state vector
"""
function qkf_filter!(data::QKData{T, 2}, model::QKModel{T,T2}) where {T <: Real, T2 <: Real}

    @unpack T_bar, Y, M = data
    @unpack N = model.state
    @unpack P = model.aug_state
    @unpack aug_mean, aug_cov = model.moments

    # Predfine Matrix
    Z_tt = Matrix{<:Real}(undef, P, T_bar + 1)
    P_tt = Array{T, 3}(undef, P, P, T_bar + 1)
    Z_ttm1 = Matrix{<:Real}(undef, P, T_bar)
    P_ttm1 = Array{T, 3}(undef, P, P, T_bar)
    K_t = Array{T, 3}(undef, P, M, T_bar)
    Y_ttm1 = Vector{<:Real}(undef, T_bar)
    M_ttm1 = Array{T, 3}(undef, M, M, T_bar)
    ll_t = Vector{<:Real}(undef, T_bar)

    #Initalize: Z₀₀ = μ̃ᵘ, P₀₀ = Σ̃ᵘ
    Z_tt[:, 1] = aug_mean
    P_tt[:, :, 1] = aug_cov
    
    # Loop over time
    for t in 1:T_bar
        Z_ttm1[:, t] = predict_Z_ttm1(Z_tt[:, t], model)
        P_ttm1[:, :, t] = predict_P_ttm1(P_tt[:, :, t], Z_tt[:, t], model, t)

        Y_ttm1[t] = predict_Y_ttm1(Z_ttm1[:, t], Y, model, t)
        M_ttm1[:, :, t] = predict_M_ttm1(P_ttm1[:, :, t], model)

        K_t[:, :, t] = compute_K_t(P_ttm1[:, :, t], M_ttm1[:, :, t], model, t)

        Z_tt[:, t + 1] = update_Z_tt(K_t[:, :, t], Y[t], Y_ttm1[t], Z_ttm1[:, t], model, t)
        P_tt[:, :, t + 1] = update_P_tt(K_t[:, :, t], M_ttm1[:, :, t], P_ttm1[:, :, t], Z_ttm1[:, t], model, t)

        Z_tt[:, t + 1] = correct_Z_tt(Z_tt[:, t + 1], model, t)

        ll_t[t] = compute_loglik(Y[t], Y_ttm1[t], M_ttm1[:, :, t])
    end

    return FilterOutput(ll_t = ll_t, Z_tt = Z_tt, P_tt = P_tt,  Y_ttm1 = Y_ttm1, M_ttm1 = M_ttm1,
        K_t = K_t, Z_ttm1 = Z_ttm1, P_ttm1 = P_ttm1)

end


export qkf_filter, qkf_filter!