
"""
    qkf_smoother!(
        Z::AbstractMatrix{T},      # Filtered states   (P × (T_bar+1))
        P::AbstractArray{T, 3},    # Filtered covariances (P × P × (T_bar+1))
        Z_pred::AbstractMatrix{T}, # One-step-ahead predicted states (P × T_bar)
        P_pred::AbstractArray{T,3},
        T_bar::Int,
        Hn::Matrix{T},  Gn::Matrix{T},  H_aug::Matrix{T},  Φ_aug::Matrix{T},
        dof::Int
    ) where {T<:Real}

Perform **in-place** backward smoothing for the Quadratic Kalman Filter (QKF).

# Description

Given the forward-filtered estimates `(Z, P)` from `t=1..T_bar`, plus the 
one-step-ahead predictions `(Z_pred, P_pred)` and the special matrices 
(H_aug, G_aug) that handle the dimension reduction via 
Vech(·)/Vec(·), this function computes `Z[:,t]` and `P[:,:,t]` for 
`t = T_bar-1 .. 1` in backward fashion to produce the smoothed estimates 
(Zₜ|T_bar, PZₜ|T_bar).  

## Mathematical Form (Backward Pass)

1. Compute  
   Fₜ = (H̃ₙPᵗ|ᵗᶻH̃ₙ')(H̃ₙΦ̃G̃ₙ)'(H̃ₙPᵗ⁺¹|ᵗᶻH̃ₙ')⁻¹
   but implemented via solves (rather than explicit inverses) for numerical stability.

2. Then update (in H̃ₙ-transformed space):
   H̃ₙZₜ|ₜ = H̃ₙZₜ|ₜ + Fₜ(H̃ₙZₜ₊₁|ₜ - H̃ₙZₜ₊₁|ₜ)

3. And similarly for the covariance:
   (H̃ₙPᵗ|ᵀᶻH̃ₙ') = (H̃ₙPᵗ|ᵗᶻH̃ₙ') + Fₜ[(H̃ₙPᵗ⁺¹|ᵀᶻH̃ₙ') - (H̃ₙPᵗ⁺¹|ᵗᶻH̃ₙ')]Fₜ'

4. Finally, transform back to get `Zₜ|T` and `Pᵗ|ᵀᶻ` in the full
   (Vec/augmented) space if necessary.

# Arguments

- `Z::AbstractMatrix{T}`: On entry, `Z[:,t]` = `Zₜ|T` for each `t`.
  On exit, it will contain the smoothed states `Zₜ|T`.
- `P::AbstractArray{T,3}`: On entry, `P[:,:,t]` = `Pᵗ|ᵀᶻ`. 
  On exit, `P[:,:,t]` = `Pᵗ|ᵀᶻ`.
- `Z_pred::AbstractMatrix{T}`, `P_pred::AbstractArray{T,3}`: 
  The one-step-ahead predicted states/covariances from the forward pass,
  i.e. `Z_pred[:,t] = Zₜ|ₜ`, `P_pred[:,:,t] = P^Zₜ|ₜ`.
- `T_bar::Int`: Total time steps (excluding time 0).
- `Hn::Matrix{T}`, `Gn::Matrix{T}`: The selection/duplication operators 
  for Vec/Vech transforms of block `(x xᵀ)`. Usually size `(n(n+1) × n(n+3)/2)` or similarly.
- `H_aug::Matrix{T}`, `Φ_aug::Matrix{T}`: The augmented versions 
  (H_aug, G_aug) used in the QKF recursion.
- `dof::Int`: Dimension parameter (often `n` or `P`). Adjust to your model.

# Notes

- This function runs backward from `t = T_bar-1` down to `t = 1`, using 
  the final values `(Z[:, T_bar], P[:,:, T_bar])` as the terminal condition 
  (`Z_{T_bar|T_bar}, P^Z_{T_bar|T_bar}`).
- If your AD library supports destructive updates, this approach should 
  be AD-friendly; if not, consider the out-of-place version `qkf_smoother`.

# Example
Suppose you already ran the forward filter, so you have:
    Z, P, Z_pred, P_pred, plus your special matrices.
```
qkf_smoother!(Z, P, Z_pred, P_pred, T_bar, Hn, Gn, H_aug, Φ_aug, n)
```
"""
function qkf_smoother!(Z::AbstractMatrix{T}, P::AbstractArray{T,3},
  Z_pred::AbstractMatrix{T}, P_pred::AbstractArray{T,3}, T_bar::Int,
  Hn_aug::AbstractMatrix{T}, Gn_aug::AbstractMatrix{T},
  Phi_aug::AbstractMatrix{T}) where {T<:Real}

  for t in (T_bar-1):-1:1
      # 1) Compute reduced covariances
      M_t = Hn_aug * P[:, :, t] * Hn_aug'
      M_tp1 = Hn_aug * P_pred[:, :, t] * Hn_aug'

      # 2) Extract reduced states
      hZ_t = Hn_aug * Z[:, t]
      hZ_tp1 = Hn_aug * Z[:, t+1]  # Smoothed t+1 (already updated)
      hZ_pred = Hn_aug * Z_pred[:, t]

      # 3) Compute cross term and F_t (correct order)
      cross = (Hn_aug * Phi_aug * Gn_aug)'
      tmp = M_t * cross
      factor_tp1 = cholesky(Symmetric(M_tp1))
      F_t = (factor_tp1 \ tmp')'  # Equivalent to tmp * inv(M_tp1)

      # 4) Update smoothed state in reduced space
      diff = hZ_tp1 .- hZ_pred
      hZ_t_smooth = hZ_t .+ F_t * diff

      # 5) Project delta back to full state and update Z
      delta_hZ = hZ_t_smooth - hZ_t
      mul!(view(Z, :, t), Hn_aug', delta_hZ, 1.0, 1.0)

      # 6) Update covariance in reduced space
      M_tp1_smooth = Hn_aug * P[:, :, t+1] * Hn_aug'
      dM = M_tp1_smooth - M_tp1
      mid = F_t * dM
      new_M_t = M_t + mid * F_t'

      # 7) Project delta covariance back to full space and update P
      delta_M = new_M_t - M_t
      mul!(view(P, :, :, t), Hn_aug', delta_M * Hn_aug, 1.0, 1.0)
  end

  return nothing
end


"""
    qkf_smoother( Z, P, Z_pred, P_pred, T_bar, H_aug, G_aug, Φ_aug, dof ) -> (Z_smooth, P_smooth)

Out-of-place version of the QKF smoother. Returns new arrays rather than overwriting the input ones.
    
# Description
Identical to qkf_smoother! in logic, but it allocates new arrays Z_smooth and P_smooth for
the smoothed results. This is often simpler for AD frameworks that do not allow in-place
mutation of arrays.

# Returns
- `Z_smooth::Matrix{T}`: (P × (T_bar+1)) smoothed states
- `P_smooth::Array{T,3}`: (P × P × (T_bar+1)) smoothed covariances


# Example
```julia
Z_smooth, P_smooth = qkf_smoother(Z, P, Z_pred, P_pred, T_bar, Hn, Gn, H_aug, Φ_aug, n)
```
""" 
function qkf_smoother(Z::AbstractMatrix{T}, P::AbstractArray{T,3},
    Z_pred::AbstractMatrix{T}, P_pred::AbstractArray{T,3}, T_bar::Int,
    H_aug::AbstractMatrix{T}, G_aug::AbstractMatrix{T}, 
    Phi_aug::AbstractMatrix{T}, dof::Int) where {T<:Real}


    # Make copies (or a deep copy if needed)
    Z_smooth = copy(Z)
    P_smooth = similar(P)  # same dims, uninitialized
    P_smooth .= P  # replicate the filtered covariances

    # Just call the in-place version on the copies:
    qkf_smoother!(Z_smooth, P_smooth, Z_pred, P_pred, T_bar, G_aug, H_aug, Phi_aug)


    return (copy(Z_smooth), copy(P_smooth))

end

"""
    qkf_smoother!(filter_output::FilterOutput{T}, model::QKModel{T,T2}) where {T<:Real, T2<:Real}

Performs in-place backward smoothing for a Quadratic Kalman Filter (QKF) using the outputs obtained during filtering.

This function refines the filtered state estimates and covariance matrices by incorporating future observations,
thereby producing a set of smoothed estimates. It operates by first extracting the filtered states (Z_tt) and their
associated covariances (P_tt), as well as the one-step-ahead predictions (Z_ttm1 and P_ttm1) from the given
FilterOutput structure. It then unpacks the augmented state parameters (H_aug, G_aug, Phi_aug) along with the
state dimension (N) from the QKModel. The function makes local copies of the filtered estimates to avoid any
modification of the original filtering results, and then calls the lower-level in-place smoothing routine to
update these copies using the one-step-ahead predictions and the model parameters. The smoothed states and
covariances are finally encapsulated into a SmootherOutput structure, which is returned.

Parameters:
  - filter_output: A FilterOutput instance containing:
      • Z_tt    :: Matrix of filtered augmented state estimates for time steps 0 to T̄.
      • P_tt    :: Array of filtered state covariance matrices.
      • Z_ttm1  :: Matrix of one-step-ahead predicted augmented states.
      • P_ttm1  :: Array of one-step-ahead predicted covariance matrices.
  - model: A QKModel instance providing the necessary model parameters for smoothing, including:
      • aug_state: A structure containing:
          - H_aug  :: The augmented measurement selection matrix.
          - G_aug  :: The augmented duplication matrix for handling quadratic forms.
          - Phi_aug:: The augmented state transition matrix.
      • state: The dimension (N) of the state vector.

Returns:
  - A SmootherOutput structure containing:
      • Z_smooth :: Matrix of smoothed augmented state estimates.
      • P_smooth :: Array of smoothed state covariance matrices.
"""
function qkf_smoother!(filter_output::FilterOutput{T}, model::QKModel{T,T2}) where {T<:Real, T2<:Real}
    @unpack Z_tt, P_tt, Z_ttm1, P_ttm1 = filter_output
    @unpack H_aug, G_aug, Phi_aug = model.aug_state
    @unpack N = model.state
    T_bar = size(Z_tt, 2) - 1
    
    # Make copies to store smoothed results
    Z_smooth = copy(Z_tt)
    P_smooth = copy(P_tt)
    
    # Call original smoother implementation using one-step-ahead predictions
    qkf_smoother!(Z_smooth, P_smooth, Z_ttm1, P_ttm1, T_bar, G_aug, H_aug, Phi_aug, N)
    
    return SmootherOutput(Z_smooth, P_smooth)
end

"""
    qkf_smoother(filter_output::FilterOutput{T}, model::QKModel{T,T2}) where {T<:Real, T2<:Real}

Performs out-of-place backward smoothing for the Quadratic Kalman Filter (QKF) using filtering outputs.
This function refines the state estimates produced during filtering by incorporating future observations
through a backward smoothing pass.

# Arguments
- filter_output::FilterOutput{T}: A container holding the outputs from the filtering phase, including:
    • Z_tt: A matrix of filtered augmented state estimates.
    • P_tt: An array of filtered state covariances.
    • Z_ttm1: A matrix containing the one-step-ahead predicted states.
    • P_ttm1: An array containing the one-step-ahead predicted state covariances.
- model::QKModel{T,T2}: A model specification that provides the necessary parameters for the smoothing process.
  The model includes an `aug_state` field which is unpacked to retrieve:
    • H_aug: The augmented measurement selection matrix.
    • G_aug: The augmented duplication matrix used for handling quadratic forms.
    • Phi_aug: The augmented state transition matrix.
  Additionally, the `state` field is used to extract the dimensionality (N) of the state.

# Details
This function first creates copies of the filtered state and covariance estimates to prevent modification
of the original filtering outputs. It then invokes the in-place smoother routine (`qkf_smoother!`)
on these copies using the one-step-ahead predicted values and the model's parameters. The final smoothed
results are wrapped in a `SmootherOutput` struct and returned as fresh copies, which is particularly important
for compatibility with automatic differentiation (AD) workflows.

# Returns
- SmootherOutput: A composite structure containing:
    • Z_smooth: A matrix of smoothed augmented state estimates.
    • P_smooth: An array of smoothed state covariance matrices.
"""
function qkf_smoother(filter_output::FilterOutput{T},
    model::QKModel{T,T2}) where {T<:Real, T2<:Real}

    @unpack Z_tt, P_tt, Z_ttm1, P_ttm1 = filter_output
    @unpack H_aug, G_aug, Phi_aug = model.aug_state
    @unpack N = model.state
    T_bar = size(Z_tt, 2) - 1

    # Create copies for the smoothed results to preserve the original filtering output
    Z_smooth = copy(Z_tt)
    P_smooth = copy(P_tt)
    
    # Perform the in-place smoothing using one-step-ahead predictions and model parameters
    qkf_smoother!(Z_smooth, P_smooth, Z_ttm1, P_ttm1, T_bar, H_aug, G_aug, Phi_aug)
    
    # Return new copies of the smoothed outputs to ensure compatibility with AD workflows
    return SmootherOutput(copy(Z_smooth), copy(P_smooth))
end

# Main interface function that returns combined results
"""
    qkf(model::QKModel{T,T2}, data::QKData{T1,N}) where {T1<:Real, T<:Real, T2<:Real, N}

Execute the full Quadratic Kalman Filter (QKF) process by combining both the filtering and backward smoothing stages.
This function first applies the filtering routine to generate real-time state estimates and then refines these estimates
using the smoother to incorporate information from future observations. The resulting output is a composite object that 
encapsulates both the filtered and smoothed results.

Parameters:
  - model::QKModel{T,T2}: A model specification that includes the system dynamics, state transition parameters,
    and the augmented state representation. This object provides all necessary configurations to perform the QKF.
  - data::QKData{T1,N}: A data container comprising the observed time series measurements, formatted appropriately
    for the QKF. The parameter N indicates the dimension of the state vector, and T1 denotes the numerical type of the data.

Process:
  1. Filtering Stage: The function invokes qkf_filter with the provided data and model to compute the filtered state
     estimates and error covariances.
  2. Smoothing Stage: It then calls qkf_smoother to perform backward smoothing on the filtered results, enhancing the 
     state estimates by leveraging future information.

Returns:
  - QKFOutput{T}: A composite output object that bundles both filtering and smoothing results. This output includes:
      • filter_output: The results from the filtering phase, containing state estimates and covariances.
      • smoother_output: The refined state estimates obtained after applying the smoother.
  This combined output is useful for subsequent analysis, diagnostics, and visualization of the QKF performance.
"""
function qkf(model::QKModel{T,T2}, data::QKData{T1,N}) where {T1<:Real, T<:Real, T2<:Real, N}
    # Run filter
    filter_output = qkf_filter(data, model)
    
    # Run smoother
    smoother_output = qkf_smoother(filter_output, model)
    
    # Return combined results
    return QKFOutput{T}(filter_output, smoother_output)
end

# Add method with reversed argument order for backwards compatibility
"""
    qkf(data::QKData{T1,N}, model::QKModel{T,T2}) where {T1<:Real, T<:Real, T2<:Real, N}

This function offers a backwards-compatible interface to the Quadratic Kalman Filter (QKF) by accepting the data and model
arguments in a reversed order compared to the standard interface. Typically, the preferred order is to pass the model first
followed by the data. This method ensures that legacy code that follows the old argument order still functions correctly.

Parameters:
  - data::QKData{T1,N}: An instance containing the observed time series data formatted for the QKF. The data structure
    organizes the measurements and any associated time indices to be processed by the filter.
  - model::QKModel{T,T2}: An instance containing the model specifications which include the system dynamics, the augmented
    state representation, and all relevant parameters required to perform the filtering and smoothing operations.

Returns:
  - QKFOutput{T}: A composite output object that includes both the filtering results and the smoothed state estimates.
    This output encapsulates all intermediate steps and final results computed by invoking the standard qkf function
    with the proper argument order.

Note:
   This reversed argument order function is maintained solely for backwards compatibility. Internally, it simply calls
   qkf(model, data) to ensure that the original processing logic remains unchanged.
"""
function qkf(data::QKData{T1,N}, model::QKModel{T,T2}) where {T1<:Real, T<:Real, T2<:Real, N}
    return qkf(model, data)
end

#export qkf_smoother!, qkf_smoother, qkf