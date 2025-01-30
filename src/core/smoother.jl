
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
- `Z_smooth::Matrix{T}`: (P × (T̄+1)) smoothed states
- `P_smooth::Array{T,3}`: (P × P × (T̄+1)) smoothed covariances

# Example
```
Z_smooth, P_smooth = qkf_smoother(Z, P, Z_pred, P_pred, T̄, Hn, Gn, H̃, Φ̃, n)
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

In-place backward smoothing for the QKF using filter outputs.
"""
function qkf_smoother!(filter_output::FilterOutput{T}, model::QKModel{T,T2}) where {T<:Real, T2<:Real}
    @unpack Z_tt, P_tt, Z_ttm1, P_ttm1 = filter_output
    @unpack H_aug, G_aug, Phi_aug = model.aug_state
    T_bar = size(Z_tt, 2) - 1
    

    # Make copies to store smoothed results
    Z_smooth = copy(Z_tt)
    P_smooth = copy(P_tt)
    
    # Call original smoother implementation
    qkf_smoother!(Z_smooth, P_smooth, Z_ttm1, P_ttm1, T_bar, G_aug, H_aug, Phi_aug, N)
    
    return SmootherOutput(Z_smooth, P_smooth)
end

"""
    qkf_smoother(filter_output::FilterOutput{T}, params::QKParams{T,T2}) where {T<:Real, T2<:Real}

Out-of-place backward smoothing for the QKF using filter outputs.
"""
function qkf_smoother(filter_output::FilterOutput{T},
    model::QKModel{T,T2}) where {T<:Real, T2<:Real}

    @unpack Z_tt, P_tt, Z_ttm1, P_ttm1 = filter_output
    @unpack H_aug, G_aug, Phi_aug = model.aug_state
    @unpack N = model.state
    T_bar = size(Z_tt, 2) - 1

    # Make copies for the smoothed results
    Z_smooth = copy(Z_tt)
    P_smooth = copy(P_tt)
    
    # Call in-place version on copies
    qkf_smoother!(Z_smooth, P_smooth, Z_ttm1, P_ttm1, T_bar, H_aug, G_aug, Phi_aug)
    
    # Return fresh copies for AD
    return SmootherOutput(copy(Z_smooth), copy(P_smooth))
end

# Main interface function that returns combined results
"""
    qkf(data::QKData{T1,N}, params::QKParams{T,T2}) where {T1,T,T2,N}

Run both QKF filter and smoother, returning combined results.
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

Reversed argument order for backwards compatibility.
"""
function qkf(data::QKData{T1,N}, model::QKModel{T,T2}) where {T1<:Real, T<:Real, T2<:Real, N}
    return qkf(model, data)
end

#export qkf_smoother!, qkf_smoother, qkf