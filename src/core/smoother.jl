
"""
    qkf_smoother!(
        Z::AbstractMatrix{T},      # Filtered states   (P × (T̄+1))
        P::AbstractArray{T, 3},    # Filtered covariances (P × P × (T̄+1))
        Z_pred::AbstractMatrix{T}, # One-step-ahead predicted states (P × T̄)
        P_pred::AbstractArray{T,3},
        T̄::Int,
        Hn::Matrix{T},  Gn::Matrix{T},  Hn_tilde::Matrix{T},  Φ_tilde::Matrix{T},
        dof::Int
    ) where {T<:Real}

Perform **in-place** backward smoothing for the Quadratic Kalman Filter (QKF).

# Description

Given the forward-filtered estimates `(Z, P)` from `t=1..T̄`, plus the 
one-step-ahead predictions `(Z_pred, P_pred)` and the special matrices 
(H̃ₙ, G̃ₙ) that handle the dimension reduction via 
Vech(·)/Vec(·), this function computes `Z[:,t]` and `P[:,:,t]` for 
`t = T̄-1 .. 1` in backward fashion to produce the smoothed estimates 
(Zₜ|T, PZₜ|T).  

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
- `T̄::Int`: Total time steps (excluding time 0).
- `Hn::Matrix{T}`, `Gn::Matrix{T}`: The selection/duplication operators 
  for Vec/Vech transforms of block `(x xᵀ)`. Usually size `(n(n+1) × n(n+3)/2)` or similarly.
- `Hn_tilde::Matrix{T}`, `Φ_tilde::Matrix{T}`: The augmented versions 
  (H̃ₙ, G̃ₙ) used in the QKF recursion.
- `dof::Int`: Dimension parameter (often `n` or `P`). Adjust to your model.

# Notes

- This function runs backward from `t = T̄-1` down to `t = 1`, using 
  the final values `(Z[:, T̄], P[:,:, T̄])` as the terminal condition 
  (`Z_{T̄|T̄}, P^Z_{T̄|T̄}`).
- If your AD library supports destructive updates, this approach should 
  be AD-friendly; if not, consider the out-of-place version `qkf_smoother`.

# Example
Suppose you already ran the forward filter, so you have:
    Z, P, Z_pred, P_pred, plus your special matrices.
```
qkf_smoother!(Z, P, Z_pred, P_pred, T̄, Hn, Gn, Hn_tilde, Φ_tilde, n)
```
"""
function qkf_smoother!( Z::AbstractMatrix{T}, P::AbstractArray{T,3},
    Z_pred::AbstractMatrix{T}, P_pred::AbstractArray{T,3}, T̄::Int,
    Hn::AbstractMatrix{T}, Gn::AbstractMatrix{T}, Hn_tilde::AbstractMatrix{T},
    Φ_tilde::AbstractMatrix{T}, dof::Int) where {T <: Real}

    @assert size(Z,2) == T̄ + 1 "Z should have T̄+1 columns"
    @assert size(P,3) == T̄ + 1 "P should have T̄+1 slices"
    @assert size(Z_pred,2) == T̄ "Z_pred should have T̄ columns"
    @assert size(P_pred,3) == T̄ "P_pred should have T̄ slices"
    
    # Work backward from t=T̄-1 down to t=1
    for t in (T̄-1):-1:1
        # 1) Transform P^Z_{t|t} and P^Z_{t+1|t} with Hn_tilde
        #    We want: M_t   = ( Hn_tilde * P^Z_{t|t} * Hn_tilde' )
        #             M_t+1 = ( Hn_tilde * P^Z_{t+1|t} * Hn_tilde' )
        M_t   = Hn_tilde * P[:,:,t]   * Hn_tilde'
        M_t1  = Hn_tilde * P_pred[:,:,t] * Hn_tilde'
    
        # 2) Also transform states
        #    hZ_t   = (Hn_tilde * Z[:, t])
        #    hZ_t+1 = (Hn_tilde * Z_pred[:, t]) for the predicted,
        #             or use the smoothed Z_{t+1|T} if it is already updated
        hZ_t   = Hn_tilde * Z[:,t]
        hZ_t1  = Hn_tilde * Z[:,t+1]   # after smoothing pass (Z_{t+1|T})
        hZ_t1p = Hn_tilde * Z_pred[:,t]  # Z_{t+1|t} from forward filter
    
        # 3) Form the cross term: (tilde{H}_n * tilde{\Phi} * tilde{G}_n)'
        #    We'll store it if needed or do direct multiplication inline.
        #    Typically: cross = (Hn_tilde * Φ_tilde * Gn)' 
        cross = (Hn_tilde * Φ_tilde * Gn)'
    
        # 4) Now compute F_t = M_t * cross * (M_t1)^-1
        #    But we use a solve: F_t = M_t * cross * (M_t1 \ I)
        #    i.e. F_t = M_t * cross * inv(M_t1), so we do:
        #       S = M_t1 \ (M_t * cross)
        #    or we do factorization of M_t1. We'll do `ldiv!` approach:
        #    Let factor = lu(M_t1)  or cholesky if it's PSD. We'll assume PSD -> use cholesky:
        factor_t1 = cholesky(Symmetric(M_t1))  # or: factor_t1 = cholesky(M_t1)
        # We'll allocate a temp for multiplication M_t * cross
        tmp = similar(M_t, size(M_t,1), size(cross,2))
        mul!(tmp, M_t, cross)  # tmp = M_t * cross
        # Now solve for F_t: F_t = tmp * (M_t1^-1) = factor_t1 \ tmp
        F_t = factor_t1 \ tmp    # => size(F_t) = (size(M_t,1), size(cross,1))
    
        # 5) Update the smoothed hZ_t: 
        #    hZ_{t|T} = hZ_{t|t} + F_t ( hZ_{t+1|T} - hZ_{t+1|t} )
        @assert size(hZ_t1) == size(hZ_t1p) "Dimension mismatch in predicted vs smoothed for time t+1"
        diff = hZ_t1 .- hZ_t1p
        hZ_t_smooth = hZ_t .+ F_t * diff
    
        # 6) Update the smoothed M_t ( => (Hn_tilde P^Z_{t|T} Hn_tilde') ), then revert
        #    M_{t|T} = M_t + F_t [ (Hn_tilde P^Z_{t+1|T} Hn_tilde') - M_t1 ] F_t'    #    We have P^Z_{t+1|T} = P[:,:,t+1] at this point (the smoothed version).
        M_t1_smooth = Hn_tilde * P[:,:,t+1] * Hn_tilde'
        dM = M_t1_smooth .- M_t1
        #   mid = F_t * dM
        mid = similar(F_t, size(F_t,1), size(dM,2))
        mul!(mid, F_t, dM)
        #   new_M_t = M_t + mid * F_t'
        #   We'll do this in-place for performance
        new_M_t = copy(M_t)
        mul!(new_M_t, mid, F_t', 1.0, 1.0)  # new_M_t = new_M_t + mid * F_t'
    
        # 7) Transform back: we want P^Z_{t|T} such that 
        #    Hn_tilde * P^Z_{t|T} * Hn_tilde' = new_M_t
        #    => P^Z_{t|T} = (Hn_tilde') \ new_M_t / (Hn_tilde)   (conceptually)
        #    but in practice we handle the shape carefully. 
        #    If Hn_tilde is invertible, we do a solve again:
        factor_tilde = lu(Hn_tilde)  # or cholesky if appropriate
        # let Q = factor_tilde \ new_M_t
        Q = similar(new_M_t)
        ldiv!(Q, factor_tilde, new_M_t)  # Q = Hn_tilde^-1 * new_M_t
        # Now P^Z_{t|T} = Q * (Hn_tilde^-1)' 
        # We'll do another ldiv! with factor_tilde on Q^T if needed. 
        # Or a direct approach: P^Z_{t|T} = Q * inv(Hn_tilde')
        # We'll do factor_tilde' if it’s an LU or cholesky factor. 
        # For simplicity:
        P_tT = Q * inv(Hn_tilde')   # This is an allocation; we can refine if needed
    
        # Overwrite the slices:
        # Overwrite Z_{t|T} by transforming hZ_t_smooth back:
        # we want Z_{t|T} = Hn_tilde^-1 * hZ_t_smooth
        Z_tT = factor_tilde \ hZ_t_smooth
        Z[:, t] .= Z_tT
        P[:,:, t] .= P_tT
    end
    
    return nothing
    
end

"""
    qkf_smoother( Z, P, Z_pred, P_pred, T̄, Hn, Gn, Hn_tilde, Φ_tilde, dof ) -> (Z_smooth, P_smooth)

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
Z_smooth, P_smooth = qkf_smoother(Z, P, Z_pred, P_pred, T̄, Hn, Gn, Hn_tilde, Φ_tilde, n)
```
""" 
function qkf_smoother( Z::AbstractMatrix{T}, P::AbstractArray{T,3},
    Z_pred::AbstractMatrix{T}, P_pred::AbstractArray{T,3}, T̄::Int,
    Hn::AbstractMatrix{T}, Gn::AbstractMatrix{T}, Hn_tilde::AbstractMatrix{T},
    Φ_tilde::AbstractMatrix{T}, dof::Int ) where {T<:Real}

    # Make copies (or a deep copy if needed)
    Z_smooth = copy(Z)
    P_smooth = similar(P)  # same dims, uninitialized
    P_smooth .= P  # replicate the filtered covariances

    # Just call the in-place version on the copies:
    qkf_smoother!(Z_smooth, P_smooth, Z_pred, P_pred, T̄, Hn, Gn, Hn_tilde, Φ_tilde, dof)

    return (Z_smooth, P_smooth)
    
end