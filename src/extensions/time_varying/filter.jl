function predict_Mₜₜ₋₁(Pₜₜ₋₁::AbstractMatrix{T}, Zₜₜ₋₁::AbstractVector{T},
    params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

    @unpack M, B̃, M, P, wc, wu, wv, wuu, wuv, wvv = params
    z = Zₜₜ₋₁
    V_tmp = [wc + wu * z[2] + wv * z[1] + wuu * z[6] + wvv * z[3] + 
        (wuv / 2.0) * (z[4] + z[5])]
    M_tmp = B̃ * Pₜₜ₋₁ * B̃' + V_tmp

    if M==1
        if M_tmp[1,1] < 0.0
            Mₜₜ₋₁ = reshape([1e-04], 1, 1)
        else
            Mₜₜ₋₁ = M_tmp
        end
    else
        if isposdef(M_tmp)
            Mₜₜ₋₁ = M_tmp
        else
            Mₜₜ₋₁ = make_positive_definite(M_tmp)
        end
    end
    return Mₜₜ₋₁
end

function predict_Mₜₜ₋₁!(Mₜₜ₋₁::AbstractArray{Real, 3}, Pₜₜ₋₁::AbstractArray{Real, 3},
    Zₜₜ₋₁::AbstractMatrix{T}, tmpB::AbstractMatrix{T}, params::QKParams{T,T2},
    t::Int) where {T <: Real, T2 <: Real}
            

    @unpack B̃, V, M, P, wc, wu, wv, wuu, wuv, wvv = params
    zt = Zₜₜ₋₁[:, t]
    #Mₜₜ₋₁ = B̃Pₜₜ₋₁B̃' + V
    # tmpB = B̃Pₜₜ₋₁
    #=
    for i = 1:M
        for j = 1:P
            tmpB[i, j] = 0
            for k = 1:P
                tmpB[i, j] += B̃[i, k] * Pₜₜ₋₁[k, j, t]
            end
        end
    end

    if typeof(V) <: Function
        V_tmp = V(zt)
        trHP = tr(HessR(zt) * Pₜₜ₋₁[:,:,t])
        for i = 1:M
            for j = 1:M
                for k = 1:P
                    Mₜₜ₋₁[i, j, t] += tmpB[i, k] * B̃[j, k]
                end
                Mₜₜ₋₁[i, j, t] += V_tmp[i, j] 
            end
        end
        Mₜₜ₋₁[:, :, t] .+= 0.5 * tr(HessR(zt) * Pₜₜ₋₁[:,:,t])
    else
    #Mₜₜ₋₁ = tmpB * B̃' + V
    for i = 1:M
        for j = 1:M
            for k = 1:P
                Mₜₜ₋₁[i, j, t] += tmpB[i, k] * B̃[j, k]
            end
            Mₜₜ₋₁[i, j, t] += V[i, j]
        end
    end
    =#

    V = [wc + wu * zt[2] + wv * zt[1] + wuu * zt[6] + wvv * zt[3] + 
        (wuv / 2.0) * (zt[4] + zt[5])]    
    Mₜₜ₋₁[:,:,t] = make_positive_definite(B̃ * Pₜₜ₋₁[:,:,t] * B̃' + V)
    #Mₜₜ₋₁[:,:,t] = B̃ * Pₜₜ₋₁[:,:,t] * B̃' + [V_tmp]


    #if any(Diagonal(Mₜₜ₋₁[:,:,t] .< 0.0))
    #    println("Negative Variance")
    #end
end

"""
    update_Pₜₜ!(Pₜₜ, Kₜ, Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, tmpKM, tmpKMK, params, t)

In-place update of the **filtered covariance** `Pₜₜ[:, :, t+1]` given the one-step-ahead 
covariance `Pₜₜ₋₁[:, :, t]`, the Kalman gain `Kₜ[:, :, t]`, and a scalar "volatility" 
term derived from `Zₜₜ₋₁[:, t]`.

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
  We read `Zₜₜ₋₁[:, t]` to compute a scalar `V_tmp` from parameters `wc, wv, ...`.
- `tmpKM, tmpKMK::AbstractMatrix{<:Real}`: Temporary buffers `(P×M, P×P)` 
  if you want manual multiplication. In the final code, we do not use them, 
  but they can be placeholders for expansions.
- `params::QKParams{T,T2}`: Must contain:
  - `wc, wu, wv, wuu, wvv, wuv` (time-varying "volatility" coefficients),
  - `B̃::Matrix{T}` (size `M×P`),
  - `P::Int, M::Int`, etc. 
- `t::Int`: The time index (1-based).

# Computation
1. We form a scalar:
   `V_tmp = wc + wu*z[2] + wv*z[1] + wuu*z[6] + wvv*z[3] + (wuv/2)*(z[4] + z[5])`
from the state z = Zₜₜ₋₁[:, t].
2. Let `A = I - Kₜ[:, :, t]*B̃`.
3. Then
    `Pₜₜ[:, :, t+1] = make_positive_definite(A * Pₜₜ₋₁[:, :, t] * A' + Kₜ[:, :, t]*V_tmp*Kₜ[:, :, t])'
4. We wrap the result with make_positive_definite to ensure no negative eigenvalues from the update.

# Notes
- This is an in-place update: we store the new covariance in Pₜₜ[:, :, t+1].
- The time-varying "volatility" is effectively a scalar (since M=1?), though you can adapt for a matrix if needed.
- AD-Friendliness: The final expression is a typical linear+outer-product operation plus
   make_positive_definite. If your AD can handle in-place modifications, or you define a
   custom adjoint, it should be fine. Otherwise, consider a purely functional approach.
"""
function update_Pₜₜ!( Pₜₜ::AbstractArray{Real,3}, Kₜ::AbstractArray{Real,3},
    Mₜₜ₋₁::AbstractArray{Real,3}, Pₜₜ₋₁::AbstractArray{Real,3},
    Zₜₜ₋₁::AbstractArray{Real,2}, tmpKM::AbstractMatrix{Real},
    tmpKMK::AbstractMatrix{Real}, params::QKParams{T,T2}, t::Int ) where {T<:Real, T2<:Real}
    @unpack B̃, wc, wu, wv, wuu, wvv, wuv = params

    # 1) Compute a scalar V_tmp from the predicted state zt
    zt = Zₜₜ₋₁[:, t]
    V_tmp = wc + wu*zt[2] + wv*zt[1] + wuu*zt[6] + wvv*zt[3] + (wuv/2.0)*(zt[4] + zt[5])
    
    # 2) Form A = I - Kₜ[:, :, t]*B̃
    local A = I - Kₜ[:, :, t]*B̃
    
    # 3) In-place assign the final updated covariance
    Pₜₜ[:, :, t+1] = make_positive_definite(
        A * Pₜₜ₋₁[:, :, t] * A' .+ (Kₜ[:, :, t] * V_tmp * Kₜ[:, :, t]')
    )
end

"""
    update_Pₜₜ(Kₜ, Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, params, t)

Compute and return a **new** filtered covariance matrix, 
given a predicted covariance `Pₜₜ₋₁`, a Kalman gain `Kₜ`, 
and a scalar "volatility" derived from `Zₜₜ₋₁`.

# Arguments
- `Kₜ::AbstractMatrix{T5}`: (P×M) or (P×1) Kalman gain for time `t`.
- `Mₜₜ₋₁::AbstractMatrix{T6}`: *(Potentially unused here or if you only 
  rely on `V_tmp`?),*
- `Pₜₜ₋₁::AbstractMatrix{T3}`: The one-step-ahead covariance `(P×P)`.
- `Zₜₜ₋₁::AbstractVector{T4}`: The predicted augmented state (length P).
- `params::QKParams{T,T2}`: Must contain the scalar coefficients `wc, wu, wv, wuu, wvv, wuv`
  and `B̃::Matrix{T}`, etc.
- `t::Int`: The time index (not strictly used, but might be for logging or consistency).

# Returns
- `Matrix{T}`: A newly allocated `(P×P)` covariance matrix 
  after the update step, guaranteed to be positive-definite if `make_positive_definite` 
  fixes any negative eigenvalues.

# Details
1. We compute a scalar `V_tmp = wc + ...` from `Zₜₜ₋₁`.
2. Let `A = I - Kₜ*B̃`.
3. Build `P = A * Pₜₜ₋₁ * A' + Kₜ*V_tmp*Kₜ'`
4. This version is purely functional (returns a new matrix). It's often simpler for AD if
   you want a direct forward pass without in-place modifications.

"""
function update_Pₜₜ( Kₜ::AbstractMatrix{T5}, Mₜₜ₋₁::AbstractMatrix{T6},
    Pₜₜ₋₁::AbstractMatrix{T3}, Zₜₜ₋₁::AbstractVector{T4}, params::QKParams{T,T2},
    t::Int ) where {T<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Real, T6<:Real}
    @unpack B̃, wc, wu, wv, wuu, wvv, wuv = params

    # 1) Extract state to form a scalar "V_tmp"
    local zt = Zₜₜ₋₁
    local V_tmp = wc + wu*zt[2] + wv*zt[1] + wuu*zt[6] + wvv*zt[3] + (wuv / 2.0)*(zt[4] + zt[5])

    # 2) A = I - Kₜ*B̃
    local A = I - Kₜ*B̃

    # 3) Build the new covariance
    local P = A * Pₜₜ₋₁ * A' .+ (Kₜ * V_tmp * Kₜ')

    # 4) Ensure positivity
    if !isposdef(P)
        return make_positive_definite(P)
    else
        return P
    end
end