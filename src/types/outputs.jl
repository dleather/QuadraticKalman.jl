"""
    FilterOutput{T<:Real}

Container for outputs from the Quadratic Kalman Filter.
"""
struct FilterOutput{T<:Real}
    ll_t::Vector{T}
    Z_tt::Matrix{T}
    P_tt::Array{T,3}
    Y_ttm1::Union{Vector{T}, Matrix{T}}  # Allow for both vector and matrix measurements
    M_ttm1::Array{T,3}
    K_t::Array{T,3}
    Z_ttm1::Matrix{T}
    P_ttm1::Array{T,3}
end

# Constructor for Vector measurement case (QKData{T1,1})
function FilterOutput(output_tuple::NamedTuple{(:ll_t, :Z_tt, :P_tt, :Y_tt_minus_1, :M_tt_minus_1, :K_t,
    :Z_ttm1, :P_ttm1, :Sigma_ttm1)})
    
    return FilterOutput(
        output_tuple.ll_t,
        output_tuple.Z_tt,
        output_tuple.P_tt,
        output_tuple.Y_ttm1,
        output_tuple.M_ttm1,
        output_tuple.K_t,
        output_tuple.Z_ttm1,
        output_tuple.P_ttm1,
    )
end

# Constructor for Matrix measurement case (QKData{T,2})
function FilterOutput(output_tuple::NamedTuple{(:ll_t, :Z_tt, :P_tt, :Y_ttm1, :M_ttm1, :K_t,
    :Z_ttm1, :P_ttm1)})
    T = eltype(output_tuple.Z_tt)

    return FilterOutput(
        output_tuple.ll_t,
        output_tuple.Z_tt,
        output_tuple.P_tt,
        output_tuple.Y_ttm1,
        output_tuple.M_ttm1,
        output_tuple.K_t,
        output_tuple.Z_ttm1,
        output_tuple.P_ttm1,
    )
end

# Constructor for functional filter case
function FilterOutput(ll_t::Vector{T}, Z_tt::Matrix{T}, P_tt::Array{T,3}) where T<:Real
    P, Tp1 = size(Z_tt)
    T̄ = Tp1 - 1
    
    return FilterOutput(
        ll_t,
        Z_tt,
        P_tt,
        Vector{T}(undef, 0),        # Empty Yₜₜ₋₁
        Array{T,3}(undef, 0, 0, 0), # Empty Mₜₜ₋₁
        Array{T,3}(undef, 0, 0, 0), # Empty Kₜ
        Matrix{T}(undef, 0, 0),     # Empty Zₜₜ₋₁
        Array{T,3}(undef, 0, 0, 0), # Empty Pₜₜ₋₁
    )
end


"""
    SmootherOutput{T<:Real}

Container for outputs from the Quadratic Kalman Smoother.

# Fields
- `Z_smooth::Matrix{T}`: Smoothed states (P × (T̄+1))
- `P_smooth::Array{T,3}`: Smoothed covariances (P × P × (T̄+1))
"""
struct SmootherOutput{T<:Real}
    Z_smooth::Matrix{T}
    P_smooth::Array{T, 3}
end

"""
    SmootherOutput{NamedTuple}

Container for outputs from the Quadratic Kalman Smoother.

# Fields
- `Z_smooth::Matrix{T}`: Smoothed states (P × (T̄+1))
- `P_smooth::Array{T,3}`: Smoothed covariances (P × P × (T̄+1))
"""
function SmootherOutput(output_tuple::NamedTuple)
    return SmootherOutput(
        output_tuple.Z_smooth,
        output_tuple.P_smooth
    )
end

"""
    QKFOutput{T<:Real}

Combined container for both filter and smoother outputs.

# Fields
- `filter::FilterOutput{T}`: Results from the filter
- `smoother::SmootherOutput{T}`: Results from the smoother
"""
@with_kw struct QKFOutput{T<:Real}
    filter::FilterOutput{T}
    smoother::SmootherOutput{T}
end

# Contructor
#function QKFOutput(filter_out::FilterOutput{T}, smoother_out::SmootherOutput{T}) where T<:Real
#    return QKFOutput{T}(filter_out, smoother_out)
#end

# Export the types
#export FilterOutput, SmootherOutput, QKFOutput