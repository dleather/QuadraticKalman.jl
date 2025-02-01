"""
    FilterOutput{T<:Real}

A container for the outputs produced by the Quadratic Kalman Filter applied to state-space models with quadratic measurement equations. This structure organizes and stores all key filtering results, facilitating subsequent analysis, diagnostics, or visualization. 

Fields:
  • ll_t::Vector{T} 
      A vector containing the log-likelihood values computed at each time step.
  • Z_tt::Matrix{T} 
      A matrix representing the filtered state estimates at the current time step.
  • P_tt::Array{T,3} 
      A 3-dimensional array containing the error covariance matrices corresponding to the filtered state estimates.
  • Y_ttm1::Union{Vector{T}, Matrix{T}}
      The one-step-ahead (t-1) predicted measurements; it can be a vector for univariate cases or a matrix for multivariate cases.
  • M_ttm1::Array{T,3} 
      A 3-dimensional array holding auxiliary statistics from the filtering process.
  • K_t::Array{T,3} 
      A 3-dimensional array of Kalman gain matrices computed at each filter update.
  • Z_ttm1::Matrix{T} 
      A matrix of the one-step-ahead state predictions (prior estimates).
  • P_ttm1::Array{T,3} 
      A 3-dimensional array representing the error covariance matrices for the one-step-ahead state predictions.

Usage:
    This structure is used to encapsulate all relevant outputs from the Quadratic Kalman Filter, ensuring that users can easily access and work with the filtered estimates, prediction errors, and associated metrics that describe the performance of the filtering process.
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

Container for outputs from the Quadratic Kalman Smoother. This structure encapsulates the results produced by the smoothing algorithm, which refines state estimates by incorporating information from both past and future observations. The smoother typically yields more accurate state estimates and associated uncertainty quantification than the filter alone.

# Fields
- `Z_smooth::Matrix{T}`: A matrix containing the smoothed state estimates. The matrix dimensions are P × (T̄+1), where P represents the dimension of the state vector and T̄+1 denotes the number of time steps.
- `P_smooth::Array{T,3}`: A three-dimensional array holding the smoothed covariance matrices. Each slice P_smooth[:, :, t] corresponds to the covariance estimate for the state at time step t, with overall dimensions P × P × (T̄+1).

# Details
The smoothed states and covariances are calculated using the Quadratic Kalman Smoother, which enhances the filtering results by backward recursion. This allows for improved state estimation by considering future as well as past measurements. The structure is essential for diagnostic checks and subsequent analyses in applications where state estimation uncertainty plays a critical role.
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

Combined container for both filtering and smoothing outputs from the Quadratic Kalman algorithm.

This type encapsulates the results of the forward filtering pass—where state estimates and associated covariances are computed recursively based solely on past and present observations—and the subsequent backward smoothing pass that refines these estimates by incorporating future observations. The unified structure provides a clear and convenient interface for diagnostic analysis, visualization, and further model-based inference tasks.

# Fields
- `filter::FilterOutput{T}`: Contains the outputs of the filtering process, such as the filtered log-likelihood, state estimates, and covariance matrices at each time step. These results represent the model’s estimates obtained in real time as the data was observed.
- `smoother::SmootherOutput{T}`: Contains the outputs of the smoothing process, which refines and improves upon the filter results by leveraging information from the entire observation sequence. This typically includes the smoothed state estimates and corresponding covariance matrices, providing a more accurate reconstruction of the underlying state dynamics.

# Details
Using the QKFOutput structure, users can conveniently access both the instantaneous (filtering) and retrospectively improved (smoothing) estimates, making it easier to perform post-hoc analysis, diagnostics, or forecasting.
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