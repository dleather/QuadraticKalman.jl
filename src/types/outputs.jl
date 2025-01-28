"""
    FilterOutput{T<:Real}

Container for outputs from the Quadratic Kalman Filter.
"""
struct FilterOutput{T<:Real}
    llₜ::Vector{T}
    Zₜₜ::Matrix{T}
    Pₜₜ::Array{T,3}
    Yₜₜ₋₁::Union{Vector{T}, Matrix{T}}  # Allow for both vector and matrix measurements
    Mₜₜ₋₁::Array{T,3}
    Kₜ::Array{T,3}
    Zₜₜ₋₁::Matrix{T}
    Pₜₜ₋₁::Array{T,3}
    Σₜₜ₋₁::Array{T,3}
end

"""
    SmootherOutput{T<:Real}

Container for outputs from the Quadratic Kalman Smoother.

# Fields
- `Zₜₜ_smooth::Matrix{T}`: Smoothed states (P × (T̄+1))
- `Pₜₜ_smooth::Array{T,3}`: Smoothed covariances (P × P × (T̄+1))
"""
struct SmootherOutput{T<:Real}
    Zₜₜ_smooth::Matrix{T}
    Pₜₜ_smooth::Array{T,3}
end

"""
    QKFOutput{T<:Real}

Combined container for both filter and smoother outputs.

# Fields
- `filter::FilterOutput{T}`: Results from the filter
- `smoother::SmootherOutput{T}`: Results from the smoother
"""
struct QKFOutput{T<:Real}
    filter::FilterOutput{T}
    smoother::SmootherOutput{T}
end

# Constructor for Vector measurement case (QKData{T1,1})
function FilterOutput(output_tuple::NamedTuple{(:llₜ, :Zₜₜ, :Pₜₜ, :Yₜₜ₋₁, :Mₜₜ₋₁, :Kₜ, :Zₜₜ₋₁, :Pₜₜ₋₁, :Σₜₜ₋₁)})
    return FilterOutput(
        output_tuple.llₜ,
        output_tuple.Zₜₜ,
        output_tuple.Pₜₜ,
        output_tuple.Yₜₜ₋₁,
        output_tuple.Mₜₜ₋₁,
        output_tuple.Kₜ,
        output_tuple.Zₜₜ₋₁,
        output_tuple.Pₜₜ₋₁,
        output_tuple.Σₜₜ₋₁
    )
end

# Constructor for Matrix measurement case (QKData{T,2})
function FilterOutput(output_tuple::NamedTuple{(:llₜ, :Zₜₜ, :Pₜₜ, :Yₜₜ₋₁, :Mₜₜ₋₁, :Kₜ, :Zₜₜ₋₁, :Pₜₜ₋₁)})
    T = eltype(output_tuple.Zₜₜ)
    return FilterOutput(
        output_tuple.llₜ,
        output_tuple.Zₜₜ,
        output_tuple.Pₜₜ,
        output_tuple.Yₜₜ₋₁,
        output_tuple.Mₜₜ₋₁,
        output_tuple.Kₜ,
        output_tuple.Zₜₜ₋₁,
        output_tuple.Pₜₜ₋₁,
        zeros(T, size(output_tuple.Pₜₜ₋₁))  # Empty Σₜₜ₋₁ for this case
    )
end

# Constructor for functional filter case
function FilterOutput(llₜ::Vector{T}, Zₜₜ::Matrix{T}, Pₜₜ::Array{T,3}) where T<:Real
    P, Tp1 = size(Zₜₜ)
    T̄ = Tp1 - 1
    
    return FilterOutput(
        llₜ,
        Zₜₜ,
        Pₜₜ,
        Vector{T}(undef, 0),        # Empty Yₜₜ₋₁
        Array{T,3}(undef, 0, 0, 0), # Empty Mₜₜ₋₁
        Array{T,3}(undef, 0, 0, 0), # Empty Kₜ
        Matrix{T}(undef, 0, 0),     # Empty Zₜₜ₋₁
        Array{T,3}(undef, 0, 0, 0), # Empty Pₜₜ₋₁
        Array{T,3}(undef, 0, 0, 0)  # Empty Σₜₜ₋₁
    )
end

function SmootherOutput(output_tuple::NamedTuple)
    return SmootherOutput(
        output_tuple.Zₜₜ_smooth,
        output_tuple.Pₜₜ_smooth
    )
end

function QKFOutput(filter_out::FilterOutput{T}, smoother_out::SmootherOutput{T}) where T<:Real
    return QKFOutput{T}(filter_out, smoother_out)
end

# Export the types
export FilterOutput, SmootherOutput, QKFOutput