"""
    log_pdf_normal(μ::Real, σ²::Real, x::Real) -> Real

Compute the log of the probability density function (PDF) of a **univariate normal** 
distribution with mean `μ` and variance `σ²` at point `x`. 

# Arguments
- `μ::Real`: The mean of the normal distribution
- `σ²::Real`: The variance (σ²) of the normal distribution
- `x::Real`: The point at which to evaluate the log PDF

# Returns
- A `Real` value representing the log PDF at `x`. 
  - If `σ² > 0`, it computes `-0.5 * log(2π * σ²) - 0.5 * ((x-μ)^2 / σ²)`.
  - Otherwise (if `σ² ≤ 0`), it returns `-Inf`, signaling effectively zero probability.

# Notes
- Returning `-Inf` for non-positive variances can help keep downstream algorithms
  (like Kalman filters) from crashing when numerical errors produce a 
  slightly negative variance. 
- If negative variances are genuinely impossible (or indicative of a deeper error), 
  you could use an assertion or throw an error instead.
"""
log_pdf_normal(μ::Real, σ2::Real, x::Real) = 
    σ2 > 0 ? (-0.5 * log(2π * σ2) - 0.5*((x - μ)^2 / σ2)) : -Inf

"""
    compute_loglik!(lls::AbstractVector{T}, Y::AbstractVector{T},
                    Ypred::AbstractVector{T}, Mpred::AbstractArray{<:Real}, t::Int) 
                    where {T<:Real}

In-place computation of the log-likelihood for **univariate** observations.

# Arguments
- `lls::AbstractVector{T}`  
  A vector to store per-time-step log-likelihood values. Must have length ≥ `t`.
- `Y::AbstractVector{T}`  
  The **actual** observations (length ≥ `t`).
- `Ypred::AbstractVector{T}`  
  The **predicted** observations (length ≥ `t`).
- `Mpred::AbstractArray{<:Real}`  
  A **3D** array of prediction variances, shaped `(1,1,numSteps)`, 
  so that `Mpred[1,1,t]` is the variance at time `t`.
- `t::Int`  
  Current time index (1-based).

# Details
1. Extracts `σ² = Mpred[1,1,t]`.
2. Extracts `μ = Ypred[t]` (predicted mean).
3. Extracts `x = Y[t]` (actual observation).
4. Computes `ll = log_pdf_normal(μ, σ², x)`.
5. Stores `lls[t] = ll`.

If `σ²` is negative or zero, `log_pdf_normal` may return `-Inf`. 
In a well-behaved filter, `σ²` should be positive.
"""
function compute_loglik!(lls::AbstractVector{T}, 
                         Y::AbstractVector{T},
                         Ypred::AbstractVector{T},
                         Mpred::AbstractArray{<:Real},
                         t::Int) where {T<:Real}

    @assert size(Mpred,1) == 1 && size(Mpred,2) == 1 "Expected Mpred to be (1,1,...) for univariate"
    @assert t <= size(Mpred,3) "Index t out of range for Mpred"
    @assert t <= length(Y) && t <= length(Ypred) && t <= length(lls)

    σ2 = Mpred[1,1,t]
    μ  = Ypred[t]
    x  = Y[t]

    # Possibly clamp small or negative σ² to a small epsilon
    eps_σ2 = 1e-12
    if σ2 <= eps_σ2
        σ2 = eps_σ2
    end

    lls[t] = log_pdf_normal(μ, σ2, x)
end


"""
    compute_loglik(Y::Real, Ypred::Real, Mpred::AbstractMatrix{T}) where {T<:Real}

Compute the log-likelihood for a **single** univariate observation at one time step.

# Arguments
- `Y::Real`: The actual observation value.
- `Ypred::Real`: The predicted observation (mean).
- `Mpred::AbstractMatrix{T}`: A 1×1 matrix storing the variance at this time step,
                              i.e. `Mpred[1,1] = σ²`.

# Returns
The log-likelihood `Float64` (or `T`) computed using `log_pdf_normal(μ, σ², x)`.

# Details
1. Extracts `σ² = Mpred[1,1]`.
2. Uses `Ypred` as the mean `μ`.
3. Uses `Y` as the observation `x`.
4. If `σ²` is <= 0, clamps it to a tiny positive `1e-12` to avoid domain error.
5. Calls `log_pdf_normal(μ, σ², x)` and returns the result.
"""
function compute_loglik(Y::Real, Ypred::Real, Mpred::AbstractMatrix{T}) where {T<:Real}
    @assert size(Mpred,1) == 1 && size(Mpred,2) == 1 "Expected Mpred to be 1x1 for univariate"

    σ² = Mpred[1,1]
    μ = Ypred
    x = Y

    # minimal clamp
    eps_σ² = 1e-12
    if σ² <= eps_σ²
        σ² = eps_σ²
    end

    return log_pdf_normal(μ, σ², x)
end

"""
    logpdf_mvn(Y_pred::AbstractMatrix{T}, Σ_pred::AbstractArray{<:Real},
               Y::AbstractMatrix{T}, t::Int) -> Real
               where {T<:Real}

Compute the **log-probability density** of a multivariate normal (MVN) at time index `t`.

# Overview
This function interprets:
- The **predicted mean** μ as `Y_pred[:, t]` (i.e. column `t` of `Y_pred`).
- The **predicted covariance** Σ as `Σ_pred[:, :, t]` (the `t`-th slice in a 3D array).
- The **actual observation** x as `Y[:, t]` (column `t` of `Y`).

It then returns the log-density, ln p(x | μ, Σ) for that MVN.

# Arguments

- `Y_pred::AbstractMatrix{T}`:
  A matrix of size `(k, T_max)`, where row dimension `k` is the data dimension, 
  and column dimension is the number of time steps. We take `Y_pred[:, t]` as μ.

- `Σ_pred::AbstractArray{<:Real}`:
  A 3D array of size `(k, k, T_max)` storing the covariance for each time step. 
  We take `Σ_pred[:, :, t]` as the `k×k` covariance at time `t`.

- `Y::AbstractMatrix{T}`:
  A matrix of actual observations, also `(k, T_max)`, so `Y[:, t]` is the 
  observed vector at time `t`.

- `t::Int`:
  The time index (1-based).

# Returns
A real scalar `::Real` representing:
   ln N(x; μ, Σ, t).

# Method

1. **Views** of μ, Σ, and x are created (`@view`) to avoid copying data.
2. **Cholesky factorization** of Σ is computed via `cholesky(Symmetric(Σ))`. 
   - This yields an upper-triangular factor `C`.
3. We form the difference `diff = x - μ`.
4. We ise `solve` to efficiently compute, Σ⁻¹(x - μ), without explicit inversion.
5. The log-determinant ln| Σ | is `2 * sum(log, diag(C.U))`.
6. We add up the standard MVN log-pdf terms:
     -frac{1}{2} (ln|Σ| + (x-μ)'Σ⁻¹(x-μ)) + constant terms.
   Here the constant includes -0.5 * k * (ln 2π + 1) which is a variation of `-0.5 * (k*ln(2π) + k + ...)`.
7. Returns the final log-pdf as a `Real`.

# Notes
- **AD-Friendliness**: Zygote and other AD tools generally handle 
  `cholesky(Symmetric(...))` with built-in rules, though reverse-mode AD 
  on a large `k` can be expensive. 
- **Numerical Issues**: If Σ is not positive-definite, `cholesky` 
  will error. In a robust filter, you might ensure PSD by correction steps.

"""
function logpdf_mvn(Y_pred::AbstractMatrix{T}, Σ_pred::AbstractArray{<:Real},
                    Y::AbstractMatrix{T}, t::Int) where T <: Real

    # 1) Take views for μ, Σ, x
    μ = @view Y_pred[:, t]
    Σ = @view Σ_pred[:, :, t]
    x = @view Y[:, t]
    
    k = length(μ)

    # 2) Cholesky factorization
    C = cholesky(Symmetric(Σ))

    # 3) Form difference
    diff = x .- μ

    # 4) Solve for Σ^-1 diff via factor
    solved = C \ diff

    # 5) log(det(Σ)) from the factor
    log_det_Σ = 2 * sum(log, diag(C.U))

    # 6) Constant term for MVN
    const_term = -0.5 * k * (log(2π) + 1)  # i.e. ~ -0.5 * k * log(2π e)

    # 7) Compute the final log-pdf
    logpdf = const_term - 0.5 * (log_det_Σ + dot(diff, solved))

    return logpdf
end

"""
    compute_loglik!(lls::AbstractVector{T}, data::AbstractMatrix{T}, 
                    mean::AbstractMatrix{T}, covs::AbstractArray{<:Real}, t::Int) 
                    where T <: Real

In-place computation of the **multivariate** log-likelihood at time index `t`. 
Stores the result into `lls[t]`.

# Arguments
- `lls::AbstractVector{T}`  
  A vector of length ≥ `t`, where the computed log-likelihood is placed in `lls[t]`.
- `data::AbstractMatrix{T}`  
  The actual observation matrix, size M×T₀, where M = dimension, T₀ = total steps. 
  We'll use `data[:, t]`.
- `mean::AbstractMatrix{T}`  
  The predicted mean matrix, also M×T₀. We'll use `mean[:, t]`.
- `covs::AbstractArray{<:Real}`  
  A 3D array, size (M, M, T₀), so `covs[:, :, t]` is the covariance at time t.
- `t::Int`  
  The time index (1-based).

# Details
This calls `logpdf_mvn(mean, covs, data, t)` to compute the log PDF at time `t`.
It then writes that value into `lls[t]`.

Typically used inside a Kalman-like filter loop for each step.
"""
function compute_loglik!(lls::AbstractVector{T}, 
                         data::AbstractMatrix{T}, 
                         mean::AbstractMatrix{T}, 
                         covs::AbstractArray{<:Real}, 
                         t::Int) where {T<:Real}

    @assert t <= length(lls) "Time index t out of range for lls"
    @assert t <= size(data,2) && t <= size(mean,2) && t <= size(covs,3) "Time index t out of range for data/mean/covs"

    lls[t] = logpdf_mvn(mean, covs, data, t)
end


"""
    compute_loglik(data::AbstractMatrix{T}, mean::AbstractMatrix{T}, 
                   covs::AbstractArray{<:Real}, t::Int) -> Real
                   where T <: Real

Return the multivariate log-likelihood at time index `t`, **without** 
mutating a separate storage array.

# Arguments
- `data::AbstractMatrix{T}`: Actual observations, size M×T₀ (M=dimension).
- `mean::AbstractMatrix{T}`: Predicted means, also M×T₀.
- `covs::AbstractArray{<:Real}`: Covariance array of size (M, M, T₀).
- `t::Int`: Time index (1-based).

# Returns
A `Real` scalar = `logpdf_mvn(mean, covs, data, t)`.

# Note
If you do not need the entire log-likelihood vector, call this function 
for each time step. Otherwise, for a typical filter, the in-place version
`compute_loglik!` is often used inside a loop to fill a preallocated vector.
"""
function compute_loglik(data::AbstractMatrix{T}, 
                        mean::AbstractMatrix{T}, 
                        covs::AbstractArray{<:Real}, 
                        t::Int) where {T <: Real}
    @assert t <= size(data,2) && t <= size(mean,2) && t <= size(covs,3) "Time index t out of range"

    return logpdf_mvn(mean, covs, data, t)
end