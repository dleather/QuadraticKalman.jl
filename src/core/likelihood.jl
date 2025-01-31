"""
    log_pdf_normal(mu::Real, sigma2::Real, x::Real) -> Real

Compute the log of the probability density function (PDF) of a **univariate normal** 
distribution with mean `mu` and variance `sigma2` at point `x`. 

# Arguments
- `mu::Real`: The mean of the normal distribution
- `sigma2::Real`: The variance (σ²) of the normal distribution
- `x::Real`: The point at which to evaluate the log PDF

# Returns
- A `Real` value representing the log PDF at `x`. 
  - If `sigma2 > 0`, it computes `-0.5 * log(2π * sigma2) - 0.5 * ((x-mu)^2 / sigma2)`.
  - Otherwise (if `sigma2 ≤ 0`), it returns `-Inf`, signaling effectively zero probability.

# Notes
- Returning `-Inf` for non-positive variances can help keep downstream algorithms
  (like Kalman filters) from crashing when numerical errors produce a 
  slightly negative variance. 
- If negative variances are genuinely impossible (or indicative of a deeper error), 
  you could use an assertion or throw an error instead.
"""
log_pdf_normal(mu::Real, sigma2::Real, x::Real) = 
    sigma2 > 0 ? (-0.5 * log(2π * sigma2) - 0.5*((x - mu)^2 / sigma2)) : -Inf

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
1. Extracts `sigma2 = Mpred[1,1,t]`.
2. Extracts `mu = Ypred[t]` (predicted mean).
3. Extracts `x = Y[t]` (actual observation).
4. Computes `ll = log_pdf_normal(mu, sigma2, x)`.
5. Stores `lls[t] = ll`.

If `sigma2` is negative or zero, `log_pdf_normal` may return `-Inf`. 
In a well-behaved filter, `sigma2` should be positive.
"""
function compute_loglik!(lls::AbstractVector{T}, 
                         Y::AbstractVector{T},
                         Ypred::AbstractVector{T},
                         Mpred::AbstractArray{<:Real},
                         t::Int) where {T<:Real}

    @assert size(Mpred,1) == 1 && size(Mpred,2) == 1 "Expected Mpred to be (1,1,...) for univariate"
    @assert t <= size(Mpred,3) "Index t out of range for Mpred"
    @assert t <= length(Y) && t <= length(Ypred) && t <= length(lls)

    sigma2 = Mpred[1,1,t]
    mu  = Ypred[t]
    x  = Y[t]

    # Possibly clamp small or negative σ² to a small epsilon
    eps_sigma2 = 1e-12
    if sigma2 <= eps_sigma2
        sigma2 = eps_sigma2
    end

    lls[t] = log_pdf_normal(mu, sigma2, x)
end


"""
    compute_loglik(Y::Real, Ypred::Real, Mpred::AbstractMatrix{T}) where {T<:Real}

Compute the log-likelihood for a **single** univariate observation at one time step.

# Arguments
- `Y::Real`: The actual observation value.
- `Ypred::Real`: The predicted observation (mean).
- `Mpred::AbstractMatrix{T}`: A 1×1 matrix storing the variance at this time step,
                              i.e. `Mpred[1,1] = sigma2`.

# Returns
The log-likelihood `Float64` (or `T`) computed using `log_pdf_normal(μ, σ², x)`.

# Details
1. Extracts `sigma2 = Mpred[1,1]`.
2. Uses `Ypred` as the mean `mu`.
3. Uses `Y` as the observation `x`.
4. If `sigma2` is <= 0, clamps it to a tiny positive `1e-12` to avoid domain error.
5. Calls `log_pdf_normal(mu, sigma2, x)` and returns the result.
"""
function compute_loglik(Y::Real, Ypred::T, Mpred::AbstractArray{T}) where {T<:Real}
    @assert size(Mpred,1) == 1 && size(Mpred,2) == 1 "Expected Mpred to be 1x1 for univariate"

    sigma2 = Mpred[1,1]
    mu = Ypred
    x = Y

    # minimal clamp
    eps_sigma2 = 1e-12
    if sigma2 <= eps_sigma2
        sigma2 = eps_sigma2
    end

    return log_pdf_normal(mu, sigma2, x)
end

"""
    logpdf_mvn(Y_pred::AbstractMatrix{T}, Σ_pred::AbstractArray{<:Real},
               Y::AbstractMatrix{T}, t::Int) -> Real
               where {T<:Real}

Compute the **log-probability density** of a multivariate normal (MVN) at time index `t`.

# Overview
This function interprets:
- The **predicted mean** mu as `Y_pred[:, t]` (i.e. column `t` of `Y_pred`).
- The **predicted covariance** Sigma as `Sigma_pred[:, :, t]` (the `t`-th slice in a 3D array).
- The **actual observation** x as `Y[:, t]` (column `t` of `Y`).

It then returns the log-density, ln p(x | μ, Σ) for that MVN.

# Arguments

- `Y_pred::AbstractMatrix{T}`:
  A matrix of size `(k, T_max)`, where row dimension `k` is the data dimension, 
  and column dimension is the number of time steps. We take `Y_pred[:, t]` as mu.

- `Sigma_pred::AbstractArray{<:Real}`:
  A 3D array of size `(k, k, T_max)` storing the covariance for each time step. 
  We take `Sigma_pred[:, :, t]` as the `k×k` covariance at time `t`.

- `Y::AbstractMatrix{T}`:
  A matrix of actual observations, also `(k, T_max)`, so `Y[:, t]` is the 
  observed vector at time `t`.

- `t::Int`:
  The time index (1-based).

# Returns
A real scalar `::Real` representing:
   ln N(x; mu, Sigma, t).

# Method

1. **Views** of mu, Sigma, and x are created (`@view`) to avoid copying data.
2. **Cholesky factorization** of Sigma is computed via `cholesky(Symmetric(Sigma))`. 
   - This yields an upper-triangular factor `C`.
3. We form the difference `diff = x - mu`.
4. We ise `solve` to efficiently compute, Sigma⁻¹(x - mu), without explicit inversion.
5. The log-determinant ln| Sigma | is `2 * sum(log, diag(C.U))`.
6. We add up the standard MVN log-pdf terms:
     -frac{1}{2} (ln|Sigma| + (x-mu)'Sigma⁻¹(x-mu)) + constant terms.
   Here the constant includes -0.5 * k * (ln 2π + 1) which is a variation of `-0.5 * (k*ln(2π) + k + ...)`.
7. Returns the final log-pdf as a `Real`.

# Notes
- **AD-Friendliness**: Zygote and other AD tools generally handle 
  `cholesky(Symmetric(...))` with built-in rules, though reverse-mode AD 
  on a large `k` can be expensive. 
- **Numerical Issues**: If Σ is not positive-definite, `cholesky` 
  will error. In a robust filter, you might ensure PSD by correction steps.

"""
function logpdf_mvn(Y_pred::AbstractArray{T1}, Sigma_pred::AbstractArray{T1},
                    Y::AbstractArray{T}, t::Int) where {T <: Real, T1 <: Real}

    # 1) Take views for mu, Sigma, x
    mu = @view Y_pred[:, t]
    Sigma = @view Sigma_pred[:, :, t]
    x = @view Y[:, t + 1]
    
    k = length(mu)

    # 2) Cholesky factorization
    C = cholesky(Symmetric(Sigma))

    # 3) Form difference
    diff = x .- mu

    # 4) Solve for Σ^-1 diff via factor
    solved = C \ diff

    # 5) log(det(Σ)) from the factor
    log_det_Sigma = 2 * sum(log, diag(C.U))

    # 6) Constant term for MVN
    const_term = -0.5 * k * log(2π)

    # 7) Compute the final log-pdf
    logpdf = const_term - 0.5 * (log_det_Sigma + dot(diff, solved))

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
function compute_loglik(data::AbstractArray{T}, 
                        mean::AbstractArray{T1}, 
                        covs::AbstractArray{T1}, 
                        t::Int) where {T <: Real, T1 <: Real}
    @assert t <= size(data,2) && t <= size(mean,2) && t <= size(covs,3) "Time index t out of range"

    return logpdf_mvn(mean, covs, data, t)
end


"""
    qkf_negloglik(params::Vector{T}, data::QKData, N::Int, M::Int) where T<:Real -> Real


Compute the negative log-likelihood for a Quadratic Kalman Filter model given parameters and data.

# Arguments
- `params::Vector{T}`: Vector of model parameters to be converted into a QKModel
- `data::QKData`: Data container with observations and optional initial conditions
- `N::Int`: Dimension of the state vector
- `M::Int`: Dimension of the measurement vector

# Returns
The negative log-likelihood value computed by:
1. Converting parameters to a QKModel
2. Running the Kalman filter
3. Taking the negative sum of the per-period log-likelihoods

# Note
This function is typically used as an objective function for maximum likelihood estimation,
where the goal is to minimize the negative log-likelihood.

For optimization, you may want to wrap this function with N, M and data specified:

    `negloglik(params) = qkf_negloglik(params, data, N, M)`

"""
function qkf_negloglik(params::AbstractVector{T}, data::QKData, N::Int, M::Int) where T<:Real
    model = params_to_model(params, N, M)
    result = qkf_filter(data, model)
    return -sum(result.ll_t)
end