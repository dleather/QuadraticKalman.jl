# matrix_utils/numerical.jl

"""
    spectral_radius(A::AbstractMatrix{T}, num_iterations::Int=100) where T <: Real

Compute the spectral radius (largest absolute eigenvalue) of a matrix using the power iteration method.

# Arguments
- `A::AbstractMatrix{T}`: Square matrix to compute spectral radius for
- `num_iterations::Int=100`: Number of power iterations to perform

# Returns
-  Estimate of the spectral radius

# Description
Uses the power iteration method to estimate the spectral radius by repeatedly multiplying
a random normalized vector by the matrix. The method converges to the eigenvector 
corresponding to the largest eigenvalue in absolute value. The spectral radius is then
computed as the Rayleigh quotient of this eigenvector.

Note that this method may not converge if the matrix has multiple eigenvalues of the same
magnitude or if the starting vector is orthogonal to the dominant eigenvector.
"""
function spectral_radius(
    A::AbstractMatrix{T}; num_iters::Int=100, rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real

    # 1) Pick a random initial vector of length `size(A,1)`,
    #    then normalize.
    n = size(A,1)
    @assert size(A,2) == n "Matrix must be square"
    v = randn(rng, T, n)
    v ./= norm(v)
    
    # 2) Power iteration: repeatedly multiply, then normalize
    for _ in 1:num_iters
        Av = A * v
        v  = Av / norm(Av)
    end
    
    # 3) Final spectral radius estimate => norm(A * v)
    Av = A * v
    return norm(Av)
end

"""
    smooth_max(x::Real; threshold::Real=1e-8) -> Real

Compute a **smooth, differentiable approximation** to `max(x, threshold)`, 
avoiding the sharp corner at `x = threshold`.

# Arguments
- `x::Real`: The input value to clamp.
- `threshold::Real=1e-8`: The lower bound being enforced (default is 1e-8).

# Returns
A real scalar close to `max(x, threshold)`, but with a smooth transition 
near `x = threshold`. This helps maintain differentiability in 
gradient-based or automatic differentiation contexts.

# Details
This function uses the formula:
```math
smooth_max(x, t) = ( (x + t) + √((x - t)² + √(ε)) ) / 2
```
where `ε` is a small constant (default is `1e-8`).
"""
function smooth_max(x::T; threshold::Float64=1e-8) where T <: Real
    return ((x + threshold) + sqrt((x - threshold)^2 + sqrt(eps(Float64)))) / 2.0
end

"""
    d_eigen(A::AbstractMatrix{T}; assume_symmetric::Bool=false) 
    where {T<:Real}

Compute the eigenvalues and eigenvectors of a real matrix `A` using 
`DifferentiableEigen.eigen(A)`. 

# Arguments
- `A::AbstractMatrix{T}`: A real square matrix.
- `assume_symmetric::Bool=false`: If `true`, the function will internally use
  `Symmetric(A)` to signal that `A` is real-symmetric. This can improve stability
  or performance if `A` is indeed symmetric.

# Returns
A tuple `(vals, vecs)` where:
- `vals::Vector{T}`: The extracted real eigenvalues, skipping every other entry 
  based on `DifferentiableEigen` output format.
- `vecs::Matrix{T}`: The corresponding eigenvectors reshaped into `N×N`, 
  where `N = length(vals)`.

# Details
`DifferentiableEigen.eigen(A)` may return eigenvalues and vectors in an interleaved
format (e.g., real/imag parts). We keep only the real parts `vals[1:2:end]` and 
the matching rows from the returned vector array. If `A` is truly symmetric, 
you can set `assume_symmetric=true` to help the solver handle it as `Symmetric(A)`.

# Notes on AD
This function is typically used once to get a differentiable eigen decomposition.
Skipping "every other" value is piecewise but still valid as long as 
`DifferentiableEigen.eigen` is AD-friendly.
"""
function d_eigen(A::AbstractMatrix{T}; assume_symmetric::Bool=false) where {T<:Real}
    # Optionally treat A as a real-symmetric matrix for eigen decomposition
    local A_eff
    if assume_symmetric
        A_eff = Symmetric(A)
    else
        A_eff = A
    end

    # Use the differentiable eigen solver
    vals, vecs = DifferentiableEigen.eigen(A_eff)
    N = Int(length(vals) / 2)

    # For real-valued portion, skip every other entry
    real_vals = vals[1:2:end]
    # Likewise for vecs
    tmp_vecs   = vecs[1:2:end] 
    out_vecs   = reshape(tmp_vecs, N, N)

    return real_vals, out_vecs
end


"""
    make_positive_definite(A::AbstractMatrix{T}; clamp_threshold::Real=1e-8) 
    where {T<:Real}

Produce a **positive-definite** approximation of `A` by:
1. Symmetrizing `A`: `A_tmp = (A + A') / 2`
2. Computing eigenvalues/eigenvectors (via `d_eigen(..., assume_symmetric=true)`)
3. Clamping eigenvalues with a smooth function `smooth_max(eig_val, clamp_threshold)`
4. Reconstructing the matrix and symmetrizing again.

# Arguments
- `A::AbstractMatrix{T}`: A real square matrix (often close to symmetric).
- `clamp_threshold::Real=1e-8`: The lower bound used by `smooth_max` for eigenvalues.

# Returns
- `Matrix{T}`: A symmetric, positive-definite matrix derived from `A`.

# Details
- The function calls `smooth_max.(eig_vals, clamp_threshold)`, ensuring no 
  eigenvalue lies below `clamp_threshold`. 
- This "soft" clamp is **AD-friendly** because `smooth_max` is differentiable 
  near zero, unlike a hard `max(eig_vals, 0)`.

Use this to fix small negative eigenvalues in covariance matrices or 
other symmetric matrices that must be PSD (positive semi-definite) 
or PD (positive definite).

"""
function make_positive_definite(A::AbstractMatrix{T}; clamp_threshold::Real=1e-8) where T <: Real
    # 1) Symmetrize
    A_tmp = (A + A') / 2.0

    # 2) Eigen-decomp (assuming real-symmetric)
    eig_vals, eig_vecs = d_eigen(A_tmp; assume_symmetric=true)

    # 3) Smooth clamp each eigenvalue
    corrected_eigvals = smooth_max.(eig_vals; threshold = clamp_threshold)

    # 4) Reconstruct and symmetrize
    corrected_A_tmp = eig_vecs * Diagonal(corrected_eigvals) * eig_vecs'
    corrected_A     = (corrected_A_tmp + corrected_A_tmp') / 2.0

    return corrected_A
end

"""
    make_symmetric(A::Matrix{T}; wrap_in_Symmetric::Bool=false) where {T<:Real}

Return a symmetric version of matrix `A` by averaging it with its transpose. 
Optionally wrap it in a `Symmetric(...)` type.

# Arguments
- `A::Matrix{T}`: Real matrix.
- `wrap_in_Symmetric::Bool=false`: If true, returns `Symmetric((A + A')/2)` instead 
  of a plain matrix.

# Returns
Either a plain `Matrix{T}` or a `Symmetric{T}` wrapper (depending on the boolean),
with elements `(A_ij + A_ji)/2`.

# Description
This function discards any skew-symmetric part of `A`. If `wrap_in_Symmetric` is 
true, the result is a `Symmetric` wrapper type, which some linear algebra routines 
can leverage for efficiency.

"""
function make_symmetric(A::Matrix{T}; wrap_in_Symmetric::Bool=false) where T<:Real
    A_sym = (A + A') / 2
    return wrap_in_Symmetric ? Symmetric(A_sym) : A_sym
end

"""
    ensure_positive_definite(A::AbstractMatrix{T1}; shift::Real=1.5e-8) 
    where {T1<:Real}

Create a symmetric, diagonally-shifted version of `A` to guarantee 
positive definiteness: `(A + A')/2 + shift*I`.

# Arguments
- `A::AbstractMatrix{T1}`: Real square matrix.
- `shift::Real=1.5e-8`: Small positive value added to the diagonal.

# Returns
- `Matrix{T1}`: A symmetric matrix guaranteed to have eigenvalues ≥ `shift`.

# Details
1. First symmetrize `A` via `(A + A')/2.0`.
2. Then add a diagonal `shift * I`. 
3. This ensures all eigenvalues are at least `shift`, provided `A` isn't extremely 
   large or indefinite. It's simpler but less minimal than individually clamping 
   eigenvalues (see `make_positive_definite`).

Often used in numerical filters (e.g. Kalman) or optimizers to avoid 
non-invertible covariance matrices. It's a straightforward "always safe" fix.

"""
function ensure_positive_definite(A::AbstractMatrix{T1}; shift::Real=1.5e-8) where T1<:Real
    return (A + A')/2.0 + shift*I
end

"""
    issymmetric(mat::AbstractMatrix{T}) where T <: Real

Check if a matrix is approximately symmetric by comparing it with its transpose.

# Arguments
- `mat::AbstractMatrix{T}`: Matrix to check for symmetry

# Returns
- `Bool`: `true` if the matrix is approximately symmetric, `false` otherwise
"""
function issymmetric(mat::AbstractMatrix{T}) where T <: Real
    return all(mat .≈ mat')
end

#export spectral_radius, make_positive_definite, make_symmetric, ensure_positive_definite,
#    issymmetric