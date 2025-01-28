"""
    File: selection.jl

Matrix utilities for handling **vectorized** forms of symmetric matrices and 
their **augmentations** within the Quadratic Kalman Filter or related 
linear algebra workflows.

# Overview

This file provides a set of functions to construct or manipulate selection and 
duplication matrices, as well as auxiliary transforms that preserve symmetry 
in vectorized operations:

1. **`selection_matrix(n::Int, ::Type{T}=Float64)`**  
   - Produces a `p×n²` matrix (`p = n(n+1)/2`) that extracts the lower-triangular 
     (including diagonal) elements of an `n×n` matrix when multiplied by 
     its vectorized form.

2. **`duplication_matrix(n::Int, ::Type{T}=Float64)`**  
   - Builds an `n²×p` matrix that "duplicates" the lower-triangular entries 
     to form the full vectorization of a symmetric `n×n` matrix.

3. **`vech(mat::AbstractMatrix{T})`**  
   - Computes the "half-vectorization" of a symmetric matrix, returning a 
     vector of its lower-triangular elements.

4. **`compute_H̃(N::Int, H::AbstractMatrix{T})`** and **`compute_G̃(N::Int, G::AbstractMatrix{T})`**  
   - Generate the **augmented** selection (`H̃`) and duplication (`G̃`) matrices
     used in certain smoothing or state‐space algorithms. They embed the original
     `H` or `G` in a block-structured matrix that handles both the state vector 
     and the half-vectorization of a covariance.

# Usage

These matrices are commonly employed in Kalman filter variants (including 
Quadratic/Extended Kalman Filters) where we vectorize or half-vectorize 
covariance matrices:
```julia
# e.g., building a selection matrix
S = selection_matrix(4)     # selects lower-triangular parts of a 4×4
D = duplication_matrix(4)   # reconstructs from half-vectorization
h_vec = vech(symmetric_mat) # half-vectorize a 4×4

# augmented forms for state-space transformations
H_tilde = compute_H̃(N, S)
G_tilde = compute_G̃(N, D)
```
"""

"""
    selection_matrix(n::Int, ::Type{T}=Float64) where T <: Real

Compute the selection matrix S that extracts the unique elements from a vectorized symmetric matrix.

For a symmetric n×n matrix, this returns a p×n² matrix where p = n(n+1)/2, which selects only
the lower triangular elements when applied to the vectorized matrix.

# Arguments
- `n::Int`: Dimension of the original square matrix
- `T::Type`: Numeric type for the output matrix elements (default: Float64)

# Returns
- `Matrix{T}`: p×n² selection matrix where p = n(n+1)/2
"""
function selection_matrix(n::Int, ::Type{T}=Float64) where T <: Real
    p = (n*(n+1)) ÷ 2         # number of lower-triangular entries
    S = spzeros(T, p, n^2)

    # We'll iterate over all j=1..n, i=j..n in column-major style
    # and assign a 1 in the appropriate place for each row k.
    k = 1
    for j in 1:n
        for i in j:n
            # the column-major index of A[i,j] is:
            col_index = i + (j-1)*n
            S[k, col_index] = one(T)
            k += 1
        end
    end

    return S
end

"""
    duplication_matrix(n::Int)

Compute the duplication matrix D that maps the lower triangular elements of a symmetric matrix 
to its vectorized form.

# Arguments
- `n::Int`: Dimension of the original square matrix

# Returns
- `Matrix{Int}`: n²×p duplication matrix where p = n(n+1)/2
"""
function duplication_matrix(n::Int, ::Type{T}=Float64) where T <: Real
    p = div(n*(n+1), 2) # number of lower-triangular entries
    D = spzeros(T, n^2, p) # sparse matrix for memory efficiency

    # We'll keep track of the column index c for each (u,v) with u>=v
    c = 1
    for v in 1:n
        for u in v:n   # lower triangle => u >= v
            # In "vech", (u,v) is column c.

            # We want to set D[row, c] = 1 for row = (u,v) and also row=(v,u),
            # because in vec(S), (u,v) and (v,u) are different rows if u!=v.
            #
            # row index for (i,j) in column-major is i + (j-1)*n.

            # (u,v)
            row1 = u + (v-1)*n
            D[row1, c] = 1

            # (v,u), only if v != u:
            if u != v
                row2 = v + (u-1)*n
                D[row2, c] = 1
            end

            c += 1
        end
    end
    return D
end

"""
    compute_H̃(N::Int, H::AbstractMatrix{T}) where T <: Real

Compute the augmented selection matrix H̃ used in the smoothing algorithm.

# Arguments
- `N::Int`: Dimension of the original state vector
- `H::AbstractMatrix{T}`: Selection matrix for N×N symmetric matrix

# Returns
- Matrix{T}: Augmented selection matrix H̃ of size ((N*(N+3))/2) × (N*(N+1))

# Description
Constructs the augmented selection matrix H̃ by combining:
- Top left: N×N identity matrix
- Top right: N×(N²-N) zero matrix  
- Bottom left: (N²-N)×N zero matrix
- Bottom right: Original selection matrix H

The augmented matrix H̃ is used to transform between vectorized and half-vectorized 
representations in the smoothing algorithm while preserving symmetry.
"""
function compute_H̃(N::Int, H::AbstractMatrix{T}) where T <: Real
    n_rows = Int((N * (N + 3)) / 2)
    n_cols = N * (N + 1)
    # Create the identity matrix part
    top_left = Matrix{T}(I, N, N)
    
    # Create the H part
    bottom_right = H
    
    # Create the zero matrices
    top_right = zeros(T, N, n_cols - N)
    bottom_left = zeros(T, n_rows - N, N)
    
    # Combine all parts
    H̃ = [
        top_left    top_right;
        bottom_left bottom_right
    ]
    
    return H̃
end

"""
    compute_G̃(N::Int, G::AbstractMatrix{T}) where T <: Real

Compute the augmented duplication matrix G̃ used in the smoothing algorithm.

# Arguments
- `N::Int`: Dimension of the original state vector
- `G::AbstractMatrix{T}`: Duplication matrix for N×N symmetric matrix

# Returns
- Matrix{T}: Augmented duplication matrix G̃ of size (N*(N+1)) × ((N*(N+3))/2)
"""
function compute_G̃(N::Int, G::AbstractMatrix{T}) where T <: Real
    n_rows = N * (N + 1)
    n_cols = Int((N * (N + 3)) / 2)
    # Create the identity matrix part
    top_left = Matrix{T}(I, N, N)
    
    # Create the G part
    bottom_right = G
    
    # Create the zero matrices
    top_right = zeros(T, N, n_cols - N)
    bottom_left = zeros(T, n_rows - N, N)
    
    # Combine all parts
    G̃ = [
        top_left    top_right;
        bottom_left bottom_right
    ]
    
    return G̃
end

"""
    vech(mat::AbstractMatrix{T}) where T <: Real

Compute the vectorized half-vectorization of a symmetric matrix.

# Arguments
- `mat::AbstractMatrix{T}`: Symmetric matrix

# Returns
- `Vector{T}`: Vectorized half-vectorization of the input matrix
"""
function vech(mat::AbstractMatrix{T}) where T <: Real
    n = size(mat, 1)
    if n != size(mat, 2)
        throw(ArgumentError("Input matrix must be square"))
    end
    
    if !issymmetric(mat)
        throw(ArgumentError("Input matrix must be symmetric"))
    end
    
    indices = [(i, j) for j in 1:n for i in j:n]
    
    return [mat[i, j] for (i, j) in indices]
end

export selection_matrix, duplication_matrix, vech, compute_H̃, compute_G̃