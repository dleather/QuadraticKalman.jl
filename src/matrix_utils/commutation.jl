"""
    File: commnutations.jl

Utilities for creating unit vectors and commutation matrices (Λₙ) used in
the Quadratic Kalman Filter and other linear‐algebraic operations.

# Overview

This file provides functions that generate:

1. **Unit vectors** (`get_e`, `compute_e`)  
   - `get_e(k, M)` produces a length‐M vector with a `1` in position `k` and `0`s elsewhere.  
   - `compute_e(M)` returns a collection of M such unit vectors, often used for block partitioning or matrix construction.

2. **Commutation matrices** (`compute_Λ`)  
   - The commutation matrix Λₙ is an `n²×n²` matrix that rearranges or 'commutes' the vectorized elements of an `n×n` matrix.  
   - We provide two versions:  
     - `compute_Λ(N)` constructs Λₙ from scratch via Kronecker products of the unit vectors.  
     - `compute_Λ(e, N)` uses a precomputed vector of unit vectors for efficiency.  

# Purpose in the QKF
In the Quadratic Kalman Filter, these utilities help handle vectorized 
covariance matrices and augmentations involving Kronecker products. 
They’re especially useful when forming block partitions or reorganizing 
elements of big matrices.

# Example Usage
```julia
# Generate a unit vector of length 5 with a 1 at position 3
v = get_e(3, 5)

# Produce a vector of 5 unit vectors
all_e = compute_e(5)

# Build commutation matrix Λ₄ of size 16×16
Λ = compute_Λ(4)

# Build commutation matrix Λ₄ using precomputed unit vectors
e = compute_e(4)
Λ2 = compute_Λ(e, 4)
```
"""

"""
    get_e(k::Int, M::Int, ::Type{T}=Float64) where T <: Real

Create a unit vector of length M with a 1 in position k and 0s elsewhere.

# Arguments
- `k::Int`: Position of the 1 entry (1 ≤ k ≤ M)
- `M::Int`: Length of the vector
- `T::Type`: Numeric type for the vector elements (default: Float64)

# Returns
- Vector{T}: Unit vector of length M with 1 at position k
"""
function get_e(k::Int, M::Int, ::Type{T}=Float64) where T <: Real
    @assert M >= 1 "M must be positive"
    @assert 1 <= k <= M "k must be between 1 and M"
    
    return [i == k ? one(T) : zero(T) for i in 1:M]
end

"""
    compute_e(M::Int, ::Type{T}=Float64) where T <: Real

Create a vector of M unit vectors, where each unit vector has a 1 in a different position.

# Arguments
- `M::Int`: Length of each unit vector and number of vectors to create
- `T::Type`: Numeric type for the vector elements (default: Float64)

# Returns
- Vector{Vector{T}}: Vector of M unit vectors, where the kth vector has a 1 in position k

# Description
Creates a vector of column selection vectors, where each vector is of size M and has all 
components equal to 0 except for one component which equals 1. The kth vector has the 1 
in position k.
"""
function compute_e(M::Int, ::Type{T}=Float64) where T <: Real
    @assert M >= 1 "M must be positive"
    return [get_e(k, M, T) for k in 1:M]
end

"""
    compute_Λ(N::Int, ::Type{T}=Float64) where T <: Real

Compute the commutation matrix Λₙ of size n² × n².

# Arguments
- `N::Int`: Dimension of the original matrix
- `T::Type`: Numeric type for the matrix elements (default: Float64)

# Returns
- Matrix{T}: The commutation matrix Λₙ

# Description
Computes Λₙ, which is an n² × n² matrix partitioned into n × n blocks, where the (i,j)th block 
is eᵢeⱼ' (outer product of unit vectors). This matrix is used in the quadratic Kalman filter
to handle vectorized covariance matrices.

The matrix can be constructed either using explicit block assignments or using Kronecker products.
This implementation uses the more efficient Kronecker product approach.
"""
function compute_Λ(N::Int, ::Type{T}=Float64) where T <: Real
    # Get the unit vectors e₁,...,eₙ
    e = compute_e(N, T)
    
    # Construct N×N blocks where block (i,j) is eᵢeⱼ'
    # Each block is N×N, resulting in an N²×N² matrix when combined
    Λ = [ Matrix(kron(e[i], e[j])') for i in 1:N, j in 1:N ]
    
    # Vertically concatenate all blocks into final N²×N² commutation matrix
    return reduce(vcat, Λ)
end

"""
    compute_Λ(e::AbstractVector{AbstractVector{T}}, N::Int) where T <: Real

Compute the commutation matrix Λₙ of size N² × N² using precomputed unit vectors.

# Arguments
- `e::AbstractVector{AbstractVector{T}}`: Vector of unit vectors eᵢ
- `N::Int`: Dimension of the original matrix

# Returns
- `Matrix{T}`: The N² × N² commutation matrix Λₙ

# Description
Constructs Λₙ, which is an N² × N² matrix partitioned into N × N blocks, where the (i,j)th block 
is eᵢeⱼ' (outer product of unit vectors). This version uses precomputed unit vectors for efficiency.

The matrix is constructed by explicitly filling each N × N block using the outer products of the
appropriate unit vectors.
"""
function compute_Λ(e::AbstractVector{AbstractVector{T}}, N::Int) where T <: Real
    # Initialize N² × N² matrix of zeros
    Λ = zeros(T, N^2, N^2)
    
    # Fill each N × N block
    for i in 1:N
        for j in 1:N
            # Calculate block indices
            i0 = (i-1) * N + 1  # Start row of block (i,j)
            i1 = i * N          # End row of block (i,j)
            j0 = (j-1) * N + 1  # Start column of block (i,j)
            j1 = j * N          # End column of block (i,j)
            
            # Set block (i,j) to outer product eⱼeᵢ'
            Λ[i0:i1, j0:j1] .= e[j] * e[i]'
        end
    end
    
    return Λ
end

export compute_Λ, compute_e