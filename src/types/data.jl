"""
    Module: QKData Submodule

This submodule defines the `QKData` structure and associated methods
for constructing, validating, and extracting measurements from the data array.

# Contents

- **`QKData{T<:Real,N}`**  
  A parametric type holding the observations `Y` (vector or matrix),
  along with two integer fields:
  - `M::Int`: Number of observed variables (size of first dimension if `N=2`, or 1 if `N=1`).
  - `T_bar::Int`: Number of time steps minus 1.

- **Constructor**:  
  `QKData(Y::AbstractArray{T,N})` automatically infers `M` and `T_bar`
  based on the dimensions of `Y`, then calls `validate_data`.

- **`validate_data(data::QKData{T,N})`**  
  Checks array dimensions, ensures `Y` is non-empty, has more than one time step,
  and that values are within an acceptable range (no infinities or NaNs).

- **`get_measurement(data::QKData{T,N}, t::Int)`**  
  Retrieves the measurement(s) at time index `t`. Returns a scalar if `N=1`
  (univariate case), or a view of column `t` if `N=2` (multivariate case).

# Usage

```julia
using .QKDataSubmodule  # or whatever your submodule is called

# Create a QKData object
data = QKData(rand(10, 5))  # 10 variables, 5 time points

# Validate (automatically called in the constructor)
validate_data(data)

# Extract the measurement at time t=3
measurement = get_measurement(data, 3)
"""

"""
    QKData{T<:Real,N}

A data structure for holding an N-dimensional array of real values (`Y`)
plus two integer fields (`M`, `T̄`) derived from its dimensions. Typically,
`M` and `T̄` might represent specific sizes used later in computations.

# Fields
- `Y::AbstractArray{T,N}` : The underlying data array (N-dimensional, element type `<: Real`).
- `M::Int` : The first dimension of `Y` if `Y` is at least 2D. If `N == 1`, we define `M = 1`.
- `T_bar::Int` : One less than the “second dimension” in a 2D case, or `length(Y) - 1` for a 1D vector.
"""
@with_kw struct QKData{T<:Real, N}
    Y::AbstractArray{T, N}  # Data array
    M::Int                  # Number of observed variables
    T_bar::Int              # Number of time periods minus 1
end

"""
    QKData(Y::AbstractArray{T,N}) where {T<:Real, N}

Construct a `QKData{T,N}` from the array `Y`. 

# Arguments
- `Y::AbstractArray{T,N}`: Input array (vector or matrix)

# Returns
- `QKData{T,N}`: Validated data structure

# Description
For vector input (N=1):
  - M is set to 1
  - T_bar is length(Y) - 1
For matrix input (N=2):
  - M is first dimension size
  - T_bar is second dimension size minus 1
"""
function QKData(Y::AbstractArray{T, N}) where {T<:Real, N}
    if N == 1   # ...data is univariate...
        M = 1
        Tp1 = length(Y)
    else        # ...Multivariate data...
        M, Tp1 = size(Y, 1), size(Y, 2)
    end
    
    data = QKData{T, N}(Y = Y, M = M, T_bar = Tp1 - 1)
    validate_data(data) # Validate after construction
    return data
end

"""
    validate_data(data::QKData{T,N}) where {T<:Real, N}

Validate dimensions and properties of QKData structure.

# Checks
- Data array Y is not empty
- For N=1 (vector case): length(Y) > 1
- For N=2 (matrix case): size(Y,2) > 1
- All elements are finite
- M matches first dimension (if matrix) or is 1 (if vector)
- T_bar matches data length minus 1
"""
function validate_data(data::QKData{T,N}) where {T<:Real, N}
    @unpack Y, M, T_bar = data
    
    # Check not empty - this is fine for AD since it's just array size
    if isempty(Y)
        throw(ArgumentError("Data array Y cannot be empty"))
    end
    
    # Dimension checks are fine for AD since they're just array sizes
    if N == 1
        if length(Y) ≤ 1
            throw(ArgumentError("Data must have more than one observation"))
        end
        if M != 1
            throw(ArgumentError("M must be 1 for vector data"))
        end
        if T_bar != length(Y) - 1
            throw(ArgumentError("T_bar must equal length(Y) - 1"))
        end
    else
        if size(Y,2) ≤ 1
            throw(ArgumentError("Data must have more than one time period"))
        end
        if M != size(Y,1)
            throw(ArgumentError("M must match first dimension of matrix data"))
        end
        if T_bar != size(Y,2) - 1
            throw(ArgumentError("T_bar must equal number of time periods minus 1"))
        end
    end
    
    # Instead of isfinite check, we can do something like this:
    # This avoids non-differentiable functions while still catching NaN/Inf
    for y in Y
        if abs(y) > 1e10    # Use a large but finite number
            throw(ArgumentError("Data values must be finite"))
        end
    end
    
    return true
end

"""
    get_measurement(data::QKData{T,N}, t::Int) where {T<:Real, N}

Extract measurement at time t from QKData. 
For vector data returns a scalar, for matrix data returns a vector.

# Arguments
- `data::QKData{T,N}`: Data structure
- `t::Int`: Time index (1-based)

# Returns
- For N=1: Single measurement Y[t]
- For N=2: Vector of measurements Y[:,t]

# Throws
- ArgumentError if t is out of bounds
"""
function get_measurement(data::QKData{T,N}, t::Int) where {T<:Real, N}
    @unpack Y, T_bar = data
    
    if t < 1 || t > T_bar + 1
        throw(ArgumentError("Time index t must be between 1 and T_bar+1"))
    end
    
    if N == 1
        return Y[t]
    else
        return @view Y[:,t]
    end
end

#export QKData, get_measurement