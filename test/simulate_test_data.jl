using Random
using LinearAlgebra
import QuadraticKalman as QK

"""
Simulate data from a 2-dimensional quadratic state space model:

State equation:
Xₜ₊₁ = μ + Φ Xₜ + Ω εₜ₊₁

Measurement equation:
Yₜ = a + B Xₜ + ∑ᵢ₌₁ᵐ Xₜ' Cᵢ Xₜ + α Yₜ₋₁ + D ηₜ

Returns:
- X: (n×T) matrix of states
- Y: (m×T) matrix of observations
- params: NamedTuple of true parameters
"""
function simulate_test_data(; 
    T::Int=100,     # Time periods
    n::Int=2,       # State dimension
    m::Int=2,       # Observation dimension
    seed::Int=42    # Random seed
)
    Random.seed!(seed)
    
    # Generate stable transition parameters
    Phi = [0.5 0.1; 0.1 0.3]  # Make sure eigenvalues < 1
    mu = [0.1, 0.2]
    Omega = [0.3 0.0; 0.1 0.2]
    
    # Generate measurement parameters
    a = [0.0, 0.0]
    B = [1.0 0.0; 0.0 1.0]
    C = Array{Float64}(undef, n, n, m)
    C[:,:,1] = [0.2 0.1; 0.1 0.0]  # For first measurement
    C[:,:,2] = [0.0 0.1; 0.1 0.2]  # For second measurement
    D = [0.2 0.0; 0.0 0.2]
    alpha = zeros(m, m)
    
    # Simulate states
    X = zeros(n, T)
    X[:,1] = (I - Phi) \ mu  # Start from unconditional mean
    
    for t in 1:(T-1)
        shock = randn(2)
        X[:,t+1] = mu + Phi * X[:,t] + Omega * shock
    end
    
    # Simulate observations
    Y = zeros(m, T)
    for t in 1:T
        noise = randn(2)
        xt = X[:,t]
        
        # Linear terms
        Y[:,t] = a + B * xt

        if t > 1
            Y[:,t] += alpha * Y[:,t-1]
        end
        
        # Quadratic terms
        for i in 1:m
            Y[i,t] += xt' * C[:,:,i] * xt
        end
        
        # Add measurement noise
        Y[:,t] += D * noise
    end
    
    params = (
        mu=mu, Phi=Phi, Omega=Omega,
        a=a, B=B, C=C, D=D, alpha=alpha
    )
    
    return X, Y, params
end

X, Y, params = simulate_test_data()

# Save data to CSV files that both Julia and R can read
using CSV, DataFrames

# Convert state matrix to DataFrame with column names x1, x2
X_df = DataFrame(X', :auto)
rename!(X_df, [Symbol("x$i") for i in 1:size(X,1)])

# Convert measurement matrix to DataFrame with column names y1, y2 
Y_df = DataFrame(Y', :auto) 
rename!(Y_df, [Symbol("y$i") for i in 1:size(Y,1)])

# Save to CSV files
CSV.write("test/simulated_data/simulated_states.csv", X_df)
CSV.write("test/simulated_data/simulated_measurements.csv", Y_df)

# Save parameters as a JSON file for reference
using JSON
open("test/simulated_data/simulated_params.json", "w") do f
    JSON.print(f, params)
end
