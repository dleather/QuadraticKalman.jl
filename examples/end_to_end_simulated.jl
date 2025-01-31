using QuadraticKalman
using Random, LinearAlgebra, Statistics


# Step 1: Set Parameters
N = 2               # Number of states
M = 2               # Number of measurements
T = 100             # Number of time periods to simulate
seed = 2314         # Random seed
Random.seed!(seed)


# Generate stable state transition parameters
# X_{t+1} = mu +Phi * X_t + Omega * epsilon_t
Phi = [0.5 0.1; 0.1 0.3]    # Autoregressive matrix
mu = [0.1, 0.2]             # State drift vector


Omega = [0.3 0.0; 0.1 0.2]  # State noise covariance matrix

# Generate measurement parameters
# Y_t = a + B * X_t + alpha * Y_{t-1} + \sum_{i=1}^m e_i * X_t' * C_i * X_t + D * epsilon_t 
A = [0.0, 0.0]              # Measurement drift vector
B = [1.0 0.0; 0.0 1.0]      # Measurement state matrix
C = [[0.2 0.1; 0.1 0.0],    # First measurement quadratic matrix
     [0.0 0.1; 0.1 0.2]]    # Second measurement quadratic matrix
D = [0.2 0.0; 0.0 0.2]      # Measurement noise covariance matrix
alpha = zeros(M, M)         # Measurement autoregressive matrix

# Step 2: Simulate states
X = zeros(N, T)
X[:,1] = (I - Phi) \ mu  # Start from unconditional mean
for t in 1:(T-1)
    shock = randn(N)
    X[:,t+1] = mu + Phi * X[:,t] + Omega * shock
end

# Simulate observations
Y = zeros(M, T)
for t in 1:T
    noise = randn(M)

    xt = X[:,t]
    
    # Linear terms
    Y[:,t] = A + B * xt

    if t > 1
        Y[:,t] += alpha * Y[:,t-1]
    end
    
    # Quadratic terms
    for i in 1:M
        Y[i,t] += xt' * C[i] * xt
    end

    # Add measurement noise
    Y[:,t] += D * noise
end

    
# Step 3: Define Model Parameters
model = QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha)

# Step 4: Create Data Structure
data = QKData(Y)

# Step 5: Run the Filter and Smoother
results = qkf_filter(data, model)
results_smoother = qkf_smoother(results, model)

# Step 6: Analyze Results
println("Filter Log-Likelihoods: ", sum(results.ll_t))
# Compute MSE between smoothed state estimates and true states
X_smooth = results_smoother.Z_smooth[1:N, :]
function l2_norm(x, y)
    return sqrt(sum(abs2, x .- y))
end
rmse_states = sqrt(mean([l2_norm(X_smooth[:, t], X[:, t]).^2 for t in 1:T]))
println("\nRMSE of smoothed state estimates: ", rmse_states)

# Plot smoothed states
using Plots
# Extract smoothed states and their variances
X_smooth = results_smoother.Z_smooth[1:N, :]
P_smooth = results_smoother.P_smooth[1:N, 1:N, :]

# Create time vector
t = 1:T

# Initialize plot
# Initialize plot
p = plot(layout=(N,1), size=(1000, 350*N),  # Increased vertical size
        legend=:bottom,                     # Move legend to bottom
        legend_columns=3,                  # Arrange legend items horizontally
        legendfontsize=8,                  # Smaller legend text
        palette=:Paired,
        dpi=300,
        titlefont=10,
        guidefont=9,
        tickfont=8,
        grid=true,
        gridalpha=0.3)               # Add space below each subplot

# Plot each state dimension
for i in 1:N
    ci = 1.96 * sqrt.(reshape(P_smooth[i,i,:], :))
    
    plot!(p[i], t, X[i,:], 
        label="True State $i", 
        color=:black, 
        linewidth=2,
        linestyle=:solid,
        subplot=i)

    plot!(p[i], t, X_smooth[i,:], 
        label="Kalman Smoothed", 
        color=:dodgerblue,
        linewidth=1.5,
        linestyle=:dash,
        subplot=i)
        
    plot!(p[i], t, X_smooth[i,:] + ci, 
        fillrange=X_smooth[i,:] - ci,
        fillalpha=0.2, 
        label="95% CI",
        color=:dodgerblue,
        linealpha=0,
        subplot=i)

    # Formatting adjustments
    title!(p[i], "State $i: Estimation vs Truth")
    xlabel!(p[i], "Time Period")
    ylabel!(p[i], "State Value")
    ylims!(p[i], (minimum(X[i,:])-0.5, maximum(X[i,:])+0.5))
    
    # Position legend below plot area
    plot!(p[i], legend=:bottom, legendfont=8, legendspacing=2)
end

# Add overall title
plot!(p, plot_title="Kalman Smoother Performance", titlefont=12)







