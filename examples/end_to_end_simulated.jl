using QuadraticKalman
using Random, LinearAlgebra, Statistics, Plots


# Step 1: Set Parameters
N = 2                       # Number of states
M = 2                       # Number of measurements
T = 100                     # Number of time periods to simulate
seed = 2314                 # Random seed
Random.seed!(seed)


# Generate stable state transition parameters
# X_{t+1} = mu +Phi * X_t + Omega * epsilon_t
Phi = [0.5 0.1; 0.1 0.3]    # Autoregressive matrix
mu = [0.1, 0.2]             # State drift vector


Sigma = [0.6 0.15; 0.15 0.4]  # State noise covariance matrix
Omega = cholesky(Sigma).L

# Generate measurement parameters
# Y_t = a + B * X_t + alpha * Y_{t-1} + \sum_{i=1}^m e_i * X_t' * C_i * X_t + D * epsilon_t 
A = [0.0, 0.0]              # Measurement drift vector
B = [1.0 0.0; 0.0 1.0]      # Measurement state matrix
C = [[0.2 0.1; 0.1 0.0],    # First measurement quadratic matrix
     [0.0 0.1; 0.1 0.2]]    # Second measurement quadratic matrix
V = [0.2 0.0; 0.0 0.2]      # Measurement noise covariance matrix
D = cholesky(V).L
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

# Step 7: Plot Results
plot(kalman_filter_truth_plot(X, results))
plot(kalman_smoother_truth_plot(X, results_smoother))
plot(kalman_filter_plot(results))
plot(kalman_smoother_plot(results_smoother))

# Step 8: Convert model to params
params = QuadraticKalman.model_to_params(model)

# Step 9: Convert params to model
model_from_params = params_to_model(params, N, M)

# Step 10: use loglik helper function
negloglik = qkf_negloglik(params, data, N, M)

# Test automatic differention
nll(params) = qkf_negloglik(params, data, N, M)
using ForwardDiff
using FiniteDiff

# Test that we can compute gradients
grad = ForwardDiff.gradient(nll, params)
grad_fd = FiniteDiff.finite_difference_gradient(nll, params)
abs_diff = maximum(abs.(grad - grad_fd))
rel_diff = norm(grad - grad_fd) / (norm(grad) + eps())
println("Maximum absolute difference: ", abs_diff)
println("Maximum relative difference: ", rel_diff)

# Test that we can compute Hessians 
hess = ForwardDiff.hessian(nll, params) 
println("Hessian condition number: ", cond(hess))



