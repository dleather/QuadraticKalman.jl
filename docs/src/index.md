# QuadraticKalman.jl

A Julia package implementing the quadratic Kalman filter and smoother.

## Installation

```julia
using Pkg
Pkg.add("QuadraticKalman")
```

## Overview
QuadraticKalman.jl implements Kalman filtering and smoothing for state-space models with quadratic measurement equations. It extends standard implementations by handling autoregressive measurement components, performs gradient and Hessian computations with automatic differentiation, and provides efficient parameter-model conversions with high numerical stability.

## Quick Start
```julia
using QuadraticKalman, Random, Plots, LinearAlgebra, ForwardDiff, Plots
Random.seed!(2314)

# Define model dimensions
N = 2  # Number of states
M = 2  # Number of measurements

# Generate state-space parameters
μ = [0.1, 0.2]                  # State drift vector
Φ = [0.5 0.1; 0.1 0.3]          # State transition matrix
Σ = [0.6 0.15; 0.15 0.4]        # State noise covariance matrix
Ω = cholesky(Σ).L              # Scale for state noise

# Define measurement parameters
A = [0.0, 0.0]                # Measurement drift vector
B = [1.0 0.0; 0.0 1.0]        # Measurement matrix
C = [
    [0.2 0.1; 0.1 0.0],        # Quadratic effect for first measurement
    [0.0 0.1; 0.1 0.2]         # Quadratic effect for second measurement
]
V = [0.2 0.0; 0.0 0.2]        # Measurement noise covariance matrix
D = cholesky(V).L             # Scale for measurement noise
α = zeros(M, M)               # Autoregressive measurement matrix

# For demonstration purposes, simulate dummy measurements
Y = randn(M, 100)  # Replace with actual or simulated data

# Create the model and data objects
model = QKModel(N, M, μ, Φ, Ω, A, B, C, D, α)
data = QKData(Y)

# Run the Kalman filter and smoother
results_filter = qkf_filter(data, model)
results_smoother = qkf_smoother(results_filter, model)

# Compute negative log-likelihood and gradients
nll(params) = qkf_negloglik(params, data, N, M)
grad = ForwardDiff.gradient(nll, params)
hess = ForwardDiff.hessian(nll, params)

# Plot results
p1 = plot(kalman_filter_truth_plot(Y, results_filter))
p2 = plot(kalman_smoother_truth_plot(Y, results_smoother))
p3 = plot(kalman_filter_plot(results_filter))

p4 = plot(kalman_smoother_plot(results_smoother))
plot(p1, p2, p3, p4, layout=(2,2), size=(800,600))
```

## License

MIT License
Copyright (c) 2024 David Leather

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.