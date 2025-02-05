# QuadraticKalman.jl


[![Build
Status](https://github.com/dleather/QuadraticKalman.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dleather/QuadraticKalman.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/dleather/QuadraticKalman.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dleather/QuadraticKalman.jl)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://dleather.github.io/QuadraticKalman.jl/)

A Julia package implementing Kalman filtering and smoothing for
state-space models with quadratic measurement equations, based on the
methodology developed in [Monfort et. al. (2013, Journal of
Econometrics)](https://www.sciencedirect.com/science/article/abs/pii/S0304407615000123).

The package implements filtering and smoothing for the following
state-space model with Gaussian noise:

State equation:

$X_{t} = \mu + \Phi X_{t-1} + \Omega \epsilon_t$, where
$\epsilon_t \sim \mathcal{N}(0, I).$

Measurement equation:

$Y_t = A + B X_t + \alpha Y_{t-1} + \sum_{i=1}^M e_i X_t^\prime C_i X_t + D \eta_t$,
where $\eta_t \sim \mathcal{N}(0, I)$ and $e_i$ is the basis vector.

![Quadratic Kalman Smoother Results](smoother_example.png)

## 📖 Documentation

👉 **[Read the Docs](https://dleather.github.io/QuadraticKalman.jl/)**

Check out the latest documentation for installation, API reference, and
usage examples.

## Features

- Kalman filtering and smoothing for quadratic state-space models.
- Extends original implementation by allowing for autoregressive
  measurement equations.
- Gradiant and hessian of negative log-likelihood computed using
  automatic differentiation using ForwardDiff.jl.
- Visualization tools for filtered and smoothed states.
- Efficient parameter-model conversion for optimization.
- Automatically reparametrizes model parameters to ensure
  positive-definiteness of covariance matrices in an unconstrained
  parameter space.
- Benchmarking results show an improvement in speed of 9.7x - 62x faster than R code depending on the dimensionality of the problem.
- Numerically stable implementation.
- TODO: Add support for state-dependent measurement noise.

## Installation

``` julia
using Pkg
Pkg.add("QuadraticKalman")
```

## Quick Start

``` julia
using QuadraticKalman, Random, Plots, LinearAlgebra
Random.seed!(2314)

# Define model parameters
N = 2  # Number of states
M = 2  # Number of measurements

# Generate stable state transition parameters
μ = [0.1, 0.2]                 # N x 1 vector
Φ = [0.5 0.1; 0.1 0.3]         # N x N matrix
Σ = [0.6 0.15; 0.15 0.4]       # N x N matrix
Ω = cholesky(Σ).L     

# Generate measurement parameters
A = [0.0, 0.0]                  # M x 1 vector
B = [1.0 0.0; 0.0 1.0]          # M x N matrix
C = [[0.2 0.1; 0.1 0.0],        # M x 1 vector of N x N matrices
     [0.0 0.1; 0.1 0.2]]    
V = [0.2 0.0; 0.0 0.2]          # M x M matrix
D = cholesky(V).L
α = zeros(M, M)                 # M x M matrix

# Simulate data (see example in Documentation)
# X, Y = ... simulation code ... # X is N x T and Y is M x T

# Create model and run filter/smoother
model = QKModel(N, M, μ, Φ, Ω, A, B, C, D, α)
data = QKData(Y)
results_filter = qkf_filter(data, model)
results_smoother = qkf_smoother(results_filter, model)

# Visualize smoothed states
p = plot(kalman_smoother_truth_plot(X, results_smoother))
savefig(p, "smoother_example.png")  # Save plot for README

# Parameter estimation example
params = model_to_params(model)
nll(p) = qkf_negloglik(p, data, N, M)
grad = ForwardDiff.gradient(nll, params)
hess = ForwardDiff.hessian(nll, params)
```
## Benchmarks
| T | N | M | Julia Median Time | Julia Min Time | R Median Time | R Min Time |
|-------|---|---|-------------------|----------------|---------------|------------|
| 10    | 1 | 1 | 0.13105          | 0.1271         | 7.74455       | 7.1367     |
| 100   | 1 | 1 | 1.249            | 1.1979         | 84.9028       | 80.7364    |
| 1000  | 1 | 1 | 13.7519          | 12.8033        | 851.7696      | 837.7759   |
| 10    | 2 | 2 | 0.8839           | 0.7449         | 8.57615       | 7.8857     |
| 100   | 2 | 2 | 9.08795          | 8.1247         | 94.19605      | 89.3201    |
| 1000  | 2 | 2 | 94.4929          | 87.5616        | 1013.1734     | 926.0593   |
| 10    | 5 | 5 | 28.94405         | 25.1419        | 791.0381      | 764.5707   |
| 100   | 5 | 5 | 324.25255        | 318.0287       | 8591.90765    | 8376.8481  |
| 1000  | 5 | 5 | 3267.50065       | 3249.1087      | 93159.1726    | 85904.4203 |

*Note: Times are measured in milliseconds. Lower values indicate better performance. Benchmarked on 100 runs each. Julia code used `BenchmarkTools.jl` package where R code uses `microbenchmark` package. Effort was made to remove error-checking and compilation steps from R code to make comparison as relevant as possible.*

![Quadratic Filter Benchmark](benchmarks/results/scaling_comparison.png)


## Citations

If you use this package in your research, please cite the original
paper:

``` bibtex
@article{monfort2015quadratic,
  title = {A Quadratic Kalman Filter},
  journal = {Journal of Econometrics},
  volume = {187},
  number = {1},
  pages = {43-56},
  year = {2015},
  issn = {0304-4076},
  doi = {https://doi.org/10.1016/j.jeconom.2015.01.003},
  url = {https://www.sciencedirect.com/science/article/pii/S0304407615000123},
  author = {Alain Monfort and Jean-Paul Renne and Guillaume Roussellet},
}
```

## License

MIT License

Copyright (c) 2025 David Leather

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
“Software”), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
