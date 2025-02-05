if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite")
}
library(jsonlite)

if (!requireNamespace("tidyverse", quietly = TRUE)) {
  install.packages("tidyverse")
}
library(tidyverse)

if (!requireNamespace("microbenchmark", quietly = TRUE)) {
  install.packages("microbenchmark")
}
library(microbenchmark)

# Check if we're in QuadraticKalman.jl root and navigate to benchmarks
if (basename(getwd()) == "QuadraticKalman.jl") {
  setwd("benchmarks")
}

# Check if we're in R/ root and navigate to ".."
if (basename(getwd()) == "r") {
  setwd("..")
}

# Source the R code
source("r/augmented_model.R")
source("r/checking_arguments.R")
source("r/loglik.R")
source("r/predict.R")
source("r/update.R")
source("r/quadratic_kalman_filtering.R")
source("r/quadratic_kalman_smoothing.R")
source("r/unconditional_moments.R")

# Read the simulated data
Y <- read.csv(paste0("data/scenario_", scenario_id, "_data.csv"))
params <- fromJSON(paste0("data/scenario_", scenario_id, "_params.json"))

# Convert parameters from JSON format to R matrices/arrays
N <- params$N
M <- params$M
mu <- matrix(unlist(params$mu), ncol = 1)
Phi <- matrix(unlist(params$Phi), nrow = N, byrow = TRUE)
Omega <- matrix(unlist(params$Omega), nrow = N, byrow = TRUE)
A <- matrix(unlist(params$A), ncol = 1)
B <- matrix(unlist(params$B), nrow = M, byrow = TRUE)
D <- matrix(unlist(params$D), nrow = M, byrow = TRUE)
alpha <- matrix(unlist(params$alpha), nrow = M, byrow = TRUE)

# Convert C array (Need to hand code)
C <- array(NA, dim = c(N, N, M))
for (i in 1:M) {
  C[,,i] <- diag(0.01, nrow = N, ncol = N)
}
# Convert data to matrices
Y <- as.matrix(t(Y))

# Get dimensions
n <- nrow(Phi)  # state dimension
m <- nrow(B)    # measurement dimension
T <- nrow(Y)    # time periods

# Initialize filter using unconditional moments
init <- initialize.filter(mu, Phi, Omega)
Z0 <- init$Z0
V0 <- init$V0
observable <- t(Y[2:T,])
setwd("r/")

bm_results <- microbenchmark(
  {
    # Run the Quadratic Kalman Filter
    qkf_results <- QKF(
        Z0 = Z0,
        V0 = V0,
        Mu.t = mu,
        Phi.t = Phi,
        Omega.t = Omega,
        A.t = A,
        B.t = B,
        C.t = C,
        D.t = D,
        observable = observable)
  }, times = 100L)

median_time <- median(bm_results$time) / 1e6  # microbenchmark returns times in nanoseconds
minimum_time <- min(bm_results$time) / 1e6
setwd("../")
